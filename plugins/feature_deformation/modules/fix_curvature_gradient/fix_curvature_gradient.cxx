#include "fix_curvature_gradient.h"

#include "common/hash.h"

#include "../feature_deformation/curvature.h"
#include "../feature_deformation/gradient.h"
#include "../feature_deformation/grid.h"

#include "vtkAlgorithm.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

vtkStandardNewMacro(fix_curvature_gradient);

fix_curvature_gradient::fix_curvature_gradient() : hash(-1)
{
    this->SetNumberOfInputPorts(1);
    this->SetNumberOfOutputPorts(1);
}

fix_curvature_gradient::~fix_curvature_gradient()
{
}

int fix_curvature_gradient::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    const std::array<double, 2> time_range{ 0.0, static_cast<double>(this->NumSteps) };

    std::vector<double> timesteps(this->NumSteps + 1);
    std::iota(timesteps.begin(), timesteps.end(), 0);

    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(), timesteps.data(), static_cast<int>(timesteps.size()));

    this->results.resize(this->NumSteps + 1uLL);

    return 1;
}

int fix_curvature_gradient::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
        return 1;
    }

    return 0;
}

int fix_curvature_gradient::RequestData(vtkInformation*, vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    auto original_grid = vtkImageData::GetData(input_vector[0]);

    if (original_grid == nullptr)
    {
        std::cerr << std::endl << "All input slots must be connected." << std::endl;
        return 0;
    }

    auto vector_field_original = GetInputArrayToProcess(0, original_grid);
    auto vector_field_deformed = GetInputArrayToProcess(1, original_grid);

    if (vector_field_original == nullptr || vector_field_deformed == nullptr)
    {
        std::cerr << std::endl << "Input vector fields must be provided." << std::endl;
        return 0;
    }

    const auto hash = joaat_hash(this->NumSteps, this->StepSize, this->Error,
        vector_field_original->GetMTime(), vector_field_deformed->GetMTime());

    if (hash != this->hash)
    {
        compute_finite_differences(original_grid, vector_field_original, vector_field_deformed);

        this->hash = hash;
    }

    const auto time = output_vector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    const auto time_step = std::min(std::max(static_cast<std::size_t>(time), 0uLL), static_cast<std::size_t>(this->NumSteps));

    auto output_grid = vtkImageData::GetData(output_vector);
    output_grid->ShallowCopy(this->results[static_cast<std::size_t>(time_step)]);

    std::cout << "Showing step: " << static_cast<std::size_t>(time_step) << std::endl;

    return 1;
}

void fix_curvature_gradient::compute_finite_differences(vtkImageData* original_grid,
    vtkDataArray* vector_field_original, vtkDataArray* vector_field_deformed)
{
    // Normalize input fields
    auto vector_field_original_normalized = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field_original_normalized->DeepCopy(vector_field_original);

    auto vector_field = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field->DeepCopy(vector_field_deformed);

    Eigen::Vector3d u;

    for (int i = 0; i < vector_field_original->GetNumberOfTuples(); ++i)
    {
        vector_field_original_normalized->GetTuple(i, u.data());
        u.normalize();
        vector_field_original_normalized->SetTuple(i, u.data());

        vector_field->GetTuple(i, u.data());
        u.normalize();
        vector_field->SetTuple(i, u.data());
    }

    // Calculate initial gradient differences
    const grid original_vector_field(original_grid, vector_field_original_normalized);

    auto original_curvature = curvature_and_torsion(original_vector_field, gradient_method_t::differences);
    original_curvature.curvature_gradient->SetName("Curvature Gradient (Original)");

    auto deformed_curvature = curvature_and_torsion(grid(original_vector_field, vector_field), gradient_method_t::differences);

    // Calculate initial error
    vtkSmartPointer<vtkDoubleArray> errors;
    double error_avg{}, error_max{};

    std::tie(errors, error_avg, error_max) = calculate_error_field(original_curvature, deformed_curvature);

    const auto original_error_avg = error_avg;
    const auto original_error_max = error_max;

    // Set initial output
    this->results[0] = vtkSmartPointer<vtkImageData>::New();
    this->results[0]->CopyStructure(original_grid);

    output_copy(this->results[0], vector_field,
        errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
        deformed_curvature.curvature_gradient, deformed_curvature.torsion,
        deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient,
        original_curvature.curvature_gradient);

    // Iteratively solve
    const auto step = solve(original_vector_field, vector_field, original_curvature, deformed_curvature, error_max);

    // If converged or stopped, later results stay the same
    if (error_max <= this->Error)
    {
        for (std::size_t i = step + 1uLL; i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[step];
        }

        std::cout << "Optimization converged." << std::endl;
    }
}

int fix_curvature_gradient::solve(const grid& original_vector_field, vtkSmartPointer<vtkDoubleArray> vector_field,
    const curvature_and_torsion_t& original_curvature, curvature_and_torsion_t deformed_curvature, double error_max)
{
    // Grid information
    const bool twoD = original_vector_field.dimensions()[2] == 1;
    const auto h = original_vector_field.get_spacing()[0];
    const auto num_dimensions = twoD ? 2 : 3;
    const auto& dimensions = original_vector_field.dimensions();
    const auto num_nodes = static_cast<std::size_t>(dimensions[0]) * dimensions[1] * dimensions[2];

    auto get_index = [&dimensions](int i, int j, int k, int d) -> Eigen::Index
    { return i + dimensions[0] * (j + dimensions[1] * (k + dimensions[2] * static_cast<Eigen::Index>(d))); };

    // Iteratively solve finite differences Ax = b
    int step = 0;

    Eigen::Vector3d u;
    Eigen::Matrix3d jacobian;
    Eigen::Vector3d gradient_ux_uy, gradient_ux_sqr, gradient_uy_sqr, gradient_alpha, gradient_beta;

    vtkSmartPointer<vtkDoubleArray> errors;
    double error_avg{};

    for (; step < this->NumSteps /* && error_max > this->Error*/; ++step)
    {
        // Create right hand side vector
        Eigen::VectorXd b;
        b.resize(num_dimensions * num_nodes);

        for (int d = 0; d < num_dimensions; ++d)
        {
            for (int k = 0; k < dimensions[2]; ++k)
            {
                for (int j = 0; j < dimensions[1]; ++j)
                {
                    for (int i = 0; i < dimensions[0]; ++i)
                    {
                        b(get_index(i, j, k, d)) = original_curvature.curvature_gradient->GetComponent(get_index(i, j, k, 0), d)
                            - deformed_curvature.curvature_gradient->GetComponent(get_index(i, j, k, 0), d);
                    }
                }
            }
        }

        // Compute derivatives of different fields
        auto jacobian_of_vector_field = gradient_field(grid(original_vector_field, vector_field), gradient_method_t::differences);

        auto ux_uy = vtkSmartPointer<vtkDoubleArray>::New();
        ux_uy->SetNumberOfComponents(1);
        ux_uy->SetNumberOfTuples(num_nodes);

        auto ux_sqr = vtkSmartPointer<vtkDoubleArray>::New();
        ux_sqr->SetNumberOfComponents(1);
        ux_sqr->SetNumberOfTuples(num_nodes);

        auto uy_sqr = vtkSmartPointer<vtkDoubleArray>::New();
        uy_sqr->SetNumberOfComponents(1);
        uy_sqr->SetNumberOfTuples(num_nodes);

        auto alphas = vtkSmartPointer<vtkDoubleArray>::New();
        alphas->SetNumberOfComponents(1);
        alphas->SetNumberOfTuples(num_nodes);

        auto betas = vtkSmartPointer<vtkDoubleArray>::New();
        betas->SetNumberOfComponents(1);
        betas->SetNumberOfTuples(num_nodes);

        for (std::size_t i = 0; i < num_nodes; ++i)
        {
            vector_field->GetTuple(i, u.data());
            jacobian_of_vector_field->GetTuple(i, jacobian.data());

            ux_uy->SetValue(i, u(0) * u(1));
            ux_sqr->SetValue(i, u(0) * u(0));
            uy_sqr->SetValue(i, u(1) * u(1));

            alphas->SetValue(i, u(1) * jacobian(1, 1) + 2.0 * u(0) * jacobian(1, 0) - u(1) * jacobian(0, 0));
            betas->SetValue(i, u(0) * jacobian(1, 1) - 2.0 * u(1) * jacobian(0, 1) - u(0) * jacobian(0, 0));
        }

        auto gradients_ux_uy = gradient_field(grid(original_vector_field, ux_uy), gradient_method_t::differences);
        auto gradients_ux_sqr = gradient_field(grid(original_vector_field, ux_sqr), gradient_method_t::differences);
        auto gradients_uy_sqr = gradient_field(grid(original_vector_field, uy_sqr), gradient_method_t::differences);
        auto gradients_alpha = gradient_field(grid(original_vector_field, alphas), gradient_method_t::differences);
        auto gradients_beta = gradient_field(grid(original_vector_field, betas), gradient_method_t::differences);

        // Create finite differences matrix
        Eigen::SparseMatrix<double> A;
        A.resize(num_dimensions * num_nodes, num_dimensions * num_nodes);

        if (twoD)
        {
            for (int j = 0; j < dimensions[1]; ++j)
            {
                for (int i = 0; i < dimensions[0]; ++i)
                {
                    // Position information
                    auto index = get_index(i, j, 0, 0);

                    const auto left = (i == 0);
                    const auto right = (i == dimensions[0] - 1);

                    const auto bottom = (j == 0);
                    const auto top = (j == dimensions[1] - 1);

                    // Get pre-computed field values
                    vector_field->GetTuple(index, u.data());
                    gradients_ux_uy->GetTuple(index, gradient_ux_uy.data());
                    gradients_ux_sqr->GetTuple(index, gradient_ux_sqr.data());
                    gradients_uy_sqr->GetTuple(index, gradient_uy_sqr.data());
                    gradients_alpha->GetTuple(index, gradient_alpha.data());
                    gradients_beta->GetTuple(index, gradient_beta.data());

                    const auto alpha = alphas->GetValue(index);
                    const auto beta = betas->GetValue(index);

                    // Derivative of the curvature in x direction
                    {
                        const auto row_index = get_index(i, j, 0, 0);

                        {
                            // v
                            {
                                // .x
                                A.coeffRef(row_index, get_index(i, j, 0, 0)) += gradient_alpha[0]; // i, j, X

                                // .y
                                A.coeffRef(row_index, get_index(i, j, 0, 1)) += gradient_beta[0]; // i, j, Y
                            }

                            // dv/dx
                            {
                                const auto offset_left = left ? 0 : 1;
                                const auto offset_right = right ? 0 : 1;
                                const auto denom = ((left || right) ? 1.0 : 2.0) * h;

                                // .x
                                A.coeffRef(row_index, get_index(i - offset_left, j, 0, 0)) -= (alpha - gradient_ux_uy[0]) / denom; // i - 1, j, X
                                A.coeffRef(row_index, get_index(i + offset_right, j, 0, 0)) += (alpha - gradient_ux_uy[0]) / denom; // i + 1, j, X

                                // .y
                                A.coeffRef(row_index, get_index(i - offset_left, j, 0, 1)) -= (beta + gradient_ux_sqr[0]) / denom; // i - 1, j, Y
                                A.coeffRef(row_index, get_index(i + offset_right, j, 0, 1)) += (beta + gradient_ux_sqr[0]) / denom; // i + 1, j, Y
                            }

                            // dv/dy
                            {
                                const auto offset_bottom = bottom ? 0 : 1;
                                const auto offset_top = top ? 0 : 1;
                                const auto denom = ((bottom || top) ? 1.0 : 2.0) * h;

                                // .x
                                A.coeffRef(row_index, get_index(i, j - offset_bottom, 0, 0)) += gradient_uy_sqr[0] / denom; // i, j - 1, X
                                A.coeffRef(row_index, get_index(i, j + offset_top, 0, 0)) -= gradient_uy_sqr[0] / denom; // i, j + 1, X

                                // .y
                                A.coeffRef(row_index, get_index(i, j - offset_bottom, 0, 1)) -= gradient_ux_uy[0] / denom; // i, j - 1, Y
                                A.coeffRef(row_index, get_index(i, j + offset_top, 0, 1)) += gradient_ux_uy[0] / denom; // i, j + 1, Y
                            }

                            // dv/dx�
                            {
                                const auto offset = left ? 1 : (right ? -1 : 0);
                                const auto denom = h * h;

                                // .x
                                A.coeffRef(row_index, get_index(i + offset, j, 0, 0)) += (2.0 * u[0] * u[1]) / denom; // i, j, X

                                A.coeffRef(row_index, get_index(i + offset - 1, j, 0, 0)) -= (u[0] * u[1]) / denom; // i - 1, j, X
                                A.coeffRef(row_index, get_index(i + offset + 1, j, 0, 0)) -= (u[0] * u[1]) / denom; // i + 1, j, X

                                // .y
                                A.coeffRef(row_index, get_index(i + offset, j, 0, 1)) -= (2.0 * u[0] * u[0]) / denom; // i, j, Y

                                A.coeffRef(row_index, get_index(i + offset - 1, j, 0, 1)) += (u[0] * u[0]) / denom; // i - 1, j, Y
                                A.coeffRef(row_index, get_index(i + offset + 1, j, 0, 1)) += (u[0] * u[0]) / denom; // i + 1, j, Y
                            }

                            // dv/dxy
                            {
                                const auto offset_left = left ? 0 : 1;
                                const auto offset_right = right ? 0 : 1;
                                const auto offset_bottom = bottom ? 0 : 1;
                                const auto offset_top = top ? 0 : 1;
                                const auto denom = ((left || right) ? 1.0 : 2.0) * ((bottom || top) ? 1.0 : 2.0) * h * h;

                                // .x
                                A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 0, 0)) -= (u[1] * u[1]) / denom; // i - 1, j - 1, X
                                A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 0, 0)) += (u[1] * u[1]) / denom; // i + 1, j - 1, X
                                A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 0, 0)) += (u[1] * u[1]) / denom; // i - 1, j + 1, X
                                A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 0, 0)) -= (u[1] * u[1]) / denom; // i + 1, j + 1, X

                                // .y
                                A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 0, 1)) += (u[0] * u[1]) / denom; // i - 1, j - 1, Y
                                A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 0, 1)) -= (u[0] * u[1]) / denom; // i + 1, j - 1, Y
                                A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 0, 1)) -= (u[0] * u[1]) / denom; // i - 1, j + 1, Y
                                A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 0, 1)) += (u[0] * u[1]) / denom; // i + 1, j + 1, Y
                            }
                        }
                    }

                    // Derivative of the curvature in y direction
                    {
                        const auto row_index = get_index(i, j, 0, 1);

                        {
                            // v
                            {
                                // .x
                                A.coeffRef(row_index, get_index(i, j, 0, 0)) += gradient_alpha[1]; // i, j, X

                                // .y
                                A.coeffRef(row_index, get_index(i, j, 0, 1)) += gradient_beta[1]; // i, j, Y
                            }

                            // dv/dx
                            {
                                const auto offset_left = left ? 0 : 1;
                                const auto offset_right = right ? 0 : 1;
                                const auto denom = ((left || right) ? 1.0 : 2.0) * h;

                                // .x
                                A.coeffRef(row_index, get_index(i - offset_left, j, 0, 0)) += gradient_ux_uy[1] / denom; // i - 1, j, X
                                A.coeffRef(row_index, get_index(i + offset_right, j, 0, 0)) -= gradient_ux_uy[1] / denom; // i + 1, j, X

                                // .y
                                A.coeffRef(row_index, get_index(i - offset_left, j, 0, 1)) -= gradient_ux_sqr[1] / denom; // i - 1, j, Y
                                A.coeffRef(row_index, get_index(i + offset_right, j, 0, 1)) += gradient_ux_sqr[1] / denom; // i + 1, j, Y
                            }

                            // dv/dy
                            {
                                const auto offset_bottom = bottom ? 0 : 1;
                                const auto offset_top = top ? 0 : 1;
                                const auto denom = ((bottom || top) ? 1.0 : 2.0) * h;

                                // .x
                                A.coeffRef(row_index, get_index(i, j - offset_bottom, 0, 0)) -= (alpha - gradient_uy_sqr[1]) / denom; // i, j - 1, X
                                A.coeffRef(row_index, get_index(i, j + offset_top, 0, 0)) += (alpha - gradient_uy_sqr[1]) / denom; // i, j + 1, X

                                // .y
                                A.coeffRef(row_index, get_index(i, j - offset_bottom, 0, 1)) -= (beta + gradient_ux_uy[1]) / denom; // i, j - 1, Y
                                A.coeffRef(row_index, get_index(i, j + offset_top, 0, 1)) += (beta + gradient_ux_uy[1]) / denom; // i, j + 1, Y
                            }

                            // dv/dy�
                            {
                                const auto offset = bottom ? 1 : (top ? -1 : 0);
                                const auto denom = h * h;

                                // .x
                                A.coeffRef(row_index, get_index(i, j + offset, 0, 0)) += (2.0 * u[1] * u[1]) / denom; // i, j, X

                                A.coeffRef(row_index, get_index(i, j + offset - 1, 0, 0)) -= (u[1] * u[1]) / denom; // i, j - 1, X
                                A.coeffRef(row_index, get_index(i, j + offset + 1, 0, 0)) -= (u[1] * u[1]) / denom; // i, j + 1, X

                                // .y
                                A.coeffRef(row_index, get_index(i, j + offset, 0, 1)) -= (2.0 * u[0] * u[1]) / denom; // i, j, Y

                                A.coeffRef(row_index, get_index(i, j + offset - 1, 0, 1)) += (u[0] * u[1]) / denom; // i, j - 1, Y
                                A.coeffRef(row_index, get_index(i, j + offset + 1, 0, 1)) += (u[0] * u[1]) / denom; // i, j + 1, Y
                            }

                            // dv/dxy
                            {
                                const auto offset_left = left ? 0 : 1;
                                const auto offset_right = right ? 0 : 1;
                                const auto offset_bottom = bottom ? 0 : 1;
                                const auto offset_top = top ? 0 : 1;
                                const auto denom = ((left || right) ? 1.0 : 2.0) * ((bottom || top) ? 1.0 : 2.0) * h * h;

                                // .x
                                A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 0, 0)) -= (u[0] * u[1]) / denom; // i - 1, j - 1, X
                                A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 0, 0)) += (u[0] * u[1]) / denom; // i + 1, j - 1, X
                                A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 0, 0)) += (u[0] * u[1]) / denom; // i - 1, j + 1, X
                                A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 0, 0)) -= (u[0] * u[1]) / denom; // i + 1, j + 1, X

                                // .y
                                A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 0, 1)) += (u[0] * u[0]) / denom; // i - 1, j - 1, Y
                                A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 0, 1)) -= (u[0] * u[0]) / denom; // i + 1, j - 1, Y
                                A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 0, 1)) -= (u[0] * u[0]) / denom; // i - 1, j + 1, Y
                                A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 0, 1)) += (u[0] * u[0]) / denom; // i + 1, j + 1, Y
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // TODO: 3D?
        }

        // Solve for x and calculate (maximum) step size
        A.makeCompressed();

        const Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
        const Eigen::VectorXd x = solver.solve(b);

        const Eigen::VectorXd residual_vector = A * x - b;

        auto residuals = vtkSmartPointer<vtkDoubleArray>::New();
        residuals->SetName("Residual");
        residuals->SetNumberOfComponents(num_dimensions);
        residuals->SetNumberOfTuples(num_nodes);

        auto step_size = this->StepSize;

        for (std::size_t i = 0; i < num_nodes; ++i)
        {
            //auto length = 0.0;

            for (int d = 0; d < num_dimensions; ++d)
            {
                residuals->SetComponent(i, d, std::abs(residual_vector(i + d * num_nodes)));

                //length += std::abs(vector_field->GetComponent(i, d) * x(i + d * num_nodes));
            }

            //step_size = std::min(step_size, this->StepSize / (2.0 * length));
        }

        std::cout << " Step size: " << step_size << std::endl;

        // Update result
        auto update = vtkSmartPointer<vtkDoubleArray>::New();
        update->SetName("Update");
        update->SetNumberOfComponents(3);
        update->SetNumberOfTuples(num_nodes);
        update->Fill(0.0);

        for (int d = 0; d < num_dimensions; ++d)
        {
            for (std::size_t i = 0; i < num_nodes; ++i)
            {
                update->SetComponent(i, d, x(i + d * num_nodes));
                vector_field->SetComponent(i, d, vector_field->GetComponent(i, d) + step_size * x(i + d * num_nodes));
            }
        }

        // Normalize vector field
        for (std::size_t i = 0; i < num_nodes; ++i)
        {
            vector_field->GetTuple(i, u.data());
            u.normalize();
            vector_field->SetTuple(i, u.data());
        }

        // Prepare for next time step and output (intermediate) results
        deformed_curvature = curvature_and_torsion(grid(original_vector_field, vector_field), gradient_method_t::differences);

        std::tie(errors, error_avg, error_max) = calculate_error_field(original_curvature, deformed_curvature);

        this->results[step + 1uLL] = vtkSmartPointer<vtkImageData>::New();
        this->results[step + 1uLL]->CopyStructure(this->results[0]);

        output_copy(this->results[step + 1uLL], vector_field, residuals,
            errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
            deformed_curvature.curvature_gradient, deformed_curvature.torsion,
            deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient,
            original_curvature.curvature_gradient, update);
    }

    return step;
}

double fix_curvature_gradient::calculate_error(const int index, const int index_block,
    const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature) const
{
    Eigen::VectorXd original_gradient, deformed_gradient;
    original_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);
    deformed_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);

    original_curvature.curvature_gradient->GetTuple(index, original_gradient.data());
    deformed_curvature.curvature_gradient->GetTuple(index_block, deformed_gradient.data());

    return (deformed_gradient - original_gradient).norm();
}

std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> fix_curvature_gradient::calculate_error_field(
    const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature) const
{
    auto errors = vtkSmartPointer<vtkDoubleArray>::New();
    errors->SetName("Error");
    errors->SetNumberOfComponents(1);
    errors->SetNumberOfTuples(original_curvature.curvature_gradient->GetNumberOfTuples());

    double error_max = std::numeric_limits<double>::min();
    std::vector<double> error_avgs(original_curvature.curvature_gradient->GetNumberOfTuples());

    for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
    {
        const auto error = calculate_error(i, i, original_curvature, deformed_curvature);

        errors->SetValue(i, error);

        error_max = std::max(error_max, error);
        error_avgs[i] = error;
    }

    // Pyramid sum for more accuracy
    for (vtkIdType offset = 1; offset < original_curvature.curvature_gradient->GetNumberOfTuples(); offset *= 2)
    {
        for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); i += 2LL * offset)
        {
            if (i + offset < original_curvature.curvature_gradient->GetNumberOfTuples())
            {
                error_avgs[i] += error_avgs[i + offset];
            }
        }
    }

    error_avgs[0] /= original_curvature.curvature_gradient->GetNumberOfTuples();

    return std::make_tuple(errors, error_avgs[0], error_max);
}