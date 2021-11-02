#include "optimizer.h"

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
#include "vtkResampleWithDataSet.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredGrid.h"

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

vtkStandardNewMacro(optimizer);

optimizer::optimizer() : hash(-1)
{
    this->SetNumberOfInputPorts(2);
    this->SetNumberOfOutputPorts(1);
}

optimizer::~optimizer()
{
}

int optimizer::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    const std::array<double, 2> time_range{ 0.0, static_cast<double>(this->NumSteps) };

    std::vector<double> timesteps(this->NumSteps + 1);
    std::iota(timesteps.begin(), timesteps.end(), 0);

    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(), timesteps.data(), static_cast<int>(timesteps.size()));

    this->results.resize(this->NumSteps + 1uLL);

    return 1;
}

int optimizer::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkStructuredGrid");
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkStructuredGrid");
        return 1;
    }

    return 0;
}

int optimizer::RequestData(vtkInformation*, vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    auto original_grid = vtkStructuredGrid::GetData(input_vector[0]);
    auto deformed_grid = vtkStructuredGrid::GetData(input_vector[1]);

    if (original_grid == nullptr || deformed_grid == nullptr)
    {
        std::cerr << std::endl << "All input slots must be connected." << std::endl;
        return 0;
    }

    auto vector_field_original = GetInputArrayToProcess(0, original_grid);

    if (vector_field_original == nullptr)
    {
        std::cerr << std::endl << "Input vector field must be provided." << std::endl;
        return 0;
    }

    const auto hash = joaat_hash(this->ErrorDefinition, this->Method, this->NumSteps, this->StepSize,
        this-StepSizeMethod, this->StepSizeControl, this->Error, this->StepSizeMin, this->StepSizeMax,
        this->LineSearchSteps, this->GradientMethod, this->GradientKernel, this->GradientStep, this->CheckWolfe,
        vector_field_original->GetMTime(), deformed_grid->GetMTime());

    if (hash != this->hash)
    {
        switch (static_cast<method_t>(this->Method))
        {
        case method_t::gradient:
        case method_t::nonlinear_conjugate:
            compute_gradient_descent(original_grid, deformed_grid, vector_field_original);

            break;
        case method_t::finite_differences:
        default:
            compute_finite_differences(original_grid, deformed_grid, vector_field_original);
        }

        this->hash = hash;
    }

    const auto time = output_vector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    const auto time_step = std::min(std::max(static_cast<std::size_t>(time), 0uLL), static_cast<std::size_t>(this->NumSteps));

    auto output_grid = vtkStructuredGrid::GetData(output_vector);
    output_grid->ShallowCopy(this->results[static_cast<std::size_t>(time_step)]);

    if (!this->CSVOutput) std::cout << "Showing step: " << static_cast<std::size_t>(time_step) << std::endl;

    return 1;
}

void optimizer::compute_finite_differences(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
    vtkDataArray* vector_field_original)
{
    // Get input grid and data
    const grid original_vector_field(original_grid, vector_field_original);

    const std::array<int, 3> dimension = original_vector_field.dimensions();
    const bool twoD = dimension[2] == 1;
    const auto num_nodes = original_grid->GetNumberOfPoints();

    if (num_nodes != deformed_grid->GetNumberOfPoints())
    {
        std::cerr << "Number of grid nodes must match." << std::endl;
        return;
    }

    std::array<double, 3> pointA{}, pointB{};
    original_grid->GetPoint(0, pointA.data());
    original_grid->GetPoint(1, pointB.data());

    const auto h = pointB[0] - pointA[0];

    // Get initial node positions, and compute deformation
    auto deformed_positions = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_positions->SetName("Deformed Position");
    deformed_positions->SetNumberOfComponents(3);
    deformed_positions->SetNumberOfTuples(num_nodes);

    auto positions = vtkSmartPointer<vtkDoubleArray>::New();
    positions->SetNumberOfComponents(3);
    positions->SetNumberOfTuples(num_nodes);

    std::array<double, 3> point{};

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        original_grid->GetPoint(i, point.data());
        positions->SetTuple(i, point.data());

        deformed_grid->GetPoint(i, point.data());
        deformed_positions->SetTuple(i, point.data());
    }

    const grid deformation_field(original_grid, deformed_positions);

    auto jacobian_field = gradient_field(deformation_field, gradient_method_t::differences, 0);
    jacobian_field->SetName("Jacobian");

    auto vector_field_deformed = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field_deformed->SetName("Vector Field");
    vector_field_deformed->SetNumberOfComponents(3);
    vector_field_deformed->SetNumberOfTuples(num_nodes);

    Eigen::Matrix3d jacobian;
    Eigen::Vector3d vector;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        jacobian_field->GetTuple(i, jacobian.data());
        vector_field_original->GetTuple(i, vector.data());

        vector = jacobian * vector;

        vector_field_deformed->SetTuple(i, vector.data());
    }

    // Calculate initial gradient difference
    const auto original_curvature = curvature_and_torsion(original_vector_field, gradient_method_t::differences, 0);

    auto original_curvature_gradients_deformed = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvature_gradients_deformed->SetName("Curvature Gradient (Original)");
    original_curvature_gradients_deformed->SetNumberOfComponents(3);
    original_curvature_gradients_deformed->SetNumberOfTuples(num_nodes);

    Eigen::Vector3d original_gradient;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        original_curvature.curvature_gradient->GetTuple(i, original_gradient.data());
        jacobian_field->GetTuple(i, jacobian.data());

        original_gradient = jacobian * original_gradient;

        original_curvature_gradients_deformed->SetTuple(i, original_gradient.data());
    }

    // Resample vector field on original regular grid
    deformed_grid->GetPointData()->AddArray(vector_field_deformed);
    deformed_grid->GetPointData()->AddArray(original_curvature_gradients_deformed);

    auto resampler = vtkSmartPointer<vtkResampleWithDataSet>::New();
    resampler->SetInputDataObject(original_grid);
    resampler->SetSourceData(deformed_grid);
    resampler->Update();

    auto vector_field = vtkSmartPointer<vtkDoubleArray>::New();
    auto original_curvature_gradients = vtkSmartPointer<vtkDoubleArray>::New();

    vector_field->DeepCopy(static_cast<vtkStructuredGrid*>(resampler->GetOutput())->GetPointData()->GetArray("Vector Field"));
    original_curvature_gradients->DeepCopy(static_cast<vtkStructuredGrid*>(resampler->GetOutput())->GetPointData()->GetArray("Curvature Gradient (Original)"));

    auto valid_field = static_cast<vtkStructuredGrid*>(resampler->GetOutput())->GetPointData()->GetArray("vtkValidPointMask");

    auto get_index = [&dimension](int i, int j, int k, int d) -> Eigen::Index
        { return i + dimension[0] * (j + dimension[1] * (k + dimension[2] * static_cast<Eigen::Index>(d))); };

    Eigen::Vector3d vector_temp, curvature_gradient, curvature_gradient_temp;

    for (int k = 0; k < dimension[2]; ++k)
    {
        for (int j = 0; j < dimension[1]; ++j)
        {
            for (int i = 0; i < dimension[0]; ++i)
            {
                const auto index = get_index(i, j, k, 0);

                // Fix vectors and curvature gradients at the boundaries by averaging their valid neighbors
                if (valid_field->GetComponent(index, 0) == 0.0)
                {
                    vector.setZero();
                    curvature_gradient.setZero();

                    int num = 0;

                    for (int kk = -1; kk <= 1; ++kk)
                    {
                        if (k + kk < 0 || k + kk >= dimension[2]) continue;

                        for (int jj = -1; jj <= 1; ++jj)
                        {
                            if (j + jj < 0 || j + jj >= dimension[1]) continue;

                            for (int ii = -1; ii <= 1; ++ii)
                            {
                                if (i + ii < 0 || i + ii >= dimension[0]) continue;

                                const auto neighbor_index = get_index(i + ii, j + jj, k + kk, 0);

                                if (valid_field->GetComponent(neighbor_index, 0) == 1.0)
                                {
                                    vector_field_deformed->GetTuple(neighbor_index, vector_temp.data());
                                    original_curvature_gradients_deformed->GetTuple(neighbor_index, curvature_gradient_temp.data());

                                    vector += vector_temp;
                                    curvature_gradient += curvature_gradient_temp;

                                    ++num;
                                }
                            }
                        }
                    }

                    vector /= num;
                    curvature_gradient /= num;

                    vector_field->SetTuple(index, vector.data());
                    original_curvature_gradients->SetTuple(index, curvature_gradient.data());
                }
            }
        }
    }

    // Calculate initial error
    auto deformed_curvature = curvature_and_torsion(grid(original_grid, vector_field), gradient_method_t::differences, 0);

    vtkSmartPointer<vtkDoubleArray> errors;
    double error_avg{}, error_max{};

    std::tie(errors, error_avg, error_max)
        = calculate_error_field(original_curvature, deformed_curvature, nullptr, error_definition_t::vector_difference);

    const auto original_error_avg = error_avg;
    const auto original_error_max = error_max;

    // Set initial output
    this->results[0] = create_output(dimension, positions);

    output_copy(this->results[0], vector_field, jacobian_field,
        errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
        deformed_curvature.curvature_gradient, deformed_curvature.torsion,
        deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient,
        original_curvature_gradients, valid_field);

    // Iteratively solve finite differences Ax = b
    auto vector_field_updated = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field_updated->DeepCopy(vector_field);

    int step = 0;

    for (; step < this->NumSteps && error_max > this->Error; ++step)
    {
        // Create right hand side vector
        Eigen::VectorXd b;
        b.resize((twoD ? 2 : 3) * num_nodes);

        for (int d = 0; d < (twoD ? 2 : 3); ++d)
        {
            for (int i = 0; i < num_nodes; ++i)
            {
                b(i + d * num_nodes) = original_curvature_gradients->GetComponent(i, d) - deformed_curvature.curvature_gradient->GetComponent(i, d);
            }
        }

        // Temporary variables
        Eigen::Vector3d u;
        Eigen::Matrix3d jacobian;
        Eigen::Vector3d gradient_ux_uy, gradient_ux_sqr, gradient_uy_sqr, gradient_alpha, gradient_beta;

        // Normalize vector field
        for (int i = 0; i < num_nodes; ++i)
        {
            vector_field->GetTuple(i, u.data());
            u.normalize();
            vector_field->SetTuple(i, u.data());
        }

        // Compute derivatives of different fields
        auto jacobian_of_vector_field = gradient_field(grid(original_grid, vector_field), gradient_method_t::differences, 0);

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

        for (int i = 0; i < num_nodes; ++i)
        {
            vector_field->GetTuple(i, u.data());
            jacobian_of_vector_field->GetTuple(i, jacobian.data());

            ux_uy->SetValue(i, u(0) * u(1));
            ux_sqr->SetValue(i, u(0) * u(0));
            uy_sqr->SetValue(i, u(1) * u(1));

            alphas->SetValue(i, u(1) * jacobian(1, 1) + 2.0 * u(0) * jacobian(1, 0) - u(1) * jacobian(0, 0));
            betas->SetValue(i, u(0) * jacobian(1, 1) - 2.0 * u(1) * jacobian(0, 1) - u(0) * jacobian(0, 0));
        }

        auto gradients_ux_uy = gradient_field(grid(original_grid, ux_uy), gradient_method_t::differences, 0);
        auto gradients_ux_sqr = gradient_field(grid(original_grid, ux_sqr), gradient_method_t::differences, 0);
        auto gradients_uy_sqr = gradient_field(grid(original_grid, uy_sqr), gradient_method_t::differences, 0);
        auto gradients_alpha = gradient_field(grid(original_grid, alphas), gradient_method_t::differences, 0);
        auto gradients_beta = gradient_field(grid(original_grid, betas), gradient_method_t::differences, 0);

        // Create finite differences matrix
        Eigen::SparseMatrix<double> A;
        A.resize((twoD ? 2 : 3) * num_nodes, (twoD ? 2 : 3) * num_nodes);

        if (twoD)
        {
            auto get_index = [&dimension](int i, int j, int d) -> Eigen::Index
                { return i + dimension[0] * (j + dimension[1] * static_cast<Eigen::Index>(d)); };

            for (int j = 0; j < dimension[1]; ++j)
            {
                for (int i = 0; i < dimension[0]; ++i)
                {
                    // Position information
                    auto index = get_index(i, j, 0);

                    const auto left = (i == 0);
                    const auto right = (i == dimension[0] - 1);

                    const auto bottom = (j == 0);
                    const auto top = (j == dimension[1] - 1);

                    // Get pre-computed field values
                    vector_field->GetTuple(index, u.data());
                    jacobian_of_vector_field->GetTuple(index, jacobian.data());
                    gradients_ux_uy->GetTuple(index, gradient_ux_uy.data());
                    gradients_ux_sqr->GetTuple(index, gradient_ux_sqr.data());
                    gradients_uy_sqr->GetTuple(index, gradient_uy_sqr.data());
                    gradients_alpha->GetTuple(index, gradient_alpha.data());
                    gradients_beta->GetTuple(index, gradient_beta.data());

                    const auto alpha = alphas->GetValue(index);
                    const auto beta = betas->GetValue(index);

                    // In x direction
                    {
                        const auto row_index = get_index(i, j, 0);

                        // v
                        {
                            // .x
                            A.coeffRef(row_index, get_index(i, j, 0)) += gradient_alpha[0]; // i, j, X

                            // .y
                            A.coeffRef(row_index, get_index(i, j, 1)) += gradient_beta[0]; // i, j, Y
                        }

                        // dv/dx
                        {
                            const auto offset_left = left ? 0 : 1;
                            const auto offset_right = right ? 0 : 1;
                            const auto denom = ((left || right) ? 1.0 : 2.0) * h;

                            // .x
                            A.coeffRef(row_index, get_index(i - offset_left, j, 0)) -= (alpha - gradient_ux_uy[0]) / denom; // i - 1, j, X
                            A.coeffRef(row_index, get_index(i + offset_right, j, 0)) += (alpha - gradient_ux_uy[0]) / denom; // i + 1, j, X

                            // .y
                            A.coeffRef(row_index, get_index(i - offset_left, j, 1)) -= (beta + gradient_ux_sqr[0]) / denom; // i - 1, j, Y
                            A.coeffRef(row_index, get_index(i + offset_right, j, 1)) += (beta + gradient_ux_sqr[0]) / denom; // i + 1, j, Y
                        }

                        // dv/dy
                        {
                            const auto offset_bottom = bottom ? 0 : 1;
                            const auto offset_top = top ? 0 : 1;
                            const auto denom = ((bottom || top) ? 1.0 : 2.0) * h;

                            // .x
                            A.coeffRef(row_index, get_index(i, j - offset_bottom, 0)) += gradient_uy_sqr[0] / denom; // i, j - 1, X
                            A.coeffRef(row_index, get_index(i, j + offset_top, 0)) -= gradient_uy_sqr[0] / denom; // i, j + 1, X

                            // .y
                            A.coeffRef(row_index, get_index(i, j - offset_bottom, 1)) -= gradient_ux_uy[0] / denom; // i, j - 1, Y
                            A.coeffRef(row_index, get_index(i, j + offset_top, 1)) += gradient_ux_uy[0] / denom; // i, j + 1, Y
                        }

                        // dv/dx²
                        {
                            const auto offset = left ? 1 : (right ? -1 : 0);

                            // .x
                            A.coeffRef(row_index, get_index(i + offset, j, 0)) += (2.0 * u[0] * u[1]) / (h * h); // i, j, X

                            A.coeffRef(row_index, get_index(i + offset - 1, j, 0)) -= (u[0] * u[1]) / (h * h); // i - 1, j, X
                            A.coeffRef(row_index, get_index(i + offset + 1, j, 0)) -= (u[0] * u[1]) / (h * h); // i + 1, j, X

                            // .y
                            A.coeffRef(row_index, get_index(i + offset, j, 1)) -= (2.0 * u[0] * u[0]) / (h * h); // i, j, Y

                            A.coeffRef(row_index, get_index(i + offset - 1, j, 1)) += (u[0] * u[0]) / (h * h); // i - 1, j, Y
                            A.coeffRef(row_index, get_index(i + offset + 1, j, 1)) += (u[0] * u[0]) / (h * h); // i + 1, j, Y
                        }

                        // dv/dxy
                        {
                            const auto offset_left = left ? 0 : 1;
                            const auto offset_right = right ? 0 : 1;
                            const auto offset_bottom = bottom ? 0 : 1;
                            const auto offset_top = top ? 0 : 1;
                            const auto denom = ((left || right) ? 1.0 : 2.0) * ((bottom || top) ? 1.0 : 2.0) * h * h;

                            // .x
                            A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 0)) -= (u[1] * u[1]) / denom; // i - 1, j - 1, X
                            A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 0)) += (u[1] * u[1]) / denom; // i + 1, j - 1, X
                            A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 0)) += (u[1] * u[1]) / denom; // i - 1, j + 1, X
                            A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 0)) -= (u[1] * u[1]) / denom; // i + 1, j + 1, X

                            // .y
                            A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 1)) += (u[0] * u[1]) / denom; // i - 1, j - 1, Y
                            A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 1)) -= (u[0] * u[1]) / denom; // i + 1, j - 1, Y
                            A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 1)) -= (u[0] * u[1]) / denom; // i - 1, j + 1, Y
                            A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 1)) += (u[0] * u[1]) / denom; // i + 1, j + 1, Y
                        }
                    }

                    // In y direction
                    {
                        const auto row_index = get_index(i, j, 1);

                        // v
                        {
                            // .x
                            A.coeffRef(row_index, get_index(i, j, 0)) += gradient_alpha[1]; // i, j, X

                            // .y
                            A.coeffRef(row_index, get_index(i, j, 1)) += gradient_beta[1]; // i, j, Y
                        }

                        // dv/dx
                        {
                            const auto offset_left = left ? 0 : 1;
                            const auto offset_right = right ? 0 : 1;
                            const auto denom = ((left || right) ? 1.0 : 2.0) * h;

                            // .x
                            A.coeffRef(row_index, get_index(i - offset_left, j, 0)) += gradient_ux_uy[1] / denom; // i - 1, j, X
                            A.coeffRef(row_index, get_index(i + offset_right, j, 0)) -= gradient_ux_uy[1] / denom; // i + 1, j, X

                            // .y
                            A.coeffRef(row_index, get_index(i - offset_left, j, 1)) -= gradient_ux_sqr[1] / denom; // i - 1, j, Y
                            A.coeffRef(row_index, get_index(i + offset_right, j, 1)) += gradient_ux_sqr[1] / denom; // i + 1, j, Y
                        }

                        // dv/dy
                        {
                            const auto offset_bottom = bottom ? 0 : 1;
                            const auto offset_top = top ? 0 : 1;
                            const auto denom = ((bottom || top) ? 1.0 : 2.0) * h;

                            // .x
                            A.coeffRef(row_index, get_index(i, j - offset_bottom, 0)) -= (alpha - gradient_uy_sqr[1]) / denom; // i, j - 1, X
                            A.coeffRef(row_index, get_index(i, j + offset_top, 0)) += (alpha - gradient_uy_sqr[1]) / denom; // i, j + 1, X

                            // .y
                            A.coeffRef(row_index, get_index(i, j - offset_bottom, 1)) -= (beta + gradient_ux_uy[1]) / denom; // i, j - 1, Y
                            A.coeffRef(row_index, get_index(i, j + offset_top, 1)) += (beta + gradient_ux_uy[1]) / denom; // i, j + 1, Y
                        }

                        // dv/dy²
                        {
                            const auto offset = bottom ? 1 : (top ? -1 : 0);

                            // .x
                            A.coeffRef(row_index, get_index(i, j + offset, 0)) += (2.0 * u[1] * u[1]) / (h * h); // i, j, X

                            A.coeffRef(row_index, get_index(i, j + offset - 1, 0)) -= (u[1] * u[1]) / (h * h); // i, j - 1, X
                            A.coeffRef(row_index, get_index(i, j + offset + 1, 0)) -= (u[1] * u[1]) / (h * h); // i, j + 1, X

                            // .y
                            A.coeffRef(row_index, get_index(i, j + offset, 1)) -= (2.0 * u[0] * u[1]) / (h * h); // i, j, Y

                            A.coeffRef(row_index, get_index(i, j + offset - 1, 1)) += (u[0] * u[1]) / (h * h); // i, j - 1, Y
                            A.coeffRef(row_index, get_index(i, j + offset + 1, 1)) += (u[0] * u[1]) / (h * h); // i, j + 1, Y
                        }

                        // dv/dxy
                        {
                            const auto offset_left = left ? 0 : 1;
                            const auto offset_right = right ? 0 : 1;
                            const auto offset_bottom = bottom ? 0 : 1;
                            const auto offset_top = top ? 0 : 1;
                            const auto denom = ((left || right) ? 1.0 : 2.0) * ((bottom || top) ? 1.0 : 2.0) * h * h;

                            // .x
                            A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 0)) -= (u[0] * u[1]) / denom; // i - 1, j - 1, X
                            A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 0)) += (u[0] * u[1]) / denom; // i + 1, j - 1, X
                            A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 0)) += (u[0] * u[1]) / denom; // i - 1, j + 1, X
                            A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 0)) -= (u[0] * u[1]) / denom; // i + 1, j + 1, X

                            // .y
                            A.coeffRef(row_index, get_index(i - offset_left, j - offset_bottom, 1)) += (u[0] * u[0]) / denom; // i - 1, j - 1, Y
                            A.coeffRef(row_index, get_index(i + offset_right, j - offset_bottom, 1)) -= (u[0] * u[0]) / denom; // i + 1, j - 1, Y
                            A.coeffRef(row_index, get_index(i - offset_left, j + offset_top, 1)) -= (u[0] * u[0]) / denom; // i - 1, j + 1, Y
                            A.coeffRef(row_index, get_index(i + offset_right, j + offset_top, 1)) += (u[0] * u[0]) / denom; // i + 1, j + 1, Y
                        }
                    }
                }
            }
        }
        else
        {
            // TODO: 3D?
        }

        // Solve for x
        const Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
        const Eigen::VectorXd x = solver.solve(b);

        // Update result
        for (int d = 0; d < (twoD ? 2 : 3); ++d)
        {
            for (int i = 0; i < num_nodes; ++i)
            {
                vector_field_updated->SetComponent(i, d,
                    vector_field->GetComponent(i, d) + this->StepSize * x(i + d * num_nodes));
            }
        }

        // Prepare for next time step and output (intermediate) results
        vector_field->DeepCopy(vector_field_updated);

        deformed_curvature = curvature_and_torsion(grid(original_grid, vector_field), gradient_method_t::differences, 0);

        std::tie(errors, error_avg, error_max)
            = calculate_error_field(original_curvature, deformed_curvature, nullptr, error_definition_t::vector_difference);

        this->results[step + 1uLL] = create_output(dimension, positions);

        output_copy(this->results[step + 1uLL], vector_field, jacobian_field,
            errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
            deformed_curvature.curvature_gradient, deformed_curvature.torsion,
            deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient,
            original_curvature_gradients);
    }

    // If converged or stopped, later results stay the same
    if (error_max <= this->Error)
    {
        for (std::size_t i = step + 1uLL; i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[step];
        }
    }
}

void optimizer::compute_gradient_descent(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
    vtkDataArray* vector_field_original)
{
    // Get input grid and data
    const grid original_vector_field(original_grid, vector_field_original);

    const auto num_nodes = original_grid->GetNumberOfPoints();

    if (num_nodes != deformed_grid->GetNumberOfPoints())
    {
        std::cerr << "Number of grid nodes must match." << std::endl;
        return;
    }

    // Get initial node positions, and compute deformation
    auto positions = vtkSmartPointer<vtkDoubleArray>::New();
    positions->SetName("Deformed Position");
    positions->SetNumberOfComponents(3);
    positions->SetNumberOfTuples(num_nodes);

    std::array<double, 3> point{};

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        deformed_grid->GetPoint(i, point.data());
        positions->SetTuple(i, point.data());
    }

    const grid deformation_field(original_grid, positions);

    auto jacobian_field = gradient_field(deformation_field, static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);
    jacobian_field->SetName("Jacobian");

    auto vector_field_deformed = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field_deformed->SetName("Deformed Vectors");
    vector_field_deformed->SetNumberOfComponents(3);
    vector_field_deformed->SetNumberOfTuples(num_nodes);

    Eigen::Matrix3d jacobian;
    Eigen::Vector3d vector;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        jacobian_field->GetTuple(i, jacobian.data());
        vector_field_original->GetTuple(i, vector.data());

        vector = jacobian * vector;

        vector_field_deformed->SetTuple(i, vector.data());
    }

    const grid deformed_vector_field(deformed_grid, vector_field_deformed, jacobian_field);

    const std::array<int, 3> dimension = original_vector_field.dimensions();
    const bool twoD = dimension[2] == 1;

    // Calculate initial gradient difference
    const auto original_curvature = curvature_and_torsion(original_vector_field,
        static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);
    auto deformed_curvature = curvature_and_torsion(deformed_vector_field,
        static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);

    vtkSmartPointer<vtkDoubleArray> errors;
    double error_avg{}, error_max{}, min_error_avg, min_error_max;
    int min_error_avg_step, min_error_max_step;

    std::tie(errors, error_avg, error_max)
        = calculate_error_field(original_curvature, deformed_curvature, jacobian_field,
            static_cast<error_definition_t>(this->ErrorDefinition));

    const auto original_error_avg = min_error_avg = error_avg;
    const auto original_error_max = min_error_max = error_max;

    min_error_avg_step = min_error_max_step = 0;

    std::vector<double> errors_avg, errors_max;
    errors_avg.push_back(original_error_avg);
    errors_max.push_back(original_error_max);

    auto original_curvature_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvature_gradients->SetName("Curvature Gradient (Original)");
    original_curvature_gradients->SetNumberOfComponents(3);
    original_curvature_gradients->SetNumberOfTuples(num_nodes);

    Eigen::Vector3d original_gradient;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        original_curvature.curvature_gradient->GetTuple(i, original_gradient.data());
        jacobian_field->GetTuple(i, jacobian.data());

        original_gradient = jacobian * original_gradient;

        original_curvature_gradients->SetTuple(i, original_gradient.data());
    }

    // Set initial output
    if (!this->CSVOutput) std::cout << "Setting initial output..." << std::endl;

    this->results[0] = create_output(dimension, positions);

    output_copy(this->results[0], vector_field_deformed, jacobian_field, positions,
        errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
        deformed_curvature.curvature_gradient, deformed_curvature.torsion,
        deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient,
        original_curvature_gradients);

    // Apply optimization
    bool converged = false;
    bool stopped = false;

    double step_size = this->StepSize;
    int step;

    vtkSmartPointer<vtkDoubleArray> previous_gradient_descent = nullptr;

    for (step = 0; step < this->NumSteps && !converged && !stopped; ++step)
    {
        if (!this->CSVOutput) std::cout << "Optimization step: " << (step + 1) << "/" << this->NumSteps << std::endl;

        vtkSmartPointer<vtkDoubleArray> gradient_descent, descent, deformed_positions;

        vtkSmartPointer<vtkDoubleArray> new_errors;
        double new_error_avg{}, new_error_max{};

        // Setup dynamic step size control
        const auto step_size_control = static_cast<step_size_control_t>(this->StepSizeControl);
        const auto max_line_search_steps = (step_size_control == step_size_control_t::dynamic ? this->LineSearchSteps + 4 : 0);
        bool tendency_left = true;
        std::size_t step_size_index;
        std::array<step_size_t, 4> step_size_errors{};

        auto remove_largest_boundary = [](std::array<step_size_t, 4>& step_size_errors) {
            std::sort(step_size_errors.begin(), step_size_errors.end(),
                [](const step_size_t& lhs, step_size_t& rhs) { return lhs.step_size < rhs.step_size; });

            const auto min = std::min_element(step_size_errors.begin(), step_size_errors.end(),
                [](const step_size_t& lhs, step_size_t& rhs) { return lhs.error_avg < rhs.error_avg; });

            if (*min == step_size_errors[2] || *min == step_size_errors[3])
            {
                step_size_errors[0] = step_size_errors[1];
                step_size_errors[1] = step_size_errors[2];
                step_size_errors[2] = step_size_errors[3];
            }
        };

        // Perform line search (last iteration is the one producing results)
        for (int line_search_step = 0; line_search_step <= max_line_search_steps; ++line_search_step)
        {
            if (step_size_control == step_size_control_t::dynamic)
            {
                if (line_search_step < max_line_search_steps)
                {
                    if (!this->CSVOutput) std::cout << " Line search step " << (line_search_step + 1) << "/" << max_line_search_steps << std::endl;

                    step_size_index = std::min(line_search_step, 3);

                    switch (line_search_step)
                    {
                    case 0:
                        step_size = this->StepSizeMin;

                        break;
                    case 1:
                        step_size = this->StepSizeMax;

                        break;
                    case 2:
                        step_size = std::pow(10.0, std::log10(this->StepSizeMin) + (1.0 / 3.0) * std::log10(this->StepSizeMax / this->StepSizeMin));

                        break;
                    case 3:
                        step_size = std::pow(10.0, std::log10(this->StepSizeMin) + (2.0 / 3.0) * std::log10(this->StepSizeMax / this->StepSizeMin));

                        break;
                    case 4:
                        remove_largest_boundary(step_size_errors);

                        step_size = std::pow(10.0, std::log10(step_size_errors[0].step_size)
                            + (1.0 / 2.0) * std::log10(step_size_errors[1].step_size / step_size_errors[0].step_size));

                        tendency_left = true;

                        break;
                    default:
                    {
                        remove_largest_boundary(step_size_errors);

                        const bool left = step_size >= step_size_errors[0].step_size && step_size < step_size_errors[1].step_size;

                        if (left && (new_error_avg > step_size_errors[0].error_avg || new_error_avg > step_size_errors[1].error_avg))
                        {
                            tendency_left = false;
                        }
                        else if (!left && (new_error_avg > step_size_errors[1].error_avg || new_error_avg > step_size_errors[2].error_avg))
                        {
                            tendency_left = true;
                        }

                        if (tendency_left)
                        {
                            step_size = std::pow(10.0, std::log10(step_size_errors[0].step_size)
                                + (1.0 / 2.0) * std::log10(step_size_errors[1].step_size / step_size_errors[0].step_size));
                        }
                        else
                        {
                            step_size = std::pow(10.0, std::log10(step_size_errors[1].step_size)
                                + (1.0 / 2.0) * std::log10(step_size_errors[2].step_size / step_size_errors[1].step_size));
                        }
                    }
                    }

                    step_size_errors[step_size_index].step_size = step_size;
                }
                else
                {
                    // Set to element with minimum error
                    const auto min = std::min_element(step_size_errors.begin(), step_size_errors.end(),
                        [](const step_size_t& lhs, step_size_t& rhs) { return lhs.error_avg < rhs.error_avg; });

                    step_size = min->step_size;

                    if (step_size == this->StepSizeMin || step_size == this->StepSizeMax)
                    {
                        if (!this->CSVOutput) std::cout << "    Step size is equal to lower or upper bound. Stopping." << std::endl;

                        stopped = true;
                    }
                }
            }

            // Perform gradient descent
            if (line_search_step == max_line_search_steps)
            {
                if (!this->CSVOutput) std::cout << " Using step size: " << step_size << std::endl;
            }

            std::tie(gradient_descent, descent) = compute_descent(dimension, original_grid,
                vector_field_original, positions, errors, original_curvature, previous_gradient_descent);

            auto descent_dir = vtkSmartPointer<vtkDoubleArray>::New();
            if (this->CheckWolfe) descent_dir->DeepCopy(descent);

            std::tie(deformed_positions, descent) = apply_descent(dimension,
                step_size, positions, errors, descent);

            // Update jacobians and vector field
            const grid new_deformation(original_grid, deformed_positions);

            jacobian_field = gradient_field(new_deformation, static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);
            jacobian_field->SetName("Jacobian");

            Eigen::Matrix3d jacobian;
            Eigen::Vector3d vector;

            for (vtkIdType i = 0; i < num_nodes; ++i)
            {
                jacobian_field->GetTuple(i, jacobian.data());
                vector_field_original->GetTuple(i, vector.data());

                vector = jacobian * vector;

                vector_field_deformed->SetTuple(i, vector.data());
            }

            // Calculate new gradient difference
            const grid new_deformed_vector_field(dimension, deformed_positions, vector_field_deformed, jacobian_field);

            deformed_curvature = curvature_and_torsion(new_deformed_vector_field,
                static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);

            std::tie(new_errors, new_error_avg, new_error_max)
                = calculate_error_field(original_curvature, deformed_curvature, jacobian_field,
                    static_cast<error_definition_t>(this->ErrorDefinition));

            // Store errors for dynamic step size control
            if (step_size_control == step_size_control_t::dynamic && line_search_step < max_line_search_steps)
            {
                step_size_errors[step_size_index].error_avg = new_error_avg;
                step_size_errors[step_size_index].error_max = new_error_max;
            }

            if (step_size_control == step_size_control_t::dynamic && this->CheckWolfe)
            {
                auto new_gradient_descent = compute_descent(dimension, original_grid,
                    vector_field_original, deformed_positions, new_errors, original_curvature, nullptr).first;

                Eigen::VectorXd full_direction, full_gradient, new_full_gradient;
                full_gradient.resize(3 * num_nodes);
                new_full_gradient.resize(3 * num_nodes);

                for (vtkIdType i = 0; i < num_nodes; ++i)
                {
                    full_direction[i * 3 + 0] = descent_dir->GetComponent(i, 0);
                    full_direction[i * 3 + 1] = descent_dir->GetComponent(i, 1);
                    full_direction[i * 3 + 2] = descent_dir->GetComponent(i, 2);

                    full_gradient[i * 3 + 0] = -gradient_descent->GetComponent(i, 0);
                    full_gradient[i * 3 + 1] = -gradient_descent->GetComponent(i, 1);
                    full_gradient[i * 3 + 2] = -gradient_descent->GetComponent(i, 2);

                    new_full_gradient[i * 3 + 0] = -new_gradient_descent->GetComponent(i, 0);
                    new_full_gradient[i * 3 + 1] = -new_gradient_descent->GetComponent(i, 1);
                    new_full_gradient[i * 3 + 2] = -new_gradient_descent->GetComponent(i, 2);
                }

                const auto satisfies_wolfe = satisfies_armijo(error_avg, new_error_avg, full_direction, full_gradient, step_size, 10e-4) &&
                    satisfies_wolfe_curv(full_direction, full_gradient, new_full_gradient, 0.1); // todo parameter

                if (satisfies_wolfe)
                {
                    if (!this->CSVOutput) std::cout << "  Step size satisfies Wolfe conditions: " << step_size << std::endl;
                }
                else
                {
                    if (!this->CSVOutput) std::cout << "  Step size does not satisfy Wolfe conditions: " << step_size << std::endl;
                }
            }
        }

        if (new_error_max > error_max)
        {
            if (!this->CSVOutput) std::cout << "    New maximum error increased from " << error_max << " to " << new_error_max << "." << std::endl;
        }
        if (new_error_avg > error_avg)
        {
            if (!this->CSVOutput) std::cout << "    New average error increased from " << error_avg << " to " << new_error_avg << "." << std::endl;
        }

        errors_max.push_back(new_error_max);
        errors_avg.push_back(new_error_avg);

        error_max = new_error_max;
        error_avg = new_error_avg;

        if (new_error_max < min_error_max && !std::isnan(new_error_max))
        {
            min_error_max = new_error_max;
            min_error_max_step = step + 1;
        }
        if (new_error_avg < min_error_avg && !std::isnan(new_error_avg))
        {
            min_error_avg = new_error_avg;
            min_error_avg_step = step + 1;
        }

        if (error_max <= this->Error)
        {
            converged = true;
        }

        // Transform original curvature to deformed grid for comparison
        for (vtkIdType i = 0; i < num_nodes; ++i)
        {
            original_curvature.curvature_gradient->GetTuple(i, original_gradient.data());
            jacobian_field->GetTuple(i, jacobian.data());

            original_gradient = jacobian * original_gradient;

            original_curvature_gradients->SetTuple(i, original_gradient.data());
        }

        // Set output for this step
        this->results[step + 1uLL] = create_output(dimension, deformed_positions);

        output_copy(this->results[step + 1uLL], vector_field_deformed, jacobian_field, deformed_positions,
            new_errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
            deformed_curvature.curvature_gradient, deformed_curvature.torsion,
            deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient, gradient_descent, descent,
            original_curvature_gradients);

        output_copy(this->results[step], gradient_descent, descent);

        // Prepare next iteration
        positions = deformed_positions;
        errors = new_errors;
        previous_gradient_descent = gradient_descent;
    }

    // If converged or stopped, later results stay the same
    if (converged)
    {
        if (!this->CSVOutput) std::cout << "Optimization converged." << std::endl;

        for (std::size_t i = step + 1uLL; i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[step];
        }
    }
    else if (stopped)
    {
        if (!this->CSVOutput) std::cout << "Optimization stopped." << std::endl;

        for (std::size_t i = std::max(1, step); i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[std::max(1, step) - 1uLL];
        }
    }
    else
    {
        if (!this->CSVOutput) std::cout << "Finished computation without convergence." << std::endl;
    }

    if (!this->CSVOutput)
    {
        std::cout << " Original error (avg): " << original_error_avg << std::endl;
        std::cout << " Original error (max): " << original_error_max << std::endl;

        if (std::isnan(error_avg) || std::isnan(error_max))
        {
            std::cout << " Error increase is NAN." << std::endl;
        }
        else
        {
            std::cout << " Error (avg): " << error_avg << std::endl;
            std::cout << " Error (max): " << error_max << std::endl;
            std::cout << " Error (avg) increase: " << (error_avg - original_error_avg) << std::endl;
            std::cout << " Error (max) increase: " << (error_max - original_error_max) << std::endl;
        }

        std::cout << " Minimum error (avg): " << min_error_avg << " in step " << min_error_avg_step << std::endl;
        std::cout << " Minimum error (max): " << min_error_max << " in step " << min_error_max_step << std::endl;
    }
    else
    {
        // Output minimum error information
        std::cout << original_error_avg << "," << original_error_max << ",";

        if (std::isnan(error_avg) || std::isnan(error_max))
        {
            std::cout << "NAN,NAN,NAN,NAN,";
        }
        else
        {
            std::cout << error_avg << "," << error_max << "," << (error_avg - original_error_avg) << "," << (error_max - original_error_max) << ",";
        }

        std::cout << min_error_avg << "," << min_error_max << "," << min_error_avg_step << "," << min_error_max_step << std::endl;

        // Output errors
        std::ofstream file("_errors.csv", std::ios_base::app | std::ios_base::binary);

        file << "Step";

        for (int i = 0; i < errors_avg.size(); ++i)
        {
            file << "," << i;
        }

        file << std::endl;
        file << "Error (avg)";

        for (const auto& err : errors_avg)
        {
            file << "," << err;
        }

        file << std::endl;
        file << "Error (max)";

        for (const auto& err : errors_max)
        {
            file << "," << err;
        }

        file << std::endl;
        file.close();
    }
}

std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> optimizer::compute_descent(
    const std::array<int, 3>& dimension, const vtkStructuredGrid* original_grid,
    const vtkDataArray* vector_field_original, const vtkDataArray* positions,
    const vtkDataArray* errors, const curvature_and_torsion_t& original_curvature,
    const vtkDataArray* previous_gradient_descent) const
{
    using duration_t = std::chrono::milliseconds;
    const std::string duration_str(" ms");

    const auto start = std::chrono::steady_clock::now();

    auto descent = vtkSmartPointer<vtkDoubleArray>::New();
    descent->SetName("Descent");
    descent->SetNumberOfComponents(3);
    descent->SetNumberOfTuples(vector_field_original->GetNumberOfTuples());
    descent->Fill(0.0);

    auto gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    gradient_descent->SetName("Gradient Descent");
    gradient_descent->SetNumberOfComponents(3);
    gradient_descent->SetNumberOfTuples(vector_field_original->GetNumberOfTuples());
    gradient_descent->Fill(0.0);

    // Domain information
    const bool twoD = dimension[2] == 1;

    Eigen::Vector3d origin, right, top, front;
    const_cast<vtkStructuredGrid*>(original_grid)->GetPoint(0, origin.data());
    const_cast<vtkStructuredGrid*>(original_grid)->GetPoint(1, right.data());
    const_cast<vtkStructuredGrid*>(original_grid)->GetPoint(dimension[0], top.data());
    const_cast<vtkStructuredGrid*>(original_grid)->GetPoint(twoD ? 0LL : (static_cast<vtkIdType>(dimension[0]) * dimension[1]), front.data());

    const Eigen::Vector3d cell_sizes(right[0] - origin[0], top[1] - origin[1], front[2] - origin[2]);
    const Eigen::Vector3d infinitesimal_steps = this->GradientStep * cell_sizes;

    // For each 11x11(x11) block of nodes, calculate partial derivatives of the
    // curvature gradient difference in direction of the degrees of freedom.
    // Use gradient descent to perform a single step for respective center
    // vertex, minimizing its curvature gradient difference.
    const auto block_offset = 5;
    const auto block_inner_offset = 2;
    const auto block_size = (2 * block_offset + 1);

    const auto num_blocks = dimension[0] * dimension[1] * dimension[2];

    for (int z = 0; z < dimension[2]; ++z)
    {
        #pragma omp parallel for
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                // Define block extent
                const std::array<std::array<int, 2>, 3> block_offsets{
                    std::array<int, 2>{std::min(x - block_offset, 0) + block_offset,
                        block_offset - (std::max(x + block_offset, dimension[0] - 1) - (dimension[0] - 1))},
                    std::array<int, 2>{std::min(y - block_offset, 0) + block_offset,
                        block_offset - (std::max(y + block_offset, dimension[1] - 1) - (dimension[1] - 1))},
                    std::array<int, 2>{std::min(z - block_offset, 0) + block_offset,
                        block_offset - (std::max(z + block_offset, dimension[2] - 1) - (dimension[2] - 1))} };

                const std::array<int, 3> block_sizes{
                    block_offsets[0][1] + block_offsets[0][0] + 1,
                    block_offsets[1][1] + block_offsets[1][0] + 1,
                    block_offsets[2][1] + block_offsets[2][0] + 1 };

                const auto block_size = block_sizes[0] * block_sizes[1] * block_sizes[2];

                // Create grid block
                auto original_position_block = vtkSmartPointer<vtkDoubleArray>::New();
                original_position_block->SetNumberOfComponents(3);
                original_position_block->SetNumberOfTuples(0);

                auto original_vector_block = vtkSmartPointer<vtkDoubleArray>::New();
                original_vector_block->SetNumberOfComponents(3);
                original_vector_block->SetNumberOfTuples(0);

                auto new_position_block = vtkSmartPointer<vtkDoubleArray>::New();
                new_position_block->SetNumberOfComponents(3);
                new_position_block->SetNumberOfTuples(0);

                auto new_vector_block = vtkSmartPointer<vtkDoubleArray>::New();
                new_vector_block->SetNumberOfComponents(3);
                new_vector_block->SetNumberOfTuples(0);

                original_position_block->SetNumberOfTuples(block_size);
                original_vector_block->SetNumberOfTuples(block_size);
                new_position_block->SetNumberOfTuples(block_size);
                new_vector_block->SetNumberOfTuples(block_size);

                Eigen::Vector3d temp;

                for (int zz = (twoD ? 0 : -block_offsets[2][0]); zz <= (twoD ? 0 : block_offsets[2][1]); ++zz)
                {
                    const auto index_zz = zz + (twoD ? 0 : block_offsets[2][0]);
                    const auto index_z = z + zz;

                    for (int yy = -block_offsets[1][0]; yy <= block_offsets[1][1]; ++yy)
                    {
                        const auto index_yy = yy + block_offsets[1][0];
                        const auto index_y = y + yy;

                        for (int xx = -block_offsets[0][0]; xx <= block_offsets[0][1]; ++xx)
                        {
                            const auto index_xx = xx + block_offsets[0][0];
                            const auto index_x = x + xx;

                            const auto index_block = index_xx + block_sizes[0] * (index_yy + block_sizes[1] * index_zz);
                            const auto index_orig = index_x + dimension[0] * (index_y + dimension[1] * index_z);

                            const_cast<vtkStructuredGrid*>(original_grid)->GetPoint(index_orig, temp.data());
                            original_position_block->SetTuple(index_block, temp.data());

                            const_cast<vtkDataArray*>(vector_field_original)->GetTuple(index_orig, temp.data());
                            original_vector_block->SetTuple(index_block, temp.data());

                            const_cast<vtkDataArray*>(positions)->GetTuple(index_orig, temp.data());
                            new_position_block->SetTuple(index_block, temp.data());
                        }
                    }
                }

                // For each degree of freedom, calculate derivative
                grid block_deformation(block_sizes, original_position_block, new_position_block);

                Eigen::Vector3d temp_vector{};
                Eigen::Matrix3d temp_jacobian{};

                const auto index_block_center = block_offsets[0][0] + block_sizes[0]
                    * (block_offsets[1][0] + block_sizes[1] * block_offsets[2][0]);
                const auto index_center = x + dimension[0] * (y + dimension[1] * z);

                for (int d = 0; d < (twoD ? 2 : 3); ++d)
                {
                    // Move node in respective direction
                    new_position_block->SetComponent(index_block_center, d,
                        new_position_block->GetComponent(index_block_center, d) + infinitesimal_steps[d]);

                    // Compute Jacobians of deformation
                    auto jacobians = gradient_field(block_deformation, static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);

                    // Compute new vectors after deformation
                    for (vtkIdType i = 0; i < new_vector_block->GetNumberOfTuples(); ++i)
                    {
                        original_vector_block->GetTuple(i, temp_vector.data());
                        jacobians->GetTuple(i, temp_jacobian.data());

                        temp_vector = temp_jacobian * temp_vector;

                        new_vector_block->SetTuple(i, temp_vector.data());
                    }

                    // Calculate cuvature and torsion
                    grid block(block_sizes, new_position_block, new_vector_block, jacobians);

                    const auto curvature = curvature_and_torsion(block, static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);

                    // Calculate difference between original and deformed curvature gradient for all neighboring nodes
                    const auto num_block_nodes =
                        (std::min(block_inner_offset, block_offsets[0][1]) + std::min(block_inner_offset, block_offsets[0][0]) + 1) *
                        (std::min(block_inner_offset, block_offsets[1][1]) + std::min(block_inner_offset, block_offsets[1][0]) + 1) *
                        (twoD ? 1 : (std::min(block_inner_offset, block_offsets[2][1]) + std::min(block_inner_offset, block_offsets[2][0]) + 1));

                    for (int zz = (twoD ? 0 : -std::min(block_inner_offset, block_offsets[2][0]));
                        zz <= (twoD ? 0 : std::min(block_inner_offset, block_offsets[2][1])); ++zz)
                    {
                        const auto index_zz = zz + (twoD ? 0 : block_offsets[2][0]);
                        const auto index_z = z + zz;

                        for (int yy = -std::min(block_inner_offset, block_offsets[1][0]);
                            yy <= std::min(block_inner_offset, block_offsets[1][1]); ++yy)
                        {
                            const auto index_yy = yy + block_offsets[1][0];
                            const auto index_y = y + yy;

                            for (int xx = -std::min(block_inner_offset, block_offsets[0][0]);
                                xx <= std::min(block_inner_offset, block_offsets[0][1]); ++xx)
                            {
                                const auto index_xx = xx + block_offsets[0][0];
                                const auto index_x = x + xx;

                                const auto index_block = index_xx + block_sizes[0] * (index_yy + block_sizes[1] * index_zz);
                                const auto index_orig = index_x + dimension[0] * (index_y + dimension[1] * index_z);

                                if (const_cast<vtkDataArray*>(errors)->GetComponent(index_orig, 0) > this->Error)
                                {
                                    const auto error = calculate_error(index_orig, index_block, original_curvature, curvature, jacobians,
                                        static_cast<error_definition_t>(this->ErrorDefinition));

                                    const auto gradient = (error
                                        - const_cast<vtkDataArray*>(errors)->GetComponent(index_orig, 0)) / infinitesimal_steps[d];

                                    gradient_descent->SetComponent(index_center, d,
                                        gradient_descent->GetComponent(index_center, d) - gradient / num_block_nodes);
                                }
                            }
                        }
                    }

                    // Reset new positions
                    new_position_block->SetComponent(index_block_center, d,
                        new_position_block->GetComponent(index_block_center, d) - infinitesimal_steps[d]);
                }
            }
        }
    }

    // Compute descent direction using Polak-Ribière
    // c.f. https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    if (static_cast<method_t>(this->Method) == method_t::nonlinear_conjugate && previous_gradient_descent != nullptr)
    {
        Eigen::VectorXd current_gradient, old_gradient;
        current_gradient.resize(3uLL * num_blocks);
        old_gradient.resize(3uLL * num_blocks);

        for (int index = 0; index < num_blocks; ++index)
        {
            current_gradient[3uLL * index + 0] = gradient_descent->GetComponent(index, 0);
            current_gradient[3uLL * index + 1] = gradient_descent->GetComponent(index, 1);
            current_gradient[3uLL * index + 2] = gradient_descent->GetComponent(index, 2);

            old_gradient[3uLL * index + 0] = const_cast<vtkDataArray*>(previous_gradient_descent)->GetComponent(index, 0);
            old_gradient[3uLL * index + 1] = const_cast<vtkDataArray*>(previous_gradient_descent)->GetComponent(index, 1);
            old_gradient[3uLL * index + 2] = const_cast<vtkDataArray*>(previous_gradient_descent)->GetComponent(index, 2);
        }

        const auto beta = current_gradient.dot(current_gradient - old_gradient) / old_gradient.dot(old_gradient);

        for (int index = 0; index < num_blocks; ++index)
        {
            descent->SetComponent(index, 0, gradient_descent->GetComponent(index, 0)
                + beta * const_cast<vtkDataArray*>(previous_gradient_descent)->GetComponent(index, 0));
            descent->SetComponent(index, 1, gradient_descent->GetComponent(index, 1)
                + beta * const_cast<vtkDataArray*>(previous_gradient_descent)->GetComponent(index, 1));
            descent->SetComponent(index, 2, gradient_descent->GetComponent(index, 2)
                + beta * const_cast<vtkDataArray*>(previous_gradient_descent)->GetComponent(index, 2));
        }
    }
    else
    {
        descent->DeepCopy(gradient_descent);
        descent->SetName("Descent");
    }

    if (!this->CSVOutput) std::cout << "  Finished computing gradient descent after " <<
        std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now() - start).count() << duration_str << std::endl;

    return std::make_pair(gradient_descent, descent);
}

std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> optimizer::apply_descent(
    const std::array<int, 3>& dimension, const double step_size, const vtkDataArray* positions,
    const vtkDataArray* errors, const vtkDataArray* descent_direction) const
{
    auto deformed_positions = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_positions->SetName("Deformed Position");
    deformed_positions->SetNumberOfComponents(3);
    deformed_positions->SetNumberOfTuples(descent_direction->GetNumberOfTuples());

    auto adjusted_gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    adjusted_gradient_descent->SetName("Descent");
    adjusted_gradient_descent->SetNumberOfComponents(3);
    adjusted_gradient_descent->SetNumberOfTuples(descent_direction->GetNumberOfTuples());

    const bool twoD = dimension[2] == 1;

    const auto step_size_method = static_cast<step_size_method_t>(this->StepSizeMethod);

    for (int z = 0; z < dimension[2]; ++z)
    {
        #pragma omp parallel for
        for (int y = 0; y < dimension[1]; ++y)
        {
            Eigen::Vector3d position, descent;

            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index = x + dimension[0] * (y + dimension[1] * z);

                const_cast<vtkDataArray*>(positions)->GetTuple(index, position.data());
                const_cast<vtkDataArray*>(descent_direction)->GetTuple(index, descent.data());

                if (step_size_method == step_size_method_t::error && descent.norm() != 0.0)
                {
                    auto error_sum = 0.0;
                    auto error_num = 0;

                    const auto kernel_size = 2;

                    for (int zz = (twoD ? 0 : -kernel_size); zz <= (twoD ? 0 : kernel_size); ++zz)
                    {
                        const auto index_z = z + zz;

                        if (index_z < 0 || index_z >= dimension[2]) continue;

                        for (int yy = -kernel_size; yy <= kernel_size; ++yy)
                        {
                            const auto index_y = y + yy;

                            if (index_y < 0 || index_y >= dimension[1]) continue;

                            for (int xx = -kernel_size; xx <= kernel_size; ++xx)
                            {
                                const auto index_x = x + xx;

                                if (index_x < 0 || index_x >= dimension[0]) continue;

                                const auto index_kernel = index_x + dimension[0] * (index_y + dimension[1] * index_z);

                                error_sum += const_cast<vtkDataArray*>(errors)->GetComponent(index_kernel, 0);
                                ++error_num;
                            }
                        }
                    }

                    const auto error = error_sum / error_num;

                    descent *= error / descent.norm();
                }
                else if (step_size_method == step_size_method_t::normalized && descent.norm() != 0.0)
                {
                    descent /= descent.norm();
                }

                descent *= step_size;
                position += descent;

                adjusted_gradient_descent->SetTuple(index, descent.data());
                deformed_positions->SetTuple(index, position.data());
            }
        }
    }

    return std::make_pair(deformed_positions, adjusted_gradient_descent);
}

double optimizer::calculate_error(const int index, const int index_block, const curvature_and_torsion_t& original_curvature,
    const curvature_and_torsion_t& deformed_curvature, const vtkDataArray* jacobian_field, const error_definition_t error_definition) const
{
    Eigen::VectorXd original_gradient, deformed_gradient;
    original_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);
    deformed_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);

    Eigen::Matrix3d jacobian;
    jacobian.setIdentity();

    original_curvature.curvature_gradient->GetTuple(index, original_gradient.data());
    deformed_curvature.curvature_gradient->GetTuple(index_block, deformed_gradient.data());

    if (original_curvature.curvature_gradient->GetNumberOfComponents() == 3)
    {
        if (jacobian_field != nullptr)
        {
            const_cast<vtkDataArray*>(jacobian_field)->GetTuple(index_block, jacobian.data());
        }

        deformed_gradient = jacobian.inverse() * deformed_gradient;
    }

    switch (error_definition)
    {
    case error_definition_t::vector_difference:
        return (deformed_gradient - original_gradient).norm();

        break;
    case error_definition_t::angle:
        return std::acos(std::min(std::max(original_gradient.normalized().dot(deformed_gradient.normalized()), -1.0), 1.0));

        break;
    case error_definition_t::length_difference:
        return std::abs(deformed_gradient.norm() - original_gradient.norm());

        break;
    default:
        std::cerr << "Unknown error definition." << std::endl;
        return 0.0;
    }
}

std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> optimizer::calculate_error_field(
    const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature,
    const vtkDataArray* jacobian_field, const error_definition_t error_definition) const
{
    auto errors = vtkSmartPointer<vtkDoubleArray>::New();
    errors->SetName("Error");
    errors->SetNumberOfComponents(1);
    errors->SetNumberOfTuples(original_curvature.curvature_gradient->GetNumberOfTuples());

    double error_max = std::numeric_limits<double>::min();
    std::vector<double> error_avgs(original_curvature.curvature_gradient->GetNumberOfTuples());

    for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
    {
        const auto error = calculate_error(i, i, original_curvature, deformed_curvature, jacobian_field, error_definition);

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

bool optimizer::satisfies_armijo(const double old_value, const double new_value, const Eigen::VectorXd& direction,
    const Eigen::VectorXd& gradient, const double step_size, const double constant) const
{
    return new_value <= (old_value + constant * step_size * direction.dot(gradient));
}

bool optimizer::satisfies_wolfe_curv(const Eigen::VectorXd& direction, const Eigen::VectorXd& old_gradient,
    const Eigen::VectorXd& new_gradient, const double constant) const
{
    return -direction.dot(new_gradient) <= -constant * direction.dot(old_gradient);
}

vtkSmartPointer<vtkStructuredGrid> optimizer::create_output(const std::array<int, 3>& dimension, const vtkDoubleArray* positions) const
{
    auto grid = vtkSmartPointer<vtkStructuredGrid>::New();

    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(positions->GetNumberOfTuples());

    std::array<double, 3> point{};

    for (vtkIdType i = 0; i < positions->GetNumberOfTuples(); ++i)
    {
        const_cast<vtkDoubleArray*>(positions)->GetTuple(i, point.data());
        points->SetPoint(i, point.data());
    }

    grid->SetDimensions(dimension.data());
    grid->SetPoints(points);

    return grid;
}
