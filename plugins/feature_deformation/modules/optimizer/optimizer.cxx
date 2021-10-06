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
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredGrid.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

    const auto hash = joaat_hash(this->NumSteps, this->StepSize, this-StepSizeMethod, this->StepSizeControl,
        this->Error, this->Adjustment, this->MaxAdjustments, this->Threshold, this->Increase, this->Stop,
        this->GradientMethod, this->GradientKernel,
        vector_field_original->GetMTime(), deformed_grid->GetMTime());

    if (hash != this->hash)
    {
        compute(original_grid, deformed_grid, vector_field_original);

        this->hash = hash;
    }

    const auto time = output_vector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    const auto time_step = std::min(std::max(static_cast<std::size_t>(time), 0uLL), static_cast<std::size_t>(this->NumSteps));

    auto output_grid = vtkStructuredGrid::GetData(output_vector);
    output_grid->ShallowCopy(this->results[static_cast<std::size_t>(time_step)]);

    std::cout << "Showing step: " << static_cast<std::size_t>(time_step) << std::endl;

    return 1;
}

void optimizer::compute(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
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
    const auto deformed_curvature = curvature_and_torsion(deformed_vector_field,
        static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);

    vtkSmartPointer<vtkDoubleArray> errors;
    double error_avg, error_max;

    std::tie(errors, error_avg, error_max)
        = calculate_error_field(original_curvature, deformed_curvature, jacobian_field,
            static_cast<error_definition_t>(this->ErrorDefinition));

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

    // Set initial step sizes
    auto step_sizes = vtkSmartPointer<vtkDoubleArray>::New();
    step_sizes->SetName("Step Sizes");
    step_sizes->SetNumberOfComponents(1);
    step_sizes->SetNumberOfTuples(num_nodes);
    step_sizes->Fill(this->StepSize);

    // Set initial output
    std::cout << "Setting initial output..." << std::endl;

    this->results[0] = create_output(dimension, positions);

    output_copy(this->results[0], vector_field_deformed, jacobian_field, positions, step_sizes,
        errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
        deformed_curvature.curvature_gradient, deformed_curvature.torsion,
        deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient,
        original_curvature_gradients);

    // Apply optimization
    bool converged = false;
    bool stopped = false;

    const auto original_step_size = this->StepSize;
    int step;

    for (step = 0; step < this->NumSteps && !converged && !stopped; ++step)
    {
        std::cout << "Optimization step: " << (step + 1) << "/" << this->NumSteps << std::endl;

        vtkSmartPointer<vtkDoubleArray> gradient_descent, deformed_positions;

        gradient_descent = compute_gradient_descent(dimension, original_grid,
            vector_field_original, positions, errors, original_curvature);

        std::tie(deformed_positions, gradient_descent) = apply_gradient_descent(dimension,
            step_sizes, positions, errors, gradient_descent);

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

        const auto deformed_curvature = curvature_and_torsion(new_deformed_vector_field,
            static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);

        vtkSmartPointer<vtkDoubleArray> new_errors;
        double new_error_avg, new_error_max;

        std::tie(new_errors, new_error_avg, new_error_max)
            = calculate_error_field(original_curvature, deformed_curvature, jacobian_field,
                static_cast<error_definition_t>(this->ErrorDefinition));

        if (new_error_max > error_max)
        {
            std::cout << "    New maximum error increased from " << error_max << " to " << new_error_max << "." << std::endl;
        }
        if (new_error_avg > error_avg)
        {
            std::cout << "    New average error increased from " << error_avg << " to " << new_error_avg << "." << std::endl;
        }

        const auto step_size_control = static_cast<step_size_control_t>(this->StepSizeControl);

        if (step_size_control == step_size_control_t::dynamic)
        {
            bool rewind = false;

            //if (new_error_max > this->Threshold * error_max || new_error_avg > this->Threshold * error_avg)
            //{
            //    if (step_sizes->GetComponent(0, 0) > std::pow(2.0, -this->MaxAdjustments) * original_step_size)
            //    {
            //        step_sizes->Fill(step_sizes->GetComponent(i, 0) * this->Adjustment);
            //        rewind = true;
            //    }
            //    else if (this->Stop)
            //    {
            //        stopped = true;
            //    }
            //}

            if (new_error_max > this->Threshold * error_max || new_error_avg > this->Threshold * error_avg)
            {
                for (vtkIdType i = 0; i < num_nodes; ++i)
                {
                    const auto new_error = new_errors->GetComponent(i, 0);
                    const auto old_error = errors->GetComponent(i, 0);

                    if (new_error > this->Threshold * old_error)
                    {
                        if (step_sizes->GetComponent(i, 0) > std::pow(2.0, -this->MaxAdjustments) * original_step_size)
                        {
                            step_sizes->SetComponent(i, 0, step_sizes->GetComponent(i, 0) * this->Adjustment);
                            rewind = true;
                        }
                        else if (this->Stop)
                        {
                            stopped = true;
                        }
                    }
                }
            }

            if (rewind)
            {
                --step;
                continue;
            }
            else if (stopped)
            {
                continue;
            }
        }

        error_max = new_error_max;
        error_avg = new_error_avg;

        if (error_max < this->Error)
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

        output_copy(this->results[step + 1uLL], vector_field_deformed, jacobian_field, deformed_positions, step_sizes,
            new_errors, deformed_curvature.curvature, deformed_curvature.curvature_vector,
            deformed_curvature.curvature_gradient, deformed_curvature.torsion,
            deformed_curvature.torsion_vector, deformed_curvature.torsion_gradient, gradient_descent,
            original_curvature_gradients);

        output_copy(this->results[step], gradient_descent);

        // Prepare next iteration
        positions = deformed_positions;
        errors = new_errors;

        if (step_size_control == step_size_control_t::dynamic && this->Increase)
        {
            for (vtkIdType i = 0; i < num_nodes; ++i)
            {
                step_sizes->SetComponent(i, 0, step_sizes->GetComponent(i, 0) / this->Adjustment);
            }
        }
    }

    // If converged, later results stay the same
    if (converged)
    {
        std::cout << "Optimization converged." << std::endl;

        for (std::size_t i = step + 1uLL; i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[step];
        }
    }
    else if (stopped)
    {
        std::cout << "Optimization stopped." << std::endl;

        for (std::size_t i = std::max(1, step); i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[std::max(1, step) - 1uLL];
        }
    }
    else
    {
        std::cout << "Finished computation without convergence." << std::endl;
    }
}

vtkSmartPointer<vtkDoubleArray> optimizer::compute_gradient_descent(const std::array<int, 3>& dimension,
    const vtkStructuredGrid* original_grid, const vtkDataArray* vector_field_original, const vtkDataArray* positions,
    const vtkDataArray* errors, const curvature_and_torsion_t& original_curvature) const
{
    using duration_t = std::chrono::milliseconds;
    const std::string duration_str(" ms");

    const auto start = std::chrono::steady_clock::now();

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
    const auto infinitesimal_step = 1.0e-3;
    const Eigen::Vector3d infinitesimal_steps = infinitesimal_step * cell_sizes;

    // For each 7x7(x7) block of nodes, calculate partial derivatives of the
    // curvature gradient difference in direction of the degrees of freedom.
    // Use gradient descent to perform a single step for respective center
    // vertex, minimizing its curvature gradient difference.
    const auto block_offset = 3;
    const auto block_inner_offset = (block_offset + 1) / 2;
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

                // Temporary variables
                Eigen::VectorXd original_gradient, deformed_gradient;
                Eigen::Matrix3d jacobian;
                original_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);
                deformed_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);

                Eigen::Vector3d temp;

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

    std::cout << "  Finished computing gradient descent after " <<
        std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now() - start).count() << duration_str << std::endl;

    return gradient_descent;
}

std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> optimizer::apply_gradient_descent(
    const std::array<int, 3>& dimension, vtkDataArray* step_sizes, const vtkDataArray* positions,
    const vtkDataArray* errors, const vtkDataArray* gradient_descent) const
{
    auto deformed_positions = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_positions->SetName("Deformed Position");
    deformed_positions->SetNumberOfComponents(3);
    deformed_positions->SetNumberOfTuples(gradient_descent->GetNumberOfTuples());

    auto adjusted_gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    adjusted_gradient_descent->SetName("Gradient Descent");
    adjusted_gradient_descent->SetNumberOfComponents(3);
    adjusted_gradient_descent->SetNumberOfTuples(gradient_descent->GetNumberOfTuples());

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
                const_cast<vtkDataArray*>(gradient_descent)->GetTuple(index, descent.data());

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

                descent *= step_sizes->GetComponent(index, 0);
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
    Eigen::Matrix3d jacobian;
    original_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);
    deformed_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);

    original_curvature.curvature_gradient->GetTuple(index, original_gradient.data());
    deformed_curvature.curvature_gradient->GetTuple(index_block, deformed_gradient.data());

    if (original_curvature.curvature_gradient->GetNumberOfComponents() == 3)
    {
        const_cast<vtkDataArray*>(jacobian_field)->GetTuple(index_block, jacobian.data());

        deformed_gradient = jacobian.inverse() * deformed_gradient;
    }

    switch (error_definition)
    {
    case error_definition_t::vector_difference:
        return (deformed_gradient - original_gradient).norm();

        break;
    case error_definition_t::angle:
        return std::acos(original_gradient.normalized().dot(deformed_gradient.normalized()));

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
    double error_avg = 0.0;

    #pragma omp parallel for
    for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
    {
        const auto error = calculate_error(i, i, original_curvature, deformed_curvature, jacobian_field, error_definition);

        errors->SetValue(i, error);

        error_max = std::max(error_max, error);
        error_avg += error;
    }

    error_avg /= original_curvature.curvature_gradient->GetNumberOfTuples();

    return std::make_tuple(errors, error_avg, error_max);
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
