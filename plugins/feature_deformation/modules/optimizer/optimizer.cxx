#include "optimizer.h"

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
#include "vtkStructuredGrid.h"

#include <array>
#include <chrono>
#include <iostream>
#include <limits>

vtkStandardNewMacro(optimizer);

optimizer::optimizer()
{
    this->SetNumberOfInputPorts(2);
    this->SetNumberOfOutputPorts(1);
}

optimizer::~optimizer()
{
}

int optimizer::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
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

int optimizer::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    using duration_t = std::chrono::milliseconds;
    const std::string duration_str(" ms");

    std::chrono::time_point<std::chrono::steady_clock> start, start_inner, start_block;

    // Get input grid and data
    std::cout << "Reading data...";
    start = std::chrono::steady_clock::now();

    auto original_grid = vtkStructuredGrid::GetData(input_vector[0]);
    auto deformed_grid = vtkStructuredGrid::GetData(input_vector[1]);

    if (original_grid == nullptr || deformed_grid == nullptr)
    {
        std::cerr << std::endl << "All input slots must be connected." << std::endl;
        return 0;
    }

    auto vector_field_original = GetInputArrayToProcess(0, original_grid);
    auto vector_field_deformed_ = GetInputArrayToProcess(1, deformed_grid);
    auto jacobian_field_ = GetInputArrayToProcess(2, deformed_grid);

    if (vector_field_original == nullptr || vector_field_deformed_ == nullptr || jacobian_field_ == nullptr)
    {
        std::cerr << std::endl << "All input fields must be provided." << std::endl;
        return 0;
    }

    auto vector_field_deformed = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field_deformed->DeepCopy(vector_field_deformed_);

    auto jacobian_field = vtkSmartPointer<vtkDoubleArray>::New();
    jacobian_field->DeepCopy(jacobian_field_);

    const grid original_vector_field(original_grid, vector_field_original);
    const grid deformed_vector_field(deformed_grid, vector_field_deformed, jacobian_field);

    std::cout << "  " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
        - start).count() << duration_str << std::endl;

    // Calculate initial gradient difference
    std::cout << "Calculating initial gradient difference...";
    start = std::chrono::steady_clock::now();

    auto original_curvature = curvature_and_torsion(original_vector_field);
    auto deformed_curvature = curvature_and_torsion(deformed_vector_field);

    auto gradient_difference = vtkSmartPointer<vtkDoubleArray>::New();
    gradient_difference->SetName("Error");
    gradient_difference->SetNumberOfComponents(1);
    gradient_difference->SetNumberOfTuples(original_curvature.curvature_gradient->GetNumberOfTuples());

    auto original_gradient_difference = vtkSmartPointer<vtkDoubleArray>::New();
    original_gradient_difference->SetName("Original Error");
    original_gradient_difference->SetNumberOfComponents(1);
    original_gradient_difference->SetNumberOfTuples(original_curvature.curvature_gradient->GetNumberOfTuples());

    Eigen::VectorXd original_gradient, deformed_gradient;
    original_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);
    deformed_gradient.resize(original_curvature.curvature_gradient->GetNumberOfComponents(), 1);

    double error_min = std::numeric_limits<double>::max();
    double error_max = std::numeric_limits<double>::min();

    for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
    {
        original_curvature.curvature_gradient->GetTuple(i, original_gradient.data());
        deformed_curvature.curvature_gradient->GetTuple(i, deformed_gradient.data());

        const auto difference = (deformed_gradient - original_gradient).norm();

        gradient_difference->SetValue(i, difference);
        original_gradient_difference->SetValue(i, difference);

        error_min = std::min(error_min, difference);
        error_max = std::max(error_max, difference);
    }

    std::cout << "  " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
        - start).count() << duration_str << std::endl;

    // Iterative optimization
    std::cout << "Initializing optimization...";
    start = std::chrono::steady_clock::now();

    bool converged = false;

    const bool twoD = original_vector_field.dimensions()[2] == 1;
    const auto block_offset = 3;
    const auto block_size = (2 * block_offset + 1);
    const auto node_weight = 1.0 / (block_size * block_size * (twoD ? 1 : block_size));

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

    auto gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    gradient_descent->SetNumberOfComponents(3);
    gradient_descent->SetNumberOfTuples(vector_field_original->GetNumberOfTuples());

    auto deformed_positions = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_positions->SetName("Deformed Position");
    deformed_positions->SetNumberOfComponents(3);
    deformed_positions->SetNumberOfTuples(vector_field_original->GetNumberOfTuples());

    std::array<double, 3> point{};

    for (vtkIdType i = 0; i < vector_field_original->GetNumberOfTuples(); ++i)
    {
        deformed_grid->GetPoint(i, point.data());
        deformed_positions->SetTuple(i, point.data());
    }

    std::cout << "  " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
        - start).count() << duration_str << std::endl;

    Eigen::Vector3d temp;

    for (int step = 0; step < this->NumSteps && !converged; ++step)
    {
        std::cout << "Optimization step: " << (step + 1) << "/" << this->NumSteps << std::endl;
        start = std::chrono::steady_clock::now();

        gradient_descent->Fill(0.0);

        // For each 7x7(x7) block of nodes, calculate partial derivatives of the
        // curvature gradient difference in direction of the degrees of freedom.
        // Use gradient descent to perform a single step for respective center
        // vertex, minimizing its curvature gradient difference.
        std::cout << "  Computing gradient descent..." << std::endl;
        start_inner = std::chrono::steady_clock::now();

        const auto num_blocks =
            original_vector_field.dimensions()[0] *
            original_vector_field.dimensions()[1] *
            original_vector_field.dimensions()[2];
        std::size_t block_index = 0;

        for (int z = 0; z < original_vector_field.dimensions()[2]; ++z)
        {
            for (int y = 0; y < original_vector_field.dimensions()[1]; ++y)
            {
                for (int x = 0; x < original_vector_field.dimensions()[0]; ++x)
                {
                    std::cout << "    Block: " << (++block_index) << "/" << num_blocks;

                    start_block = std::chrono::steady_clock::now();

                    const std::array<std::array<int, 2>, 3> block_offsets{
                        std::array<int, 2>{std::min(x - block_offset, 0) + block_offset,
                            block_offset - (std::max(x + block_offset, original_vector_field.dimensions()[0] - 1) - (original_vector_field.dimensions()[0] - 1))},
                        std::array<int, 2>{std::min(y - block_offset, 0) + block_offset,
                            block_offset - (std::max(y + block_offset, original_vector_field.dimensions()[1] - 1) - (original_vector_field.dimensions()[1] - 1))},
                        std::array<int, 2>{std::min(z - block_offset, 0) + block_offset,
                            block_offset - (std::max(z + block_offset, original_vector_field.dimensions()[2] - 1) - (original_vector_field.dimensions()[2] - 1))} };

                    const std::array<int, 3> block_sizes{
                        block_offsets[0][1] + block_offsets[0][0] + 1,
                        block_offsets[1][1] + block_offsets[1][0] + 1,
                        block_offsets[2][1] + block_offsets[2][0] + 1 };

                    const auto block_size = block_sizes[0] * block_sizes[1] * block_sizes[2];

                    original_position_block->SetNumberOfTuples(block_size);
                    original_vector_block->SetNumberOfTuples(block_size);
                    new_position_block->SetNumberOfTuples(block_size);
                    new_vector_block->SetNumberOfTuples(block_size);

                    // Create grid block
                    std::chrono::time_point<std::chrono::steady_clock> start_block_part;
                    start_block_part = std::chrono::steady_clock::now();

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
                                const auto index_orig = index_x + original_vector_field.dimensions()[0]
                                    * (index_y + original_vector_field.dimensions()[1] * index_z);

                                original_grid->GetPoint(index_orig, temp.data());
                                original_position_block->SetTuple(index_block, temp.data());

                                vector_field_original->GetTuple(index_orig, temp.data());
                                original_vector_block->SetTuple(index_block, temp.data());

                                deformed_positions->GetTuple(index_orig, temp.data());
                                new_position_block->SetTuple(index_block, temp.data());
                            }
                        }
                    }

                    std::cout << "  " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
                        - start_block_part).count() << duration_str;

                    // For each degree of freedom, calculate derivative
                    start_block_part = std::chrono::steady_clock::now();

                    grid block_deformation(block_sizes, original_position_block, new_position_block);

                    Eigen::Vector3d temp_vector{};
                    Eigen::Matrix3d temp_jacobian{};

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
                                const auto index_block_center = block_offsets[0][0] + block_sizes[0]
                                    * (block_offsets[1][0] + block_sizes[1] * block_offsets[2][0]);
                                const auto index_orig = index_x + original_vector_field.dimensions()[0]
                                    * (index_y + original_vector_field.dimensions()[1] * index_z);

                                for (int d = 0; d < (twoD ? 2 : 3); ++d)
                                {
                                    // Move node first in respective direction
                                    new_position_block->SetComponent(index_block, d,
                                        new_position_block->GetComponent(index_block, d) + this->StepSize);

                                    // Compute Jacobians of deformation
                                    auto jacobians = gradient_field(block_deformation);

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

                                    const auto curvature = curvature_and_torsion(block);

                                    // Calculate difference between original and deformed curvature gradient for central node
                                    const auto index_center = x + original_vector_field.dimensions()[0]
                                        * (y + original_vector_field.dimensions()[1] * z);

                                    original_curvature.curvature_gradient->GetTuple(index_center, original_gradient.data());
                                    curvature.curvature_gradient->GetTuple(index_block_center, deformed_gradient.data());

                                    const auto difference = (deformed_gradient - original_gradient).norm();

                                    gradient_descent->SetComponent(index_orig, d, gradient_descent->GetComponent(index_orig, d)
                                        - node_weight *(difference - gradient_difference->GetValue(index_orig))); // TODO: weight?

                                    // Reset new positions
                                    new_position_block->SetComponent(index_block, d,
                                        new_position_block->GetComponent(index_block, d) - this->StepSize);
                                }
                            }
                        }
                    }

                    std::cout << "  " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
                        - start_block_part).count() << duration_str;

                    std::cout << "  -> " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
                        - start_block).count() << duration_str << std::endl;
                }
            }
        }

        std::cout << "  Finished computing gradient descent after " <<
            std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now() - start_inner).count() << duration_str << std::endl;

        // Apply gradient descent
        std::cout << "  Applying gradient descent...";
        start_inner = std::chrono::steady_clock::now();

        Eigen::Vector3d position, descent;

        for (int z = 0; z < original_vector_field.dimensions()[2]; ++z)
        {
            for (int y = 0; y < original_vector_field.dimensions()[1]; ++y)
            {
                for (int x = 0; x < original_vector_field.dimensions()[0]; ++x)
                {
                    const auto index = x + original_vector_field.dimensions()[0]
                        * (y + original_vector_field.dimensions()[1] * z);

                    deformed_positions->GetTuple(index, position.data());
                    gradient_descent->GetTuple(index, descent.data());

                    position += descent;

                    deformed_positions->SetTuple(index, position.data());
                }
            }
        }

        // Update jacobians and vector field to build an updated grid
        grid new_deformation(original_grid, deformed_positions);
        jacobian_field = gradient_field(new_deformation);

        Eigen::Matrix3d jacobian;
        Eigen::Vector3d vector;

        for (int z = 0; z < original_vector_field.dimensions()[2]; ++z)
        {
            for (int y = 0; y < original_vector_field.dimensions()[1]; ++y)
            {
                for (int x = 0; x < original_vector_field.dimensions()[0]; ++x)
                {
                    const auto index = x + original_vector_field.dimensions()[0]
                        * (y + original_vector_field.dimensions()[1] * z);

                    jacobian_field->GetTuple(index, jacobian.data());
                    vector_field_original->GetTuple(index, vector.data());

                    vector = jacobian * vector;

                    vector_field_deformed->SetTuple(index, vector.data());
                }
            }
        }

        // Calculate new gradient difference
        const grid new_deformed_vector_field(deformed_vector_field.dimensions(),
            deformed_positions, vector_field_deformed, jacobian_field);

        deformed_curvature = curvature_and_torsion(new_deformed_vector_field);

        double new_error_min = std::numeric_limits<double>::max();
        double new_error_max = std::numeric_limits<double>::min();

        for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
        {
            original_curvature.curvature_gradient->GetTuple(i, original_gradient.data());
            deformed_curvature.curvature_gradient->GetTuple(i, deformed_gradient.data());

            const auto difference = (deformed_gradient - original_gradient).norm();

            gradient_difference->SetValue(i, difference);

            new_error_min = std::min(new_error_min, difference);
            new_error_max = std::max(new_error_max, difference);
        }

        std::cout << "  " << std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
            - start_inner).count() << duration_str << std::endl;

        // Calculate new min, max error
        if (new_error_min > error_min)
        {
            std::cout << "New minimum error increased from " << error_min << " to " << new_error_min << "." << std::endl;
        }
        if (new_error_max > error_max)
        {
            std::cout << "New maximum error increased from " << error_max << " to " << new_error_max << "." << std::endl;
        }

        error_min = new_error_min;
        error_max = new_error_max;

        if (error_max < this->Error)
        {
            converged = true;

            std::cout << "  Optimization converged." << std::endl;
        }

        std::cout << "  Optimization step complete after " <<
            std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now()
                - start).count() << duration_str << std::endl;
    }

    // Set output
    std::cout << "Setting output..." << std::endl;

    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(deformed_positions->GetNumberOfTuples());

    for (vtkIdType i = 0; i < deformed_positions->GetNumberOfTuples(); ++i)
    {
        deformed_positions->GetTuple(i, point.data());
        points->SetPoint(i, point.data());
    }

    jacobian_field->SetName("Jacobian");

    auto output_grid = vtkStructuredGrid::GetData(output_vector);
    output_grid->SetDimensions(deformed_vector_field.dimensions().data());
    output_grid->SetPoints(points);

    output_grid->GetPointData()->AddArray(vector_field_deformed);
    output_grid->GetPointData()->AddArray(jacobian_field);
    output_grid->GetPointData()->AddArray(deformed_positions);
    output_grid->GetPointData()->AddArray(gradient_difference);
    output_grid->GetPointData()->AddArray(original_gradient_difference);
    output_grid->GetPointData()->AddArray(deformed_curvature.curvature);
    output_grid->GetPointData()->AddArray(deformed_curvature.curvature_vector);
    output_grid->GetPointData()->AddArray(deformed_curvature.curvature_gradient);
    output_grid->GetPointData()->AddArray(deformed_curvature.torsion);
    output_grid->GetPointData()->AddArray(deformed_curvature.torsion_vector);
    output_grid->GetPointData()->AddArray(deformed_curvature.torsion_gradient);

    return 1;
}
