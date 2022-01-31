#include "optimizer.h"

#include "ppm_io.h"

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
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

const auto block_offset = 5;
const auto block_inner_offset = 2;
const auto block_size = (2 * block_offset + 1);

#define __output_input
#define __output_curvatures

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

    if (original_grid->GetNumberOfPoints() != deformed_grid->GetNumberOfPoints())
    {
        std::cerr << std::endl << "Number of grid nodes must match." << std::endl;
        return 0;
    }

    auto vector_field_original = GetInputArrayToProcess(0, original_grid);

    if (vector_field_original == nullptr)
    {
        std::cerr << std::endl << "Input vector field must be provided." << std::endl;
        return 0;
    }

    auto feature_mapping = GetInputArrayToProcess(1, deformed_grid);
    auto feature_mapping_original = GetInputArrayToProcess(2, deformed_grid);

    const auto hash = joaat_hash(this->IgnoreBorder, this->NumSteps, this->StepSize,
        this-StepSizeMethod, this->Error, this->GradientMethod, this->GradientKernel, this->GradientStep,
        /*vector_field_original->GetMTime(),*/ deformed_grid->GetMTime(), // TODO: WTF?!
        feature_mapping->GetMTime(), feature_mapping_original->GetMTime());

    this->IgnoreBorder = std::max(this->IgnoreBorder, 0);

    if (hash != this->hash)
    {
        try
        {
            compute_gradient_descent(original_grid, deformed_grid, vector_field_original, feature_mapping_original, feature_mapping);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error computing gradient descent: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown error computing gradient descent" << std::endl;
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

void optimizer::compute_gradient_descent(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
    vtkDataArray* vector_field_original, vtkDataArray* original_feature_mapping, vtkDataArray* feature_mapping)
{
    const auto num_nodes = original_grid->GetNumberOfPoints();

    // Get initial node positions, and compute deformation
    auto original_positions = vtkSmartPointer<vtkDoubleArray>::New();
    original_positions->SetName("Positions (Original)");
    original_positions->SetNumberOfComponents(3);
    original_positions->SetNumberOfTuples(num_nodes);

    auto deformed_positions = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_positions->SetName("Positions");
    deformed_positions->SetNumberOfComponents(3);
    deformed_positions->SetNumberOfTuples(num_nodes);

    std::array<double, 3> point{};

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        original_grid->GetPoint(i, point.data());
        original_positions->SetTuple(i, point.data());

        deformed_grid->GetPoint(i, point.data());
        deformed_positions->SetTuple(i, point.data());
    }

    auto jacobian_field = gradient_field(grid(original_grid, deformed_positions), static_cast<gradient_method_t>(this->GradientMethod), this->GradientKernel);
    jacobian_field->SetName("Jacobian");

    // Calculate direction from node position to B-spline for directional derivative
    auto original_derivative_direction = vtkSmartPointer<vtkDoubleArray>::New();
    original_derivative_direction->SetName("Derivative Direction (Original)");
    original_derivative_direction->SetNumberOfComponents(3);
    original_derivative_direction->SetNumberOfTuples(num_nodes);

    auto transformed_derivative_direction = vtkSmartPointer<vtkDoubleArray>::New();
    transformed_derivative_direction->SetName("Derivative Direction (Transformed)");
    transformed_derivative_direction->SetNumberOfComponents(3);
    transformed_derivative_direction->SetNumberOfTuples(num_nodes);

    auto deformed_derivative_direction = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_derivative_direction->SetName("Derivative Direction");
    deformed_derivative_direction->SetNumberOfComponents(3);
    deformed_derivative_direction->SetNumberOfTuples(num_nodes);

    Eigen::Matrix3d jacobian;
    Eigen::Vector3d original_direction, transformed_direction, deformed_direction;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        jacobian_field->GetTuple(i, jacobian.data());
        original_feature_mapping->GetTuple(i, original_direction.data());
        feature_mapping->GetTuple(i, deformed_direction.data());

        original_direction.normalize();
        transformed_direction = (jacobian * original_direction).normalized();
        deformed_direction.normalize();

        original_derivative_direction->SetTuple(i, original_direction.data());
        transformed_derivative_direction->SetTuple(i, transformed_direction.data());
        deformed_derivative_direction->SetTuple(i, deformed_direction.data());
    }

    // Adjust vector field
    auto original_vector_field = vtkSmartPointer<vtkDoubleArray>::New();
    original_vector_field->SetName("Vector Field (Original)");
    original_vector_field->SetNumberOfComponents(3);
    original_vector_field->SetNumberOfTuples(num_nodes);

    auto deformed_vector_field = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_vector_field->SetName("Vector Field");
    deformed_vector_field->SetNumberOfComponents(3);
    deformed_vector_field->SetNumberOfTuples(num_nodes);

    Eigen::Vector3d vector;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        jacobian_field->GetTuple(i, jacobian.data());
        vector_field_original->GetTuple(i, vector.data());

        original_vector_field->SetTuple(i, vector.data());

        vector = jacobian * vector;

        deformed_vector_field->SetTuple(i, vector.data());
    }

    const std::array<int, 3> dimension = grid(original_grid, original_vector_field).dimensions();
    const bool twoD = dimension[2] == 1;

    // Calculate initial curvature
    const auto original_curvature = blockwise_curvature(dimension, 0.0, original_positions, original_vector_field, nullptr, original_derivative_direction);

    auto original_curvatures = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvatures->SetName("Curvature (Original)");
    original_curvatures->SetNumberOfComponents(1);
    original_curvatures->SetNumberOfTuples(num_nodes);

    auto original_curvature_vector = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvature_vector->SetName("Curvature Vector (Original)");
    original_curvature_vector->SetNumberOfComponents(3);
    original_curvature_vector->SetNumberOfTuples(num_nodes);

    auto transformed_curvature_vector = vtkSmartPointer<vtkDoubleArray>::New();
    transformed_curvature_vector->SetName("Curvature Vector (Transformed)");
    transformed_curvature_vector->SetNumberOfComponents(3);
    transformed_curvature_vector->SetNumberOfTuples(num_nodes);

    auto original_curvature_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvature_gradients->SetName("Curvature Gradient (Original)");
    original_curvature_gradients->SetNumberOfComponents(3);
    original_curvature_gradients->SetNumberOfTuples(num_nodes);

    auto transformed_curvature_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    transformed_curvature_gradients->SetName("Curvature Gradient (Transformed)");
    transformed_curvature_gradients->SetNumberOfComponents(3);
    transformed_curvature_gradients->SetNumberOfTuples(num_nodes);

    auto original_curvature_vector_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvature_vector_gradients->SetName("Curvature Vector Gradient (Original)");
    original_curvature_vector_gradients->SetNumberOfComponents(9);
    original_curvature_vector_gradients->SetNumberOfTuples(num_nodes);

    auto transformed_curvature_vector_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    transformed_curvature_vector_gradients->SetName("Curvature Vector Gradient (Transformed)");
    transformed_curvature_vector_gradients->SetNumberOfComponents(9);
    transformed_curvature_vector_gradients->SetNumberOfTuples(num_nodes);

    auto original_curvature_directional_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    original_curvature_directional_gradients->SetName("Directional Curvature Vector Gradient (Original)");
    original_curvature_directional_gradients->SetNumberOfComponents(3);
    original_curvature_directional_gradients->SetNumberOfTuples(num_nodes);

    auto transformed_curvature_directional_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    transformed_curvature_directional_gradients->SetName("Directional Curvature Vector Gradient (Transformed)");
    transformed_curvature_directional_gradients->SetNumberOfComponents(3);
    transformed_curvature_directional_gradients->SetNumberOfTuples(num_nodes);

    double curvature{};
    Eigen::Vector3d original_vector, original_gradient, original_directional_gradient;
    Eigen::Matrix3d original_vector_gradient, inv_transp_jacobian;

    for (vtkIdType i = 0; i < num_nodes; ++i)
    {
        original_curvature.curvature->GetTuple(i, &curvature);
        original_curvature.curvature_vector->GetTuple(i, original_vector.data());
        original_curvature.curvature_gradient->GetTuple(i, original_gradient.data());
        original_curvature.curvature_vector_gradient->GetTuple(i, original_vector_gradient.data());
        original_curvature.curvature_directional_gradient->GetTuple(i, original_directional_gradient.data());

        original_curvatures->SetTuple(i, &curvature);
        original_curvature_vector->SetTuple(i, original_vector.data());
        original_curvature_gradients->SetTuple(i, original_gradient.data());
        original_curvature_vector_gradients->SetTuple(i, original_vector_gradient.data());
        original_curvature_directional_gradients->SetTuple(i, original_directional_gradient.data());

        jacobian_field->GetTuple(i, jacobian.data());
        inv_transp_jacobian = jacobian.inverse().transpose();

        original_vector = jacobian * original_vector;
        original_gradient = inv_transp_jacobian * original_gradient;
        original_vector_gradient = inv_transp_jacobian * original_vector_gradient;
        original_directional_gradient = inv_transp_jacobian * original_directional_gradient;

        transformed_curvature_vector->SetTuple(i, original_vector.data());
        transformed_curvature_gradients->SetTuple(i, original_gradient.data());
        transformed_curvature_vector_gradients->SetTuple(i, original_vector_gradient.data());
        transformed_curvature_directional_gradients->SetTuple(i, original_directional_gradient.data());
    }



    // ----------------------------------------------------------------------------------------------------------------------------------------------------------



    // Calculate initial error
    const auto deformed_curvature = blockwise_curvature(dimension, 0.0, deformed_positions, deformed_vector_field, jacobian_field, deformed_derivative_direction);

    vtkSmartPointer<vtkDoubleArray> errors;
    double error_avg{}, error_max{}, min_error_avg, min_error_max;
    int min_error_avg_step, min_error_max_step;

    std::tie(errors, error_avg, error_max) = calculate_error_field(original_curvature, deformed_curvature, jacobian_field);

    const auto original_error_avg = min_error_avg = error_avg;
    const auto original_error_max = min_error_max = error_max;

    min_error_avg_step = min_error_max_step = 0;

    std::vector<double> errors_avg, errors_max;
    errors_avg.push_back(original_error_avg);
    errors_max.push_back(original_error_max);

    deformed_curvature.curvature->SetName("Curvature");
    deformed_curvature.curvature_vector->SetName("Curvature Vector");
    deformed_curvature.curvature_gradient->SetName("Curvature Gradient");
    deformed_curvature.curvature_vector_gradient->SetName("Curvature Vector Gradient");
    deformed_curvature.curvature_directional_gradient->SetName("Directional Curvature Vector Gradient");

    // Initialize arrays that have no meaning in the initial grid
    auto change = vtkSmartPointer<vtkDoubleArray>::New();
    change->SetName("Change");
    change->SetNumberOfComponents(1);
    change->SetNumberOfTuples(errors->GetNumberOfTuples());
    change->Fill(0.0);

    auto gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    gradient_descent->SetName("Gradient Descent");
    gradient_descent->SetNumberOfComponents(1);
    gradient_descent->SetNumberOfTuples(errors->GetNumberOfTuples());
    gradient_descent->Fill(0.0);

    auto adjusted_gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    adjusted_gradient_descent->SetName("Gradient Descent (Adjusted)");
    adjusted_gradient_descent->SetNumberOfComponents(1);
    adjusted_gradient_descent->SetNumberOfTuples(errors->GetNumberOfTuples());
    adjusted_gradient_descent->Fill(0.0);

    auto valid_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    valid_gradients->SetName("Valid Gradients");
    valid_gradients->SetNumberOfComponents(1);
    valid_gradients->SetNumberOfTuples(errors->GetNumberOfTuples());
    valid_gradients->Fill(1.0);

    // Set initial output
    if (!this->CSVOutput) std::cout << "Setting initial output..." << std::endl;

    this->results[0] = create_output(dimension, deformed_positions);

    output_copy(this->results[0],
#ifdef __output_input
        original_positions, deformed_positions, jacobian_field,                                             // grid positions and Jacobian (const.)
        original_derivative_direction, transformed_derivative_direction, deformed_derivative_direction,     // derivative directions (const.)
        original_vector_field,                                                                              // original vector field (const.)
#endif
#ifdef __output_curvatures
        original_curvatures, original_curvature_vector, transformed_curvature_vector,                       // \ 
        original_curvature_gradients, transformed_curvature_gradients,                                      //  | curvature and
        original_curvature_vector_gradients, transformed_curvature_vector_gradients,                        //  | curvature gradients (const.)
        original_curvature_directional_gradients, transformed_curvature_directional_gradients,              // / 
        deformed_curvature.curvature, deformed_curvature.curvature_vector,                                  // \ 
        deformed_curvature.curvature_gradient, deformed_curvature.curvature_vector_gradient,                //  | curvature and curvature gradients
        deformed_curvature.curvature_directional_gradient,                                                  // / 
#endif
        deformed_vector_field,                                                                              // vector field
        gradient_descent, adjusted_gradient_descent, valid_gradients,                                       // gradient descent
        errors, change);                                                                                    // error field and its difference



    // ----------------------------------------------------------------------------------------------------------------------------------------------------------



    // Apply optimization
    bool converged = false;
    bool stopped = false;

    double step_size = this->StepSize;
    int step;

    for (step = 0; step < this->NumSteps && !converged && !stopped; ++step)
    {
        if (!this->CSVOutput) std::cout << "Optimization step: " << (step + 1) << "/" << this->NumSteps << std::endl;

        // Perform line search (last iteration is the one producing results)
        bool all_gradients_valid{};

        std::tie(gradient_descent, valid_gradients, all_gradients_valid) = compute_descent(dimension, deformed_vector_field,
            original_curvature, jacobian_field, deformed_positions, errors, deformed_derivative_direction);

        std::tie(deformed_vector_field, adjusted_gradient_descent) = apply_descent(dimension, step_size, deformed_vector_field, errors, gradient_descent);

        // Calculate new gradient difference
        const auto deformed_curvature = blockwise_curvature(dimension, 0.0, deformed_positions, deformed_vector_field, jacobian_field, deformed_derivative_direction);

        vtkSmartPointer<vtkDoubleArray> new_errors;
        double new_error_avg{}, new_error_max{};

        std::tie(new_errors, new_error_avg, new_error_max) = calculate_error_field(original_curvature, deformed_curvature, jacobian_field);

        if (new_error_max > error_max)
        {
            if (!this->CSVOutput) std::cout << "    New maximum error increased from " << error_max << " to " << new_error_max << "." << std::endl;

            stopped = this->AbortWhenGrowing;
        }
        if (new_error_avg > error_avg)
        {
            if (!this->CSVOutput) std::cout << "    New average error increased from " << error_avg << " to " << new_error_avg << "." << std::endl;

            stopped = this->AbortWhenGrowing;
        }
        if (!all_gradients_valid)
        {
            if (!this->CSVOutput) std::cout << "    Invalid gradients found for current gradient step size " << this->GradientStep << "." << std::endl;

            stopped = this->AbortWhenGrowing;
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

        deformed_curvature.curvature->SetName("Curvature");
        deformed_curvature.curvature_vector->SetName("Curvature Vector");
        deformed_curvature.curvature_gradient->SetName("Curvature Gradient");
        deformed_curvature.curvature_vector_gradient->SetName("Curvature Vector Gradient");
        deformed_curvature.curvature_directional_gradient->SetName("Directional Curvature Vector Gradient");

        // Calculate change
        for (vtkIdType i = 0; i < change->GetNumberOfTuples(); ++i)
        {
            change->SetValue(i, new_errors->GetValue(i) - errors->GetValue(i));
        }

        // Set output for this step
        this->results[step + 1uLL] = create_output(dimension, deformed_positions);

        output_copy(this->results[step + 1uLL],
#ifdef __output_input
            original_positions, deformed_positions, jacobian_field,                                             // grid positions and Jacobian (const.)
            original_derivative_direction, transformed_derivative_direction, deformed_derivative_direction,     // derivative directions (const.)
            original_vector_field,                                                                              // original vector field (const.)
#endif
#ifdef __output_curvatures
            original_curvatures, original_curvature_vector, transformed_curvature_vector,                       // \ 
            original_curvature_gradients, transformed_curvature_gradients,                                      //  | curvature and
            original_curvature_vector_gradients, transformed_curvature_vector_gradients,                        //  | curvature gradients (const.)
            original_curvature_directional_gradients, transformed_curvature_directional_gradients,              // / 
            deformed_curvature.curvature, deformed_curvature.curvature_vector,                                  // \ 
            deformed_curvature.curvature_gradient, deformed_curvature.curvature_vector_gradient,                //  | curvature and curvature gradients
            deformed_curvature.curvature_directional_gradient,                                                  // / 
#endif
            deformed_vector_field,                                                                              // vector field
            gradient_descent, adjusted_gradient_descent, valid_gradients,                                       // gradient descent
            new_errors, change);                                                                                // new error field and its difference

        output_copy(this->results[step], gradient_descent, adjusted_gradient_descent, valid_gradients);

        // Prepare next iteration
        errors = new_errors;
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

        for (std::size_t i = step + 1uLL; i <= this->NumSteps; ++i)
        {
            this->results[i] = this->results[step];
        }
    }
    else
    {
        if (!this->CSVOutput) std::cout << "Finished computation without convergence." << std::endl;
    }

    // Output information
    output_info(original_error_avg, original_error_max, error_avg, error_max, min_error_avg, min_error_max,
        min_error_avg_step, min_error_max_step, errors_avg, errors_max);
}

std::tuple<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>, bool> optimizer::compute_descent(
    const std::array<int, 3>& dimension, const vtkDataArray* vector_field,
    const curvature_and_torsion_t& original_curvature, const vtkDataArray* jacobian_field,
    const vtkDataArray* positions, const vtkDataArray* errors, vtkDoubleArray* derivative_direction) const
{
    using duration_t = std::chrono::milliseconds;
    const std::string duration_str(" ms");

    const auto start = std::chrono::steady_clock::now();

    auto gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    gradient_descent->SetName("Gradient Descent");
    gradient_descent->SetNumberOfComponents(1);
    gradient_descent->SetNumberOfTuples(vector_field->GetNumberOfTuples());
    gradient_descent->Fill(0.0);

    auto valid_gradients = vtkSmartPointer<vtkDoubleArray>::New();
    valid_gradients->SetName("Valid Gradients");
    valid_gradients->SetNumberOfComponents(1);
    valid_gradients->SetNumberOfTuples(vector_field->GetNumberOfTuples());
    valid_gradients->Fill(0.0);

    auto good_first_direction = vtkSmartPointer<vtkDoubleArray>::New();
    good_first_direction->SetName("Good First Direction");
    good_first_direction->SetNumberOfComponents(1);
    good_first_direction->SetNumberOfTuples(vector_field->GetNumberOfTuples());
    good_first_direction->Fill(0.0);

    auto good_second_direction = vtkSmartPointer<vtkDoubleArray>::New();
    good_second_direction->SetName("Good Second Direction");
    good_second_direction->SetNumberOfComponents(1);
    good_second_direction->SetNumberOfTuples(vector_field->GetNumberOfTuples());
    good_second_direction->Fill(0.0);

    // Rotate vectors in (counter-)clock-wise direction, and calculate corresponding curvature (gradients)
    const bool twoD = dimension[2] == 1;

    bool all_valid = true;

    for (int d = 0; d < 2; ++d)
    {
        const auto sign = std::pow(-1.0, d);

        const double alpha = sign * this->GradientStep;

        const auto curvature_after_rotation = blockwise_curvature(dimension, alpha, positions, vector_field, jacobian_field, derivative_direction);

        // Calculate gradient
        for (int z = 0; z < dimension[2]; ++z)
        {
            #pragma omp parallel for reduction(&&: all_valid)
            for (int y = 0; y < dimension[1]; ++y)
            {
                for (int x = 0; x < dimension[0]; ++x)
                {
                    const auto index_center = x + dimension[0] * (y + dimension[1] * z);

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

                    // Calculate difference between original and deformed curvature gradient for all neighboring nodes
                    double old_error, new_error;
                    old_error = new_error = 0.0; // TODO: Gaussian weight?

                    for (int zz = (twoD ? 0 : -std::min(block_inner_offset, block_offsets[2][0]));
                        zz <= (twoD ? 0 : std::min(block_inner_offset, block_offsets[2][1])); ++zz)
                    {
                        const auto index_z = z + zz;

                        if (twoD || (index_z >= this->IgnoreBorder && index_z < dimension[2] - this->IgnoreBorder))
                        {
                            for (int yy = -std::min(block_inner_offset, block_offsets[1][0]);
                                yy <= std::min(block_inner_offset, block_offsets[1][1]); ++yy)
                            {
                                const auto index_y = y + yy;

                                if (index_y >= this->IgnoreBorder && index_y < dimension[1] - this->IgnoreBorder)
                                {
                                    for (int xx = -std::min(block_inner_offset, block_offsets[0][0]);
                                        xx <= std::min(block_inner_offset, block_offsets[0][1]); ++xx)
                                    {
                                        const auto index_x = x + xx;

                                        if (index_x >= this->IgnoreBorder && index_x < dimension[0] - this->IgnoreBorder)
                                        {
                                            const auto index_neighbor = index_x + dimension[0] * (index_y + dimension[1] * index_z);

                                            old_error += const_cast<vtkDataArray*>(errors)->GetComponent(index_neighbor, 0);
                                            new_error += calculate_error(index_neighbor, original_curvature,
                                                curvature_after_rotation, jacobian_field);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Use old and new error to compute gradient
                    const auto gradient = (new_error - old_error) / this->GradientStep;

                    // Check that exactly one rotation direction yields an improvement, i.e.,
                    // the gradient is positive in one, and negative in the other direction.
                    // If this is not the case, it is not a valid gradient, or we are at a local extremum!
                    if (d == 0)
                    {
                        good_first_direction->SetValue(index_center, (gradient < 0) ? 1.0 : 0.0);

                        if (gradient < 0)
                        {
                            gradient_descent->SetComponent(index_center, 0, -gradient);
                        }
                    }
                    else
                    {
                        good_second_direction->SetValue(index_center, (gradient < 0) ? 1.0 : 0.0);

                        if (gradient < 0)
                        {
                            gradient_descent->SetComponent(index_center, 0, gradient);
                        }

                        const bool good_first = good_first_direction->GetValue(index_center) != 0.0;
                        const bool good_second = good_second_direction->GetValue(index_center) != 0.0;

                        valid_gradients->SetValue(index_center, (good_first != good_second) ? 1.0 : 0.0);
                        all_valid = all_valid && (good_first != good_second);
                    }
                }
            }
        }
    }

    if (!this->CSVOutput) std::cout << "  Finished computing gradient descent after " <<
        std::chrono::duration_cast<duration_t>(std::chrono::steady_clock::now() - start).count() << duration_str << std::endl;

    return std::make_tuple(gradient_descent, valid_gradients, all_valid);
}

std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> optimizer::apply_descent(
    const std::array<int, 3>& dimension, const double step_size, const vtkDataArray* vector_field,
    const vtkDataArray* errors, const vtkDataArray* gradient_descent) const
{
    auto deformed_vector_field = vtkSmartPointer<vtkDoubleArray>::New();
    deformed_vector_field->SetName("Deformed Vectors");
    deformed_vector_field->SetNumberOfComponents(3);
    deformed_vector_field->SetNumberOfTuples(gradient_descent->GetNumberOfTuples());

    auto adjusted_gradient_descent = vtkSmartPointer<vtkDoubleArray>::New();
    adjusted_gradient_descent->SetName("Gradient Descent (Adjusted)");
    adjusted_gradient_descent->SetNumberOfComponents(1);
    adjusted_gradient_descent->SetNumberOfTuples(gradient_descent->GetNumberOfTuples());

    const bool twoD = dimension[2] == 1;

    const auto step_size_method = static_cast<step_size_method_t>(this->StepSizeMethod);

    adjusted_gradient_descent->Fill(0.0);
    deformed_vector_field->DeepCopy(const_cast<vtkDataArray*>(vector_field));

    for (int z = 0; z < dimension[2]; ++z)
    {
        #pragma omp parallel for
        for (int y = 0; y < dimension[1]; ++y)
        {
            Eigen::Vector3d old_vector, new_vector;

            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index = x + dimension[0] * (y + dimension[1] * z);

                const_cast<vtkDataArray*>(vector_field)->GetTuple(index, old_vector.data());
                auto descent = const_cast<vtkDataArray*>(gradient_descent)->GetComponent(index, 0);

                if (step_size_method == step_size_method_t::error)
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

                    descent = error / std::abs(descent);
                }
                else if (step_size_method == step_size_method_t::normalized)
                {
                    descent /= std::abs(descent);
                }

                descent *= step_size;

                new_vector = old_vector;
                new_vector[0] = old_vector[0] * std::cos(descent) - old_vector[1] * std::sin(descent);
                new_vector[1] = old_vector[0] * std::sin(descent) + old_vector[1] * std::cos(descent);

                adjusted_gradient_descent->SetComponent(index, 0, descent);
                deformed_vector_field->SetTuple(index, new_vector.data());
            }
        }
    }

    return std::make_pair(deformed_vector_field, adjusted_gradient_descent);
}

curvature_and_torsion_t optimizer::blockwise_curvature(const std::array<int, 3>& dimension,
    const double rotation, const vtkDataArray* positions, const vtkDataArray* vector_field,
    const vtkDataArray* jacobian_field, const vtkDataArray* derivative_direction) const
{
    const auto num_blocks = dimension[0] * dimension[1] * dimension[2];

    auto block_of_them_all = curvature_and_torsion_t::create(num_blocks);

    // For each 11x11(x11) block of nodes, calculate partial derivatives of the
    // curvature gradient difference in direction of the degrees of freedom.
    // Use gradient descent to perform a single step for respective center
    // vertex, minimizing its curvature gradient difference.
    const bool twoD = dimension[2] == 1;

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

                const auto index_block_center = block_offsets[0][0] + block_sizes[0]
                    * (block_offsets[1][0] + block_sizes[1] * block_offsets[2][0]);
                const auto index_center = x + dimension[0] * (y + dimension[1] * z);

                // Create grid block
                auto new_position_block = vtkSmartPointer<vtkDoubleArray>::New();
                new_position_block->SetNumberOfComponents(3);
                new_position_block->SetNumberOfTuples(0);
                new_position_block->SetNumberOfTuples(block_size);

                auto new_vector_block = vtkSmartPointer<vtkDoubleArray>::New();
                new_vector_block->SetNumberOfComponents(3);
                new_vector_block->SetNumberOfTuples(0);
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

                            const_cast<vtkDataArray*>(positions)->GetTuple(index_orig, temp.data());
                            new_position_block->SetTuple(index_block, temp.data());

                            const_cast<vtkDataArray*>(vector_field)->GetTuple(index_orig, temp.data());
                            new_vector_block->SetTuple(index_block, temp.data());
                        }
                    }
                }

                // Rotate central vector
                Eigen::Vector3d old_vector, new_vector;
                new_vector_block->GetTuple(index_block_center, new_vector.data());

                old_vector = new_vector;

                new_vector[0] = old_vector[0] * std::cos(rotation) - old_vector[1] * std::sin(rotation);
                new_vector[1] = old_vector[0] * std::sin(rotation) + old_vector[1] * std::cos(rotation);

                new_vector_block->SetTuple(index_block_center, new_vector.data());

                // Calculate cuvature and torsion
                grid block(block_sizes, new_position_block, new_vector_block, jacobian_field);

                const auto curvature = curvature_and_torsion(block, static_cast<gradient_method_t>(this->GradientMethod),
                    this->GradientKernel, derivative_direction);

                // Copy central values
                std::vector<double> buffer(9);
                auto copy_content = [&buffer, index_center, index_block_center](vtkDoubleArray* source, vtkDoubleArray* target)
                {
                    source->GetTuple(index_block_center, buffer.data());
                    target->SetTuple(index_center, buffer.data());
                };

                copy_content(curvature.first_derivative, block_of_them_all.first_derivative);
                copy_content(curvature.second_derivative, block_of_them_all.second_derivative);

                copy_content(curvature.curvature, block_of_them_all.curvature);
                copy_content(curvature.curvature_vector, block_of_them_all.curvature_vector);
                copy_content(curvature.curvature_gradient, block_of_them_all.curvature_gradient);
                copy_content(curvature.curvature_vector_gradient, block_of_them_all.curvature_vector_gradient);
                copy_content(curvature.curvature_directional_gradient, block_of_them_all.curvature_directional_gradient);

                copy_content(curvature.torsion, block_of_them_all.torsion);
                copy_content(curvature.torsion_vector, block_of_them_all.torsion_vector);
                copy_content(curvature.torsion_gradient, block_of_them_all.torsion_gradient);
                copy_content(curvature.torsion_vector_gradient, block_of_them_all.torsion_vector_gradient);
                copy_content(curvature.torsion_directional_gradient, block_of_them_all.torsion_directional_gradient);
            }
        }
    }

    return block_of_them_all;
}

double optimizer::calculate_error(const int index, const curvature_and_torsion_t& original_curvature,
    const curvature_and_torsion_t& deformed_curvature, const vtkDataArray* jacobian_field) const
{
    Eigen::Vector3d original_gradient, deformed_gradient;
    Eigen::Matrix3d jacobian;

    original_curvature.curvature_directional_gradient->GetTuple(index, original_gradient.data());
    deformed_curvature.curvature_directional_gradient->GetTuple(index, deformed_gradient.data());

    const_cast<vtkDataArray*>(jacobian_field)->GetTuple(index, jacobian.data());
    const auto inv_transp_jacobian = jacobian.inverse().transpose();

    original_gradient = inv_transp_jacobian * original_gradient; // TODO: correct?

    return (deformed_gradient - original_gradient).norm();
}

std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> optimizer::calculate_error_field(
    const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature,
    const vtkDataArray* jacobian_field) const
{
    auto errors = vtkSmartPointer<vtkDoubleArray>::New();
    errors->SetName("Error");
    errors->SetNumberOfComponents(1);
    errors->SetNumberOfTuples(original_curvature.curvature_vector_gradient->GetNumberOfTuples());

    double error_max = std::numeric_limits<double>::min();
    std::vector<double> error_avgs(original_curvature.curvature_vector_gradient->GetNumberOfTuples());

    for (vtkIdType i = 0; i < original_curvature.curvature_vector_gradient->GetNumberOfTuples(); ++i)
    {
        const auto error = calculate_error(i, original_curvature, deformed_curvature, jacobian_field);

        errors->SetValue(i, error);

        error_max = std::max(error_max, error);
        error_avgs[i] = error;
    }

    // Pyramid sum for more accuracy
    for (vtkIdType offset = 1; offset < original_curvature.curvature_vector_gradient->GetNumberOfTuples(); offset *= 2)
    {
        for (vtkIdType i = 0; i < original_curvature.curvature_vector_gradient->GetNumberOfTuples(); i += 2LL * offset)
        {
            if (i + offset < original_curvature.curvature_vector_gradient->GetNumberOfTuples())
            {
                error_avgs[i] += error_avgs[i + offset];
            }
        }
    }

    error_avgs[0] /= original_curvature.curvature_vector_gradient->GetNumberOfTuples();

    return std::make_tuple(errors, error_avgs[0], error_max);
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

void optimizer::output_info(const double original_error_avg, const double original_error_max, const double error_avg, const double error_max,
    const double min_error_avg, const double min_error_max, const int min_error_avg_step, const int min_error_max_step,
    const std::vector<double>& errors_avg, const std::vector<double>& errors_max) const
{
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
