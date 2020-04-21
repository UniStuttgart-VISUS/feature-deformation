#include "feature_deformation.h"

#include "b-spline.h"
#include "displacement.h"
#include "hash.h"
#include "smoothing.h"

#include "vtkCell.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkImageData.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPointSet.h"
#include "vtkPolyData.h"
#include "vtkPolyhedron.h"
#include "vtkRectilinearGrid.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredGrid.h"
#include "vtkUnstructuredGrid.h"

#include "Eigen/Dense"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{
    template <typename T>
    void create_or_get_data_object(const int index, vtkAlgorithm* output_algorithm, vtkInformationVector* output_info)
    {
        auto output = T::SafeDownCast(output_info->GetInformationObject(index)->Get(vtkDataObject::DATA_OBJECT()));

        if (!output)
        {
            output = T::New();
            output_info->GetInformationObject(index)->Set(vtkDataObject::DATA_OBJECT(), output);
            output_algorithm->GetOutputPortInformation(index)->Set(vtkDataObject::DATA_EXTENT_TYPE(), output->GetExtentType());
        }
    }

    int calc_index_point(const std::array<int, 3>& dimension, int x, int y, int z)
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    }

    int calc_index_cell(const std::array<int, 3>& dimension, int x, int y, int z)
    {
        x = std::min(std::max(0, x), dimension[0] - 2);
        y = std::min(std::max(0, y), dimension[1] - 2);
        z = std::min(std::max(0, z), dimension[2] - 2);

        return (z * (dimension[1] - 1) + y) * (dimension[0] - 1) + x;
    }
}

vtkStandardNewMacro(feature_deformation);

feature_deformation::feature_deformation()
{
    this->SetNumberOfInputPorts(3);
    this->SetNumberOfOutputPorts(4);
}

feature_deformation::~feature_deformation() {}

int feature_deformation::ProcessRequest(vtkInformation* request, vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Create an output object of the correct type.
    if (request->Has(vtkDemandDrivenPipeline::REQUEST_DATA_OBJECT()))
    {
        return this->RequestDataObject(request, input_vector, output_vector);
    }

    // Generate the data
    if (request->Has(vtkDemandDrivenPipeline::REQUEST_INFORMATION()))
    {
        return this->RequestInformation(request, input_vector, output_vector);
    }

    if (request->Has(vtkDemandDrivenPipeline::REQUEST_DATA()))
    {
        return this->RequestData(request, input_vector, output_vector);
    }

    if (request->Has(vtkStreamingDemandDrivenPipeline::REQUEST_UPDATE_EXTENT()))
    {
        return this->RequestUpdateExtent(request, input_vector, output_vector);
    }

    return this->Superclass::ProcessRequest(request, input_vector, output_vector);
}

int feature_deformation::RequestDataObject(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    create_or_get_data_object<vtkPolyData>(0, this, output_vector);
    create_or_get_data_object<vtkPolyData>(1, this, output_vector);
    create_or_get_data_object<vtkMultiBlockDataSet>(2, this, output_vector);
    create_or_get_data_object<vtkImageData>(3, this, output_vector);

    return 1;
}

int feature_deformation::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    // If iterative smoothing is selected, create time step values based on animation parameters
    std::array<double, 2> time_range;
    std::vector<double> time_steps;

    if (this->Method == 1)
    {
        time_range = { 0.0, 1.0 };
        time_steps.resize(this->MaxNumIterations);

        for (std::size_t i = 0; i < time_steps.size(); ++i)
        {
            time_steps[i] = i * (1.0 / (this->MaxNumIterations - 1));
        }
    }
    else
    {
        time_range = { 0.0, 0.0 };
        time_steps = { 0.0 };
    }

    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(1)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(2)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(3)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);

    return 1;
}

int feature_deformation::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
        info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
        return 1;
    }
    else if (port == 2)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
        info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
        return 1;
    }

    return 0;
}

int feature_deformation::FillOutputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
        return 1;
    }
    else if (port == 2)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
        return 1;
    }
    else if (port == 3)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
        return 1;
    }

    return 1;
}

int feature_deformation::RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    return 1;
}

int feature_deformation::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Output info
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Starting deformation" << std::endl << std::endl;

    // Get time
    const auto time = output_vector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

    // Get parameters
    cache_parameter_lines();
    cache_parameter_smoothing(time);
    cache_parameter_displacement();
    cache_parameter_output_grid();

    if (this->parameter_lines.method == smoothing::method_t::smoothing)
    {
        std::cout << "Time: " << time << std::endl << std::endl;
    }

    // Get input
    cache_input_grid(input_vector[0]);
    cache_input_lines(input_vector[1]);
    cache_input_geometry(input_vector[2]);

    std::cout << std::endl;

    // Sanity checks
    if (!this->input_lines.valid || !parameter_checks())
    {
        return 0;
    }

    // Smooth line and store results in cache
    if (this->parameter_lines.modified || this->parameter_smoothing.modified || this->input_lines.modified)
    {
        std::cout << "Smoothing line" << std::endl;

        // Create smoother
        smoothing smoother(this->input_lines.selected_line, this->parameter_lines.method, this->parameter_smoothing.variant,
            this->parameter_smoothing.lambda, this->parameter_smoothing.mu, this->parameter_smoothing.max_num_iterations);

        // Straighten the selected line
        while (smoother.has_step())
        {
            // Perform smoothing step
            smoother.next_step();
        }

        // Replace the input line, in case resampling was done by the smoother
        const auto zero_displacement = smoother.get_displacement();

        for (std::size_t i = 0; i < zero_displacement.size(); ++i)
        {
            this->input_lines.lines[this->parameter_lines.selected_line_id][i][0] = zero_displacement[i].first[0];
            this->input_lines.lines[this->parameter_lines.selected_line_id][i][1] = zero_displacement[i].first[1];
            this->input_lines.lines[this->parameter_lines.selected_line_id][i][2] = zero_displacement[i].first[2];
        }

        // Store results in cache
        std::tie(this->results_smoothing.positions, this->results_smoothing.displacements) = get_displacements(smoother.get_displacement());

        this->results_smoothing.valid = true;
        this->results_smoothing.modified = true;
    }
    else
    {
        std::cout << "Loading smoothed line from cache" << std::endl;
    }

    if (!this->results_smoothing.valid)
    {
        std::cerr << "Smoothing results not valid" << std::endl;
        return 0;
    }

    std::cout << std::endl;

    // Displace grid points and store results in cache
    if (this->input_grid.valid && this->parameter_output_grid.output_deformed_grid)
    {
        std::cout << "Displacing grid points..." << std::endl;

        // Upload new points, if input grid was modified
        if (this->input_grid.modified)
        {
            std::cout << "  uploading points to the GPU" << std::endl;

            this->results_grid_displacement.displacement = std::make_shared<cuda::displacement>(
                std::array<double, 3>{ this->input_grid.origin[0], this->input_grid.origin[1], this->input_grid.origin[2] },
                std::array<double, 3>{ this->input_grid.spacing[0], this->input_grid.spacing[1], this->input_grid.spacing[2] },
                this->input_grid.dimension);
        }

        // Pre-compute B-Spline mapping, if relevant parameters changed, or the input grid or lines were modified
        if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
            this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
        {
            const uint32_t precomputation_hash = hash(this->parameter_lines.selected_line_id, this->parameter_displacement.bspline_parameters.degree,
                this->parameter_displacement.bspline_parameters.iterations, this->input_grid.hash, this->input_lines.hash);

            if (precomputation_hash != this->results_grid_displacement.hash)
            {
                std::cout << "  precomputing B-Spline mapping on the GPU" << std::endl;

                this->results_grid_displacement.displacement->precompute(this->parameter_displacement.parameters, this->results_smoothing.positions);

                this->results_grid_displacement.hash = precomputation_hash;
            }
        }

        // Displace grid points
        if (this->parameter_displacement.modified || this->input_grid.modified || this->results_smoothing.modified)
        {
            std::cout << "  calculating new positions on the GPU" << std::endl;

            this->results_grid_displacement.displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters,
                this->results_smoothing.positions, this->results_smoothing.displacements);
        }

        this->results_grid_displacement.valid = true;
        this->results_grid_displacement.modified = true;
    }

    // Displace line points and store results in cache
    {
        std::cout << "Displacing line points..." << std::endl;

        // Upload new points, if input lines were modified
        if (this->input_lines.modified)
        {
            std::cout << "  uploading points to the GPU" << std::endl;

            std::vector<std::array<float, 3>> line_points;

            for (const auto& line : this->input_lines.lines)
            {
                for (const auto& point : line)
                {
                    line_points.push_back(point);
                }
            }

            this->results_line_displacement.displacement = std::make_shared<cuda::displacement>(line_points);
        }

        // Pre-compute B-Spline mapping, if relevant parameters changed, or the lines were modified
        if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
            this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
        {
            const uint32_t precomputation_hash = hash(this->parameter_lines.selected_line_id, this->parameter_displacement.bspline_parameters.degree,
                this->parameter_displacement.bspline_parameters.iterations, this->input_lines.hash);

            if (precomputation_hash != this->results_line_displacement.hash)
            {
                std::cout << "  precomputing B-Spline mapping on the GPU" << std::endl;

                this->results_line_displacement.displacement->precompute(this->parameter_displacement.parameters, this->results_smoothing.positions);

                this->results_line_displacement.hash = precomputation_hash;
            }
        }

        // Displace line points
        if (this->parameter_displacement.modified || this->results_smoothing.modified)
        {
            std::cout << "  calculating new positions on the GPU" << std::endl;

            this->results_line_displacement.displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters,
                this->results_smoothing.positions, this->results_smoothing.displacements);
        }

        this->results_line_displacement.valid = true;
        this->results_line_displacement.modified = true;
    }

    // Displace geometry points and store results in cache
    if (this->input_geometry.valid)
    {
        std::cout << "Displacing geometry points..." << std::endl;

        // Upload new points, if input geometry was modified
        if (this->input_geometry.modified)
        {
            std::cout << "  uploading points to the GPU" << std::endl;

            this->results_geometry_displacement.displacement = std::make_shared<cuda::displacement>(this->input_geometry.geometry);
        }

        // Pre-compute B-Spline mapping, if relevant parameters changed, or the input geometry or lines were modified
        if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
            this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
        {
            const uint32_t precomputation_hash = hash(this->parameter_lines.selected_line_id, this->parameter_displacement.bspline_parameters.degree,
                this->parameter_displacement.bspline_parameters.iterations, this->input_lines.hash, this->input_geometry.hash);

            if (precomputation_hash != this->results_geometry_displacement.hash)
            {
                std::cout << "  precomputing B-Spline mapping on the GPU" << std::endl;

                this->results_geometry_displacement.displacement->precompute(this->parameter_displacement.parameters, this->results_smoothing.positions);

                this->results_geometry_displacement.hash = precomputation_hash;
            }
        }

        // Displace grid points
        if (this->parameter_displacement.modified || this->results_smoothing.modified || this->input_geometry.modified)
        {
            std::cout << "  calculating new positions on the GPU" << std::endl;

            this->results_geometry_displacement.displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters,
                this->results_smoothing.positions, this->results_smoothing.displacements);
        }

        this->results_geometry_displacement.valid = true;
        this->results_geometry_displacement.modified = true;
    }

    std::cout << std::endl;

    // Output grid
    if (this->results_grid_displacement.valid && (this->results_grid_displacement.modified || this->parameter_output_grid.modified))
    {
        if (this->parameter_output_grid.output_deformed_grid)
        {
            std::cout << "Creating deformed grid output" << std::endl;

            // Create structured or unstructured grid
            auto out_deformed_grid_info = output_vector->GetInformationObject(2);
            vtkSmartPointer<vtkPointSet> output_deformed_grid = nullptr;

            if (this->parameter_output_grid.remove_cells)
            {
                output_deformed_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
            }
            else
            {
                output_deformed_grid = vtkSmartPointer<vtkStructuredGrid>::New();
                vtkStructuredGrid::SafeDownCast(output_deformed_grid)->SetExtent(this->input_grid.extent.data());
            }

            vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->SetBlock(0u, output_deformed_grid);
            vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->GetMetaData(0u)->Set(vtkCompositeDataSet::NAME(), "Grid");

            // Deform grid
            create_undeformed_grid(output_deformed_grid, this->input_grid.extent, this->input_grid.dimension, this->input_grid.origin, this->input_grid.spacing);
            set_output_deformed_grid(output_deformed_grid, *this->results_grid_displacement.displacement);

            if (this->parameter_output_grid.remove_cells)
            {
                // Create second grid for the "removed" cells
                auto output_deformed_grid_removed = vtkSmartPointer<vtkUnstructuredGrid>::New();
                output_deformed_grid_removed->ShallowCopy(vtkUnstructuredGrid::SafeDownCast(output_deformed_grid));

                vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->SetBlock(1u, output_deformed_grid_removed);
                vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->GetMetaData(1u)->Set(vtkCompositeDataSet::NAME(), "Removed Cells");

                // Create cells
                create_cells(vtkUnstructuredGrid::SafeDownCast(output_deformed_grid), output_deformed_grid_removed, this->input_grid.dimension, this->input_grid.spacing);
            }

            // Create displacement field and use it to "deform" the velocities
            create_displacement_field(output_deformed_grid);

            if (this->input_grid.input_data.valid && this->parameter_output_grid.output_vector_field)
            {
                std::cout << "Calculating deformed velocities" << std::endl;

                deform_velocities(output_deformed_grid, this->input_grid.input_data.data, this->input_grid.dimension, this->input_grid.spacing);

                // Resample the deformed grid on the original one
                if (this->parameter_output_grid.output_resampled_grid)
                {
                    std::cout << "Creating resampled grid output" << std::endl;

                    auto out_resampled_grid_info = output_vector->GetInformationObject(3);
                    auto output_resampled_grid = vtkImageData::SafeDownCast(out_resampled_grid_info->Get(vtkDataObject::DATA_OBJECT()));

                    output_resampled_grid->DeepCopy(this->input_grid.grid);

                    resample_grid(output_deformed_grid, output_resampled_grid, this->input_grid.input_data.data->GetName(),
                        this->input_grid.dimension, this->input_grid.origin, this->input_grid.spacing);

                    out_resampled_grid_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
                }
            }

            out_deformed_grid_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
            this->Modified();
        }
    }
    else if (this->parameter_output_grid.output_deformed_grid && !this->results_grid_displacement.modified)
    {
        std::cout << "Loading deformed grid output from cache" << std::endl;
    }

    // Output lines
    if (this->results_line_displacement.valid && this->results_line_displacement.modified)
    {
        auto out_deformed_lines_info = output_vector->GetInformationObject(0);
        auto output_deformed_lines = vtkPolyData::SafeDownCast(out_deformed_lines_info->Get(vtkDataObject::DATA_OBJECT()));

        set_output_deformed_lines(vtkPolyData::SafeDownCast(input_vector[1]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())), 
            output_deformed_lines, *this->results_line_displacement.displacement, this->input_lines.modified, this->output_lines, "lines");

        out_deformed_lines_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
        this->Modified();
    }
    else if (!this->results_line_displacement.modified)
    {
        std::cout << "Loading deformed line output from cache" << std::endl;
    }

    // Output geometry
    if (this->results_geometry_displacement.valid && this->results_geometry_displacement.modified)
    {
        auto out_deformed_geometry_info = output_vector->GetInformationObject(1);
        auto output_deformed_geometry = vtkPolyData::SafeDownCast(out_deformed_geometry_info->Get(vtkDataObject::DATA_OBJECT()));

        set_output_deformed_lines(vtkPolyData::SafeDownCast(input_vector[2]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())),
            output_deformed_geometry, *this->results_geometry_displacement.displacement, this->input_geometry.modified, this->output_geometry, "geometry");

        out_deformed_geometry_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
        this->Modified();
    }
    else if (!this->results_geometry_displacement.modified)
    {
        std::cout << "Loading deformed geometry output from cache" << std::endl;
    }

    // Output info
    std::cout << std::endl << "Finished deformation" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // Set modification to false for the next run
    this->parameter_lines.modified = false;
    this->parameter_smoothing.modified = false;
    this->parameter_displacement.modified = false;
    this->parameter_output_grid.modified = false;

    this->input_grid.modified = false;
    this->input_grid.input_data.modified = false;
    this->input_lines.modified = false;
    this->input_geometry.modified = false;

    this->results_smoothing.modified = false;
    this->results_grid_displacement.modified = false;
    this->results_line_displacement.modified = false;
    this->results_geometry_displacement.modified = false;

    return 1;
}


/// Parameters

void feature_deformation::cache_parameter_lines()
{
    if (this->parameter_lines.selected_line_id != this->LineID ||
        this->parameter_lines.method != static_cast<smoothing::method_t>(this->Method))
    {
        // Set cached parameters
        this->parameter_lines.selected_line_id = this->LineID;

        this->parameter_lines.method = static_cast<smoothing::method_t>(this->Method);

        // Parameters have been modified
        this->parameter_lines.modified = true;
    }
}

void feature_deformation::cache_parameter_smoothing(const double time)
{
    // In case of animation, set modified number of iterations
    int num_iterations = this->MaxNumIterations;

    if (this->parameter_lines.method == smoothing::method_t::smoothing)
    {
        if (this->Interpolator == 0)
        {
            // Linear
            num_iterations *= std::min(time, 1.0);
        }
        else
        {
            // Exponential
            if (time == 0.0)
            {
                num_iterations = 0;
            }
            else if (time < 1.0)
            {
                num_iterations = std::pow(2.0, time * std::log2(num_iterations + 1)) - 1;
            }
        }
    }

    if (this->parameter_smoothing.variant != static_cast<smoothing::variant_t>(this->Variant) ||
        this->parameter_smoothing.lambda != static_cast<float>(this->Lambda) ||
        this->parameter_smoothing.mu != static_cast<float>(this->Mu) ||
        this->parameter_smoothing.max_num_iterations != num_iterations)
    {
        // Set cached parameters
        this->parameter_smoothing.variant = static_cast<smoothing::variant_t>(this->Variant);

        this->parameter_smoothing.lambda = static_cast<float>(this->Lambda);
        this->parameter_smoothing.mu = static_cast<float>(this->Mu);

        this->parameter_smoothing.max_num_iterations = num_iterations;

        // Parameters have been modified
        this->parameter_smoothing.modified = true;
    }
}

void feature_deformation::cache_parameter_displacement()
{
    if (this->parameter_displacement.method != static_cast<cuda::displacement::method_t>(this->Weight) ||
        this->parameter_displacement.idw_parameters.exponent != static_cast<float>(this->EpsilonScalar) ||
        this->parameter_displacement.idw_parameters.neighborhood != static_cast<float>(this->VoronoiDistance) ||
        this->parameter_displacement.projection_parameters.gauss_parameter != static_cast<float>(this->GaussParameter) ||
        this->parameter_displacement.bspline_parameters.degree != this->SplineDegree ||
        this->parameter_displacement.bspline_parameters.gauss_parameter != static_cast<float>(this->GaussParameter) ||
        this->parameter_displacement.bspline_parameters.iterations != this->Subdivisions)
    {
        // Set cached parameters
        this->parameter_displacement.method = static_cast<cuda::displacement::method_t>(this->Weight);

        switch (this->parameter_displacement.method)
        {
        case cuda::displacement::method_t::greedy:
        case cuda::displacement::method_t::voronoi:
            this->parameter_displacement.parameters.inverse_distance_weighting.exponent = static_cast<float>(this->EpsilonScalar);
            this->parameter_displacement.parameters.inverse_distance_weighting.neighborhood = this->VoronoiDistance;

            break;
        case cuda::displacement::method_t::greedy_joints:
            this->parameter_displacement.parameters.inverse_distance_weighting.exponent = static_cast<float>(this->EpsilonScalar);

            break;
        case cuda::displacement::method_t::projection:
            this->parameter_displacement.parameters.projection.gauss_parameter = static_cast<float>(this->GaussParameter);

            break;
        case cuda::displacement::method_t::b_spline:
        case cuda::displacement::method_t::b_spline_joints:
            this->parameter_displacement.parameters.b_spline.degree = this->SplineDegree;
            this->parameter_displacement.parameters.b_spline.gauss_parameter = static_cast<float>(this->GaussParameter);
            this->parameter_displacement.parameters.b_spline.iterations = this->Subdivisions;

            break;
        }

        // Store parameters separately
        this->parameter_displacement.idw_parameters.exponent = static_cast<float>(this->EpsilonScalar);
        this->parameter_displacement.idw_parameters.neighborhood = this->VoronoiDistance;
        this->parameter_displacement.projection_parameters.gauss_parameter = static_cast<float>(this->GaussParameter);
        this->parameter_displacement.bspline_parameters.degree = this->SplineDegree;
        this->parameter_displacement.bspline_parameters.gauss_parameter = static_cast<float>(this->GaussParameter);
        this->parameter_displacement.bspline_parameters.iterations = this->Subdivisions;

        // Parameters have been modified
        this->parameter_displacement.modified = true;
    }
}

void feature_deformation::cache_parameter_output_grid()
{
    if (this->parameter_output_grid.output_deformed_grid != (this->OutputDeformedGrid != 0) ||
        this->parameter_output_grid.output_vector_field != (this->OutputVectorField != 0) ||
        this->parameter_output_grid.output_resampled_grid != (this->OutputResampledGrid != 0) ||
        this->parameter_output_grid.remove_cells != (this->RemoveCells != 0) ||
        this->parameter_output_grid.remove_cells_scalar != static_cast<float>(this->RemoveCellsScalar))
    {
        // Set cached parameters
        this->parameter_output_grid.output_deformed_grid = (this->OutputDeformedGrid != 0);
        this->parameter_output_grid.output_vector_field = (this->OutputVectorField != 0);
        this->parameter_output_grid.output_resampled_grid = (this->OutputResampledGrid != 0);

        this->parameter_output_grid.remove_cells = (this->RemoveCells != 0);
        this->parameter_output_grid.remove_cells_scalar = static_cast<float>(this->RemoveCellsScalar);

        // Parameters have been modified
        this->parameter_output_grid.modified = true;
    }
}

bool feature_deformation::parameter_checks() const
{
    if (this->Method == 1)
    {
        if (this->Lambda <= 0.0)
        {
            std::cerr << "Lambda must be larger than zero to achieve smoothing." << std::endl;
            return false;
        }

        if (this->Mu > 0.0)
        {
            std::cerr << "Mu must not be larger than zero to achieve inflation (or zero for pure Gaussian smoothing)." << std::endl;
            return false;
        }

        if (this->Mu == 0.0 && static_cast<smoothing::variant_t>(this->Variant) == smoothing::variant_t::fixed_arclength)
        {
            std::cerr << "Mu must not be zero to achieve inflation." << std::endl;
            return false;
        }

        if (-this->Mu < this->Lambda && static_cast<smoothing::variant_t>(this->Variant) == smoothing::variant_t::fixed_arclength)
        {
            std::cerr << "The absolute of mu must not be smaller than lambda to achieve inflation." << std::endl;
            return false;
        }
    }

    return true;
}


/// Input

void feature_deformation::cache_input_grid(vtkInformationVector* input_grid_vector)
{
    if ((input_grid_vector == nullptr || input_grid_vector->GetInformationObject(0) == nullptr) && this->parameter_output_grid.output_deformed_grid)
    {
        std::cout << "Warning: Cannot show output grid because there is no input" << std::endl;
        this->parameter_output_grid.output_deformed_grid = false;
    }

    if (input_grid_vector != nullptr && input_grid_vector->GetInformationObject(0) != nullptr && this->parameter_output_grid.output_deformed_grid)
    {
        // Get grid
        this->input_grid.grid = vtkImageData::SafeDownCast(input_grid_vector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

        if (this->input_grid.grid != nullptr)
        {
            // Get extents
            this->input_grid.grid->GetExtent(this->input_grid.extent.data());
            this->input_grid.grid->GetDimensions(this->input_grid.dimension.data());

            // Get origin and spacing
            std::array<double, 3> origin_data;
            std::array<double, 3> spacing_data;

            this->input_grid.grid->GetOrigin(origin_data.data());
            this->input_grid.grid->GetSpacing(spacing_data.data());

            this->input_grid.origin << static_cast<float>(origin_data[0]), static_cast<float>(origin_data[1]), static_cast<float>(origin_data[2]);
            this->input_grid.spacing << static_cast<float>(spacing_data[0]), static_cast<float>(spacing_data[1]), static_cast<float>(spacing_data[2]);

            this->input_grid.origin += Eigen::Vector3f(this->input_grid.extent[0], this->input_grid.extent[2],
                this->input_grid.extent[4]).cwiseProduct(this->input_grid.spacing);

            // Compute hash
            const auto new_hash = hash(this->input_grid.dimension[0], this->input_grid.dimension[1], this->input_grid.dimension[2],
                origin_data[0], origin_data[1], origin_data[2], spacing_data[0], spacing_data[1], spacing_data[2]);

            if (this->input_grid.hash != new_hash)
            {
                std::cout << "Loading input grid" << std::endl;

                // Input has been modified
                this->input_grid.hash = new_hash;
                this->input_grid.modified = true;
            }
            else
            {
                std::cout << "Loading input grid from cache" << std::endl;
            }

            this->input_grid.valid = true;
        }
        else
        {
            this->input_grid.valid = false;
        }

        // Get data array
        this->input_grid.input_data.data = this->GetInputArrayToProcess(0, &input_grid_vector);

        if (this->input_grid.input_data.data != nullptr && this->parameter_output_grid.output_vector_field &&
            this->input_grid.input_data.hash != this->GetInputArrayToProcess(0, &input_grid_vector)->GetMTime())
        {
            std::cout << "Loading input vector field" << std::endl;

            // Input has been modified
            this->input_grid.input_data.hash = this->GetInputArrayToProcess(0, &input_grid_vector)->GetMTime();
            this->input_grid.input_data.modified = true;

            // Input is now (preliminarily) valid
            this->input_grid.input_data.valid = true;

            // Sanity check
            if (this->input_grid.input_data.data->GetNumberOfComponents() != 3)
            {
                std::cerr << "Input array must be a vector field" << std::endl;

                this->input_grid.input_data.valid = false;
            }
        }
        else if (this->input_grid.input_data.data != nullptr && this->parameter_output_grid.output_vector_field)
        {
            std::cout << "Loading input vector field from cache" << std::endl;
        }
        else
        {
            this->input_grid.input_data.valid = false;
        }
    }
}

void feature_deformation::cache_input_lines(vtkInformationVector* input_lines_vector)
{
    // Get lines
    auto vtk_input_lines = vtkPolyData::SafeDownCast(input_lines_vector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

    if (vtk_input_lines != nullptr)
    {
        // Calculate hash
        std::array<double, 6> bounds;
        vtk_input_lines->GetBounds(bounds.data());

        const auto new_hash = hash(vtk_input_lines->GetNumberOfCells(), vtk_input_lines->GetNumberOfPoints(),
            bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);

        if (this->input_lines.hash != new_hash || this->parameter_lines.modified)
        {
            std::cout << "Loading input lines" << std::endl;

            // Input has been modified
            this->input_lines.hash = new_hash;
            this->input_lines.modified = true;

            // Input is now (preliminarily) valid
            this->input_lines.valid = true;

            // Sanity checks
            if (vtk_input_lines->GetLines()->GetNumberOfCells() == 0)
            {
                this->input_lines.lines.clear();
                this->input_lines.selected_line.clear();

                std::cout << "No lines -- nothing to do" << std::endl;

                this->input_lines.valid = false;
                return;
            }

            if (this->parameter_lines.selected_line_id >= vtk_input_lines->GetLines()->GetNumberOfCells())
            {
                std::cerr << "Selected line does not exist" << std::endl;

                this->input_lines.valid = false;
                return;
            }

            // Get lines and extract selected line
            this->input_lines.lines.resize(vtk_input_lines->GetLines()->GetNumberOfCells());

            auto point_list = vtkSmartPointer<vtkIdList>::New();
            std::size_t line_index = 0;

            vtk_input_lines->GetLines()->InitTraversal();
            while (vtk_input_lines->GetLines()->GetNextCell(point_list))
            {
                this->input_lines.lines[line_index].resize(point_list->GetNumberOfIds());

                if (line_index == this->parameter_lines.selected_line_id)
                {
                    this->input_lines.selected_line.resize(point_list->GetNumberOfIds());
                }

                for (vtkIdType point_index = 0; point_index < point_list->GetNumberOfIds(); ++point_index)
                {
                    std::array<double, 3> point;
                    vtk_input_lines->GetPoints()->GetPoint(point_list->GetId(point_index), point.data());

                    this->input_lines.lines[line_index][point_index] = { static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]) };

                    if (line_index == this->parameter_lines.selected_line_id)
                    {
                        this->input_lines.selected_line[point_index] << static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]);
                    }
                }

                ++line_index;
            }

            if (this->input_lines.selected_line.size() < 3)
            {
                std::cout << "Line consists only of one segment -- nothing to do" << std::endl;

                this->input_lines.valid = false;
                return;
            }
        }
        else
        {
            std::cout << "Loading input lines from cache" << std::endl;
        }
    }
    else
    {
        this->input_lines.valid = false;
    }
}

void feature_deformation::cache_input_geometry(vtkInformationVector* input_geometry_vector)
{
    if (input_geometry_vector != nullptr && input_geometry_vector->GetInformationObject(0) != nullptr)
    {
        // Get geometry
        auto vtk_input_geometry = vtkPolyData::SafeDownCast(input_geometry_vector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

        if (vtk_input_geometry != nullptr)
        {
            // Calculate hash
            std::array<double, 6> bounds;
            vtk_input_geometry->GetBounds(bounds.data());

            const auto new_hash = hash(vtk_input_geometry->GetNumberOfCells(), vtk_input_geometry->GetNumberOfPoints(),
                bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);

            if (this->input_geometry.hash != new_hash)
            {
                std::cout << "Loading input geometry" << std::endl;

                // Input has been modified
                this->input_geometry.hash = new_hash;
                this->input_geometry.modified = true;

                // Input is now valid
                this->input_geometry.valid = true;

                // Get geometry
                this->input_geometry.geometry.resize(vtk_input_geometry->GetNumberOfPoints());

                #pragma omp parallel for
                for (vtkIdType p = 0; p < vtk_input_geometry->GetNumberOfPoints(); ++p)
                {
                    std::array<double, 3> point;
                    vtk_input_geometry->GetPoints()->GetPoint(p, point.data());

                    this->input_geometry.geometry[p] = { static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]) };
                }
            }
            else
            {
                std::cout << "Loading input geometry from cache" << std::endl;
            }
        }
        else
        {
            this->input_geometry.valid = false;
        }
    }
    else
    {
        this->input_geometry.valid = false;
    }
}


/// Convenience

std::pair<std::vector<std::array<float, 4>>, std::vector<std::array<float, 4>>> feature_deformation::get_displacements(
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& displacement) const
{
    std::vector<std::array<float, 4>> positions, displacements;
    positions.resize(displacement.size());
    displacements.resize(displacement.size());

    for (std::size_t i = 0; i < displacements.size(); ++i)
    {
        positions[i] = { displacement[i].first[0], displacement[i].first[1], displacement[i].first[2], 1.0f };
        displacements[i] = { displacement[i].second[0], displacement[i].second[1], displacement[i].second[2], 0.0f };
    }

    return std::make_pair(positions, displacements);
}


/// Output

void feature_deformation::create_undeformed_grid(vtkPointSet* output_deformed_grid, const std::array<int, 6>& extent,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const
{
    // Create point nodes
    auto coords = vtkSmartPointer<vtkPoints>::New();
    coords->SetNumberOfPoints(dimension[0] * dimension[1] * dimension[2]);

    auto tex_coords = vtkSmartPointer<vtkFloatArray>::New();
    tex_coords->SetNumberOfComponents(3);
    tex_coords->SetNumberOfTuples(dimension[0] * dimension[1] * dimension[2]);
    tex_coords->SetName("Original Coordinates");

    vtkIdType index = 0;

    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x, ++index)
            {
                const Eigen::Vector3f point = origin + Eigen::Vector3f(x, y, z).cwiseProduct(spacing);

                coords->SetPoint(index, point.data());
                tex_coords->SetTuple(index, point.data());
            }
        }
    }

    output_deformed_grid->SetPoints(coords);
    output_deformed_grid->GetPointData()->AddArray(tex_coords);
}

void feature_deformation::create_cells(vtkUnstructuredGrid* output_deformed_grid, vtkUnstructuredGrid* output_deformed_grid_removed,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& spacing) const
{
    const auto threshold = static_cast<float>(this->parameter_output_grid.remove_cells_scalar) * spacing.norm();

    // Create cells
    output_deformed_grid->Allocate((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));
    output_deformed_grid_removed->Allocate((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));

    auto handiness = vtkSmartPointer<vtkFloatArray>::New();
    handiness->SetNumberOfComponents(1);
    handiness->Allocate((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));
    handiness->SetName("Handiness");

    for (int z = 0; z < dimension[2] - 1; ++z)
    {
        for (int y = 0; y < dimension[1] - 1; ++y)
        {
            for (int x = 0; x < dimension[0] - 1; ++x)
            {
                // Create point IDs
                const auto point0 = calc_index_point(dimension, x + 0, y + 0, z + 0);
                const auto point1 = calc_index_point(dimension, x + 1, y + 0, z + 0);
                const auto point2 = calc_index_point(dimension, x + 0, y + 1, z + 0);
                const auto point3 = calc_index_point(dimension, x + 1, y + 1, z + 0);
                const auto point4 = calc_index_point(dimension, x + 0, y + 0, z + 1);
                const auto point5 = calc_index_point(dimension, x + 1, y + 0, z + 1);
                const auto point6 = calc_index_point(dimension, x + 0, y + 1, z + 1);
                const auto point7 = calc_index_point(dimension, x + 1, y + 1, z + 1);

                const std::array<vtkIdType, 8> point_ids{
                    point0,
                    point1,
                    point2,
                    point3,
                    point4,
                    point5,
                    point6,
                    point7
                };

                // Calculate distances between points and compare to threshold
                bool discard_cell = false;

                if (this->parameter_output_grid.remove_cells)
                {
                    // Get all cell points
                    std::vector<Eigen::Vector3d> cell_points(point_ids.size());

                    for (std::size_t point_index = 0; point_index < point_ids.size(); ++point_index)
                    {
                        output_deformed_grid->GetPoints()->GetPoint(point_ids[point_index], cell_points[point_index].data());
                    }

                    // Pairwise calculate the distance between all points and compare the result with the threshold
                    for (std::size_t i = 0; i < cell_points.size() - 1; ++i)
                    {
                        for (std::size_t j = i + 1; j < cell_points.size(); ++j)
                        {
                            discard_cell |= (cell_points[i] - cell_points[j]).norm() > threshold;
                        }
                    }
                }

                // Create cell faces
                if (!discard_cell)
                {
                    auto faces = vtkSmartPointer<vtkCellArray>::New();

                    vtkIdType face0[4] = { point0, point1, point3, point2 }; // front
                    vtkIdType face1[4] = { point6, point7, point5, point4 }; // back
                    vtkIdType face2[4] = { point4, point5, point1, point0 }; // bottom
                    vtkIdType face3[4] = { point2, point3, point7, point6 }; // top
                    vtkIdType face4[4] = { point0, point2, point6, point4 }; // left
                    vtkIdType face5[4] = { point1, point5, point7, point3 }; // right

                    faces->InsertNextCell(4, face0);
                    faces->InsertNextCell(4, face1);
                    faces->InsertNextCell(4, face2);
                    faces->InsertNextCell(4, face3);
                    faces->InsertNextCell(4, face4);
                    faces->InsertNextCell(4, face5);

                    output_deformed_grid->InsertNextCell(VTK_POLYHEDRON, 8, point_ids.data(), 6, faces->GetData()->GetPointer(0));

                    // Calculate handiness
                    std::array<Eigen::Vector3d, 4> points;
                    output_deformed_grid->GetPoints()->GetPoint(point0, points[0].data());
                    output_deformed_grid->GetPoints()->GetPoint(point1, points[1].data());
                    output_deformed_grid->GetPoints()->GetPoint(point2, points[2].data());
                    output_deformed_grid->GetPoints()->GetPoint(point4, points[3].data());

                    const auto vector_1 = points[1] - points[0];
                    const auto vector_2 = points[2] - points[0];
                    const auto vector_3 = points[3] - points[0];

                    handiness->InsertNextValue(static_cast<float>(vector_1.cross(vector_2).dot(vector_3)));
                }
                else
                {
                    auto faces = vtkSmartPointer<vtkCellArray>::New();

                    vtkIdType face0[4] = { point0, point1, point3, point2 }; // front
                    vtkIdType face1[4] = { point6, point7, point5, point4 }; // back
                    vtkIdType face2[4] = { point4, point5, point1, point0 }; // bottom
                    vtkIdType face3[4] = { point2, point3, point7, point6 }; // top
                    vtkIdType face4[4] = { point0, point2, point6, point4 }; // left
                    vtkIdType face5[4] = { point1, point5, point7, point3 }; // right

                    faces->InsertNextCell(4, face0);
                    faces->InsertNextCell(4, face1);
                    faces->InsertNextCell(4, face2);
                    faces->InsertNextCell(4, face3);
                    faces->InsertNextCell(4, face4);
                    faces->InsertNextCell(4, face5);

                    output_deformed_grid_removed->InsertNextCell(VTK_POLYHEDRON, 8, point_ids.data(), 6, faces->GetData()->GetPointer(0));

                    // Calculate handiness
                    std::array<Eigen::Vector3d, 4> points;
                    output_deformed_grid_removed->GetPoints()->GetPoint(point0, points[0].data());
                    output_deformed_grid_removed->GetPoints()->GetPoint(point1, points[1].data());
                    output_deformed_grid_removed->GetPoints()->GetPoint(point2, points[2].data());
                    output_deformed_grid_removed->GetPoints()->GetPoint(point4, points[3].data());

                    const auto vector_1 = points[1] - points[0];
                    const auto vector_2 = points[2] - points[0];
                    const auto vector_3 = points[3] - points[0];
                }
            }
        }
    }

    output_deformed_grid->GetCellData()->AddArray(handiness);
    output_deformed_grid->BuildLinks();

    output_deformed_grid_removed->BuildLinks();
}

void feature_deformation::set_output_deformed_grid(vtkPointSet* output_deformed_grid, const cuda::displacement& grid_displacement) const
{
    // Set displaced points
    const auto& displaced_grid = grid_displacement.get_results();

    for (vtkIdType i = 0; i < output_deformed_grid->GetNumberOfPoints(); ++i)
    {
        output_deformed_grid->GetPoints()->SetPoint(i, displaced_grid[i].data());
    }

    // Create displacement ID array
    const auto& displacement_ids = grid_displacement.get_displacement_info();

    auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
    displacement_id_array->SetNumberOfComponents(4);
    displacement_id_array->SetNumberOfTuples(displacement_ids.size());
    displacement_id_array->SetName("Displacement Information");

    std::memcpy(displacement_id_array->GetPointer(0), displacement_ids.data(), displacement_ids.size() * sizeof(float4));

    output_deformed_grid->GetPointData()->AddArray(displacement_id_array);
}

void feature_deformation::set_output_deformed_lines(vtkPolyData* input_lines, vtkPolyData* output_deformed_lines, const cuda::displacement& line_displacement,
    const bool modified, cache_output_lines_t& output_lines, const std::string& name) const
{
    // Create output geometry
    if (modified || !output_lines.valid)
    {
        std::cout << "Creating deformed " << name << " output" << std::endl;

        output_lines.data = vtkSmartPointer<vtkPolyData>::New();
        output_lines.data->DeepCopy(input_lines);
        output_lines.valid = true;

        auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
        displacement_id_array->SetNumberOfComponents(4);
        displacement_id_array->SetNumberOfTuples(output_lines.data->GetNumberOfPoints());
        displacement_id_array->SetName("Displacement Information");
        displacement_id_array->FillValue(0.0f);

        auto displacement_distance_array = vtkSmartPointer<vtkFloatArray>::New();
        displacement_distance_array->SetNumberOfComponents(1);
        displacement_distance_array->SetNumberOfTuples(output_lines.data->GetNumberOfPoints());
        displacement_distance_array->SetName("B-Spline Distance");
        displacement_distance_array->FillValue(0.0f);

        output_lines.data->GetPointData()->AddArray(displacement_id_array);
        output_lines.data->GetPointData()->AddArray(displacement_distance_array);
    }
    else
    {
        std::cout << "Updating deformed " << name << " output" << std::endl;
    }

    // Set displaced points
    const auto& displaced_lines = line_displacement.get_results();

    for (vtkIdType i = 0; i < output_lines.data->GetNumberOfPoints(); ++i)
    {
        output_lines.data->GetPoints()->SetPoint(i, displaced_lines[i].data());
    }

    // Set displacement ID array
    const auto& displacement_ids = line_displacement.get_displacement_info();

    std::memcpy(vtkFloatArray::SafeDownCast(output_lines.data->GetPointData()->GetArray("Displacement Information"))->GetPointer(0),
        displacement_ids.data(), displacement_ids.size() * sizeof(float4));

    // In case of the B-Spline, store distance on B-Spline for neighboring points
    if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
        this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints) &&
        this->OutputBSplineDistance)
    {
        auto displacement_distance_array = vtkFloatArray::SafeDownCast(output_lines.data->GetPointData()->GetArray("B-Spline Distance"));

        vtkIdType index = 0;
        vtkIdType cell_index = 0;

        for (vtkIdType l = 0; l < output_lines.data->GetLines()->GetNumberOfCells(); ++l)
        {
            const auto num_points = output_lines.data->GetLines()->GetData()->GetValue(cell_index);

            displacement_distance_array->SetValue(index, std::abs(displacement_ids[index].w - displacement_ids[index + 1].w));

            for (vtkIdType i = 1; i < num_points - 1; ++i)
            {
                displacement_distance_array->SetValue(index + i, 0.5f * (std::abs(displacement_ids[index + i - 1].w - displacement_ids[index + i].w)
                    + std::abs(displacement_ids[index + i].w - displacement_ids[index + i + 1].w)));
            }

            displacement_distance_array->SetValue(index + num_points - 1, std::abs(displacement_ids[index + num_points - 2].w - displacement_ids[index + num_points - 1].w));

            index += num_points;
            cell_index += num_points + 1;
        }
    }

    // Cache output
    output_deformed_lines->DeepCopy(output_lines.data);
}

void feature_deformation::create_displacement_field(vtkPointSet* output_deformed_grid) const
{
    // Create displacement field
    auto displacement_map = vtkSmartPointer<vtkDoubleArray>::New();
    displacement_map->SetNumberOfComponents(3);
    displacement_map->SetNumberOfTuples(output_deformed_grid->GetPoints()->GetNumberOfPoints());
    displacement_map->SetName("Displacement Map");

    for (vtkIdType p = 0; p < output_deformed_grid->GetPoints()->GetNumberOfPoints(); ++p)
    {
        Eigen::Vector3d displaced_point;
        output_deformed_grid->GetPoints()->GetPoint(p, displaced_point.data());

        displacement_map->SetTuple(p, displaced_point.data());
    }

    output_deformed_grid->GetPointData()->AddArray(displacement_map);
}

void feature_deformation::deform_velocities(vtkPointSet* output_deformed_grid, vtkDataArray* data_array,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& spacing) const
{
    // Setup velocity interpolation for support of point and cell data
    std::function<Eigen::Vector3d(int, int, int)> get_velocity;

    if (data_array->GetNumberOfTuples() == dimension[0] * dimension[1] * dimension[2])
    {
        get_velocity = [&dimension, &data_array](const int x, const int y, const int z) -> Eigen::Vector3d
        {
            Eigen::Vector3d velocity;
            data_array->GetTuple(calc_index_point(dimension, x, y, z), velocity.data());

            return velocity;
        };
    }
    else
    {
        get_velocity = [&dimension, &data_array](const int x, const int y, const int z) -> Eigen::Vector3d
        {
            std::array<Eigen::Vector3d, 8> velocities;
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 1, z - 1), velocities[0].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 1, z - 1), velocities[1].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 0, z - 1), velocities[2].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 0, z - 1), velocities[3].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 1, z - 0), velocities[4].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 1, z - 0), velocities[5].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 0, z - 0), velocities[6].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 0, z - 0), velocities[7].data());

            return 0.125 * (velocities[0] + velocities[1] + velocities[2] + velocities[3] + velocities[4] + velocities[5] + velocities[6] + velocities[7]);
        };
    }

    // Setup finite differences for the calculation of the Jacobian
    auto displacement_map = vtkDoubleArray::SafeDownCast(output_deformed_grid->GetPointData()->GetArray("Displacement Map"));

    auto calc_jacobian = [displacement_map](const int center, const int index, const int max, const int component, double h, const int offset) -> double
    {
        double left, right;

        if (center == 0) // Forward difference
        {
            left = displacement_map->GetComponent(index, component);
            right = displacement_map->GetComponent(index + offset, component);
        }
        else if (center == max) // Backward difference
        {
            left = displacement_map->GetComponent(index - offset, component);
            right = displacement_map->GetComponent(index, component);
        }
        else // Central difference
        {
            left = displacement_map->GetComponent(index - offset, component);
            right = displacement_map->GetComponent(index + offset, component);

            h *= 2.0;
        }

        return (right - left) / h;
    };

    // Calculate Jacobian and use it to calculate the velocities at the deformed grid
    auto jacobian = vtkSmartPointer<vtkDoubleArray>::New();
    jacobian->SetNumberOfComponents(9);
    jacobian->SetNumberOfTuples(dimension[0] * dimension[1] * dimension[2]);
    jacobian->SetName("Jacobian");

    // ... at the point nodes
    auto velocities_p = vtkSmartPointer<vtkDoubleArray>::New();
    velocities_p->SetNumberOfComponents(3);
    velocities_p->SetNumberOfTuples(dimension[0] * dimension[1] * dimension[2]);
    velocities_p->SetName(data_array->GetName());

    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index_p = calc_index_point(dimension, x, y, z);

                // Calculate Jacobian
                const auto Jxdx = calc_jacobian(x, index_p, dimension[0] - 1, 0, spacing[0], 1);
                const auto Jxdy = calc_jacobian(y, index_p, dimension[1] - 1, 0, spacing[1], dimension[0]);
                const auto Jxdz = calc_jacobian(z, index_p, dimension[2] - 1, 0, spacing[2], dimension[0] * dimension[1]);
                const auto Jydx = calc_jacobian(x, index_p, dimension[0] - 1, 1, spacing[0], 1);
                const auto Jydy = calc_jacobian(y, index_p, dimension[1] - 1, 1, spacing[1], dimension[0]);
                const auto Jydz = calc_jacobian(z, index_p, dimension[2] - 1, 1, spacing[2], dimension[0] * dimension[1]);
                const auto Jzdx = calc_jacobian(x, index_p, dimension[0] - 1, 2, spacing[0], 1);
                const auto Jzdy = calc_jacobian(y, index_p, dimension[1] - 1, 2, spacing[1], dimension[0]);
                const auto Jzdz = calc_jacobian(z, index_p, dimension[2] - 1, 2, spacing[2], dimension[0] * dimension[1]);

                Eigen::Matrix3d Jacobian;
                Jacobian << Jxdx, Jxdy, Jxdz, Jydx, Jydy, Jydz, Jzdx, Jzdy, Jzdz;

                jacobian->SetTuple(index_p, Jacobian.data());

                // Calculate velocities
                auto velocity = get_velocity(x, y, z);
                velocity = Jacobian * velocity;

                velocities_p->SetTuple(index_p, velocity.data());
            }
        }
    }

    output_deformed_grid->GetPointData()->AddArray(jacobian);
    output_deformed_grid->GetPointData()->AddArray(velocities_p);

    // ... at the cell centers
    auto velocities_c = vtkSmartPointer<vtkDoubleArray>::New();
    velocities_c->SetNumberOfComponents(3);
    velocities_c->SetNumberOfTuples((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));
    velocities_c->SetName(data_array->GetName());

    for (int z = 0; z < dimension[2] - 1; ++z)
    {
        for (int y = 0; y < dimension[1] - 1; ++y)
        {
            for (int x = 0; x < dimension[0] - 1; ++x)
            {
                const auto index_c = calc_index_cell(dimension, x, y, z);

                std::array<Eigen::Vector3d, 8> velocities;
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 0, z + 0), velocities[0].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 0, z + 0), velocities[1].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 1, z + 0), velocities[2].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 1, z + 0), velocities[3].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 0, z + 1), velocities[4].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 0, z + 1), velocities[5].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 1, z + 1), velocities[6].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 1, z + 1), velocities[7].data());

                const Eigen::Vector3d velocity = 0.125 * (velocities[0] + velocities[1] + velocities[2]
                    + velocities[3] + velocities[4] + velocities[5] + velocities[6] + velocities[7]);

                velocities_c->SetTuple(index_c, velocity.data());
            }
        }
    }

    output_deformed_grid->GetCellData()->AddArray(velocities_c);
}

void feature_deformation::resample_grid(vtkPointSet* output_deformed_grid, vtkImageData* output_resampled_grid, const std::string& velocity_name,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const
{
    // Resample original grid
    auto velocities_deformed = vtkDoubleArray::SafeDownCast(output_deformed_grid->GetPointData()->GetArray(velocity_name.c_str()));

    auto velocities_resampled = vtkSmartPointer<vtkDoubleArray>::New();
    velocities_resampled->SetNumberOfComponents(3);
    velocities_resampled->SetNumberOfTuples(velocities_deformed->GetNumberOfTuples());
    velocities_resampled->SetName(velocities_deformed->GetName());

    const Eigen::Vector3d origin_d(static_cast<double>(origin[0]), static_cast<double>(origin[1]), static_cast<double>(origin[2]));
    const Eigen::Vector3d spacing_d(static_cast<double>(spacing[0]), static_cast<double>(spacing[1]), static_cast<double>(spacing[2]));

    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            vtkCell* cell = nullptr;

            for (int x = 0; x < dimension[0]; ++x)
            {
                Eigen::Vector3d point = origin_d + Eigen::Vector3d(x, y, z).cwiseProduct(spacing_d);

                // Find cell of the deformed grid, in which the point lies
                int subID;
                Eigen::Vector3d pcoords;
                std::array<double, 8> weights;

                cell = output_deformed_grid->FindAndGetCell(point.data(), cell, 0, 0.0, subID, pcoords.data(), weights.data());

                // Use weights to interpolate the velocity
                if (cell != nullptr)
                {
                    if (cell->GetNumberOfPoints() == 8)
                    {
                        auto point_ids = cell->GetPointIds();

                        Eigen::Vector3d velocity_sum{ 0.0, 0.0, 0.0 };
                        double weight_sum = 0.0;

                        for (vtkIdType i = 0; i < point_ids->GetNumberOfIds(); ++i)
                        {
                            Eigen::Vector3d velocity;
                            velocities_deformed->GetTuple(point_ids->GetId(i), velocity.data());

                            velocity_sum += weights[i] * velocity;
                            weight_sum += weights[i];
                        }

                        const Eigen::Vector3d velocity = velocity_sum / weight_sum;

                        velocities_resampled->SetTuple(calc_index_point(dimension, x, y, z), velocity.data());
                    }
                    else
                    {
                        std::clog << cell->GetNumberOfPoints() << std::endl;
                        velocities_resampled->SetTuple3(calc_index_point(dimension, x, y, z), 0.0, 0.0, 0.0);
                    }
                }
                else
                {
                    velocities_resampled->SetTuple3(calc_index_point(dimension, x, y, z), 0.0, 0.0, 0.0);
                }
            }
        }
    }

    output_resampled_grid->GetPointData()->AddArray(velocities_resampled);
}
