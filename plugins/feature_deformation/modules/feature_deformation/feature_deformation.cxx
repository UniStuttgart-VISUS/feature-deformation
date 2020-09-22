#include "feature_deformation.h"

#include "algorithm_compute_gauss.h"
#include "algorithm_compute_tearing.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_displacement_precomputation.h"
#include "algorithm_geometry_input.h"
#include "algorithm_grid_input.h"
#include "algorithm_grid_output_creation.h"
#include "algorithm_grid_output_set.h"
#include "algorithm_grid_output_update.h"
#include "algorithm_grid_output_vectorfield.h"
#include "algorithm_line_input.h"
#include "algorithm_geometry_output_creation.h"
#include "algorithm_geometry_output_set.h"
#include "algorithm_geometry_output_update.h"
#include "algorithm_line_output_creation.h"
#include "algorithm_line_output_set.h"
#include "algorithm_line_output_update.h"
#include "algorithm_smoothing.h"
#include "algorithm_vectorfield_input.h"

#include "displacement.h"
//#define __disable_performance_measure
#include "performance.h"
#include "smoothing.h"

#include "vtkAlgorithm.h"
#include "vtkDataObject.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointSet.h"
#include "vtkPolyData.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

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

vtkStandardNewMacro(feature_deformation);

feature_deformation::feature_deformation() : frames(0)
{
    this->PerformanceLog = new char[1024];

    this->alg_grid_input = std::make_shared<algorithm_grid_input>();
    this->alg_line_input = std::make_shared<algorithm_line_input>();
    this->alg_geometry_input = std::make_shared<algorithm_geometry_input>();
    this->alg_vectorfield_input = std::make_shared<algorithm_vectorfield_input>();

    this->alg_compute_gauss = std::make_shared<algorithm_compute_gauss>();
    this->alg_compute_tearing = std::make_shared<algorithm_compute_tearing>();

    this->alg_smoothing = std::make_shared<algorithm_smoothing>();

    this->alg_displacement_creation_lines = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation_lines = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation_lines = std::make_shared<algorithm_displacement_computation>();

    this->alg_displacement_creation_grid = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation_grid = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation_grid = std::make_shared<algorithm_displacement_computation>();

    this->alg_displacement_creation_geometry = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation_geometry = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation_geometry = std::make_shared<algorithm_displacement_computation>();

    this->alg_line_output_creation = std::make_shared<algorithm_line_output_creation>();
    this->alg_line_output_update = std::make_shared<algorithm_line_output_update>();
    this->alg_line_output_set = std::make_shared<algorithm_line_output_set>();

    this->alg_grid_output_creation = std::make_shared<algorithm_grid_output_creation>();
    this->alg_grid_output_set = std::make_shared<algorithm_grid_output_set>();
    this->alg_grid_output_update = std::make_shared<algorithm_grid_output_update>();
    this->alg_grid_output_vectorfield = std::make_shared<algorithm_grid_output_vectorfield>();

    this->alg_geometry_output_creation = std::make_shared<algorithm_geometry_output_creation>();
    this->alg_geometry_output_update = std::make_shared<algorithm_geometry_output_update>();
    this->alg_geometry_output_set = std::make_shared<algorithm_geometry_output_set>();

    this->SetNumberOfInputPorts(3);
    this->SetNumberOfOutputPorts(3);
}

feature_deformation::~feature_deformation()
{
    delete[] this->PerformanceLog;
}

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
    create_or_get_data_object<vtkMultiBlockDataSet>(1, this, output_vector);
    create_or_get_data_object<vtkMultiBlockDataSet>(2, this, output_vector);

    return 1;
}

int feature_deformation::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    // If iterative smoothing is selected, create time step values based on animation parameters
    std::array<double, 2> time_range;

    if (this->Method == 1)
    {
        time_range = { 0.0, 1.0 };
    }
    else
    {
        time_range = { 0.0, 0.0 };
    }

    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(1)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(2)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);

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
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPointSet");
        info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
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
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
        return 1;
    }
    else if (port == 2)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
        return 1;
    }

    return 1;
}

void feature_deformation::RemoveAllGeometryInputs()
{
    this->SetInputConnection(2, nullptr);
}

int feature_deformation::RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    return 1;
}

int feature_deformation::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Set logging
    const bool quiet = this->Quiet != 0;

    this->alg_grid_input->be_quiet(quiet);
    this->alg_line_input->be_quiet(quiet);
    this->alg_geometry_input->be_quiet(quiet);
    this->alg_vectorfield_input->be_quiet(quiet);
    this->alg_compute_gauss->be_quiet(quiet);
    this->alg_compute_tearing->be_quiet(quiet);
    this->alg_smoothing->be_quiet(quiet);
    this->alg_displacement_creation_lines->be_quiet(quiet);
    this->alg_displacement_precomputation_lines->be_quiet(quiet);
    this->alg_displacement_computation_lines->be_quiet(quiet);
    this->alg_displacement_creation_grid->be_quiet(quiet);
    this->alg_displacement_precomputation_grid->be_quiet(quiet);
    this->alg_displacement_computation_grid->be_quiet(quiet);
    this->alg_displacement_creation_geometry->be_quiet(quiet);
    this->alg_displacement_precomputation_geometry->be_quiet(quiet);
    this->alg_displacement_computation_geometry->be_quiet(quiet);
    this->alg_line_output_creation->be_quiet(quiet);
    this->alg_line_output_update->be_quiet(quiet);
    this->alg_line_output_set->be_quiet(quiet);
    this->alg_grid_output_creation->be_quiet(quiet);
    this->alg_grid_output_set->be_quiet(quiet);
    this->alg_grid_output_update->be_quiet(quiet);
    this->alg_grid_output_vectorfield->be_quiet(quiet);
    this->alg_geometry_output_creation->be_quiet(quiet);
    this->alg_geometry_output_update->be_quiet(quiet);
    this->alg_geometry_output_set->be_quiet(quiet);

    // Create performance measure
    __init_perf_file(this->PerformanceLog, std::ios_base::app | std::ios_base::out, performance::style::csv);

    if (!quiet)
    {
        __add_perf(std::cout, performance::style::colored_message);
    }

    // Output info
    if (!quiet) std::cout << "------------------------------------------------------" << std::endl;
    if (!quiet) std::cout << "Starting deformation, frame: " << this->frames++ << std::endl << std::endl;

    // Get time
    const auto time = output_vector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

    // Get parameters
    __next_perf_measure("process parameters");

    process_parameters(time);

    if (this->parameters.smoothing_method == smoothing::method_t::smoothing)
    {
        if (!quiet) std::cout << "Time: " << time << std::endl << std::endl;
    }

    // Get input
    __next_perf_measure("get feature lines input");

    if (!this->alg_line_input->run(vtkPolyData::SafeDownCast(input_vector[1]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())), this->parameters.selected_line_id))
    {
        return 0;
    }

    if (input_vector[0] != nullptr && input_vector[0]->GetInformationObject(0) != nullptr &&
        (this->parameters.output_deformed_grid || this->parameters.compute_gauss))
    {
        __next_perf_measure("get grid input");

        this->alg_grid_input->run(vtkImageData::SafeDownCast(input_vector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())));

        if (this->parameters.output_vector_field)
        {
            auto velocity_array = GetInputArrayToProcess(0, input_vector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

            if (velocity_array != nullptr)
            {
                __next_perf_measure("get vector field input");

                this->alg_vectorfield_input->run(this->alg_grid_input, velocity_array->GetName());
            }
        }
    }

    if (input_vector[2] != nullptr)
    {
        __next_perf_measure("get geometry input");

        std::vector<vtkPointSet*> geometry_sets;

        for (vtkIdType i = 0; i < input_vector[2]->GetNumberOfInformationObjects(); ++i)
        {
            geometry_sets.push_back(vtkPointSet::SafeDownCast(input_vector[2]->GetInformationObject(i)->Get(vtkDataObject::DATA_OBJECT())));
        }

        this->alg_geometry_input->run(geometry_sets);
    }

    if (!quiet) std::cout << std::endl;

    // Pre-compute Gauss parameter and tearing
    if (this->parameters.compute_gauss && (this->parameters.displacement_method == cuda::displacement::method_t::b_spline || 
        this->parameters.displacement_method == cuda::displacement::method_t::b_spline_joints))
    {
        __next_perf_measure("pre-compute gauss parameter");

        this->alg_compute_gauss->run(this->alg_line_input, this->alg_grid_input, this->parameters.smoothing_method,
            this->parameters.variant, this->parameters.lambda, this->parameters.max_num_iterations, this->parameters.displacement_method,
            this->parameters.bspline_parameters, this->parameters.num_subdivisions, this->parameters.remove_cells_scalar,
            this->parameters.check_handedness, this->parameters.check_convexity, this->parameters.check_volume, this->parameters.volume_percentage);

        this->parameters.displacement_parameters.b_spline.gauss_parameter
            = this->parameters.bspline_parameters.gauss_parameter = this->alg_compute_gauss->get_results().gauss_parameter;
    }

    if (this->parameters.compute_tearing && this->parameters.remove_cells)
    {
        __next_perf_measure("pre-compute tearing");

        this->alg_compute_tearing->run(this->alg_line_input, this->alg_grid_input, this->parameters.smoothing_method,
            this->parameters.variant, this->parameters.lambda, this->parameters.max_num_iterations, this->parameters.displacement_method,
            this->parameters.displacement_parameters, this->parameters.bspline_parameters, this->parameters.remove_cells_scalar);
    }

    // Smooth line
    __next_perf_measure("smooth feature lines");

    if (!this->alg_smoothing->run(this->alg_line_input, this->parameters.smoothing_method, this->parameters.variant, this->parameters.lambda, this->parameters.num_iterations))
    {
        return 0;
    }

    if (!quiet) std::cout << std::endl;

    // Displace points
    const std::array<std::tuple<std::string, std::shared_ptr<algorithm_input>, std::shared_ptr<algorithm_displacement_creation>,
        std::shared_ptr<algorithm_displacement_precomputation>, std::shared_ptr<algorithm_displacement_computation>>, 3> displacement_inputs
    {
        std::make_tuple("line", std::static_pointer_cast<algorithm_input>(this->alg_line_input),
            this->alg_displacement_creation_lines, this->alg_displacement_precomputation_lines, this->alg_displacement_computation_lines),

        std::make_tuple("grid", std::static_pointer_cast<algorithm_input>(this->alg_grid_input),
            this->alg_displacement_creation_grid, this->alg_displacement_precomputation_grid, this->alg_displacement_computation_grid),

        std::make_tuple("geometry", std::static_pointer_cast<algorithm_input>(this->alg_geometry_input),
            this->alg_displacement_creation_geometry, this->alg_displacement_precomputation_geometry, this->alg_displacement_computation_geometry),
    };

    for (const auto& displacement_input : displacement_inputs)
    {
        if (std::get<1>(displacement_input)->get_points().valid)
        {
            if (!quiet) std::cout << "Displacing " << std::get<0>(displacement_input) << " points..." << std::endl;

            __next_perf_measure("upload points to GPU");

            std::get<2>(displacement_input)->run(std::get<1>(displacement_input));

            __next_perf_measure("pre-compute on GPU");

            std::get<3>(displacement_input)->run(std::get<2>(displacement_input), this->alg_smoothing, this->alg_line_input,
                this->parameters.displacement_method, this->parameters.displacement_parameters, this->parameters.bspline_parameters);

            __next_perf_measure("displace points on GPU");

            std::get<4>(displacement_input)->run(std::get<2>(displacement_input), this->alg_smoothing,
                this->parameters.displacement_method, this->parameters.displacement_parameters);
        }
    }

    if (!quiet) std::cout << std::endl;

    // Output lines
    __next_perf_measure("create output feature lines");

    this->alg_line_output_creation->run(this->alg_line_input);

    __next_perf_measure("update output feature lines");

    this->alg_line_output_update->run(this->alg_line_output_creation, this->alg_displacement_computation_lines,
        this->parameters.displacement_method, this->parameters.output_bspline_distance);

    __next_perf_measure("output feature lines");

    this->alg_line_output_set->run(this->alg_line_output_update, output_vector->GetInformationObject(0), time);

    // Output geometry
    __next_perf_measure("create output geometry");

    this->alg_geometry_output_creation->run(this->alg_geometry_input);

    __next_perf_measure("update output geometry");

    this->alg_geometry_output_update->run(this->alg_geometry_output_creation, this->alg_displacement_computation_geometry,
        this->parameters.displacement_method, this->parameters.output_bspline_distance);

    __next_perf_measure("output geometry");

    this->alg_geometry_output_set->run(this->alg_geometry_output_update, output_vector->GetInformationObject(1), time);

    // Output grid
    if (this->parameters.output_deformed_grid)
    {
        __next_perf_measure("create output grid");

        this->alg_grid_output_creation->run(this->alg_grid_input, this->parameters.remove_cells);

        __next_perf_measure("update output grid");

        this->alg_grid_output_update->run(this->alg_grid_input, this->alg_grid_output_creation, this->alg_displacement_computation_grid,
            this->alg_compute_tearing, this->parameters.remove_cells, this->parameters.remove_cells_scalar);

        if (this->parameters.output_vector_field)
        {
            __next_perf_measure("output vector field");

            this->alg_grid_output_vectorfield->run(this->alg_grid_input, this->alg_grid_output_update, this->alg_vectorfield_input);
        }

        __next_perf_measure("output grid");

        this->alg_grid_output_set->run(this->alg_grid_output_update, output_vector->GetInformationObject(2), time);
    }

    // Output info
    if (!quiet) std::cout << std::endl << "Finished deformation" << std::endl;
    if (!quiet) std::cout << "------------------------------------------------------" << std::endl;

    return 1;
}

void feature_deformation::process_parameters(double time)
{
    // Line parameters
    this->parameters.selected_line_id = this->LineID;

    // Smoothing parameters
    this->parameters.smoothing_method = static_cast<smoothing::method_t>(this->Method);
    this->parameters.variant = static_cast<smoothing::variant_t>(this->Variant);
    this->parameters.lambda = static_cast<float>(this->Lambda);
    this->parameters.num_iterations = this->MaxNumIterations;
    this->parameters.max_num_iterations = this->MaxNumIterations;

    if (this->parameters.smoothing_method == smoothing::method_t::smoothing)
    {
        if (this->Inverse)
        {
            time = 1.0 - time;
        }

        if (this->Interpolator == 0)
        {
            // Linear
            this->parameters.num_iterations *= std::min(time, 1.0);
        }
        else if (this->Interpolator == 1)
        {
            // Exponential
            if (time == 0.0)
            {
                this->parameters.num_iterations = 0;
            }
            else
            {
                this->parameters.num_iterations = std::pow(this->parameters.num_iterations + 1, time) - 1;
            }
        }
        else
        {
            // First linear, then exponential
            const auto connection_time = this->InterpolatorThreshold;
            const auto connection_value = std::pow(this->parameters.num_iterations + 1, connection_time) - 1;

            if (time == 0.0)
            {
                this->parameters.num_iterations = 0;
            }
            else if (time <= connection_time)
            {
                this->parameters.num_iterations = std::min((time / connection_time) * connection_value, connection_value);
            }
            else
            {
                this->parameters.num_iterations = std::pow(this->parameters.num_iterations + 1, time) - 1;
            }
        }
    }

    // Displacement parameters
    this->parameters.displacement_method = static_cast<cuda::displacement::method_t>(this->Weight);

    switch (this->parameters.displacement_method)
    {
    case cuda::displacement::method_t::greedy:
    case cuda::displacement::method_t::voronoi:
        this->parameters.displacement_parameters.inverse_distance_weighting.exponent = static_cast<float>(this->EpsilonScalar);
        this->parameters.displacement_parameters.inverse_distance_weighting.neighborhood = this->VoronoiDistance;

        break;
    case cuda::displacement::method_t::greedy_joints:
        this->parameters.displacement_parameters.inverse_distance_weighting.exponent = static_cast<float>(this->EpsilonScalar);

        break;
    case cuda::displacement::method_t::projection:
        this->parameters.displacement_parameters.projection.gauss_parameter = static_cast<float>(this->GaussParameter);

        break;
    case cuda::displacement::method_t::b_spline:
    case cuda::displacement::method_t::b_spline_joints:
        this->parameters.displacement_parameters.b_spline.degree = this->SplineDegree;
        this->parameters.displacement_parameters.b_spline.gauss_parameter = static_cast<float>(this->GaussParameter);
        this->parameters.displacement_parameters.b_spline.iterations = this->Subdivisions;

        break;
    }

    this->parameters.idw_parameters.exponent = static_cast<float>(this->EpsilonScalar);
    this->parameters.idw_parameters.neighborhood = this->VoronoiDistance;
    this->parameters.projection_parameters.gauss_parameter = static_cast<float>(this->GaussParameter);
    this->parameters.bspline_parameters.degree = this->SplineDegree;
    this->parameters.bspline_parameters.gauss_parameter = static_cast<float>(this->GaussParameter);
    this->parameters.bspline_parameters.iterations = this->Subdivisions;

    // Pre-computation parameters
    this->parameters.compute_gauss = (this->ComputeGauss != 0);
    this->parameters.check_handedness = (this->CheckHandedness != 0);
    this->parameters.check_convexity = (this->CheckConvexity != 0);
    this->parameters.check_volume = (this->CheckVolume != 0);
    this->parameters.volume_percentage = this->VolumePercentage;
    this->parameters.num_subdivisions = this->GaussSubdivisions;
    this->parameters.compute_tearing = (this->ComputeTearing != 0);

    // Output parameters
    this->parameters.output_bspline_distance = (this->OutputBSplineDistance != 0);
    this->parameters.output_deformed_grid = (this->OutputDeformedGrid != 0);
    this->parameters.output_vector_field = (this->OutputVectorField != 0);
    this->parameters.remove_cells = (this->RemoveCells != 0);
    this->parameters.remove_cells_scalar = static_cast<float>(this->RemoveCellsScalar);
}
