#include "feature_deformation.h"

#include "algorithm_compute_gauss.h"
#include "algorithm_compute_tearing.h"
#include "algorithm_displacement_assessment.h"
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

vtkStandardNewMacro(feature_deformation);

feature_deformation::feature_deformation() : frames(0)
{
    this->ParameterLog = new char[1024];
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
    this->alg_displacement_assess_lines = std::make_shared<algorithm_displacement_assessment>();

    this->alg_displacement_creation_grid = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation_grid = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation_grid = std::make_shared<algorithm_displacement_computation>();
    this->alg_displacement_assess_grid = std::make_shared<algorithm_displacement_assessment>();

    this->alg_displacement_creation_geometry = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation_geometry = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation_geometry = std::make_shared<algorithm_displacement_computation>();
    this->alg_displacement_assess_geometry = std::make_shared<algorithm_displacement_assessment>();

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
    delete[] this->ParameterLog;
    delete[] this->PerformanceLog;
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
    this->alg_displacement_assess_lines->be_quiet(quiet);
    this->alg_displacement_creation_grid->be_quiet(quiet);
    this->alg_displacement_precomputation_grid->be_quiet(quiet);
    this->alg_displacement_computation_grid->be_quiet(quiet);
    this->alg_displacement_assess_grid->be_quiet(quiet);
    this->alg_displacement_creation_geometry->be_quiet(quiet);
    this->alg_displacement_precomputation_geometry->be_quiet(quiet);
    this->alg_displacement_computation_geometry->be_quiet(quiet);
    this->alg_displacement_assess_geometry->be_quiet(quiet);
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
        std::shared_ptr<algorithm_displacement_precomputation>, std::shared_ptr<algorithm_displacement_computation>,
        std::shared_ptr<algorithm_displacement_assessment>>, 3> displacement_inputs
    {
        std::make_tuple("line", std::static_pointer_cast<algorithm_input>(this->alg_line_input),
            this->alg_displacement_creation_lines, this->alg_displacement_precomputation_lines,
            this->alg_displacement_computation_lines, this->alg_displacement_assess_lines),

        std::make_tuple("grid", std::static_pointer_cast<algorithm_input>(this->alg_grid_input),
            this->alg_displacement_creation_grid, this->alg_displacement_precomputation_grid,
            this->alg_displacement_computation_grid, this->alg_displacement_assess_grid),

        std::make_tuple("geometry", std::static_pointer_cast<algorithm_input>(this->alg_geometry_input),
            this->alg_displacement_creation_geometry, this->alg_displacement_precomputation_geometry,
            this->alg_displacement_computation_geometry, this->alg_displacement_assess_geometry),
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

            __next_perf_measure("assess quality on GPU");

            std::get<5>(displacement_input)->run(std::get<4>(displacement_input), this->alg_smoothing,
                this->parameters.displacement_method, this->parameters.displacement_parameters,
                this->parameters.bspline_parameters, this->parameters.assess_mapping);
        }
    }

    if (!quiet) std::cout << std::endl;

    // Output lines
    __next_perf_measure("create output feature lines");

    this->alg_line_output_creation->run(this->alg_line_input);

    __next_perf_measure("update output feature lines");

    this->alg_line_output_update->run(this->alg_line_output_creation, this->alg_displacement_computation_lines,
        this->alg_displacement_assess_lines, this->parameters.displacement_method, this->parameters.output_bspline_distance);

    __next_perf_measure("output feature lines");

    this->alg_line_output_set->run(this->alg_line_output_update, output_vector->GetInformationObject(0), time);

    // Output geometry
    __next_perf_measure("create output geometry");

    this->alg_geometry_output_creation->run(this->alg_geometry_input);

    __next_perf_measure("update output geometry");

    this->alg_geometry_output_update->run(this->alg_geometry_output_creation, this->alg_displacement_computation_geometry,
        this->alg_displacement_assess_geometry, this->parameters.displacement_method, this->parameters.output_bspline_distance);

    __next_perf_measure("output geometry");

    this->alg_geometry_output_set->run(this->alg_geometry_output_update, output_vector->GetInformationObject(1), time);

    // Output grid
    if (this->parameters.output_deformed_grid)
    {
        __next_perf_measure("create output grid");

        this->alg_grid_output_creation->run(this->alg_grid_input, this->parameters.remove_cells);

        __next_perf_measure("update output grid");

        this->alg_grid_output_update->run(this->alg_grid_input, this->alg_grid_output_creation, this->alg_displacement_computation_grid,
            this->alg_displacement_assess_grid, this->alg_compute_tearing, this->parameters.remove_cells, this->parameters.remove_cells_scalar);

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
    // Set up parameter logging
    std::ofstream parameter_log(this->ParameterLog, std::ios_base::app | std::ios_base::out);

#define __log_header(name) parameter_log << std::endl << name << std::endl << "--------------------------" << std::endl;
#define __log_parameter(name, value) parameter_log.width(55); parameter_log.fill(' '); parameter_log << std::left << name << std::flush; \
                                     parameter_log.width(0); parameter_log << value << std::endl;

#define __set_parameter(lhs, rhs) this->parameters.lhs = rhs; __log_parameter(#lhs, rhs);
#define __set_parameter_with_cast(lhs, rhs, cast) this->parameters.lhs = cast(rhs); __log_parameter(#lhs, rhs);
#define __set_parameter_bool(lhs, rhs) this->parameters.lhs = rhs != 0; __log_parameter(#lhs, (rhs != 0 ? "true" : "false"));

    // Line parameters
    __log_header("line");

    __set_parameter(selected_line_id, this->LineID);

    // Smoothing parameters
    __log_header("smoothing");

    __set_parameter_with_cast(smoothing_method, this->Method, static_cast<smoothing::method_t>);
    __set_parameter_with_cast(variant, this->Variant, static_cast<smoothing::variant_t>);
    __set_parameter(max_num_iterations, this->MaxNumIterations);

    float lambda = static_cast<float>(this->Lambda);
    int num_iterations = this->MaxNumIterations;

    if (this->parameters.smoothing_method == smoothing::method_t::smoothing)
    {
        if (this->Inverse)
        {
            time = 1.0 - time;
        }

        if (this->Interpolator == 0)
        {
            // Linear
            num_iterations *= std::min(time, 1.0);
        }
        else if (this->Interpolator == 1)
        {
            // Exponential
            if (time == 0.0)
            {
                num_iterations = 0;
            }
            else
            {
                num_iterations = std::pow(num_iterations + 1, time) - 1;
            }
        }
        else if (this->Interpolator == 2)
        {
            // "Quadratic"
            if (time == 0.0)
            {
                num_iterations = 0;
            }
            else
            {
                num_iterations = num_iterations * std::pow(time, this->Exponent);
            }
        }
        else if (this->Interpolator == 3)
        {
            // First linear, then exponential
            const auto connection_time = this->InterpolatorThreshold;
            const auto connection_value = std::pow(num_iterations + 1, connection_time) - 1;

            if (time == 0.0)
            {
                num_iterations = 0;
            }
            else if (time <= connection_time)
            {
                num_iterations = std::min((time / connection_time) * connection_value, connection_value);
            }
            else
            {
                num_iterations = std::pow(num_iterations + 1, time) - 1;
            }
        }

        if (this->InterpolateSmoothingFactor)
        {
            lambda *= std::min(1.0, std::max(0.1, time));
        }
    }

    __set_parameter(lambda, lambda);
    __set_parameter(num_iterations, num_iterations);

    // Displacement parameters
    __log_header("displacement");

    __set_parameter_with_cast(displacement_method, this->Weight, static_cast<cuda::displacement::method_t>);

    switch (this->parameters.displacement_method)
    {
    case cuda::displacement::method_t::greedy:
    case cuda::displacement::method_t::voronoi:
        __set_parameter_with_cast(displacement_parameters.inverse_distance_weighting.exponent, this->EpsilonScalar, static_cast<float>);
        __set_parameter(displacement_parameters.inverse_distance_weighting.neighborhood, this->VoronoiDistance);

        break;
    case cuda::displacement::method_t::greedy_joints:
        __set_parameter_with_cast(displacement_parameters.inverse_distance_weighting.exponent, this->EpsilonScalar, static_cast<float>);

        break;
    case cuda::displacement::method_t::projection:
        __set_parameter_with_cast(displacement_parameters.projection.gauss_parameter, this->GaussParameter, static_cast<float>);

        break;
    case cuda::displacement::method_t::b_spline:
    case cuda::displacement::method_t::b_spline_joints:
        __set_parameter(displacement_parameters.b_spline.degree, this->SplineDegree);
        __set_parameter_with_cast(displacement_parameters.b_spline.gauss_parameter, this->GaussParameter, static_cast<float>);
        __set_parameter(displacement_parameters.b_spline.iterations, this->Subdivisions);

        break;
    }

    __set_parameter_with_cast(idw_parameters.exponent, this->EpsilonScalar, static_cast<float>);
    __set_parameter(idw_parameters.neighborhood, this->VoronoiDistance);
    __set_parameter_with_cast(projection_parameters.gauss_parameter, this->GaussParameter, static_cast<float>);
    __set_parameter(bspline_parameters.degree, this->SplineDegree);
    __set_parameter_with_cast(bspline_parameters.gauss_parameter, this->GaussParameter, static_cast<float>);
    __set_parameter(bspline_parameters.iterations, this->Subdivisions);

    // Pre-computation parameters
    __log_header("precomputation");

    __set_parameter_bool(compute_gauss, this->ComputeGauss);
    __set_parameter_bool(check_handedness, this->CheckHandedness);
    __set_parameter_bool(check_convexity, this->CheckConvexity);
    __set_parameter_bool(check_volume, this->CheckVolume);
    __set_parameter(volume_percentage, this->VolumePercentage);
    __set_parameter(num_subdivisions, this->GaussSubdivisions);
    __set_parameter_bool(compute_tearing, this->ComputeTearing);

    // Assessment parameters
    __log_header("assessment");

    __set_parameter_bool(assess_mapping, this->AssessMapping);

    // Output parameters
    __log_header("output");

    __set_parameter_bool(output_bspline_distance, this->OutputBSplineDistance);
    __set_parameter_bool(output_deformed_grid, this->OutputDeformedGrid);
    __set_parameter_bool(output_vector_field, this->OutputVectorField);
    __set_parameter_bool(remove_cells, this->RemoveCells);
    __set_parameter_with_cast(remove_cells_scalar, this->RemoveCellsScalar, static_cast<float>);

    parameter_log << std::endl << "-----------------------------------------------------------------------------------------------" << std::endl;

#undef __log_header
#undef __log_parameter
#undef __set_parameter
#undef __set_parameter_with_cast
#undef __set_parameter_bool
}
