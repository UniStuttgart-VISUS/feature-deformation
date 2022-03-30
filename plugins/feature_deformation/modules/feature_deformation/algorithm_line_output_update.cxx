#include "algorithm_line_output_update.h"

#include "algorithm_displacement_assessment.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_computation_twisting.h"
#include "algorithm_line_output_creation.h"
#include "hash.h"

#include "vtkCellArray.h"
#include "vtkDataArray.h"
#include "vtkFloatArray.h"
#include "vtkIdTypeArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include <cmath>
#include <iostream>
#include <memory>

void algorithm_line_output_update::set_input(const std::shared_ptr<const algorithm_line_output_creation> output_lines,
    std::shared_ptr<const algorithm_displacement_computation> displacement,
    std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting,
    const std::shared_ptr<const algorithm_displacement_assessment> assessment,
    const cuda::displacement::method_t displacement_method, const bool minimal_output, const bool output_bspline_distance)
{
    this->output_lines = output_lines;
    this->displacement = displacement;
    this->displacement_twisting = displacement_twisting;
    this->assessment = assessment;
    this->displacement_method = displacement_method;
    this->minimal_output = minimal_output;
    this->output_bspline_distance = output_bspline_distance;
}

std::uint32_t algorithm_line_output_update::calculate_hash() const
{
    if (!(this->output_lines->is_valid() && this->displacement->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->displacement_twisting->get_hash(),
        this->assessment->get_hash(), this->displacement_method, this->minimal_output, this->output_bspline_distance);
}

bool algorithm_line_output_update::run_computation()
{
    if (!this->is_quiet()) std::cout << "Updating deformed lines output" << std::endl;

    // Set displaced points
    const auto& displaced_lines = this->displacement_twisting->is_valid()
        ? this->displacement->get_results().displacements->get_results_twisting()
        : this->displacement->get_results().displacements->get_results();

    #pragma omp parallel for
    for (vtkIdType i = 0; i < this->output_lines->get_results().lines->GetNumberOfPoints(); ++i)
    {
        this->output_lines->get_results().lines->GetPoints()->SetPoint(i, displaced_lines[i].data());
    }

    // Set displacement ID array
    if (!this->minimal_output)
    {
        const auto displacement_ids = displacement->get_results().displacements->get_displacement_info();

        auto displacement_info = this->output_lines->get_results().lines->GetPointData()->GetArray("Displacement Information");
        auto mapping = this->output_lines->get_results().lines->GetPointData()->GetArray("Mapping to B-Spline");
        auto mapping_original = this->output_lines->get_results().lines->GetPointData()->GetArray("Mapping to B-Spline (Original)");

        std::memcpy(displacement_info->GetVoidPointer(0),
            std::get<0>(displacement_ids).data(), std::get<0>(displacement_ids).size() * sizeof(float4));

        std::memcpy(mapping->GetVoidPointer(0),
            std::get<1>(displacement_ids).data(), std::get<1>(displacement_ids).size() * sizeof(float3));

        std::memcpy(mapping_original->GetVoidPointer(0),
            std::get<2>(displacement_ids).data(), std::get<2>(displacement_ids).size() * sizeof(float3));

        displacement_info->Modified();
        mapping->Modified();
        mapping_original->Modified();

        // In case of the B-Spline, store distance on B-Spline for neighboring points
        if ((this->displacement_method == cuda::displacement::method_t::b_spline ||
            this->displacement_method == cuda::displacement::method_t::b_spline_joints) &&
            this->output_bspline_distance)
        {
            auto displacement_distance_array = vtkFloatArray::SafeDownCast(this->output_lines->get_results().lines->GetPointData()->GetArray("B-Spline Distance"));

            vtkIdType index = 0;
            vtkIdType cell_index = 0;

            for (vtkIdType l = 0; l < this->output_lines->get_results().lines->GetLines()->GetNumberOfCells(); ++l)
            {
                const auto num_points = this->output_lines->get_results().lines->GetLines()->GetData()->GetValue(cell_index);

                displacement_distance_array->SetValue(index, std::abs(std::get<0>(displacement_ids)[index].w - std::get<0>(displacement_ids)[index + 1].w));

                for (vtkIdType i = 1; i < num_points - 1; ++i)
                {
                    displacement_distance_array->SetValue(index + i, 0.5f * (std::abs(std::get<0>(displacement_ids)[index + i - 1].w - std::get<0>(displacement_ids)[index + i].w)
                        + std::abs(std::get<0>(displacement_ids)[index + i].w - std::get<0>(displacement_ids)[index + i + 1].w)));
                }

                displacement_distance_array->SetValue(index + num_points - 1,
                    std::abs(std::get<0>(displacement_ids)[index + num_points - 2].w - std::get<0>(displacement_ids)[index + num_points - 1].w));

                index += num_points;
                cell_index += num_points + 1;
            }

            displacement_distance_array->Modified();
        }
    }

    this->results.lines = this->output_lines->get_results().lines;

    return true;
}

void algorithm_line_output_update::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading deformed line output from cache" << std::endl;
}

const algorithm_line_output_update::results_t& algorithm_line_output_update::get_results() const
{
    return this->results;
}
