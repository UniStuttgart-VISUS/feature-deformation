#include "algorithm_geometry_output_update.h"

#include "algorithm_displacement_assessment.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_computation_twisting.h"
#include "algorithm_geometry_output_creation.h"
#include "hash.h"

#include "vtkCellArray.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkIdTypeArray.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPointSet.h"
#include "vtkSmartPointer.h"

#include <cmath>
#include <iostream>
#include <memory>

void algorithm_geometry_output_update::set_input(const std::shared_ptr<const algorithm_geometry_output_creation> output_geometry,
    std::shared_ptr<const algorithm_displacement_computation> displacement,
    std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting,
    const std::shared_ptr<const algorithm_displacement_assessment> assessment,
    const cuda::displacement::method_t displacement_method, const bool output_bspline_distance)
{
    this->output_geometry = output_geometry;
    this->displacement = displacement;
    this->displacement_twisting = displacement_twisting;
    this->assessment = assessment;
    this->displacement_method = displacement_method;
    this->output_bspline_distance = output_bspline_distance;
}

std::uint32_t algorithm_geometry_output_update::calculate_hash() const
{
    if (!(this->output_geometry->is_valid() && this->displacement->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->displacement_twisting->get_hash(),
        this->assessment->get_hash(), this->displacement_method, this->output_bspline_distance);
}

bool algorithm_geometry_output_update::run_computation()
{
    if (!this->is_quiet()) std::cout << "Updating deformed geometry output" << std::endl;

    // Set displaced points
    const auto& displaced_geometry = this->displacement->get_results().displacements->get_results();

    std::size_t global_id = 0;

    for (unsigned int block_index = 0; block_index < this->output_geometry->get_results().geometry->GetNumberOfBlocks(); ++block_index)
    {
        auto block = vtkPointSet::SafeDownCast(this->output_geometry->get_results().geometry->GetBlock(block_index));

        for (vtkIdType i = 0; i < block->GetNumberOfPoints(); ++i, ++global_id)
        {
            block->GetPoints()->SetPoint(i, displaced_geometry[global_id].data());
        }
    }

    // Set displacement ID arrays
    const auto displacement_ids = this->displacement->get_results().displacements->get_displacement_info();

    std::size_t global_data_index = 0;

    for (unsigned int block_index = 0; block_index < this->output_geometry->get_results().geometry->GetNumberOfBlocks(); ++block_index)
    {
        auto block = vtkPointSet::SafeDownCast(this->output_geometry->get_results().geometry->GetBlock(block_index));
        auto displacement_array = vtkFloatArray::SafeDownCast(block->GetPointData()->GetArray("Displacement Information"));
        auto mapping_array = vtkFloatArray::SafeDownCast(block->GetPointData()->GetArray("Mapping to B-Spline"));
        auto mapping_original_array = vtkFloatArray::SafeDownCast(block->GetPointData()->GetArray("Mapping to B-Spline (Original)"));

        std::memcpy(displacement_array->GetPointer(0), &std::get<0>(displacement_ids)[global_data_index], displacement_array->GetNumberOfTuples() * sizeof(float4));
        std::memcpy(mapping_array->GetPointer(0), &std::get<1>(displacement_ids)[global_data_index], mapping_array->GetNumberOfTuples() * sizeof(float3));
        std::memcpy(mapping_original_array->GetPointer(0), &std::get<2>(displacement_ids)[global_data_index], mapping_original_array->GetNumberOfTuples() * sizeof(float3));

        displacement_array->Modified();
        mapping_array->Modified();
        mapping_original_array->Modified();

        global_data_index += displacement_array->GetNumberOfTuples();
    }

    // In case of the B-Spline, store distance on B-Spline for neighboring points
    if ((this->displacement_method == cuda::displacement::method_t::b_spline ||
        this->displacement_method == cuda::displacement::method_t::b_spline_joints) &&
        this->output_bspline_distance)
    {
        for (unsigned int block_index = 0; block_index < this->output_geometry->get_results().geometry->GetNumberOfBlocks(); ++block_index)
        {
            auto block = vtkPointSet::SafeDownCast(this->output_geometry->get_results().geometry->GetBlock(block_index));
            auto displacement_distance_array = vtkFloatArray::SafeDownCast(block->GetPointData()->GetArray("B-Spline Distance"));

            if (vtkPolyData::SafeDownCast(block) != nullptr)
            {
                auto poly_block = vtkPolyData::SafeDownCast(block);

                vtkIdType index = 0;
                vtkIdType cell_index = 0;

                for (vtkIdType l = 0; l < poly_block->GetLines()->GetNumberOfCells(); ++l)
                {
                    const auto num_points = poly_block->GetLines()->GetData()->GetValue(cell_index);

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
            }

            displacement_distance_array->Modified();
        }
    }

    // Create displacement field
    for (unsigned int block_index = 0; block_index < this->output_geometry->get_results().geometry->GetNumberOfBlocks(); ++block_index)
    {
        auto block = vtkPointSet::SafeDownCast(this->output_geometry->get_results().geometry->GetBlock(block_index));
        auto displacement_map_array = vtkDoubleArray::SafeDownCast(block->GetPointData()->GetArray("Displacement Map"));

        #pragma omp parallel for
        for (vtkIdType p = 0; p < block->GetPoints()->GetNumberOfPoints(); ++p)
        {
            Eigen::Vector3d displaced_point;
            block->GetPoints()->GetPoint(p, displaced_point.data());

            displacement_map_array->SetTuple(p, displaced_point.data());
        }

        displacement_map_array->Modified();
    }

    // Set input as output
    this->results.geometry = this->output_geometry->get_results().geometry;

    return true;
}

void algorithm_geometry_output_update::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading deformed geometry output from cache" << std::endl;
}

const algorithm_geometry_output_update::results_t& algorithm_geometry_output_update::get_results() const
{
    return this->results;
}
