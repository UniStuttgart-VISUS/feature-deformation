#include "algorithm_grid_output_update.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_grid_output_creation.h"
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

void algorithm_grid_output_update::set_input(std::shared_ptr<const algorithm_grid_output_creation> output_grids,
    std::shared_ptr<const algorithm_displacement_computation> displacement, cuda::displacement::method_t displacement_method,
    bool output_bspgrid_distance)
{
    this->output_grids = output_grids;
    this->displacement = displacement;
    this->displacement_method = displacement_method;
    this->output_bspgrid_distance = output_bspgrid_distance;
}

std::uint32_t algorithm_grid_output_update::calculate_hash() const
{
    if (!(this->output_grids->is_valid() && this->displacement->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->displacement_method, this->output_bspgrid_distance);
}

bool algorithm_grid_output_update::run_computation()
{
    std::cout << "Updating deformed grids output" << std::endl;

    // Set displaced points
    const auto& displaced_grids = displacement->get_results().displacements->get_results();

    #pragma omp parallel for
    for (vtkIdType i = 0; i < this->output_grids->get_results().grids->GetNumberOfPoints(); ++i)
    {
        this->output_grids->get_results().grids->GetPoints()->SetPoint(i, displaced_grids[i].data());
    }

    // Set displacement ID array
    const auto& displacement_ids = displacement->get_results().displacements->get_displacement_info();

    std::memcpy(vtkFloatArray::SafeDownCast(this->output_grids->get_results().grids->GetPointData()->GetArray("Displacement Information"))->GetPointer(0),
        displacement_ids.data(), displacement_ids.size() * sizeof(float4));

    // In case of the B-Spgrid, store distance on B-Spgrid for neighboring points
    if ((this->displacement_method == cuda::displacement::method_t::b_spgrid ||
        this->displacement_method == cuda::displacement::method_t::b_spgrid_joints) &&
        this->output_bspgrid_distance)
    {
        auto displacement_distance_array = vtkFloatArray::SafeDownCast(this->output_grids->get_results().grids->GetPointData()->GetArray("B-Spgrid Distance"));

        vtkIdType index = 0;
        vtkIdType cell_index = 0;

        for (vtkIdType l = 0; l < this->output_grids->get_results().grids->Getgrids()->GetNumberOfCells(); ++l)
        {
            const auto num_points = this->output_grids->get_results().grids->Getgrids()->GetData()->GetValue(cell_index);

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

    this->results.grids = this->output_grids->get_results().grids;

    return true;
}

void algorithm_grid_output_update::cache_load() const
{
    std::cout << "Loading deformed grid output from cache" << std::endl;
}

const algorithm_grid_output_update::results_t& algorithm_grid_output_update::get_results() const
{
    return this->results;
}
