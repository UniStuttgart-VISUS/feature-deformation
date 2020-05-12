#include "algorithm_grid_output_set.h"

#include "algorithm_grid_output_update.h"
#include "hash.h"

#include "vtkInformation.h"

#include <iostream>
#include <memory>

void algorithm_grid_output_set::set_input(const std::shared_ptr<const algorithm_grid_output_update> output_grid,
    vtkInformation* const output_information, const double data_time)
{
    this->output_grid = output_grid;
    this->output_information = output_information;
}

std::uint32_t algorithm_grid_output_set::calculate_hash() const
{
    return this->output_grid->get_hash();
}

bool algorithm_grid_output_set::run_computation()
{
    auto output_deformed_grid = vtkMultiBlockDataSet::SafeDownCast(this->output_information->Get(vtkDataObject::DATA_OBJECT()));

    output_deformed_grid->ShallowCopy(this->output_grid->get_results().grid);
    output_deformed_grid->Modified();

    this->output_information->Set(vtkDataObject::DATA_TIME_STEP(), this->data_time);

    return true;
}

void algorithm_grid_output_set::cache_load() const
{
    const_cast<algorithm_grid_output_set*>(this)->run_computation();
}
