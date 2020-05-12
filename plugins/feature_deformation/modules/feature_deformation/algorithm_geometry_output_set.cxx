#include "algorithm_geometry_output_set.h"

#include "algorithm_geometry_output_update.h"
#include "hash.h"

#include "vtkInformation.h"

#include <iostream>
#include <memory>

void algorithm_geometry_output_set::set_input(std::shared_ptr<const algorithm_geometry_output_update> output_geometry,
    vtkInformation* output_information, double data_time)
{
    this->output_geometry = output_geometry;
    this->output_information = output_information;
}

std::uint32_t algorithm_geometry_output_set::calculate_hash() const
{
    return this->output_geometry->get_hash();
}

bool algorithm_geometry_output_set::run_computation()
{
    auto output_deformed_geometry = vtkMultiBlockDataSet::SafeDownCast(this->output_information->Get(vtkDataObject::DATA_OBJECT()));

    output_deformed_geometry->ShallowCopy(this->output_geometry->get_results().geometry);
    output_deformed_geometry->Modified();

    for (unsigned int block_index = 0; block_index < output_deformed_geometry->GetNumberOfBlocks(); ++block_index)
    {
        output_deformed_geometry->GetBlock(block_index)->GetInformation()->Set(vtkDataObject::DATA_TIME_STEP(), this->data_time);
        output_deformed_geometry->GetBlock(block_index)->Modified();
    }

    this->output_information->Set(vtkDataObject::DATA_TIME_STEP(), this->data_time);

    return true;
}

void algorithm_geometry_output_set::cache_load() const
{
    auto output_deformed_geometry = vtkMultiBlockDataSet::SafeDownCast(this->output_information->Get(vtkDataObject::DATA_OBJECT()));

    output_deformed_geometry->ShallowCopy(this->output_geometry->get_results().geometry);
    output_deformed_geometry->Modified();

    for (unsigned int block_index = 0; block_index < output_deformed_geometry->GetNumberOfBlocks(); ++block_index)
    {
        output_deformed_geometry->GetBlock(block_index)->GetInformation()->Set(vtkDataObject::DATA_TIME_STEP(), this->data_time);
        output_deformed_geometry->GetBlock(block_index)->Modified();
    }

    this->output_information->Set(vtkDataObject::DATA_TIME_STEP(), this->data_time);
}
