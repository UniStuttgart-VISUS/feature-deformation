#include "algorithm_line_output_set.h"

#include "algorithm_line_output_update.h"

#include "vtkInformation.h"

#include <iostream>
#include <memory>

void algorithm_line_output_set::set_input(const std::shared_ptr<const algorithm_line_output_update> output_lines,
    vtkInformation* const output_information, const double data_time)
{
    this->output_lines = output_lines;
    this->output_information = output_information;
    this->data_time = data_time;
}

std::uint32_t algorithm_line_output_set::calculate_hash() const
{
    return this->output_lines->get_hash();
}

bool algorithm_line_output_set::run_computation()
{
    auto output_deformed_lines = vtkPolyData::SafeDownCast(this->output_information->Get(vtkDataObject::DATA_OBJECT()));

    output_deformed_lines->ShallowCopy(this->output_lines->get_results().lines);
    output_deformed_lines->Modified();

    this->output_information->Set(vtkDataObject::DATA_TIME_STEP(), this->data_time);

    return true;
}

void algorithm_line_output_set::cache_load() const
{
    const_cast<algorithm_line_output_set*>(this)->run_computation();
}
