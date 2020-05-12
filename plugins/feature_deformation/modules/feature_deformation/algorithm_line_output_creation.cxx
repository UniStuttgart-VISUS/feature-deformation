#include "algorithm_line_output_creation.h"

#include "algorithm_line_input.h"

#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include <iostream>
#include <memory>

void algorithm_line_output_creation::set_input(std::shared_ptr<const algorithm_line_input> input_lines)
{
    this->input_lines = input_lines;
}

std::uint32_t algorithm_line_output_creation::calculate_hash() const
{
    if (!this->input_lines->is_valid())
    {
        return -1;
    }

    return this->input_lines->get_hash();
}

bool algorithm_line_output_creation::run_computation()
{
    // Create output geometry
    if (!this->is_quiet()) std::cout << "Creating deformed lines output" << std::endl;

    this->results.lines = vtkSmartPointer<vtkPolyData>::New();
    this->results.lines->DeepCopy(this->input_lines->get_results().input_lines);

    auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
    displacement_id_array->SetNumberOfComponents(4);
    displacement_id_array->SetNumberOfTuples(this->results.lines->GetNumberOfPoints());
    displacement_id_array->SetName("Displacement Information");
    displacement_id_array->FillValue(0.0f);

    auto displacement_distance_array = vtkSmartPointer<vtkFloatArray>::New();
    displacement_distance_array->SetNumberOfComponents(1);
    displacement_distance_array->SetNumberOfTuples(this->results.lines->GetNumberOfPoints());
    displacement_distance_array->SetName("B-Spline Distance");
    displacement_distance_array->FillValue(0.0f);

    this->results.lines->GetPointData()->AddArray(displacement_id_array);
    this->results.lines->GetPointData()->AddArray(displacement_distance_array);

    return true;
}

const algorithm_line_output_creation::results_t& algorithm_line_output_creation::get_results() const
{
    return this->results;
}
