#include "algorithm_geometry_output_creation.h"

#include "algorithm_geometry_input.h"

#include "vtkDataObjectTypes.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include <iostream>
#include <memory>

void algorithm_geometry_output_creation::set_input(std::shared_ptr<const algorithm_geometry_input> input_geometry)
{
    this->input_geometry = input_geometry;
}

std::uint32_t algorithm_geometry_output_creation::calculate_hash() const
{
    if (!this->input_geometry->is_valid())
    {
        return -1;
    }

    return this->input_geometry->get_hash();
}

bool algorithm_geometry_output_creation::run_computation()
{
    if (!this->is_quiet()) std::cout << "Creating deformed geometry output" << std::endl;

    this->results.geometry = vtkSmartPointer<vtkMultiBlockDataSet>::New();

    for (unsigned int block_index = 0; block_index < static_cast<unsigned int>(this->input_geometry->get_results().geometry.size()); ++block_index)
    {
        this->results.geometry->SetBlock(block_index, vtkDataObjectTypes::NewDataObject(this->input_geometry->get_results().geometry[block_index]->GetClassName()));
        this->results.geometry->GetBlock(block_index)->DeepCopy(this->input_geometry->get_results().geometry[block_index]);

        auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
        displacement_id_array->SetNumberOfComponents(4);
        displacement_id_array->SetNumberOfTuples(this->input_geometry->get_results().geometry[block_index]->GetNumberOfPoints());
        displacement_id_array->SetName("Displacement Information");
        displacement_id_array->FillValue(0.0f);

        auto displacement_distance_array = vtkSmartPointer<vtkFloatArray>::New();
        displacement_distance_array->SetNumberOfComponents(1);
        displacement_distance_array->SetNumberOfTuples(this->input_geometry->get_results().geometry[block_index]->GetNumberOfPoints());
        displacement_distance_array->SetName("B-Spline Distance");
        displacement_distance_array->FillValue(0.0f);

        auto displacement_map_array = vtkSmartPointer<vtkDoubleArray>::New();
        displacement_map_array->SetNumberOfComponents(3);
        displacement_map_array->SetNumberOfTuples(this->input_geometry->get_results().geometry[block_index]->GetNumberOfPoints());
        displacement_map_array->SetName("Displacement Map");
        displacement_map_array->FillValue(0.0);

        vtkPointSet::SafeDownCast(this->results.geometry->GetBlock(block_index))->GetPointData()->AddArray(displacement_id_array);
        vtkPointSet::SafeDownCast(this->results.geometry->GetBlock(block_index))->GetPointData()->AddArray(displacement_distance_array);
        vtkPointSet::SafeDownCast(this->results.geometry->GetBlock(block_index))->GetPointData()->AddArray(displacement_map_array);
    }

    return true;
}

const algorithm_geometry_output_creation::results_t& algorithm_geometry_output_creation::get_results() const
{
    return this->results;
}
