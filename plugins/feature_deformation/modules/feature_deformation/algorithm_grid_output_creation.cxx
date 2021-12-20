#include "algorithm_grid_output_creation.h"

#include "algorithm_grid_input.h"
#include "hash.h"

#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkInformation.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPointSet.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"
#include "vtkUnstructuredGrid.h"

#include <iostream>
#include <memory>

void algorithm_grid_output_creation::set_input(const std::shared_ptr<const algorithm_grid_input> input_grid, const bool remove_cells)
{
    this->input_grid = input_grid;
    this->remove_cells = remove_cells;
}

std::uint32_t algorithm_grid_output_creation::calculate_hash() const
{
    if (!this->input_grid->is_valid())
    {
        return -1;
    }

    return jenkins_hash(this->input_grid->get_hash(), this->remove_cells);
}

bool algorithm_grid_output_creation::run_computation()
{
    if (!this->is_quiet()) std::cout << "Creating deformed grid output" << std::endl;

    // Create structured or unstructured grid
    vtkSmartPointer<vtkPointSet> output_deformed_grid = nullptr;

    if (this->remove_cells)
    {
        output_deformed_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    }
    else
    {
        output_deformed_grid = vtkSmartPointer<vtkStructuredGrid>::New();
        vtkStructuredGrid::SafeDownCast(output_deformed_grid)->SetExtent(const_cast<int*>(this->input_grid->get_results().extent.data()));
    }

    // Set points of the undeformed grid
    const auto num_cells = (this->input_grid->get_results().dimension[0] - 1) *
        (this->input_grid->get_results().dimension[1] - 1) * (this->input_grid->get_results().dimension[2] - 1);
    const auto num_points = this->input_grid->get_results().dimension[0] *
        this->input_grid->get_results().dimension[1] * this->input_grid->get_results().dimension[2];

    auto coords = vtkSmartPointer<vtkPoints>::New();
    coords->SetNumberOfPoints(num_points);

    auto tex_coords = vtkSmartPointer<vtkFloatArray>::New();
    tex_coords->SetNumberOfComponents(3);
    tex_coords->SetNumberOfTuples(num_points);
    tex_coords->SetName("Original Coordinates");

    vtkIdType index = 0;

    #pragma omp parallel for
    for (int z = 0; z < this->input_grid->get_results().dimension[2]; ++z)
    {
        for (int y = 0; y < this->input_grid->get_results().dimension[1]; ++y)
        {
            for (int x = 0; x < this->input_grid->get_results().dimension[0]; ++x, ++index)
            {
                const Eigen::Vector3f point = this->input_grid->get_results().origin + Eigen::Vector3f(x, y, z)
                    .cwiseProduct(this->input_grid->get_results().spacing);

                coords->SetPoint(index, point.data());
                tex_coords->SetTuple(index, point.data());
            }
        }
    }

    output_deformed_grid->SetPoints(coords);
    output_deformed_grid->GetPointData()->AddArray(tex_coords);

    if (this->remove_cells)
    {
        vtkUnstructuredGrid::SafeDownCast(output_deformed_grid)->Allocate(num_cells);
    }

    // Create information arrays
    auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
    displacement_id_array->SetNumberOfComponents(4);
    displacement_id_array->SetNumberOfTuples(num_points);
    displacement_id_array->SetName("Displacement Information");
    displacement_id_array->FillValue(0.0f);

    auto mapping_array = vtkSmartPointer<vtkFloatArray>::New();
    mapping_array->SetNumberOfComponents(3);
    mapping_array->SetNumberOfTuples(num_points);
    mapping_array->SetName("Mapping to B-Spline");
    mapping_array->FillValue(0.0f);

    auto mapping_array_original = vtkSmartPointer<vtkFloatArray>::New();
    mapping_array_original->SetNumberOfComponents(3);
    mapping_array_original->SetNumberOfTuples(num_points);
    mapping_array_original->SetName("Mapping to B-Spline (Original)");
    mapping_array_original->FillValue(0.0f);

    auto displacement_map = vtkSmartPointer<vtkDoubleArray>::New();
    displacement_map->SetNumberOfComponents(3);
    displacement_map->SetNumberOfTuples(num_points);
    displacement_map->SetName("Displacement Map");
    displacement_map->FillValue(0.0);

    auto jacobian = vtkSmartPointer<vtkDoubleArray>::New();
    jacobian->SetNumberOfComponents(9);
    jacobian->SetNumberOfTuples(num_points);
    jacobian->SetName("Jacobian of Deformation");
    jacobian->FillValue(0.0);

    auto velocities = vtkSmartPointer<vtkDoubleArray>::New();
    velocities->SetNumberOfComponents(3);
    velocities->SetNumberOfTuples(num_points);
    velocities->SetName("Deformed Velocities");
    velocities->FillValue(0.0);

    auto orig_divergence = vtkSmartPointer<vtkDoubleArray>::New();
    orig_divergence->SetNumberOfComponents(1);
    orig_divergence->SetNumberOfTuples(num_points);
    orig_divergence->SetName("Divergence (Original)");
    orig_divergence->FillValue(0.0);

    auto orig_curl = vtkSmartPointer<vtkDoubleArray>::New();
    orig_curl->SetNumberOfComponents(3);
    orig_curl->SetNumberOfTuples(num_points);
    orig_curl->SetName("Curl (Original)");
    orig_curl->FillValue(0.0);

    auto def_divergence = vtkSmartPointer<vtkDoubleArray>::New();
    def_divergence->SetNumberOfComponents(1);
    def_divergence->SetNumberOfTuples(num_points);
    def_divergence->SetName("Divergence (Deformed)");
    def_divergence->FillValue(0.0);

    auto def_curl = vtkSmartPointer<vtkDoubleArray>::New();
    def_curl->SetNumberOfComponents(3);
    def_curl->SetNumberOfTuples(num_points);
    def_curl->SetName("Curl (Deformed)");
    def_curl->FillValue(0.0);

    output_deformed_grid->GetPointData()->AddArray(displacement_id_array);
    output_deformed_grid->GetPointData()->AddArray(mapping_array);
    output_deformed_grid->GetPointData()->AddArray(mapping_array_original);
    output_deformed_grid->GetPointData()->AddArray(displacement_map);
    output_deformed_grid->GetPointData()->AddArray(jacobian);
    output_deformed_grid->GetPointData()->AddArray(velocities);
    output_deformed_grid->GetPointData()->AddArray(orig_divergence);
    output_deformed_grid->GetPointData()->AddArray(orig_curl);
    output_deformed_grid->GetPointData()->AddArray(def_divergence);
    output_deformed_grid->GetPointData()->AddArray(def_curl);

    // Create dataset
    this->results.grid = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    this->results.grid->SetBlock(0u, output_deformed_grid);
    this->results.grid->GetMetaData(0u)->Set(vtkCompositeDataSet::NAME(), "Grid");

    return true;
}

const algorithm_grid_output_creation::results_t& algorithm_grid_output_creation::get_results() const
{
    return this->results;
}
