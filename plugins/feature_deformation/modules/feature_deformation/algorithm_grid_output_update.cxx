#include "algorithm_grid_output_update.h"

#include "algorithm_compute_tearing.h"
#include "algorithm_displacement_assessment.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_computation_twisting.h"
#include "algorithm_displacement_computation_winding.h"
#include "algorithm_grid_output_creation.h"
#include "hash.h"
#include "jacobian.h"

#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkIdTypeArray.h"
#include "vtkInformation.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPointSet.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkUnstructuredGrid.h"

#include <cmath>
#include <iostream>
#include <memory>

void algorithm_grid_output_update::set_input(const std::shared_ptr<const algorithm_grid_input> input_grid,
    const std::shared_ptr<const algorithm_grid_output_creation> output_grid,
    std::shared_ptr<const algorithm_displacement_computation> displacement,
    std::shared_ptr<const algorithm_displacement_computation_winding> displacement_winding,
    std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting,
    const std::shared_ptr<const algorithm_displacement_assessment> assessment,
    const std::shared_ptr<const algorithm_compute_tearing> tearing, const bool remove_cells,
    const float remove_cells_scalar, const bool minimal_output)
{
    this->input_grid = input_grid;
    this->output_grid = output_grid;
    this->displacement = displacement;
    this->displacement_winding = displacement_winding;
    this->displacement_twisting = displacement_twisting;
    this->assessment = assessment;
    this->tearing = tearing;
    this->remove_cells = remove_cells;
    this->remove_cells_scalar = remove_cells_scalar;
    this->minimal_output = minimal_output;
}

std::uint32_t algorithm_grid_output_update::calculate_hash() const
{
    if (!(this->output_grid->is_valid() && this->displacement->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->displacement_winding->get_hash(), this->displacement_twisting->get_hash(),
        this->assessment->get_hash(), this->tearing->get_hash(), this->remove_cells, this->remove_cells_scalar, this->minimal_output);
}

bool algorithm_grid_output_update::run_computation()
{
    if (!this->is_quiet()) std::cout << "Updating deformed grid output" << std::endl;

    // Set displaced points and displacement map
    const auto& displaced_grid = (this->displacement_winding->is_valid() || this->displacement_twisting->is_valid())
        ? this->displacement->get_results().displacements->get_results_twisting()
        : this->displacement->get_results().displacements->get_results();

    const auto grid = vtkPointSet::SafeDownCast(this->output_grid->get_results().grid->GetBlock(0u));
    auto displacement_map = vtkDoubleArray::FastDownCast(grid->GetPointData()->GetArray("Displacement Map"));

    #pragma omp parallel for
    for (vtkIdType i = 0; i < grid->GetNumberOfPoints(); ++i)
    {
        grid->GetPoints()->SetPoint(i, displaced_grid[i].data());
        displacement_map->SetTuple(i, displaced_grid[i].data());
    }

    displacement_map->Modified();

    // Create displacement ID array
    if (!this->minimal_output)
    {
        const auto displacement_ids = this->displacement->get_results().displacements->get_displacement_info();

        auto displacement_info = grid->GetPointData()->GetArray("Displacement Information");
        auto mapping = grid->GetPointData()->GetArray("Mapping to B-Spline");
        auto mapping_original = grid->GetPointData()->GetArray("Mapping to B-Spline (Original)");

        std::memcpy(displacement_info->GetVoidPointer(0),
            std::get<0>(displacement_ids).data(), std::get<0>(displacement_ids).size() * sizeof(float4));

        std::memcpy(mapping->GetVoidPointer(0),
            std::get<1>(displacement_ids).data(), std::get<1>(displacement_ids).size() * sizeof(float3));

        std::memcpy(mapping_original->GetVoidPointer(0),
            std::get<2>(displacement_ids).data(), std::get<2>(displacement_ids).size() * sizeof(float3));

        displacement_info->Modified();
        mapping->Modified();
        mapping_original->Modified();
    }

    // Calculate Jacobian of deformation
    const auto& dimension = this->input_grid->get_results().dimension;
    const auto& spacing = this->input_grid->get_results().spacing;

    auto jacobian = grid->GetPointData()->GetArray("Jacobian of Deformation");
    auto magnitude = grid->GetPointData()->GetArray("Magnitude of Deformation");

    #pragma omp parallel for
    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index_p = calc_index_point(dimension, x, y, z);

                // Calculate Jacobian of the displacement
                const auto Jxdx = (dimension[0] > 1) ? calc_jacobian(displacement_map, x, index_p, dimension[0] - 1, 0, spacing[0], spacing[0], 1) : 1.0;
                const auto Jxdy = (dimension[1] > 1) ? calc_jacobian(displacement_map, y, index_p, dimension[1] - 1, 0, spacing[1], spacing[1], dimension[0]) : 0.0;
                const auto Jxdz = (dimension[2] > 1) ? calc_jacobian(displacement_map, z, index_p, dimension[2] - 1, 0, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;
                const auto Jydx = (dimension[0] > 1) ? calc_jacobian(displacement_map, x, index_p, dimension[0] - 1, 1, spacing[0], spacing[0], 1) : 0.0;
                const auto Jydy = (dimension[1] > 1) ? calc_jacobian(displacement_map, y, index_p, dimension[1] - 1, 1, spacing[1], spacing[1], dimension[0]) : 1.0;
                const auto Jydz = (dimension[2] > 1) ? calc_jacobian(displacement_map, z, index_p, dimension[2] - 1, 1, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;
                const auto Jzdx = (dimension[0] > 1) ? calc_jacobian(displacement_map, x, index_p, dimension[0] - 1, 2, spacing[0], spacing[0], 1) : 0.0;
                const auto Jzdy = (dimension[1] > 1) ? calc_jacobian(displacement_map, y, index_p, dimension[1] - 1, 2, spacing[1], spacing[1], dimension[0]) : 0.0;
                const auto Jzdz = (dimension[2] > 1) ? calc_jacobian(displacement_map, z, index_p, dimension[2] - 1, 2, spacing[2], spacing[2], dimension[0] * dimension[1]) : 1.0;

                Eigen::Matrix3d Jacobian;
                Jacobian << Jxdx, Jxdy, Jxdz, Jydx, Jydy, Jydz, Jzdx, Jzdy, Jzdz;

                jacobian->SetTuple(index_p, Jacobian.data());
                magnitude->SetComponent(index_p, 0, Jacobian.norm());
            }
        }
    }

    jacobian->Modified();
    magnitude->Modified();

    // Create cells and create grid to store "removed" cells
    if (this->remove_cells)
    {
        // Create new deformed grid without cells
        const auto dimension = this->input_grid->get_results().dimension;
        const auto num_cells = (dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1);

        auto points = vtkSmartPointer<vtkPoints>::New();
        points->ShallowCopy(grid->GetPoints());

        auto output_deformed_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        output_deformed_grid->Allocate(num_cells);
        output_deformed_grid->SetPoints(points);
        output_deformed_grid->GetPointData()->ShallowCopy(grid->GetPointData());

        auto output_deformed_grid_removed = vtkSmartPointer<vtkUnstructuredGrid>::New();
        output_deformed_grid_removed->Allocate(num_cells);
        output_deformed_grid_removed->SetPoints(points);

        // Create cells
        const auto is_2d = dimension[2] == 1;
        const auto threshold = this->remove_cells_scalar * this->input_grid->get_results().spacing.head(is_2d ? 2 : 3).norm();

        auto handedness = vtkSmartPointer<vtkFloatArray>::New();
        handedness->SetNumberOfComponents(1);
        handedness->Allocate(num_cells);
        handedness->SetName("Handedness");

        for (int z = 0; z < (is_2d ? 1 : (dimension[2] - 1)); ++z)
        {
            for (int y = 0; y < dimension[1] - 1; ++y)
            {
                for (int x = 0; x < dimension[0] - 1; ++x)
                {
                    // Create point IDs
                    const auto point0 = calc_index_point(dimension, x + 0, y + 0, z + 0);
                    const auto point1 = calc_index_point(dimension, x + 1, y + 0, z + 0);
                    const auto point2 = calc_index_point(dimension, x + 0, y + 1, z + 0);
                    const auto point3 = calc_index_point(dimension, x + 1, y + 1, z + 0);
                    const auto point4 = calc_index_point(dimension, x + 0, y + 0, z + 1);
                    const auto point5 = calc_index_point(dimension, x + 1, y + 0, z + 1);
                    const auto point6 = calc_index_point(dimension, x + 0, y + 1, z + 1);
                    const auto point7 = calc_index_point(dimension, x + 1, y + 1, z + 1);

                    const std::array<vtkIdType, 8> point_ids{
                        point0,
                        point1,
                        point2,
                        point3,
                        point4,
                        point5,
                        point6,
                        point7
                    };

                    // Calculate distances between points and compare to threshold
                    bool discard_cell = false;

                    if (this->remove_cells)
                    {
                        // Get all cell points
                        std::vector<Eigen::Vector3d> cell_points(is_2d ? 4 : 8);

                        for (std::size_t point_index = 0; point_index < (is_2d ? 4 : 8); ++point_index)
                        {
                            output_deformed_grid->GetPoints()->GetPoint(point_ids[point_index], cell_points[point_index].data());
                        }

                        // Pairwise calculate the distance between all points and compare the result with the threshold
                        for (std::size_t i = 0; i < cell_points.size() - 1; ++i)
                        {
                            for (std::size_t j = i + 1; j < cell_points.size(); ++j)
                            {
                                discard_cell |= (cell_points[i] - cell_points[j]).norm() > threshold;
                            }
                        }
                    }

                    // Create cell faces
                    if (!discard_cell)
                    {
                        if (!is_2d)
                        {
                            auto faces = vtkSmartPointer<vtkCellArray>::New();

                            vtkIdType face0[4] = { point0, point1, point3, point2 }; // front
                            vtkIdType face1[4] = { point6, point7, point5, point4 }; // back
                            vtkIdType face2[4] = { point4, point5, point1, point0 }; // bottom
                            vtkIdType face3[4] = { point2, point3, point7, point6 }; // top
                            vtkIdType face4[4] = { point0, point2, point6, point4 }; // left
                            vtkIdType face5[4] = { point1, point5, point7, point3 }; // right

                            faces->InsertNextCell(4, face0);
                            faces->InsertNextCell(4, face1);
                            faces->InsertNextCell(4, face2);
                            faces->InsertNextCell(4, face3);
                            faces->InsertNextCell(4, face4);
                            faces->InsertNextCell(4, face5);

                            output_deformed_grid->InsertNextCell(VTK_POLYHEDRON, 8, point_ids.data(), 6, faces->GetData()->GetPointer(0));

                            // Calculate handedness
                            std::array<Eigen::Vector3d, 4> points;
                            output_deformed_grid->GetPoints()->GetPoint(point0, points[0].data());
                            output_deformed_grid->GetPoints()->GetPoint(point1, points[1].data());
                            output_deformed_grid->GetPoints()->GetPoint(point2, points[2].data());
                            output_deformed_grid->GetPoints()->GetPoint(point4, points[3].data());

                            const auto vector_1 = points[1] - points[0];
                            const auto vector_2 = points[2] - points[0];
                            const auto vector_3 = points[3] - points[0];

                            handedness->InsertNextValue(static_cast<float>(vector_1.cross(vector_2).dot(vector_3)));
                        }
                        else
                        {
                            const std::array<vtkIdType, 8> point_ids{ point0, point1, point3, point2 };

                            output_deformed_grid->InsertNextCell(VTK_QUAD, 4, point_ids.data());

                            // Calculate handedness
                            std::array<Eigen::Vector3d, 3> points;
                            output_deformed_grid->GetPoints()->GetPoint(point0, points[0].data());
                            output_deformed_grid->GetPoints()->GetPoint(point1, points[1].data());
                            output_deformed_grid->GetPoints()->GetPoint(point2, points[2].data());

                            const auto vector_1 = points[1] - points[0];
                            const auto vector_2 = points[2] - points[0];

                            handedness->InsertNextValue(static_cast<float>(vector_1.cross(vector_2)[2]));
                        }
                    }
                    else
                    {
                        if (!is_2d)
                        {
                            auto faces = vtkSmartPointer<vtkCellArray>::New();

                            vtkIdType face0[4] = { point0, point1, point3, point2 }; // front
                            vtkIdType face1[4] = { point6, point7, point5, point4 }; // back
                            vtkIdType face2[4] = { point4, point5, point1, point0 }; // bottom
                            vtkIdType face3[4] = { point2, point3, point7, point6 }; // top
                            vtkIdType face4[4] = { point0, point2, point6, point4 }; // left
                            vtkIdType face5[4] = { point1, point5, point7, point3 }; // right

                            faces->InsertNextCell(4, face0);
                            faces->InsertNextCell(4, face1);
                            faces->InsertNextCell(4, face2);
                            faces->InsertNextCell(4, face3);
                            faces->InsertNextCell(4, face4);
                            faces->InsertNextCell(4, face5);

                            output_deformed_grid_removed->InsertNextCell(VTK_POLYHEDRON, 8, point_ids.data(), 6, faces->GetData()->GetPointer(0));
                        }
                        else
                        {
                            const std::array<vtkIdType, 8> point_ids{ point0, point1, point3, point2 };

                            output_deformed_grid_removed->InsertNextCell(VTK_QUAD, 4, point_ids.data());
                        }
                    }
                }
            }
        }

        output_deformed_grid->GetCellData()->AddArray(handedness);

        // Add tear array if possible
        if (this->tearing->is_valid())
        {
            output_deformed_grid->GetPointData()->AddArray(this->tearing->get_results().tearing_cells);
        }

        this->output_grid->get_results().grid->SetBlock(0u, output_deformed_grid);
        this->output_grid->get_results().grid->SetBlock(1u, output_deformed_grid_removed);
        this->output_grid->get_results().grid->GetMetaData(1u)->Set(vtkCompositeDataSet::NAME(), "Removed Cells");
    }

    // Set input as output
    this->results.grid = this->output_grid->get_results().grid;

    return true;
}

void algorithm_grid_output_update::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading deformed grid output from cache" << std::endl;
}

const algorithm_grid_output_update::results_t& algorithm_grid_output_update::get_results() const
{
    return this->results;
}
