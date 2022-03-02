#include "algorithm_twisting.h"

#include "hash.h"
#include "twisting.h"

#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>
#include <memory>
#include <vector>

algorithm_twisting::algorithm_twisting()
{
}

void algorithm_twisting::set_input(
    std::shared_ptr<const algorithm_vectorfield_input> vector_field,
    std::shared_ptr<const algorithm_grid_input> grid,
    std::shared_ptr<const algorithm_line_input> lines,
    std::shared_ptr<const algorithm_smoothing> straight_feature_line,
    std::shared_ptr<const algorithm_displacement_computation> displacement,
    const bool active,
    const int selected_eigenvector
)
{
    this->vector_field = vector_field;
    this->grid = grid;
    this->lines = lines;
    this->straight_feature_line = straight_feature_line;
    this->displacement = displacement;

    this->active = active;
    this->selected_eigenvector = selected_eigenvector;
}

std::uint32_t algorithm_twisting::calculate_hash() const
{
    if (!this->active || !this->vector_field->is_valid() || !this->grid->is_valid() || !this->lines->is_valid() ||
        !this->straight_feature_line->is_valid() || !this->displacement->is_valid())
    {
        return -1;
    }

    return jenkins_hash(this->vector_field->get_hash(), this->grid->get_hash(), this->lines->get_hash(),
        this->straight_feature_line->get_hash(), this->displacement->get_hash(), this->selected_eigenvector);
}

bool algorithm_twisting::run_computation()
{
    if (!this->is_quiet()) std::cout << "Twisting line" << std::endl;

    // Get straightened line
    const auto& original_positions = this->straight_feature_line->get_results().positions;
    const auto& displacements = this->straight_feature_line->get_results().displacements;

    std::vector<Eigen::Vector3d> straight_line(original_positions.size());

    for (std::size_t i = 0; i < straight_line.size(); ++i)
    {
        straight_line[i][0] = static_cast<double>(original_positions[i][0]) + displacements[i][0];
        straight_line[i][1] = static_cast<double>(original_positions[i][1]) + displacements[i][1];
        straight_line[i][2] = static_cast<double>(original_positions[i][2]) + displacements[i][2];
    }

    // Get deformed grid
    auto deformed_grid = vtkSmartPointer<vtkStructuredGrid>::New();
    vtkStructuredGrid::SafeDownCast(deformed_grid)->SetExtent(const_cast<int*>(this->grid->get_results().extent.data()));

    const auto num_points = this->grid->get_results().dimension[0] *
        this->grid->get_results().dimension[1] * this->grid->get_results().dimension[2];

    auto coords = vtkSmartPointer<vtkPoints>::New();
    coords->SetNumberOfPoints(num_points);

    const auto& displacement = this->displacement->get_results().displacements->get_results();

    #pragma omp parallel for
    for (vtkIdType i = 0; i < num_points; ++i)
    {
        coords->SetPoint(i, displacement[i].data());
    }

    deformed_grid->SetPoints(coords);

    // Create deformed velocities
    auto velocities = vtkSmartPointer<vtkDoubleArray>::New();
    velocities->SetNumberOfComponents(3);
    velocities->SetNumberOfTuples(num_points);
    velocities->SetName("velocity");
    velocities->FillValue(0.0);

    auto jacobian = vtkSmartPointer<vtkDoubleArray>::New();
    jacobian->SetNumberOfComponents(9);
    jacobian->SetNumberOfTuples(num_points);
    jacobian->SetName("jacobian");
    jacobian->FillValue(0.0);

    auto calc_index_point = [](const std::array<int, 3>& dimension, int x, int y, int z) -> int
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    };

    auto calc_jacobian = [](vtkPoints* field, const int center,
        const int index, const int max, const int component, double h_l, double h_r, const int offset) -> double
    {
        double left_diff = 0.0;
        double right_diff = 0.0;
        int num = 0;

        std::array<double, 3> point_l{}, point_r{};

        if (center != 0) // Backward difference
        {
            field->GetPoint(static_cast<std::size_t>(index) - offset, point_l.data());
            field->GetPoint(index, point_r.data());

            left_diff = (point_r[component] - point_l[component]) / h_l;
            ++num;
        }
        if (center != max) // Forward difference
        {
            field->GetPoint(index, point_l.data());
            field->GetPoint(static_cast<std::size_t>(index) + offset, point_r.data());

            right_diff = (point_r[component] - point_l[component]) / h_r;
            ++num;
        }

        return (left_diff + right_diff) / num;
    };

    const auto& dimension = this->grid->get_results().dimension;
    const auto& spacing = this->grid->get_results().spacing;

    #pragma omp parallel for
    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index_p = calc_index_point(dimension, x, y, z);

                // Calculate Jacobian of the displacement
                const auto Jxdx = (dimension[0] > 1) ? calc_jacobian(coords, x, index_p, dimension[0] - 1, 0, spacing[0], spacing[0], 1) : 1.0;
                const auto Jxdy = (dimension[1] > 1) ? calc_jacobian(coords, y, index_p, dimension[1] - 1, 0, spacing[1], spacing[1], dimension[0]) : 0.0;
                const auto Jxdz = (dimension[2] > 1) ? calc_jacobian(coords, z, index_p, dimension[2] - 1, 0, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;
                const auto Jydx = (dimension[0] > 1) ? calc_jacobian(coords, x, index_p, dimension[0] - 1, 1, spacing[0], spacing[0], 1) : 0.0;
                const auto Jydy = (dimension[1] > 1) ? calc_jacobian(coords, y, index_p, dimension[1] - 1, 1, spacing[1], spacing[1], dimension[0]) : 1.0;
                const auto Jydz = (dimension[2] > 1) ? calc_jacobian(coords, z, index_p, dimension[2] - 1, 1, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;
                const auto Jzdx = (dimension[0] > 1) ? calc_jacobian(coords, x, index_p, dimension[0] - 1, 2, spacing[0], spacing[0], 1) : 0.0;
                const auto Jzdy = (dimension[1] > 1) ? calc_jacobian(coords, y, index_p, dimension[1] - 1, 2, spacing[1], spacing[1], dimension[0]) : 0.0;
                const auto Jzdz = (dimension[2] > 1) ? calc_jacobian(coords, z, index_p, dimension[2] - 1, 2, spacing[2], spacing[2], dimension[0] * dimension[1]) : 1.0;

                Eigen::Matrix3d Jacobian;
                Jacobian << Jxdx, Jxdy, Jxdz, Jydx, Jydy, Jydz, Jzdx, Jzdy, Jzdz;

                // Calculate velocities
                Eigen::Vector3d velocity;
                this->vector_field->get_results().vector_field->GetTuple(calc_index_point(dimension, x, y, z), velocity.data());
                velocity = Jacobian * velocity;

                velocities->SetTuple(index_p, velocity.data());
                jacobian->SetTuple(index_p, Jacobian.data());
            }
        }
    }

    deformed_grid->GetPointData()->AddArray(velocities);
    deformed_grid->GetPointData()->AddArray(jacobian);

    // Create twister algorithm
    twisting twister(straight_line, deformed_grid, this->selected_eigenvector);

    if (!twister.run())
    {
        std::cerr << "ERROR: Twisting failed." << std::endl;
        return false;
    }

    const auto twisting_results = twister.get_rotations();

    // Convert results
    this->results.rotations.resize(twisting_results.first.size());

    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(this->results.rotations.size()); ++i)
    {
        this->results.rotations[i] = { static_cast<float>(twisting_results.first[i]), 0.0f, 0.0f, 0.0f };
    }

    this->results.coordinate_systems = vtkSmartPointer<vtkDoubleArray>::New();
    this->results.coordinate_systems->SetName("Coordinate System");
    this->results.coordinate_systems->SetNumberOfComponents(9);
    this->results.coordinate_systems->SetNumberOfTuples(this->lines->get_results().input_lines->GetNumberOfPoints());
    this->results.coordinate_systems->FillValue(0.0);

    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(twisting_results.second.size()); ++i)
    {
        this->results.coordinate_systems->SetTuple(this->lines->get_results().selected_line_ids[i], twisting_results.second[i].data());
    }

    return true;
}

void algorithm_twisting::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading twisted line from cache" << std::endl;
}

const algorithm_twisting::results_t& algorithm_twisting::get_results() const
{
    return this->results;
}
