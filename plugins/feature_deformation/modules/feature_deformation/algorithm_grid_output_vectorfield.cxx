#include "algorithm_grid_output_vectorfield.h"

#include "algorithm_grid_output_update.h"
#include "algorithm_vectorfield_input.h"
#include "hash.h"

#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPointSet.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>
#include <memory>

void algorithm_grid_output_vectorfield::set_input(const std::shared_ptr<const algorithm_grid_input> input_grid,
    const std::shared_ptr<const algorithm_grid_output_update> output_grid, const std::shared_ptr<const algorithm_vectorfield_input> vector_field)
{
    this->input_grid = input_grid;
    this->output_grid = output_grid;
    this->vector_field = vector_field;
}

std::uint32_t algorithm_grid_output_vectorfield::calculate_hash() const
{
    if (!this->vector_field->is_valid())
    {
        return -1;
    }

    return jenkins_hash(this->output_grid->get_hash(), this->vector_field->get_hash());
}

bool algorithm_grid_output_vectorfield::run_computation()
{
    if (!this->is_quiet()) std::cout << "Computing deformed velocity field" << std::endl;

    // Setup finite differences for the calculation of the Jacobian
    auto grid = vtkPointSet::SafeDownCast(output_grid->get_results().grid->GetBlock(0u));
    auto displacement_map = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Displacement Map"));
    auto jacobian = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Jacobian of Deformation"));
    auto velocities = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Deformed Velocities"));
    auto data_array = vector_field->get_results().vector_field;

    auto calc_index_point = [](const std::array<int, 3>& dimension, int x, int y, int z) -> int
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    };

    auto calc_jacobian = [displacement_map](const int center, const int index, const int max, const int component, double h, const int offset) -> double
    {
        double left, right;

        if (center == 0) // Forward difference
        {
            left = displacement_map->GetComponent(index, component);
            right = displacement_map->GetComponent(index + offset, component);
        }
        else if (center == max) // Backward difference
        {
            left = displacement_map->GetComponent(index - offset, component);
            right = displacement_map->GetComponent(index, component);
        }
        else // Central difference
        {
            left = displacement_map->GetComponent(index - offset, component);
            right = displacement_map->GetComponent(index + offset, component);

            h *= 2.0;
        }

        return (right - left) / h;
    };

    // Calculate Jacobian and use it to calculate the velocities at the deformed grid
    const auto& dimension = this->input_grid->get_results().dimension;
    const auto& spacing = this->input_grid->get_results().spacing;

    #pragma omp parallel for
    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index_p = calc_index_point(dimension, x, y, z);

                // Calculate Jacobian
                const auto Jxdx = calc_jacobian(x, index_p, dimension[0] - 1, 0, spacing[0], 1);
                const auto Jxdy = calc_jacobian(y, index_p, dimension[1] - 1, 0, spacing[1], dimension[0]);
                const auto Jxdz = calc_jacobian(z, index_p, dimension[2] - 1, 0, spacing[2], dimension[0] * dimension[1]);
                const auto Jydx = calc_jacobian(x, index_p, dimension[0] - 1, 1, spacing[0], 1);
                const auto Jydy = calc_jacobian(y, index_p, dimension[1] - 1, 1, spacing[1], dimension[0]);
                const auto Jydz = calc_jacobian(z, index_p, dimension[2] - 1, 1, spacing[2], dimension[0] * dimension[1]);
                const auto Jzdx = calc_jacobian(x, index_p, dimension[0] - 1, 2, spacing[0], 1);
                const auto Jzdy = calc_jacobian(y, index_p, dimension[1] - 1, 2, spacing[1], dimension[0]);
                const auto Jzdz = calc_jacobian(z, index_p, dimension[2] - 1, 2, spacing[2], dimension[0] * dimension[1]);

                Eigen::Matrix3d Jacobian;
                Jacobian << Jxdx, Jxdy, Jxdz, Jydx, Jydy, Jydz, Jzdx, Jzdy, Jzdz;

                jacobian->SetTuple(index_p, Jacobian.data());

                // Calculate velocities
                Eigen::Vector3d velocity;
                data_array->GetTuple(calc_index_point(dimension, x, y, z), velocity.data());
                velocity = (Jacobian / Jacobian.determinant()) * velocity;

                velocities->SetTuple(index_p, velocity.data());
            }
        }
    }

    jacobian->Modified();
    velocities->Modified();

    // Set input as output
    this->results.grid = this->output_grid->get_results().grid;

    return true;
}

void algorithm_grid_output_vectorfield::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading deformed velocity field from cache" << std::endl;
}

const algorithm_grid_output_vectorfield::results_t& algorithm_grid_output_vectorfield::get_results() const
{
    return this->results;
}
