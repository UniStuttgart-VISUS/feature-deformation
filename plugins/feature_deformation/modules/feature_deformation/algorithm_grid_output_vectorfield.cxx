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
    auto orig_divergence = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Divergence (Original)"));
    auto orig_curl = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Curl (Original)"));
    auto def_divergence = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Divergence (Deformed)"));
    auto def_curl = vtkDoubleArray::SafeDownCast(grid->GetPointData()->GetArray("Curl (Deformed)"));
    auto data_array = vector_field->get_results().vector_field;

    auto calc_index_point = [](const std::array<int, 3>& dimension, int x, int y, int z) -> int
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    };

    auto calc_jacobian = [](vtkDataArray* field, const int center,
        const int index, const int max, const int component, double h_l, double h_r, const int offset) -> double
    {
        double left_diff = 0.0;
        double right_diff = 0.0;
        int num = 0;

        if (center != 0) // Backward difference
        {
            const auto left = field->GetComponent(index - offset, component);
            const auto right = field->GetComponent(index, component);

            left_diff = (right - left) / h_l;
            ++num;
        }
        if (center != max) // Forward difference
        {
            const auto left = field->GetComponent(index, component);
            const auto right = field->GetComponent(index + offset, component);

            right_diff = (right - left) / h_r;
            ++num;
        }

        return (left_diff + right_diff) / num;
    };

    auto calc_jacobian_irregular = [calc_jacobian](vtkDataArray* field, const int center,
        const int index, const int max, const int component, vtkDoubleArray* h, const int offset) -> double
    {
        double h_l = 0.0;
        double h_r = 0.0;

        if (center != 0) // Backward difference
        {
            Eigen::Vector3d left, right;
            h->GetTuple(index - offset, left.data());
            h->GetTuple(index, right.data());

            h_l = (right - left).norm();
        }
        if (center != max) // Forward difference
        {
            Eigen::Vector3d left, right;
            h->GetTuple(index, left.data());
            h->GetTuple(index + offset, right.data());

            h_r = (right - left).norm();
        }

        return calc_jacobian(field, center, index, max, component, h_l, h_r, offset);
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

                // Calculate velocities
                Eigen::Vector3d velocity;
                data_array->GetTuple(calc_index_point(dimension, x, y, z), velocity.data());
                //velocity = (Jacobian / std::abs(Jacobian.determinant())) * velocity;
                velocity = Jacobian * velocity;

                velocities->SetTuple(index_p, velocity.data());
            }
        }
    }

    #pragma omp parallel for
    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index_p = calc_index_point(dimension, x, y, z);

                // Calculate divergence and curl of the original vector field
                Eigen::Matrix3d J_vel;

                {
                    const auto Jxdx = (dimension[0] > 1) ? calc_jacobian(data_array, x, index_p, dimension[0] - 1, 0, spacing[0], spacing[0], 1) : 0.0;
                    const auto Jxdy = (dimension[1] > 1) ? calc_jacobian(data_array, y, index_p, dimension[1] - 1, 0, spacing[1], spacing[1], dimension[0]) : 0.0;
                    const auto Jxdz = (dimension[2] > 1) ? calc_jacobian(data_array, z, index_p, dimension[2] - 1, 0, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;
                    const auto Jydx = (dimension[0] > 1) ? calc_jacobian(data_array, x, index_p, dimension[0] - 1, 1, spacing[0], spacing[0], 1) : 0.0;
                    const auto Jydy = (dimension[1] > 1) ? calc_jacobian(data_array, y, index_p, dimension[1] - 1, 1, spacing[1], spacing[1], dimension[0]) : 0.0;
                    const auto Jydz = (dimension[2] > 1) ? calc_jacobian(data_array, z, index_p, dimension[2] - 1, 1, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;
                    const auto Jzdx = (dimension[0] > 1) ? calc_jacobian(data_array, x, index_p, dimension[0] - 1, 2, spacing[0], spacing[0], 1) : 0.0;
                    const auto Jzdy = (dimension[1] > 1) ? calc_jacobian(data_array, y, index_p, dimension[1] - 1, 2, spacing[1], spacing[1], dimension[0]) : 0.0;
                    const auto Jzdz = (dimension[2] > 1) ? calc_jacobian(data_array, z, index_p, dimension[2] - 1, 2, spacing[2], spacing[2], dimension[0] * dimension[1]) : 0.0;

                    J_vel << Jxdx, Jxdy, Jxdz, Jydx, Jydy, Jydz, Jzdx, Jzdy, Jzdz;
                }

                const auto divergence_orig = J_vel.trace();
                const auto curl_orig = Eigen::Vector3d(J_vel(7) - J_vel(5), J_vel(2) - J_vel(6), J_vel(3) - J_vel(1));

                // Calculate divergence and curl of the deformed vector field
                // c.f. https://www.youtube.com/watch?v=00NnJBv6-q0
                Eigen::Matrix3d J_vel_def;

                {
                    const auto Jxdx = (dimension[0] > 1) ? calc_jacobian_irregular(velocities, x, index_p, dimension[0] - 1, 0, displacement_map, 1) : 0.0;
                    const auto Jxdy = (dimension[1] > 1) ? calc_jacobian_irregular(velocities, y, index_p, dimension[1] - 1, 0, displacement_map, dimension[0]) : 0.0;
                    const auto Jxdz = (dimension[2] > 1) ? calc_jacobian_irregular(velocities, z, index_p, dimension[2] - 1, 0, displacement_map, dimension[0] * dimension[1]) : 0.0;
                    const auto Jydx = (dimension[0] > 1) ? calc_jacobian_irregular(velocities, x, index_p, dimension[0] - 1, 1, displacement_map, 1) : 0.0;
                    const auto Jydy = (dimension[1] > 1) ? calc_jacobian_irregular(velocities, y, index_p, dimension[1] - 1, 1, displacement_map, dimension[0]) : 0.0;
                    const auto Jydz = (dimension[2] > 1) ? calc_jacobian_irregular(velocities, z, index_p, dimension[2] - 1, 1, displacement_map, dimension[0] * dimension[1]) : 0.0;
                    const auto Jzdx = (dimension[0] > 1) ? calc_jacobian_irregular(velocities, x, index_p, dimension[0] - 1, 2, displacement_map, 1) : 0.0;
                    const auto Jzdy = (dimension[1] > 1) ? calc_jacobian_irregular(velocities, y, index_p, dimension[1] - 1, 2, displacement_map, dimension[0]) : 0.0;
                    const auto Jzdz = (dimension[2] > 1) ? calc_jacobian_irregular(velocities, z, index_p, dimension[2] - 1, 2, displacement_map, dimension[0] * dimension[1]) : 0.0;

                    J_vel_def << Jxdx, Jxdy, Jxdz, Jydx, Jydy, Jydz, Jzdx, Jzdy, Jzdz;
                }

                Eigen::Matrix3d Jacobian;
                jacobian->GetTuple(index_p, Jacobian.data());

                const Eigen::Vector3d lame_vec_1 = Jacobian.col(0);
                const Eigen::Vector3d lame_vec_2 = Jacobian.col(1);
                const Eigen::Vector3d lame_vec_3 = Jacobian.col(2);

                const auto lame_coeff_1 = lame_vec_1.norm();
                const auto lame_coeff_2 = lame_vec_2.norm();
                const auto lame_coeff_3 = lame_vec_3.norm();

                const Eigen::Vector3d norm_lame_vec_1 = lame_vec_1 / lame_coeff_1;
                const Eigen::Vector3d norm_lame_vec_2 = lame_vec_2 / lame_coeff_2;
                const Eigen::Vector3d norm_lame_vec_3 = lame_vec_3 / lame_coeff_3;

                const auto divergence_def = (1.0 / (lame_coeff_1 * lame_coeff_2 * lame_coeff_3)) *
                    ((J_vel_def(0, 0) * lame_coeff_2 * lame_coeff_3)
                        + (J_vel_def(1, 1) * lame_coeff_1 * lame_coeff_3)
                        + (J_vel_def(2, 2) * lame_coeff_1 * lame_coeff_2));

                const Eigen::Vector3d curl_def = (1.0 / (lame_coeff_1 * lame_coeff_2 * lame_coeff_3)) *
                    (((J_vel_def(7) - J_vel_def(5)) * lame_vec_1)
                        + ((J_vel_def(2) - J_vel_def(6)) * lame_vec_2)
                        + ((J_vel_def(3) - J_vel_def(1)) * lame_vec_3));

                // Store computed values
                orig_divergence->SetComponent(index_p, 0, divergence_orig);
                orig_curl->SetTuple(index_p, curl_orig.data());
                def_divergence->SetComponent(index_p, 0, divergence_def);
                def_curl->SetTuple(index_p, curl_def.data());
            }
        }
    }

    jacobian->Modified();
    velocities->Modified();
    orig_divergence->Modified();
    orig_curl->Modified();
    def_divergence->Modified();
    def_curl->Modified();

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
