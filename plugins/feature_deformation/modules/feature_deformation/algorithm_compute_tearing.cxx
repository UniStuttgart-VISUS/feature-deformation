#include "algorithm_compute_tearing.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_displacement_precomputation.h"
#include "algorithm_grid_input.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "hash.h"
#include "smoothing.h"

#include "vtkIdTypeArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

void algorithm_compute_tearing::set_input(const std::shared_ptr<const algorithm_line_input> line_input,
    const std::shared_ptr<const algorithm_grid_input> grid_input, const smoothing::method_t smoothing_method,
    const smoothing::variant_t smoothing_variant, const float lambda, const int num_iterations,
    const cuda::displacement::method_t displacement_method, const cuda::displacement::parameter_t displacement_parameters,
    const cuda::displacement::b_spline_parameters_t bspline_parameters, const float remove_cells_scalar)
{
    this->line_input = line_input;
    this->grid_input = grid_input;
    this->smoothing_method = smoothing_method;
    this->smoothing_variant = smoothing_variant;
    this->lambda = lambda;
    this->num_iterations = num_iterations;
    this->displacement_method = displacement_method;
    this->displacement_parameters = displacement_parameters;
    this->bspline_parameters = bspline_parameters;
    this->remove_cells_scalar = remove_cells_scalar;

    this->alg_smoothing = std::make_shared<algorithm_smoothing>();
    this->alg_displacement_creation = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation = std::make_shared<algorithm_displacement_computation>();

    this->alg_smoothing->be_quiet();
    this->alg_displacement_creation->be_quiet();
    this->alg_displacement_precomputation->be_quiet();
    this->alg_displacement_computation->be_quiet();
}

std::uint32_t algorithm_compute_tearing::calculate_hash() const
{
    if (!(this->line_input->is_valid() && this->grid_input->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->line_input->get_hash(), this->grid_input->get_hash(), this->smoothing_method,
        this->smoothing_variant, this->lambda, this->num_iterations, this->displacement_method,
        this->displacement_parameters, this->remove_cells_scalar);
}

bool algorithm_compute_tearing::run_computation()
{
    if (!this->is_quiet()) std::cout << "Computing tearing" << std::endl;

    // Smooth line completely and create displacer
    this->alg_smoothing->run(this->line_input, this->smoothing_method, this->smoothing_variant, this->lambda, this->num_iterations);
    this->alg_displacement_creation->run(this->grid_input);
    this->alg_displacement_precomputation->run(this->alg_displacement_creation, this->alg_smoothing, this->line_input,
        this->displacement_method, this->displacement_parameters, this->bspline_parameters);
    this->alg_displacement_computation->run(this->alg_displacement_creation, this->alg_smoothing, this->displacement_method, this->displacement_parameters);

    const auto& displaced_positions = this->alg_displacement_computation->get_results().displacements->get_results();

    // Get grid information
    const auto dimension = this->grid_input->get_results().dimension;
    const auto spacing = this->grid_input->get_results().spacing;

    const auto is_2d = dimension[2] == 1;
    const auto threshold = this->remove_cells_scalar * this->grid_input->get_results().spacing.head(is_2d ? 2 : 3).norm();

    auto calc_index_point = [](const std::array<int, 3>& dimension, int x, int y, int z) -> int
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    };

    // Create output array
    this->results.tearing_cells = vtkSmartPointer<vtkIdTypeArray>::New();
    this->results.tearing_cells->SetNumberOfComponents(2);
    this->results.tearing_cells->SetNumberOfTuples(this->grid_input->get_results().grid->GetNumberOfPoints());
    this->results.tearing_cells->SetName("Tearing cells");
    this->results.tearing_cells->FillTypedComponent(0, 0);
    this->results.tearing_cells->FillTypedComponent(1, -1);

    // Set value to 1 for all points that are part of a cell that tears
    #pragma omp parallel for
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

                std::vector<Eigen::Vector3d> cell_points(is_2d ? 4 : 8);

                for (std::size_t point_index = 0; point_index < (is_2d ? 4 : 8); ++point_index)
                {
                    cell_points[point_index] = Eigen::Vector3d(displaced_positions[point_ids[point_index]][0],
                        displaced_positions[point_ids[point_index]][1], displaced_positions[point_ids[point_index]][2]);
                }

                // Pairwise calculate the distance between all points and compare the result with the threshold
                for (std::size_t i = 0; i < cell_points.size() - 1; ++i)
                {
                    for (std::size_t j = i + 1; j < cell_points.size(); ++j)
                    {
                        discard_cell |= (cell_points[i] - cell_points[j]).norm() > threshold;
                    }
                }

                // Set value if discarded
                if (discard_cell)
                {
                    for (std::size_t point_index = 0; point_index < (is_2d ? 4 : 8); ++point_index)
                    {
                        this->results.tearing_cells->SetTypedComponent(point_ids[point_index], 0, 1);
                    }
                }
            }
        }
    }

    // Use region growing to detect and label connected tearing regions
    int next_region = 0;

    auto hasher = [&dimension, &calc_index_point](const std::tuple<int, int, int>& key) -> std::size_t {
        return std::hash<int>()(calc_index_point(dimension, std::get<0>(key), std::get<1>(key), std::get<2>(key)));
    };

    std::unordered_set<std::tuple<int, int, int>, decltype(hasher)> todo(29, hasher);

    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x)
            {
                const auto index = calc_index_point(dimension, x, y, z);

                std::array<vtkIdType, 2> value;
                this->results.tearing_cells->GetTypedTuple(index, value.data());

                if (value[0] == 1 && value[1] == -1)
                {
                    const auto current_region = next_region++;

                    todo.insert({ x, y, z });

                    // Take first item, process it, and put unprocessed neighbors on the ToDo list
                    while (!todo.empty())
                    {
                        const auto current_coords = *todo.cbegin();
                        todo.erase(current_coords);

                        const auto current_x = std::get<0>(current_coords);
                        const auto current_y = std::get<1>(current_coords);
                        const auto current_z = std::get<2>(current_coords);
                        const auto current_index = calc_index_point(dimension, current_x, current_y, current_z);

                        this->results.tearing_cells->SetTypedComponent(current_index, 1, current_region);

                        // Add neighbors to ToDo list
                        for (int k = (is_2d ? 0 : -1); k <= (is_2d ? 0 : 1); ++k)
                        {
                            for (int j = -1; j <= 1; ++j)
                            {
                                for (int i = -1; i <= 1; ++i)
                                {
                                    if (i != 0 || j != 0 || k != 0)
                                    {
                                        const auto neighbor_x = current_x + i;
                                        const auto neighbor_y = current_y + j;
                                        const auto neighbor_z = current_z + k;
                                        const auto neighbor_index = calc_index_point(dimension, neighbor_x, neighbor_y, neighbor_z);

                                        if (neighbor_x >= 0 && neighbor_x < dimension[0] &&
                                            neighbor_y >= 0 && neighbor_y < dimension[1] &&
                                            neighbor_z >= 0 && neighbor_z < dimension[2])
                                        {
                                            std::array<vtkIdType, 2> value;
                                            this->results.tearing_cells->GetTypedTuple(neighbor_index, value.data());

                                            if (value[0] == 1 && value[1] == -1)
                                            {
                                                todo.insert({ neighbor_x, neighbor_y, neighbor_z });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

const algorithm_compute_tearing::results_t& algorithm_compute_tearing::get_results() const
{
    return this->results;
}
