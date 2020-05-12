#include "algorithm_compute_gauss.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_displacement_precomputation.h"
#include "algorithm_grid_input.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "hash.h"
#include "smoothing.h"

#include <CGAL/convex_hull_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Plane_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Tetrahedron_3.h>

#include "Eigen/Dense"

#include <iostream>

void algorithm_compute_gauss::set_input(std::shared_ptr<const algorithm_line_input> line_input,
    std::shared_ptr<const algorithm_grid_input> grid_input, const smoothing::method_t smoothing_method,
    const smoothing::variant_t smoothing_variant, const float lambda, const int num_iterations,
    cuda::displacement::method_t displacement_method, cuda::displacement::b_spline_parameters_t bspline_parameters,
    const int num_subdivisions, const float remove_cells_scalar, const bool check_handedness,
    const bool check_convexity, const bool check_volume, const float volume_percentage)
{
    this->line_input = line_input;
    this->grid_input = grid_input;
    this->smoothing_method = smoothing_method;
    this->smoothing_variant = smoothing_variant;
    this->lambda = lambda;
    this->num_iterations = num_iterations;
    this->displacement_method = displacement_method;
    this->bspline_parameters = bspline_parameters;
    this->num_subdivisions = num_subdivisions;
    this->remove_cells_scalar = remove_cells_scalar;
    this->check_handedness = check_handedness;
    this->check_convexity = check_convexity;
    this->check_volume = check_volume;
    this->volume_percentage = volume_percentage;

    this->alg_smoothing = std::make_shared<algorithm_smoothing>();
    this->alg_displacement_creation = std::make_shared<algorithm_displacement_creation>();
    this->alg_displacement_precomputation = std::make_shared<algorithm_displacement_precomputation>();
    this->alg_displacement_computation = std::make_shared<algorithm_displacement_computation>();

    this->alg_smoothing->be_quiet();
    this->alg_displacement_creation->be_quiet();
    this->alg_displacement_precomputation->be_quiet();
    this->alg_displacement_computation->be_quiet();
}

std::uint32_t algorithm_compute_gauss::calculate_hash() const
{
    if (!this->grid_input->is_valid())
    {
        std::cerr << "Cannot compute Gauss parameter without a grid input" << std::endl;
        return -1;
    }

    if (!this->line_input->is_valid())
    {
        return -1;
    }

    return jenkins_hash(this->line_input->get_hash(), this->grid_input->get_hash(), this->smoothing_method,
        this->smoothing_variant, this->lambda, this->num_iterations, this->displacement_method,
        this->bspline_parameters.degree, this->bspline_parameters.iterations, this->num_subdivisions,
        this->remove_cells_scalar, this->check_handedness, this->check_convexity, this->check_volume, this->volume_percentage);
}

bool algorithm_compute_gauss::run_computation()
{
    if (!this->is_quiet()) std::cout << "Computing Gauss parameter" << std::endl;

    // Smooth line completely and create displacer
    cuda::displacement::parameter_t displacement_parameters;
    displacement_parameters.b_spline = this->bspline_parameters;

    this->alg_smoothing->run(this->line_input, this->smoothing_method, this->smoothing_variant, this->lambda, this->num_iterations);
    this->alg_displacement_creation->run(this->grid_input);
    this->alg_displacement_precomputation->run(this->alg_displacement_creation, this->alg_smoothing, this->line_input,
        this->displacement_method, displacement_parameters, this->bspline_parameters);

    // Set range of possible Gauss parameter values
    const auto max_displacement = std::max_element(this->alg_smoothing->get_results().displacements.begin(), this->alg_smoothing->get_results().displacements.end(),
        [](const std::array<float, 4>& lhs, const std::array<float, 4>& rhs)
        {
            const Eigen::Vector3f lhs_eigen(lhs[0], lhs[1], lhs[2]);
            const Eigen::Vector3f rhs_eigen(rhs[0], rhs[1], rhs[2]);

            return lhs_eigen.squaredNorm() < rhs_eigen.squaredNorm();
        }
    );

    float min_epsilon, max_epsilon;
    const auto min_influence = Eigen::Vector3f((*max_displacement)[0], (*max_displacement)[1], (*max_displacement)[2]).norm();

    if (min_influence == 0.0f)
    {
        min_epsilon = max_epsilon = 0.0f;
    }
    else
    {
        min_epsilon = 0.0f;
        max_epsilon = 2.0f / min_influence;
    }

    // Get grid information
    const auto dimension = this->grid_input->get_results().dimension;
    const auto spacing = this->grid_input->get_results().spacing;

    const auto is_2d = dimension[2] == 1;
    const auto threshold = this->remove_cells_scalar * this->grid_input->get_results().spacing.head(is_2d ? 2 : 3).norm();

    auto calc_index_point = [](const std::array<int, 3>& dimension, int x, int y, int z) -> int
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    };

    auto project_point_onto_line = [](const Eigen::Vector3d& point, const Eigen::Vector3d& line_point) -> float
    {
        return point.dot(line_point) / line_point.squaredNorm();
    };

    // Initial guess
    this->results.gauss_parameter = (min_epsilon + max_epsilon) / 2.0f;

    if (!this->is_quiet()) std::cout << "  initial guess: " << this->results.gauss_parameter;

    // Iteratively improve parameter
    float last_good_epsilon = 0.0f;
    float last_good_small_epsilon = 0.0f;

    for (int i = 0; i < this->num_subdivisions; ++i)
    {
        // Deform grid
        displacement_parameters.b_spline.gauss_parameter = this->results.gauss_parameter;

        this->alg_displacement_computation->run(this->alg_displacement_creation, this->alg_smoothing, this->displacement_method, displacement_parameters);

        const auto displaced_positions = this->alg_displacement_computation->get_results().displacements->get_results();

        // Compute handedness
        auto min_handedness = std::numeric_limits<float>::max();
        auto max_handedness = std::numeric_limits<float>::min();

        bool good = true;
        bool wellformed = true;
        bool convex = true;
        bool large = true;

        for (int z = 0; z < (is_2d ? 1 : (dimension[2] - 1)) && good; ++z)
        {
            for (int y = 0; y < dimension[1] - 1 && good; ++y)
            {
                for (int x = 0; x < dimension[0] - 1 && good; ++x)
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

                    // Check handedness, convexity, and cell volumes
                    if (!discard_cell)
                    {
                        float handedness;

                        if (!is_2d)
                        {
                            std::array<Eigen::Vector3d, 4> points{
                                Eigen::Vector3d(displaced_positions[point0][0], displaced_positions[point0][1], displaced_positions[point0][2]),
                                Eigen::Vector3d(displaced_positions[point1][0], displaced_positions[point1][1], displaced_positions[point1][2]),
                                Eigen::Vector3d(displaced_positions[point2][0], displaced_positions[point2][1], displaced_positions[point2][2]),
                                Eigen::Vector3d(displaced_positions[point4][0], displaced_positions[point4][1], displaced_positions[point4][2])
                            };

                            const auto vector_1 = points[1] - points[0];
                            const auto vector_2 = points[2] - points[0];
                            const auto vector_3 = points[3] - points[0];

                            // Calculate handedness
                            handedness = static_cast<float>(vector_1.cross(vector_2).dot(vector_3));

                            // Create cell polyhedron
                            if (this->check_convexity)
                            {
                                using kernel_t = CGAL::Exact_predicates_inexact_constructions_kernel;

                                using delaunay_t = CGAL::Delaunay_triangulation_3<kernel_t>;
                                using polyhedron_t = CGAL::Polyhedron_3<kernel_t>;
                                using tetrahedron_t = CGAL::Tetrahedron_3<kernel_t>;

                                std::array<CGAL::Point_3<kernel_t>, 8> points{
                                    CGAL::Point_3<kernel_t>(displaced_positions[point0][0], displaced_positions[point0][1], displaced_positions[point0][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point1][0], displaced_positions[point1][1], displaced_positions[point1][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point2][0], displaced_positions[point2][1], displaced_positions[point2][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point3][0], displaced_positions[point3][1], displaced_positions[point3][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point4][0], displaced_positions[point4][1], displaced_positions[point4][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point5][0], displaced_positions[point5][1], displaced_positions[point5][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point6][0], displaced_positions[point6][1], displaced_positions[point6][2]),
                                    CGAL::Point_3<kernel_t>(displaced_positions[point7][0], displaced_positions[point7][1], displaced_positions[point7][2])
                                };

                                std::array<tetrahedron_t, 6> cell{
                                    tetrahedron_t(points[0], points[2], points[3], points[7]),
                                    tetrahedron_t(points[0], points[1], points[3], points[7]),
                                    tetrahedron_t(points[0], points[1], points[5], points[7]),
                                    tetrahedron_t(points[0], points[4], points[5], points[7]),
                                    tetrahedron_t(points[0], points[2], points[6], points[7]),
                                    tetrahedron_t(points[0], points[4], points[6], points[7])
                                };

                                const auto volume =
                                    std::abs(cell[0].volume()) + std::abs(cell[1].volume()) + std::abs(cell[2].volume()) +
                                    std::abs(cell[3].volume()) + std::abs(cell[4].volume()) + std::abs(cell[5].volume());

                                // Check convexity
                                polyhedron_t hull;
                                CGAL::convex_hull_3(points.begin(), points.end(), hull);

                                delaunay_t delaunay;
                                delaunay.insert(hull.points_begin(), hull.points_end());

                                double convex_volume = 0.0;

                                for (auto it = delaunay.finite_cells_begin(); it != delaunay.finite_cells_end(); ++it)
                                {
                                    convex_volume += delaunay.tetrahedron(it).volume();
                                }

                                good &= convex &= std::abs(convex_volume - volume) < 0.01f * spacing.head(is_2d ? 2 : 3).norm();

                                // Check volume
                                if (this->check_volume)
                                {
                                    large &= volume > this->volume_percentage * spacing.head(is_2d ? 2 : 3).prod();
                                }
                            }
                        }
                        else
                        {
                            std::array<Eigen::Vector3d, 4> points{
                                Eigen::Vector3d(displaced_positions[point0][0], displaced_positions[point0][1], displaced_positions[point0][2]),
                                Eigen::Vector3d(displaced_positions[point1][0], displaced_positions[point1][1], displaced_positions[point1][2]),
                                Eigen::Vector3d(displaced_positions[point2][0], displaced_positions[point2][1], displaced_positions[point2][2]),
                                Eigen::Vector3d(displaced_positions[point3][0], displaced_positions[point3][1], displaced_positions[point3][2])
                            };

                            const auto vector_1 = points[1] - points[0];
                            const auto vector_2 = points[2] - points[0];
                            const auto vector_3 = points[3] - points[0];

                            // Calculate handedness
                            handedness = static_cast<float>(vector_1.cross(vector_2)[2]);

                            // Check convexity
                            if (this->check_convexity)
                            {
                                good &= convex &= std::signbit(vector_3.cross(vector_1)[2]) != std::signbit(vector_3.cross(vector_2)[2]);
                            }

                            // Calculate area
                            if (this->check_volume)
                            {
                                const auto t1 = project_point_onto_line(vector_1, vector_3);
                                const auto t2 = project_point_onto_line(vector_2, vector_3);

                                const auto area =
                                    0.5 * vector_3.norm() * (vector_1 - (t1 * vector_3)).norm() +
                                    0.5 * vector_3.norm() * (vector_2 - (t2 * vector_3)).norm();

                                large &= area > this->volume_percentage * spacing.head(is_2d ? 2 : 3).prod();
                            }
                        }

                        // Apply handedness criterion
                        if (this->check_handedness)
                        {
                            min_handedness = std::min(min_handedness, handedness);
                            max_handedness = std::max(max_handedness, handedness);

                            good &= wellformed &= std::signbit(min_handedness) == std::signbit(max_handedness);
                        }
                    }
                }
            }
        }

        // Set new influence
        if (!good || !large)
        {
            max_epsilon = this->results.gauss_parameter;
        }
        else
        {
            min_epsilon = this->results.gauss_parameter;
        }

        if (good && !large)
        {
            last_good_small_epsilon = this->results.gauss_parameter;
        }
        else if (good && large)
        {
            last_good_epsilon = this->results.gauss_parameter;
        }

        if (!this->is_quiet()) std::cout << " (" << (good && large ? "good" : (wellformed ? (convex ? (large ? "" : "small") : "concave") : "ill-formed")) << ")" << std::endl;

        if (i < this->num_subdivisions - 1)
        {
            this->results.gauss_parameter = (min_epsilon + max_epsilon) / 2.0f;

            if (!this->is_quiet()) std::cout << "  checked parameter: " << this->results.gauss_parameter;
        }
        else
        {
            if (good && large)
            {
                if (!this->is_quiet()) std::cout << "  found good parameter: " << this->results.gauss_parameter << std::endl;
            }
            else if (last_good_epsilon != 0.0f)
            {
                this->results.gauss_parameter = last_good_epsilon;

                if (!this->is_quiet()) std::cout << "  found good parameter: " << last_good_epsilon << std::endl;
            }
            else if (good && !large)
            {
                if (!this->is_quiet()) std::cout << "  found good parameter, but cells may be small: " << this->results.gauss_parameter << std::endl;
            }
            else if (last_good_small_epsilon != 0.0f)
            {
                this->results.gauss_parameter = last_good_small_epsilon;

                if (!this->is_quiet()) std::cout << "  found good parameter, but cells may be small: " << this->results.gauss_parameter << std::endl;
            }
            else
            {
                this->results.gauss_parameter = 0.0f;

                if (!this->is_quiet()) std::cout << "  unable to find good parameter, using: 0.0" << std::endl;
            }
        }
    }

    return true;
}

const algorithm_compute_gauss::results_t& algorithm_compute_gauss::get_results() const
{
    return this->results;
}
