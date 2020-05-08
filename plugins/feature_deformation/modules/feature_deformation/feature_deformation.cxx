#include "feature_deformation.h"

#include "b-spline.h"
#include "displacement.h"
#include "hash.h"
#include "smoothing.h"

#include "common/math.h"

#include "vtkCell.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkDataObjectTypes.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkIdTypeArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkImageData.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointData.h"
#include "vtkPointSet.h"
#include "vtkPolyData.h"
#include "vtkPolyhedron.h"
#include "vtkRectilinearGrid.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredGrid.h"
#include "vtkUnstructuredGrid.h"

#include "Eigen/Dense"

#define __cgal
#ifdef __cgal
#include <CGAL/convex_hull_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Plane_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Tetrahedron_3.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace
{
    template <typename T>
    void create_or_get_data_object(const int index, vtkAlgorithm* output_algorithm, vtkInformationVector* output_info)
    {
        auto output = T::SafeDownCast(output_info->GetInformationObject(index)->Get(vtkDataObject::DATA_OBJECT()));

        if (!output)
        {
            output = T::New();
            output_info->GetInformationObject(index)->Set(vtkDataObject::DATA_OBJECT(), output);
            output_algorithm->GetOutputPortInformation(index)->Set(vtkDataObject::DATA_EXTENT_TYPE(), output->GetExtentType());
        }
    }

    int calc_index_point(const std::array<int, 3>& dimension, int x, int y, int z)
    {
        return (z * dimension[1] + y) * dimension[0] + x;
    }

    int calc_index_cell(const std::array<int, 3>& dimension, int x, int y, int z)
    {
        x = std::min(std::max(0, x), dimension[0] - 2);
        y = std::min(std::max(0, y), dimension[1] - 2);
        z = std::min(std::max(0, z), dimension[2] - 2);

        return (z * (dimension[1] - 1) + y) * (dimension[0] - 1) + x;
    }

    float project_point_onto_line(const Eigen::Vector3d& point, const Eigen::Vector3d& line_point)
    {
        return point.dot(line_point) / line_point.squaredNorm();
    }
}

vtkStandardNewMacro(feature_deformation);

feature_deformation::feature_deformation() : frames(0)
{
    this->SetNumberOfInputPorts(3);
    this->SetNumberOfOutputPorts(4);
}

feature_deformation::~feature_deformation() {}

int feature_deformation::ProcessRequest(vtkInformation* request, vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Create an output object of the correct type.
    if (request->Has(vtkDemandDrivenPipeline::REQUEST_DATA_OBJECT()))
    {
        return this->RequestDataObject(request, input_vector, output_vector);
    }

    // Generate the data
    if (request->Has(vtkDemandDrivenPipeline::REQUEST_INFORMATION()))
    {
        return this->RequestInformation(request, input_vector, output_vector);
    }

    if (request->Has(vtkDemandDrivenPipeline::REQUEST_DATA()))
    {
        return this->RequestData(request, input_vector, output_vector);
    }

    if (request->Has(vtkStreamingDemandDrivenPipeline::REQUEST_UPDATE_EXTENT()))
    {
        return this->RequestUpdateExtent(request, input_vector, output_vector);
    }

    return this->Superclass::ProcessRequest(request, input_vector, output_vector);
}

int feature_deformation::RequestDataObject(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    create_or_get_data_object<vtkPolyData>(0, this, output_vector);
    create_or_get_data_object<vtkMultiBlockDataSet>(1, this, output_vector);
    create_or_get_data_object<vtkMultiBlockDataSet>(2, this, output_vector);
    create_or_get_data_object<vtkImageData>(3, this, output_vector);

    return 1;
}

int feature_deformation::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    // If iterative smoothing is selected, create time step values based on animation parameters
    std::array<double, 2> time_range;

    if (this->Method == 1)
    {
        time_range = { 0.0, 1.0 };
    }
    else
    {
        time_range = { 0.0, 0.0 };
    }

    output_vector->GetInformationObject(0)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(1)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(2)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);
    output_vector->GetInformationObject(3)->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), time_range.data(), 2);

    return 1;
}

int feature_deformation::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
        info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
        return 1;
    }
    else if (port == 2)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPointSet");
        info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
        info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
        return 1;
    }

    return 0;
}

int feature_deformation::FillOutputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
        return 1;
    }
    else if (port == 2)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
        return 1;
    }
    else if (port == 3)
    {
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
        return 1;
    }

    return 1;
}

void feature_deformation::RemoveAllGeometryInputs()
{
    this->SetInputConnection(2, nullptr);
}

int feature_deformation::RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    return 1;
}

int feature_deformation::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Output info
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Starting deformation, frame: " << this->frames++ << std::endl << std::endl;

    // Get time
    const auto time = output_vector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

    // Get parameters
    process_parameters(time);

    if (this->parameters.smoothing_method == smoothing::method_t::smoothing)
    {
        std::cout << "Time: " << time << std::endl << std::endl;
    }

    // Get input
    if (!this->alg_line_input.run(vtkPolyData::SafeDownCast(input_vector[1]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())), this->parameters.selected_line_id))
    {
        return 0;
    }

    if (input_vector[0] != nullptr && input_vector[0]->GetInformationObject(0) != nullptr &&
        (this->parameters.output_deformed_grid || this->parameters.compute_gauss))
    {
        this->alg_grid_input.run(vtkImageData::SafeDownCast(input_vector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())));
    }

    if (input_vector[2] != nullptr)
    {
        std::vector<vtkPointSet*> geometry_sets;

        for (vtkIdType i = 0; i < input_vector[2]->GetNumberOfInformationObjects(); ++i)
        {
            geometry_sets.push_back(vtkPointSet::SafeDownCast(input_vector[2]->GetInformationObject(i)->Get(vtkDataObject::DATA_OBJECT())));
        }

        this->alg_geometry_input.run(geometry_sets);
    }

    std::cout << std::endl;

    // Compute Gauss parameter epsilon and tearing
    const bool is_2d = this->input_grid.dimension[2] == 1;

    if (this->input_grid.valid && (this->parameter_lines.modified || this->parameter_smoothing.modified ||
        this->parameter_displacement.modified || this->parameter_precompute.modified || this->input_grid.modified || this->input_lines.modified))
    {
        if (this->parameter_precompute.compute_gauss || this->parameter_precompute.compute_tearing)
        {
            // Create smoother
            smoothing smoother(this->input_lines.selected_line, this->parameter_lines.method, this->parameter_smoothing.variant,
                this->parameter_smoothing.lambda, this->parameter_smoothing.max_num_iterations);

            // Straighten the selected line
            while (smoother.has_step())
            {
                // Perform smoothing step
                smoother.next_step();
            }

            const auto displacements = get_displacements(smoother.get_displacement());

            // Create displacer
            auto displacement = std::make_shared<cuda::displacement>(
                std::array<double, 3>{ this->input_grid.origin[0], this->input_grid.origin[1], this->input_grid.origin[2] },
                std::array<double, 3>{ this->input_grid.spacing[0], this->input_grid.spacing[1], this->input_grid.spacing[2] },
                this->input_grid.dimension);

            // Set threshold for tearing cells
            const auto threshold = static_cast<float>(this->parameter_output_grid.remove_cells_scalar) * this->input_grid.spacing.head(is_2d ? 2 : 3).norm();

            // Find good choice for Gauss parameter
            if (this->parameter_precompute.compute_gauss)
            {
                std::cout << "Computing Gauss parameter" << std::endl;

                // Set range of possible Gauss parameter values
                const auto max_displacement = std::max_element(displacements.second.begin(), displacements.second.end(),
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

                // Initial guess
                auto epsilon = (min_epsilon + max_epsilon) / 2.0f;

                this->GaussParameter = epsilon;
                cache_parameter_displacement();

                std::cout << "  initial guess: " << this->GaussParameter;

                // Iteratively improve parameter
                float last_good_epsilon = 0.0f;
                float last_good_small_epsilon = 0.0f;

                for (int i = 0; i < this->parameter_precompute.num_subdivisions; ++i)
                {
                    // Deform grid
                    if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
                        this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
                    {
                        displacement->precompute(this->parameter_displacement.parameters, displacements.first);
                    }

                    displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters, displacements.first, displacements.second);

                    const auto displaced_positions = displacement->get_results();

                    // Compute handedness
                    auto min_handedness = std::numeric_limits<float>::max();
                    auto max_handedness = std::numeric_limits<float>::min();

                    bool good = true;
                    bool wellformed = true;
                    bool convex = true;
                    bool large = true;

                    for (int z = 0; z < (is_2d ? 1 : (this->input_grid.dimension[2] - 1)) && good; ++z)
                    {
                        for (int y = 0; y < this->input_grid.dimension[1] - 1 && good; ++y)
                        {
                            for (int x = 0; x < this->input_grid.dimension[0] - 1 && good; ++x)
                            {
                                // Create point IDs
                                const auto point0 = calc_index_point(this->input_grid.dimension, x + 0, y + 0, z + 0);
                                const auto point1 = calc_index_point(this->input_grid.dimension, x + 1, y + 0, z + 0);
                                const auto point2 = calc_index_point(this->input_grid.dimension, x + 0, y + 1, z + 0);
                                const auto point3 = calc_index_point(this->input_grid.dimension, x + 1, y + 1, z + 0);
                                const auto point4 = calc_index_point(this->input_grid.dimension, x + 0, y + 0, z + 1);
                                const auto point5 = calc_index_point(this->input_grid.dimension, x + 1, y + 0, z + 1);
                                const auto point6 = calc_index_point(this->input_grid.dimension, x + 0, y + 1, z + 1);
                                const auto point7 = calc_index_point(this->input_grid.dimension, x + 1, y + 1, z + 1);

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

#ifdef __cgal
                                        // Create cell polyhedron
                                        if (this->parameter_precompute.check_convexity)
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

                                            good &= convex &= std::abs(convex_volume - volume) < 0.01f * this->input_grid.spacing.head(is_2d ? 2 : 3).norm();

                                            // Check volume
                                            if (this->parameter_precompute.check_volume)
                                            {
                                                large &= volume > this->parameter_precompute.volume_percentage * this->input_grid.spacing.head(is_2d ? 2 : 3).prod();
                                            }
                                        }
#endif
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
                                        if (this->parameter_precompute.check_convexity)
                                        {
                                            good &= convex &= std::signbit(vector_3.cross(vector_1)[2]) != std::signbit(vector_3.cross(vector_2)[2]);
                                        }

                                        // Calculate area
                                        if (this->parameter_precompute.check_volume)
                                        {
                                            const auto t1 = project_point_onto_line(vector_1, vector_3);
                                            const auto t2 = project_point_onto_line(vector_2, vector_3);

                                            const auto area =
                                                0.5 * vector_3.norm() * (vector_1 - (t1 * vector_3)).norm() +
                                                0.5 * vector_3.norm() * (vector_2 - (t2 * vector_3)).norm();

                                            large &= area > this->parameter_precompute.volume_percentage * this->input_grid.spacing.head(is_2d ? 2 : 3).prod();
                                        }
                                    }

                                    // Apply handedness criterion
                                    if (this->parameter_precompute.check_handedness)
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
                        max_epsilon = epsilon;
                    }
                    else
                    {
                        min_epsilon = epsilon;
                    }

                    if (good && !large)
                    {
                        last_good_small_epsilon = epsilon;
                    }
                    else if (good && large)
                    {
                        last_good_epsilon = epsilon;
                    }

                    std::cout << " (" << (good && large ? "good" : (wellformed ? (convex ? (large ? "" : "small") : "concave") : "ill-formed")) << ")" << std::endl;

                    if (i < this->parameter_precompute.num_subdivisions - 1)
                    {
                        epsilon = (min_epsilon + max_epsilon) / 2.0f;

                        this->GaussParameter = epsilon;
                        cache_parameter_displacement();

                        std::cout << "  checked parameter: " << this->GaussParameter;
                    }
                    else
                    {
                        if (good && large)
                        {
                            std::cout << "  found good parameter: " << epsilon << std::endl;
                        }
                        else if (last_good_epsilon != 0.0f)
                        {
                            this->GaussParameter = last_good_epsilon;
                            cache_parameter_displacement();

                            std::cout << "  found good parameter: " << last_good_epsilon << std::endl;
                        }
                        else if (good && !large)
                        {
                            std::cout << "  found good parameter, but cells may be small: " << epsilon << std::endl;
                        }
                        else if (last_good_small_epsilon != 0.0f)
                        {
                            this->GaussParameter = last_good_small_epsilon;
                            cache_parameter_displacement();

                            std::cout << "  found good parameter, but cells may be small: " << epsilon << std::endl;
                        }
                        else
                        {
                            this->GaussParameter = 0.0f;
                            cache_parameter_displacement();

                            std::cout << "  unable to find good parameter, using: 0.0" << std::endl;
                        }
                    }
                }
            }

            // Deform grid and mark cells which are going to tear
            if (this->parameter_precompute.compute_tearing)
            {
                // Deform grid
                if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
                    this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
                {
                    displacement->precompute(this->parameter_displacement.parameters, displacements.first);
                }

                displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters, displacements.first, displacements.second);

                const auto displaced_positions = displacement->get_results();

                // Create output array
                std::cout << "Pre-computing tearing cells" << std::endl;

                precompute_tearing.removed_cells = vtkSmartPointer<vtkIdTypeArray>::New();
                precompute_tearing.removed_cells->SetNumberOfComponents(2);
                precompute_tearing.removed_cells->SetNumberOfTuples(this->input_grid.grid->GetNumberOfPoints());
                precompute_tearing.removed_cells->SetName("Tearing cells");
                precompute_tearing.removed_cells->FillTypedComponent(0, 0);
                precompute_tearing.removed_cells->FillTypedComponent(1, -1);

                // Set value to 1 for all points that are part of a cell that tears
                #pragma omp parallel for
                for (int z = 0; z < (is_2d ? 1 : (this->input_grid.dimension[2] - 1)); ++z)
                {
                    for (int y = 0; y < this->input_grid.dimension[1] - 1; ++y)
                    {
                        for (int x = 0; x < this->input_grid.dimension[0] - 1; ++x)
                        {
                            // Create point IDs
                            const auto point0 = calc_index_point(this->input_grid.dimension, x + 0, y + 0, z + 0);
                            const auto point1 = calc_index_point(this->input_grid.dimension, x + 1, y + 0, z + 0);
                            const auto point2 = calc_index_point(this->input_grid.dimension, x + 0, y + 1, z + 0);
                            const auto point3 = calc_index_point(this->input_grid.dimension, x + 1, y + 1, z + 0);
                            const auto point4 = calc_index_point(this->input_grid.dimension, x + 0, y + 0, z + 1);
                            const auto point5 = calc_index_point(this->input_grid.dimension, x + 1, y + 0, z + 1);
                            const auto point6 = calc_index_point(this->input_grid.dimension, x + 0, y + 1, z + 1);
                            const auto point7 = calc_index_point(this->input_grid.dimension, x + 1, y + 1, z + 1);

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
                                    precompute_tearing.removed_cells->SetTypedComponent(point_ids[point_index], 0, 1);
                                }
                            }
                        }
                    }
                }

                // Use region growing to detect and label connected tearing regions
                int next_region = 0;

                auto hasher = [this](const std::tuple<int, int, int>& key) -> std::size_t {
                    return std::hash<int>()(calc_index_point(this->input_grid.dimension, std::get<0>(key), std::get<1>(key), std::get<2>(key)));
                };

                std::unordered_set<std::tuple<int, int, int>, decltype(hasher)> todo(29, hasher);

                for (int z = 0; z < this->input_grid.dimension[2]; ++z)
                {
                    for (int y = 0; y < this->input_grid.dimension[1]; ++y)
                    {
                        for (int x = 0; x < this->input_grid.dimension[0]; ++x)
                        {
                            const auto index = calc_index_point(this->input_grid.dimension, x, y, z);

                            std::array<vtkIdType, 2> value;
                            precompute_tearing.removed_cells->GetTypedTuple(index, value.data());

                            if (value[0] == 1 && value[1] == -1)
                            {
                                const auto current_region = next_region++;

                                todo.insert({x, y, z});

                                // Take first item, process it, and put unprocessed neighbors on the ToDo list
                                while (!todo.empty())
                                {
                                    const auto current_coords = *todo.cbegin();
                                    todo.erase(current_coords);

                                    const auto current_x = std::get<0>(current_coords);
                                    const auto current_y = std::get<1>(current_coords);
                                    const auto current_z = std::get<2>(current_coords);
                                    const auto current_index = calc_index_point(this->input_grid.dimension, current_x, current_y, current_z);

                                    precompute_tearing.removed_cells->SetTypedComponent(current_index, 1, current_region);

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
                                                    const auto neighbor_index = calc_index_point(this->input_grid.dimension, neighbor_x, neighbor_y, neighbor_z);

                                                    if (neighbor_x >= 0 && neighbor_x < this->input_grid.dimension[0] &&
                                                        neighbor_y >= 0 && neighbor_y < this->input_grid.dimension[1] &&
                                                        neighbor_z >= 0 && neighbor_z < this->input_grid.dimension[2])
                                                    {
                                                        std::array<vtkIdType, 2> value;
                                                        precompute_tearing.removed_cells->GetTypedTuple(neighbor_index, value.data());

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

                precompute_tearing.valid = true;
            }

            std::cout << std::endl;
        }
    }

    // Smooth line
    if (!this->alg_smoothing.run(this->alg_line_input, this->parameters.smoothing_method, this->parameters.variant, this->parameters.lambda, this->parameters.num_iterations))
    {
        return 0;
    }

    std::cout << std::endl;

    // Displace grid points and store results in cache
    if (this->input_grid.valid && this->parameter_output_grid.output_deformed_grid)
    {
        std::cout << "Displacing grid points..." << std::endl;

        // Upload new points, if input grid was modified
        if (this->input_grid.modified)
        {
            std::cout << "  uploading points to the GPU" << std::endl;

            this->results_grid_displacement.displacement = std::make_shared<cuda::displacement>(
                std::array<double, 3>{ this->input_grid.origin[0], this->input_grid.origin[1], this->input_grid.origin[2] },
                std::array<double, 3>{ this->input_grid.spacing[0], this->input_grid.spacing[1], this->input_grid.spacing[2] },
                this->input_grid.dimension);
        }

        // Pre-compute B-Spline mapping, if relevant parameters changed, or the input grid or lines were modified
        if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
            this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
        {
            const uint32_t precomputation_hash = hash(this->parameter_lines.selected_line_id, this->parameter_displacement.bspline_parameters.degree,
                this->parameter_displacement.bspline_parameters.iterations, this->input_grid.hash, this->input_lines.hash, this->results_grid_displacement.displacement.get());

            if (precomputation_hash != this->results_grid_displacement.hash)
            {
                std::cout << "  precomputing B-Spline mapping on the GPU" << std::endl;

                this->results_grid_displacement.displacement->precompute(this->parameter_displacement.parameters, this->results_smoothing.positions);

                this->results_grid_displacement.hash = precomputation_hash;
            }
        }

        // Displace grid points
        if (this->parameter_displacement.modified || this->input_grid.modified || this->results_smoothing.modified)
        {
            std::cout << "  calculating new positions on the GPU" << std::endl;

            this->results_grid_displacement.displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters,
                this->results_smoothing.positions, this->results_smoothing.displacements);
        }

        this->results_grid_displacement.valid = true;
        this->results_grid_displacement.modified = true;
    }

    // Displace line points and store results in cache
    {
        std::cout << "Displacing line points..." << std::endl;

        // Upload new points, if input lines were modified
        if (this->input_lines.modified)
        {
            std::cout << "  uploading points to the GPU" << std::endl;

            this->results_line_displacement.displacement = std::make_shared<cuda::displacement>(this->input_lines.lines);
        }

        // Pre-compute B-Spline mapping, if relevant parameters changed, or the lines were modified
        if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
            this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
        {
            const uint32_t precomputation_hash = hash(this->parameter_lines.selected_line_id, this->parameter_displacement.bspline_parameters.degree,
                this->parameter_displacement.bspline_parameters.iterations, this->input_lines.hash, this->results_line_displacement.displacement.get());

            if (precomputation_hash != this->results_line_displacement.hash)
            {
                std::cout << "  precomputing B-Spline mapping on the GPU" << std::endl;

                this->results_line_displacement.displacement->precompute(this->parameter_displacement.parameters, this->results_smoothing.positions);

                this->results_line_displacement.hash = precomputation_hash;
            }
        }

        // Displace line points
        if (this->parameter_displacement.modified || this->results_smoothing.modified)
        {
            std::cout << "  calculating new positions on the GPU" << std::endl;

            this->results_line_displacement.displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters,
                this->results_smoothing.positions, this->results_smoothing.displacements);
        }

        this->results_line_displacement.valid = true;
        this->results_line_displacement.modified = true;
    }

    // Displace geometry points and store results in cache
    if (this->input_geometry.valid)
    {
        std::cout << "Displacing geometry points..." << std::endl;

        // Upload new points, if input geometry was modified
        if (this->input_geometry.modified)
        {
            std::cout << "  uploading points to the GPU" << std::endl;

            this->results_geometry_displacement.displacement = std::make_shared<cuda::displacement>(this->input_geometry.geometry);
        }

        // Pre-compute B-Spline mapping, if relevant parameters changed, or the input geometry or lines were modified
        if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
            this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints))
        {
            const uint32_t precomputation_hash = hash(this->parameter_lines.selected_line_id, this->parameter_displacement.bspline_parameters.degree,
                this->parameter_displacement.bspline_parameters.iterations, this->input_lines.hash, this->input_geometry.hash, this->results_geometry_displacement.displacement.get());

            if (precomputation_hash != this->results_geometry_displacement.hash)
            {
                std::cout << "  precomputing B-Spline mapping on the GPU" << std::endl;

                this->results_geometry_displacement.displacement->precompute(this->parameter_displacement.parameters, this->results_smoothing.positions);

                this->results_geometry_displacement.hash = precomputation_hash;
            }
        }

        // Displace grid points
        if (this->parameter_displacement.modified || this->results_smoothing.modified || this->input_geometry.modified)
        {
            std::cout << "  calculating new positions on the GPU" << std::endl;

            this->results_geometry_displacement.displacement->displace(this->parameter_displacement.method, this->parameter_displacement.parameters,
                this->results_smoothing.positions, this->results_smoothing.displacements);
        }

        this->results_geometry_displacement.valid = true;
        this->results_geometry_displacement.modified = true;
    }

    std::cout << std::endl;

    // Output grid
    if (this->results_grid_displacement.valid && (this->results_grid_displacement.modified || this->parameter_output_grid.modified))
    {
        if (this->parameter_output_grid.output_deformed_grid)
        {
            std::cout << "Creating deformed grid output" << std::endl;

            // Create structured or unstructured grid
            auto out_deformed_grid_info = output_vector->GetInformationObject(2);
            vtkSmartPointer<vtkPointSet> output_deformed_grid = nullptr;

            if (this->parameter_output_grid.remove_cells)
            {
                output_deformed_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
            }
            else
            {
                output_deformed_grid = vtkSmartPointer<vtkStructuredGrid>::New();
                vtkStructuredGrid::SafeDownCast(output_deformed_grid)->SetExtent(this->input_grid.extent.data());
            }

            vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->SetBlock(0u, output_deformed_grid);
            vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->GetMetaData(0u)->Set(vtkCompositeDataSet::NAME(), "Grid");

            // Deform grid
            create_undeformed_grid(output_deformed_grid, this->input_grid.extent, this->input_grid.dimension, this->input_grid.origin, this->input_grid.spacing);
            set_output_deformed_grid(output_deformed_grid, *this->results_grid_displacement.displacement);

            if (this->parameter_output_grid.remove_cells)
            {
                // Create second grid for the "removed" cells
                auto output_deformed_grid_removed = vtkSmartPointer<vtkUnstructuredGrid>::New();
                output_deformed_grid_removed->ShallowCopy(vtkUnstructuredGrid::SafeDownCast(output_deformed_grid));

                vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->SetBlock(1u, output_deformed_grid_removed);
                vtkMultiBlockDataSet::SafeDownCast(out_deformed_grid_info->Get(vtkDataObject::DATA_OBJECT()))->GetMetaData(1u)->Set(vtkCompositeDataSet::NAME(), "Removed Cells");

                // Create cells
                create_cells(vtkUnstructuredGrid::SafeDownCast(output_deformed_grid), output_deformed_grid_removed, this->input_grid.dimension, this->input_grid.spacing);

                // Add tear array if possible
                if (this->precompute_tearing.valid && this->precompute_tearing.removed_cells->GetNumberOfTuples() == output_deformed_grid->GetNumberOfPoints())
                {
                    output_deformed_grid->GetPointData()->AddArray(this->precompute_tearing.removed_cells);
                }
            }

            // Create displacement field and use it to "deform" the velocities
            create_displacement_field(output_deformed_grid);

            if (this->input_grid.input_data.valid && this->parameter_output_grid.output_vector_field)
            {
                std::cout << "Calculating deformed velocities" << std::endl;

                deform_velocities(output_deformed_grid, this->input_grid.input_data.data, this->input_grid.dimension, this->input_grid.spacing);

                // Resample the deformed grid on the original one
                if (this->parameter_output_grid.output_resampled_grid)
                {
                    std::cout << "Creating resampled grid output" << std::endl;

                    auto out_resampled_grid_info = output_vector->GetInformationObject(3);
                    auto output_resampled_grid = vtkImageData::SafeDownCast(out_resampled_grid_info->Get(vtkDataObject::DATA_OBJECT()));

                    output_resampled_grid->DeepCopy(this->input_grid.grid);

                    resample_grid(output_deformed_grid, output_resampled_grid, this->input_grid.input_data.data->GetName(),
                        this->input_grid.dimension, this->input_grid.origin, this->input_grid.spacing);

                    out_resampled_grid_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
                }
            }

            out_deformed_grid_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
            this->Modified();
        }
    }
    else if (this->parameter_output_grid.output_deformed_grid && !this->results_grid_displacement.modified)
    {
        std::cout << "Loading deformed grid output from cache" << std::endl;
    }

    // Output lines
    if (this->results_line_displacement.valid && this->results_line_displacement.modified)
    {
        auto out_deformed_lines_info = output_vector->GetInformationObject(0);
        auto output_deformed_lines = vtkPolyData::SafeDownCast(out_deformed_lines_info->Get(vtkDataObject::DATA_OBJECT()));

        set_output_deformed_lines(vtkPolyData::SafeDownCast(input_vector[1]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT())), 
            output_deformed_lines, *this->results_line_displacement.displacement, this->input_lines.modified, this->output_lines);

        out_deformed_lines_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
        this->Modified();
    }
    else if (!this->results_line_displacement.modified)
    {
        std::cout << "Loading deformed line output from cache" << std::endl;
    }

    // Output geometry
    if (this->results_geometry_displacement.valid && this->results_geometry_displacement.modified)
    {
        auto out_deformed_geometry_info = output_vector->GetInformationObject(1);
        auto output_deformed_geometry = vtkMultiBlockDataSet::SafeDownCast(out_deformed_geometry_info->Get(vtkDataObject::DATA_OBJECT()));

        std::vector<vtkPointSet*> input_datasets(input_vector[2]->GetNumberOfInformationObjects());

        for (int input_index = 0; input_index < input_vector[2]->GetNumberOfInformationObjects(); ++input_index)
        {
            input_datasets[input_index] = vtkPointSet::SafeDownCast(input_vector[2]->GetInformationObject(input_index)->Get(vtkDataObject::DATA_OBJECT()));
        }

        set_output_deformed_geometry(input_datasets, output_deformed_geometry,
            *this->results_geometry_displacement.displacement, this->input_geometry.modified, this->output_geometry);

        for (unsigned int block_index = 0; block_index < output_deformed_geometry->GetNumberOfBlocks(); ++block_index)
        {
            // Create displacement field
            auto displacement_map = vtkSmartPointer<vtkDoubleArray>::New();
            displacement_map->SetNumberOfComponents(3);
            displacement_map->SetNumberOfTuples(vtkPointSet::SafeDownCast(output_deformed_geometry->GetBlock(block_index))->GetPoints()->GetNumberOfPoints());
            displacement_map->SetName("Displacement Map");

            #pragma omp parallel for
            for (vtkIdType p = 0; p < vtkPointSet::SafeDownCast(output_deformed_geometry->GetBlock(block_index))->GetPoints()->GetNumberOfPoints(); ++p)
            {
                Eigen::Vector3d displaced_point;
                vtkPointSet::SafeDownCast(output_deformed_geometry->GetBlock(block_index))->GetPoints()->GetPoint(p, displaced_point.data());

                displacement_map->SetTuple(p, displaced_point.data());
            }

            vtkPointSet::SafeDownCast(output_deformed_geometry->GetBlock(block_index))->GetPointData()->AddArray(displacement_map);
        }

        for (unsigned int block_index = 0; block_index < output_deformed_geometry->GetNumberOfBlocks(); ++block_index)
        {
            output_deformed_geometry->GetBlock(block_index)->GetInformation()->Set(vtkDataObject::DATA_TIME_STEP(), time);
        }

        out_deformed_geometry_info->Set(vtkDataObject::DATA_TIME_STEP(), time);
        this->Modified();
    }
    else if (!this->results_geometry_displacement.modified)
    {
        std::cout << "Loading deformed geometry output from cache" << std::endl;
    }

    // Output info
    std::cout << std::endl << "Finished deformation" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // Set modification to false for the next run
    this->results_grid_displacement.modified = false;
    this->results_line_displacement.modified = false;
    this->results_geometry_displacement.modified = false;

    return 1;
}


/// Parameters

void feature_deformation::process_parameters(double time)
{
    // Line parameters
    this->parameters.selected_line_id = this->LineID;

    // Smoothing parameters
    this->parameters.smoothing_method = static_cast<smoothing::method_t>(this->Method);
    this->parameters.variant = static_cast<smoothing::variant_t>(this->Variant);
    this->parameters.lambda = static_cast<float>(this->Lambda);
    this->parameters.num_iterations = this->MaxNumIterations;

    if (this->parameters.smoothing_method == smoothing::method_t::smoothing)
    {
        if (this->Inverse)
        {
            time = 1.0 - time;
        }

        if (this->Interpolator == 0)
        {
            // Linear
            this->parameters.num_iterations *= std::min(time, 1.0);
        }
        else
        {
            // Exponential
            if (time == 0.0)
            {
                this->parameters.num_iterations = 0;
            }
            else if (time < 1.0)
            {
                this->parameters.num_iterations = std::pow(2.0, time * std::log2(this->parameters.num_iterations + 1)) - 1;
            }
        }
    }

    // Displacement parameters
    this->parameters.displacement_method = static_cast<cuda::displacement::method_t>(this->Weight);

    switch (this->parameters.displacement_method)
    {
    case cuda::displacement::method_t::greedy:
    case cuda::displacement::method_t::voronoi:
        this->parameters.displacement_parameters.inverse_distance_weighting.exponent = static_cast<float>(this->EpsilonScalar);
        this->parameters.displacement_parameters.inverse_distance_weighting.neighborhood = this->VoronoiDistance;

        break;
    case cuda::displacement::method_t::greedy_joints:
        this->parameters.displacement_parameters.inverse_distance_weighting.exponent = static_cast<float>(this->EpsilonScalar);

        break;
    case cuda::displacement::method_t::projection:
        this->parameters.displacement_parameters.projection.gauss_parameter = static_cast<float>(this->GaussParameter);

        break;
    case cuda::displacement::method_t::b_spline:
    case cuda::displacement::method_t::b_spline_joints:
        this->parameters.displacement_parameters.b_spline.degree = this->SplineDegree;
        this->parameters.displacement_parameters.b_spline.gauss_parameter = static_cast<float>(this->GaussParameter);
        this->parameters.displacement_parameters.b_spline.iterations = this->Subdivisions;

        break;
    }

    this->parameters.idw_parameters.exponent = static_cast<float>(this->EpsilonScalar);
    this->parameters.idw_parameters.neighborhood = this->VoronoiDistance;
    this->parameters.projection_parameters.gauss_parameter = static_cast<float>(this->GaussParameter);
    this->parameters.bspline_parameters.degree = this->SplineDegree;
    this->parameters.bspline_parameters.gauss_parameter = static_cast<float>(this->GaussParameter);
    this->parameters.bspline_parameters.iterations = this->Subdivisions;

    // Pre-computation parameters
    this->parameters.compute_gauss = (this->ComputeGauss != 0);
    this->parameters.check_handedness = (this->CheckHandedness != 0);
    this->parameters.check_convexity = (this->CheckConvexity != 0);
    this->parameters.check_volume = (this->CheckVolume != 0);
    this->parameters.volume_percentage = this->VolumePercentage;
    this->parameters.num_subdivisions = this->GaussSubdivisions;
    this->parameters.compute_tearing = (this->ComputeTearing != 0);

    // Output parameters
    this->parameters.output_deformed_grid = (this->OutputDeformedGrid != 0);
    this->parameters.output_vector_field = (this->OutputVectorField != 0);
    this->parameters.output_resampled_grid = (this->OutputResampledGrid != 0);
    this->parameters.remove_cells = (this->RemoveCells != 0);
    this->parameters.remove_cells_scalar = static_cast<float>(this->RemoveCellsScalar);
}


/// Output

void feature_deformation::create_undeformed_grid(vtkPointSet* output_deformed_grid, const std::array<int, 6>& extent,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const
{
    // Create point nodes
    auto coords = vtkSmartPointer<vtkPoints>::New();
    coords->SetNumberOfPoints(dimension[0] * dimension[1] * dimension[2]);

    auto tex_coords = vtkSmartPointer<vtkFloatArray>::New();
    tex_coords->SetNumberOfComponents(3);
    tex_coords->SetNumberOfTuples(dimension[0] * dimension[1] * dimension[2]);
    tex_coords->SetName("Original Coordinates");

    vtkIdType index = 0;

    #pragma omp parallel for
    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            for (int x = 0; x < dimension[0]; ++x, ++index)
            {
                const Eigen::Vector3f point = origin + Eigen::Vector3f(x, y, z).cwiseProduct(spacing);

                coords->SetPoint(index, point.data());
                tex_coords->SetTuple(index, point.data());
            }
        }
    }

    output_deformed_grid->SetPoints(coords);
    output_deformed_grid->GetPointData()->AddArray(tex_coords);
}

void feature_deformation::create_cells(vtkUnstructuredGrid* output_deformed_grid, vtkUnstructuredGrid* output_deformed_grid_removed,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& spacing) const
{
    const auto is_2d = dimension[2] == 1;
    const auto threshold = static_cast<float>(this->parameter_output_grid.remove_cells_scalar) * spacing.head(is_2d ? 2 : 3).norm();

    // Create cells
    output_deformed_grid->Allocate((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));
    output_deformed_grid_removed->Allocate((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));

    auto handedness = vtkSmartPointer<vtkFloatArray>::New();
    handedness->SetNumberOfComponents(1);
    handedness->Allocate((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));
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

                if (this->parameter_output_grid.remove_cells)
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
    output_deformed_grid->BuildLinks();

    output_deformed_grid_removed->BuildLinks();
}

void feature_deformation::set_output_deformed_grid(vtkPointSet* output_deformed_grid, const cuda::displacement& grid_displacement) const
{
    // Set displaced points
    const auto& displaced_grid = grid_displacement.get_results();

    #pragma omp parallel for
    for (vtkIdType i = 0; i < output_deformed_grid->GetNumberOfPoints(); ++i)
    {
        output_deformed_grid->GetPoints()->SetPoint(i, displaced_grid[i].data());
    }

    // Create displacement ID array
    const auto& displacement_ids = grid_displacement.get_displacement_info();

    auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
    displacement_id_array->SetNumberOfComponents(4);
    displacement_id_array->SetNumberOfTuples(displacement_ids.size());
    displacement_id_array->SetName("Displacement Information");

    std::memcpy(displacement_id_array->GetPointer(0), displacement_ids.data(), displacement_ids.size() * sizeof(float4));

    output_deformed_grid->GetPointData()->AddArray(displacement_id_array);
}

void feature_deformation::set_output_deformed_lines(vtkPolyData* input_lines, vtkPolyData* output_deformed_lines, const cuda::displacement& line_displacement,
    const bool modified, cache_output_lines_t& output_lines) const
{
    // Create output geometry
    if (modified || !output_lines.valid)
    {
        std::cout << "Creating deformed lines output" << std::endl;

        output_lines.data = vtkSmartPointer<vtkPolyData>::New();
        output_lines.data->DeepCopy(input_lines);
        output_lines.valid = true;

        auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
        displacement_id_array->SetNumberOfComponents(4);
        displacement_id_array->SetNumberOfTuples(output_lines.data->GetNumberOfPoints());
        displacement_id_array->SetName("Displacement Information");
        displacement_id_array->FillValue(0.0f);

        auto displacement_distance_array = vtkSmartPointer<vtkFloatArray>::New();
        displacement_distance_array->SetNumberOfComponents(1);
        displacement_distance_array->SetNumberOfTuples(output_lines.data->GetNumberOfPoints());
        displacement_distance_array->SetName("B-Spline Distance");
        displacement_distance_array->FillValue(0.0f);

        output_lines.data->GetPointData()->AddArray(displacement_id_array);
        output_lines.data->GetPointData()->AddArray(displacement_distance_array);
    }
    else
    {
        std::cout << "Updating deformed lines output" << std::endl;
    }

    // Set displaced points
    const auto& displaced_lines = line_displacement.get_results();

    #pragma omp parallel for
    for (vtkIdType i = 0; i < output_lines.data->GetNumberOfPoints(); ++i)
    {
        output_lines.data->GetPoints()->SetPoint(i, displaced_lines[i].data());
    }

    // Set displacement ID array
    const auto& displacement_ids = line_displacement.get_displacement_info();

    std::memcpy(vtkFloatArray::SafeDownCast(output_lines.data->GetPointData()->GetArray("Displacement Information"))->GetPointer(0),
        displacement_ids.data(), displacement_ids.size() * sizeof(float4));

    // In case of the B-Spline, store distance on B-Spline for neighboring points
    if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
        this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints) &&
        this->OutputBSplineDistance)
    {
        auto displacement_distance_array = vtkFloatArray::SafeDownCast(output_lines.data->GetPointData()->GetArray("B-Spline Distance"));

        vtkIdType index = 0;
        vtkIdType cell_index = 0;

        for (vtkIdType l = 0; l < output_lines.data->GetLines()->GetNumberOfCells(); ++l)
        {
            const auto num_points = output_lines.data->GetLines()->GetData()->GetValue(cell_index);

            displacement_distance_array->SetValue(index, std::abs(displacement_ids[index].w - displacement_ids[index + 1].w));

            for (vtkIdType i = 1; i < num_points - 1; ++i)
            {
                displacement_distance_array->SetValue(index + i, 0.5f * (std::abs(displacement_ids[index + i - 1].w - displacement_ids[index + i].w)
                    + std::abs(displacement_ids[index + i].w - displacement_ids[index + i + 1].w)));
            }

            displacement_distance_array->SetValue(index + num_points - 1, std::abs(displacement_ids[index + num_points - 2].w - displacement_ids[index + num_points - 1].w));

            index += num_points;
            cell_index += num_points + 1;
        }
    }

    // Cache output
    output_deformed_lines->DeepCopy(output_lines.data);
}

void feature_deformation::set_output_deformed_geometry(const std::vector<vtkPointSet*>& input_geometry, vtkMultiBlockDataSet* output_deformed_geometry,
    const cuda::displacement& geometry_displacement, bool modified, cache_output_geometry_t& output_geometry) const
{
    // Create output geometry
    if (modified || !output_geometry.valid)
    {
        std::cout << "Creating deformed geometry output" << std::endl;

        output_geometry.data = vtkSmartPointer<vtkMultiBlockDataSet>::New();

        for (unsigned int block_index = 0; block_index < static_cast<unsigned int>(input_geometry.size()); ++block_index)
        {
            output_geometry.data->SetBlock(block_index, vtkDataObjectTypes::NewDataObject(input_geometry[block_index]->GetClassName()));
            output_geometry.data->GetBlock(block_index)->DeepCopy(input_geometry[block_index]);

            auto displacement_id_array = vtkSmartPointer<vtkFloatArray>::New();
            displacement_id_array->SetNumberOfComponents(4);
            displacement_id_array->SetNumberOfTuples(input_geometry[block_index]->GetNumberOfPoints());
            displacement_id_array->SetName("Displacement Information");
            displacement_id_array->FillValue(0.0f);

            auto displacement_distance_array = vtkSmartPointer<vtkFloatArray>::New();
            displacement_distance_array->SetNumberOfComponents(1);
            displacement_distance_array->SetNumberOfTuples(input_geometry[block_index]->GetNumberOfPoints());
            displacement_distance_array->SetName("B-Spline Distance");
            displacement_distance_array->FillValue(0.0f);

            vtkPointSet::SafeDownCast(output_geometry.data->GetBlock(block_index))->GetPointData()->AddArray(displacement_id_array);
            vtkPointSet::SafeDownCast(output_geometry.data->GetBlock(block_index))->GetPointData()->AddArray(displacement_distance_array);
        }

        output_geometry.valid = true;
    }
    else
    {
        std::cout << "Updating deformed geometry output" << std::endl;
    }

    // Set displaced points
    const auto& displaced_geometry = geometry_displacement.get_results();

    std::size_t global_id = 0;

    for (unsigned int block_index = 0; block_index < output_geometry.data->GetNumberOfBlocks(); ++block_index)
    {
        auto block = vtkPointSet::SafeDownCast(output_geometry.data->GetBlock(block_index));

        for (vtkIdType i = 0; i < block->GetNumberOfPoints(); ++i, ++global_id)
        {
            block->GetPoints()->SetPoint(i, displaced_geometry[global_id].data());
        }
    }

    // Set displacement ID arrays
    const auto& displacement_ids = geometry_displacement.get_displacement_info();

    std::size_t global_data_index = 0;

    for (unsigned int block_index = 0; block_index < output_geometry.data->GetNumberOfBlocks(); ++block_index)
    {
        auto block = vtkPointSet::SafeDownCast(output_geometry.data->GetBlock(block_index));
        auto data_array = vtkFloatArray::SafeDownCast(block->GetPointData()->GetArray("Displacement Information"));

        std::memcpy(data_array->GetPointer(0), &displacement_ids[global_data_index], data_array->GetNumberOfTuples() * sizeof(float4));

        global_data_index += data_array->GetNumberOfTuples();
    }

    // In case of the B-Spline, store distance on B-Spline for neighboring points
    if ((this->parameter_displacement.method == cuda::displacement::method_t::b_spline ||
        this->parameter_displacement.method == cuda::displacement::method_t::b_spline_joints) &&
        this->OutputBSplineDistance)
    {
        for (unsigned int block_index = 0; block_index < output_geometry.data->GetNumberOfBlocks(); ++block_index)
        {
            auto block = vtkPointSet::SafeDownCast(output_geometry.data->GetBlock(block_index));
            auto displacement_distance_array = vtkFloatArray::SafeDownCast(block->GetPointData()->GetArray("B-Spline Distance"));

            if (vtkPolyData::SafeDownCast(block) != nullptr)
            {
                auto poly_block = vtkPolyData::SafeDownCast(block);

                vtkIdType index = 0;
                vtkIdType cell_index = 0;

                for (vtkIdType l = 0; l < poly_block->GetLines()->GetNumberOfCells(); ++l)
                {
                    const auto num_points = poly_block->GetLines()->GetData()->GetValue(cell_index);

                    displacement_distance_array->SetValue(index, std::abs(displacement_ids[index].w - displacement_ids[index + 1].w));

                    for (vtkIdType i = 1; i < num_points - 1; ++i)
                    {
                        displacement_distance_array->SetValue(index + i, 0.5f * (std::abs(displacement_ids[index + i - 1].w - displacement_ids[index + i].w)
                            + std::abs(displacement_ids[index + i].w - displacement_ids[index + i + 1].w)));
                    }

                    displacement_distance_array->SetValue(index + num_points - 1, std::abs(displacement_ids[index + num_points - 2].w - displacement_ids[index + num_points - 1].w));

                    index += num_points;
                    cell_index += num_points + 1;
                }
            }
        }
    }

    // Cache output
    output_deformed_geometry->DeepCopy(output_geometry.data);
}

void feature_deformation::create_displacement_field(vtkPointSet* output_deformed_grid) const
{
    // Create displacement field
    auto displacement_map = vtkSmartPointer<vtkDoubleArray>::New();
    displacement_map->SetNumberOfComponents(3);
    displacement_map->SetNumberOfTuples(output_deformed_grid->GetPoints()->GetNumberOfPoints());
    displacement_map->SetName("Displacement Map");

    #pragma omp parallel for
    for (vtkIdType p = 0; p < output_deformed_grid->GetPoints()->GetNumberOfPoints(); ++p)
    {
        Eigen::Vector3d displaced_point;
        output_deformed_grid->GetPoints()->GetPoint(p, displaced_point.data());

        displacement_map->SetTuple(p, displaced_point.data());
    }

    output_deformed_grid->GetPointData()->AddArray(displacement_map);
}

void feature_deformation::deform_velocities(vtkPointSet* output_deformed_grid, vtkDataArray* data_array,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& spacing) const
{
    // Setup velocity interpolation for support of point and cell data
    std::function<Eigen::Vector3d(int, int, int)> get_velocity;

    if (data_array->GetNumberOfTuples() == dimension[0] * dimension[1] * dimension[2])
    {
        get_velocity = [&dimension, &data_array](const int x, const int y, const int z) -> Eigen::Vector3d
        {
            Eigen::Vector3d velocity;
            data_array->GetTuple(calc_index_point(dimension, x, y, z), velocity.data());

            return velocity;
        };
    }
    else
    {
        get_velocity = [&dimension, &data_array](const int x, const int y, const int z) -> Eigen::Vector3d
        {
            std::array<Eigen::Vector3d, 8> velocities;
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 1, z - 1), velocities[0].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 1, z - 1), velocities[1].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 0, z - 1), velocities[2].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 0, z - 1), velocities[3].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 1, z - 0), velocities[4].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 1, z - 0), velocities[5].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 1, y - 0, z - 0), velocities[6].data());
            data_array->GetTuple(calc_index_cell(dimension, x - 0, y - 0, z - 0), velocities[7].data());

            return 0.125 * (velocities[0] + velocities[1] + velocities[2] + velocities[3] + velocities[4] + velocities[5] + velocities[6] + velocities[7]);
        };
    }

    // Setup finite differences for the calculation of the Jacobian
    auto displacement_map = vtkDoubleArray::SafeDownCast(output_deformed_grid->GetPointData()->GetArray("Displacement Map"));

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
    auto jacobian = vtkSmartPointer<vtkDoubleArray>::New();
    jacobian->SetNumberOfComponents(9);
    jacobian->SetNumberOfTuples(dimension[0] * dimension[1] * dimension[2]);
    jacobian->SetName("Jacobian");

    // ... at the point nodes
    auto velocities_p = vtkSmartPointer<vtkDoubleArray>::New();
    velocities_p->SetNumberOfComponents(3);
    velocities_p->SetNumberOfTuples(dimension[0] * dimension[1] * dimension[2]);
    velocities_p->SetName(data_array->GetName());

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
                auto velocity = get_velocity(x, y, z);
                velocity = (Jacobian / Jacobian.determinant()) * velocity;

                velocities_p->SetTuple(index_p, velocity.data());
            }
        }
    }

    output_deformed_grid->GetPointData()->AddArray(jacobian);
    output_deformed_grid->GetPointData()->AddArray(velocities_p);

    // ... at the cell centers
    auto velocities_c = vtkSmartPointer<vtkDoubleArray>::New();
    velocities_c->SetNumberOfComponents(3);
    velocities_c->SetNumberOfTuples((dimension[0] - 1) * (dimension[1] - 1) * (dimension[2] - 1));
    velocities_c->SetName(data_array->GetName());

    #pragma omp parallel for
    for (int z = 0; z < dimension[2] - 1; ++z)
    {
        for (int y = 0; y < dimension[1] - 1; ++y)
        {
            for (int x = 0; x < dimension[0] - 1; ++x)
            {
                const auto index_c = calc_index_cell(dimension, x, y, z);

                std::array<Eigen::Vector3d, 8> velocities;
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 0, z + 0), velocities[0].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 0, z + 0), velocities[1].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 1, z + 0), velocities[2].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 1, z + 0), velocities[3].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 0, z + 1), velocities[4].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 0, z + 1), velocities[5].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 0, y + 1, z + 1), velocities[6].data());
                velocities_p->GetTuple(calc_index_point(dimension, x + 1, y + 1, z + 1), velocities[7].data());

                const Eigen::Vector3d velocity = 0.125 * (velocities[0] + velocities[1] + velocities[2]
                    + velocities[3] + velocities[4] + velocities[5] + velocities[6] + velocities[7]);

                velocities_c->SetTuple(index_c, velocity.data());
            }
        }
    }

    output_deformed_grid->GetCellData()->AddArray(velocities_c);
}

void feature_deformation::resample_grid(vtkPointSet* output_deformed_grid, vtkImageData* output_resampled_grid, const std::string& velocity_name,
    const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const
{
    // Resample original grid
    auto velocities_deformed = vtkDoubleArray::SafeDownCast(output_deformed_grid->GetPointData()->GetArray(velocity_name.c_str()));

    auto velocities_resampled = vtkSmartPointer<vtkDoubleArray>::New();
    velocities_resampled->SetNumberOfComponents(3);
    velocities_resampled->SetNumberOfTuples(velocities_deformed->GetNumberOfTuples());
    velocities_resampled->SetName(velocities_deformed->GetName());

    const Eigen::Vector3d origin_d(static_cast<double>(origin[0]), static_cast<double>(origin[1]), static_cast<double>(origin[2]));
    const Eigen::Vector3d spacing_d(static_cast<double>(spacing[0]), static_cast<double>(spacing[1]), static_cast<double>(spacing[2]));

    for (int z = 0; z < dimension[2]; ++z)
    {
        for (int y = 0; y < dimension[1]; ++y)
        {
            vtkCell* cell = nullptr;

            for (int x = 0; x < dimension[0]; ++x)
            {
                Eigen::Vector3d point = origin_d + Eigen::Vector3d(x, y, z).cwiseProduct(spacing_d);

                // Find cell of the deformed grid, in which the point lies
                int subID;
                Eigen::Vector3d pcoords;
                std::array<double, 8> weights;

                cell = output_deformed_grid->FindAndGetCell(point.data(), cell, 0, 0.0, subID, pcoords.data(), weights.data());

                // Use weights to interpolate the velocity
                if (cell != nullptr)
                {
                    if (cell->GetNumberOfPoints() == 8)
                    {
                        auto point_ids = cell->GetPointIds();

                        Eigen::Vector3d velocity_sum{ 0.0, 0.0, 0.0 };
                        double weight_sum = 0.0;

                        for (vtkIdType i = 0; i < point_ids->GetNumberOfIds(); ++i)
                        {
                            Eigen::Vector3d velocity;
                            velocities_deformed->GetTuple(point_ids->GetId(i), velocity.data());

                            velocity_sum += weights[i] * velocity;
                            weight_sum += weights[i];
                        }

                        const Eigen::Vector3d velocity = velocity_sum / weight_sum;

                        velocities_resampled->SetTuple(calc_index_point(dimension, x, y, z), velocity.data());
                    }
                    else
                    {
                        std::clog << cell->GetNumberOfPoints() << std::endl;
                        velocities_resampled->SetTuple3(calc_index_point(dimension, x, y, z), 0.0, 0.0, 0.0);
                    }
                }
                else
                {
                    velocities_resampled->SetTuple3(calc_index_point(dimension, x, y, z), 0.0, 0.0, 0.0);
                }
            }
        }
    }

    output_resampled_grid->GetPointData()->AddArray(velocities_resampled);
}
