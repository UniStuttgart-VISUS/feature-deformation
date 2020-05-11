#include "algorithm_geometry_input.h"

#include "hash.h"

#include "vtkCellArray.h"
#include "vtkIdList.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>

void algorithm_geometry_input::set_input(std::vector<vtkPointSet*> geometry)
{
    this->input_geometry = geometry;
}

std::uint32_t algorithm_geometry_input::calculate_hash() const
{
    if (this->input_geometry.empty())
    {
        return -1;
    }

    // Calculate hash from all input geometry sets
    std::uint32_t hash = static_cast<std::uint32_t>(this->input_geometry.size());

    for (auto geometry_set : this->input_geometry)
    {
        hash = jenkins_hash(hash, geometry_set->GetPoints()->GetMTime());
    }

    return hash;
}

bool algorithm_geometry_input::run_computation()
{
    std::cout << "Loading input geometry" << std::endl;

    // Count points
    std::size_t num_points = 0;

    for (auto geometry_set : this->input_geometry)
    {
        if (geometry_set != nullptr)
        {
            num_points += geometry_set->GetNumberOfPoints();
        }
    }

    if (num_points == 0)
    {
        return false;
    }

    // Get all points
    std::size_t index = 0;
    std::size_t point_index = 0;

    this->results.points.resize(num_points);

    for (auto geometry_set : this->input_geometry)
    {
        if (geometry_set != nullptr)
        {
            if (this->input_geometry.size() > 1)
            {
                std::cout << "  input " << index++ << std::endl;
            }

            // Get geometry
            #pragma omp parallel for
            for (vtkIdType p = 0; p < geometry_set->GetNumberOfPoints(); ++p)
            {
                std::array<double, 3> point;
                geometry_set->GetPoints()->GetPoint(p, point.data());

                this->results.points[point_index + p] = { static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]) };
            }

            point_index += geometry_set->GetNumberOfPoints();
        }
    }

    // Store input also as output
    this->results.geometry = this->input_geometry;

    return true;
}

void algorithm_geometry_input::cache_load() const
{
    std::cout << "Loading input geometry from cache" << std::endl;
}

const algorithm_geometry_input::results_t& algorithm_geometry_input::get_results() const
{
    return this->results;
}

algorithm_input::points_t algorithm_geometry_input::get_points() const
{
    return algorithm_input::points_t{ this->results.points, this->get_hash(), this->is_valid() };
}
