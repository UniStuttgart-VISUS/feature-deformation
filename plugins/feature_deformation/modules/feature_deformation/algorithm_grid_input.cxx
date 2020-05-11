#include "algorithm_grid_input.h"

#include "hash.h"

#include "vtkImageData.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>

void algorithm_grid_input::set_input(vtkImageData* grid)
{
    this->input_grid = grid;
}

std::uint32_t algorithm_grid_input::calculate_hash() const
{
    if (this->input_grid == nullptr)
    {
        return -1;
    }

    return jenkins_hash(this->input_grid->GetMTime());
}

bool algorithm_grid_input::run_computation()
{
    std::cout << "Loading input grid" << std::endl;

    // Get extents
    this->input_grid->GetExtent(this->results.extent.data());
    this->input_grid->GetDimensions(this->results.dimension.data());

    // Get origin and spacing
    std::array<double, 3> origin_data;
    std::array<double, 3> spacing_data;

    this->input_grid->GetOrigin(origin_data.data());
    this->input_grid->GetSpacing(spacing_data.data());

    this->results.origin << static_cast<float>(origin_data[0]), static_cast<float>(origin_data[1]), static_cast<float>(origin_data[2]);
    this->results.spacing << static_cast<float>(spacing_data[0]), static_cast<float>(spacing_data[1]), static_cast<float>(spacing_data[2]);

    this->results.origin += Eigen::Vector3f(this->results.extent[0], this->results.extent[2],
        this->results.extent[4]).cwiseProduct(this->results.spacing);

    // Store points
    this->results.points.resize(this->results.dimension[0] * this->results.dimension[1] * this->results.dimension[2]);

    #pragma omp parallel for
    for (long long z_omp = 0; z_omp < static_cast<long long>(this->results.dimension[2]); ++z_omp)
    {
        const auto z = static_cast<std::size_t>(z_omp);

        auto index = z * this->results.dimension[1] * this->results.dimension[0];

        for (std::size_t y = 0; y < this->results.dimension[1]; ++y)
        {
            for (std::size_t x = 0; x < this->results.dimension[0]; ++x)
            {
                this->results.points[index][0] = static_cast<float>(this->results.origin[0] + x * this->results.spacing[0]);
                this->results.points[index][1] = static_cast<float>(this->results.origin[1] + y * this->results.spacing[1]);
                this->results.points[index][2] = static_cast<float>(this->results.origin[2] + z * this->results.spacing[2]);

                ++index;
            }
        }
    }

    // Set input also as output
    this->results.grid = this->input_grid;

    return true;
}

void algorithm_grid_input::cache_load() const
{
    std::cout << "Loading input grid from cache" << std::endl;
}

const algorithm_grid_input::results_t& algorithm_grid_input::get_results() const
{
    return this->results;
}

algorithm_input::points_t algorithm_grid_input::get_points() const
{
    return algorithm_input::points_t{ this->results.points, this->get_hash() };
}
