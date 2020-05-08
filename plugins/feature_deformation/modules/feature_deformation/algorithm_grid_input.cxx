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
