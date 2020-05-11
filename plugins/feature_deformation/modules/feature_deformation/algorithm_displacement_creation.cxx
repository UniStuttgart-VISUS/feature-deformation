#include "algorithm_displacement_creation.h"

#include "algorithm_input.h"
#include "displacement.h"

#include <iostream>
#include <memory>

void algorithm_displacement_creation::set_input(const algorithm_input& input)
{
    this->input = input;
}

std::uint32_t algorithm_displacement_creation::calculate_hash() const
{
    return this->input.get().get_points().hash;
}

bool algorithm_displacement_creation::run_computation()
{
    std::cout << "  uploading points to the GPU" << std::endl;

    this->results.displacements = std::make_shared<cuda::displacement>(this->input.get().get_points().points);

    return true;
}

const algorithm_displacement_creation::results_t& algorithm_displacement_creation::get_results() const
{
    return this->results;
}
