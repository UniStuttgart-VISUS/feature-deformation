#include "algorithm_displacement_computation.h"

#include "algorithm_displacement_creation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "hash.h"

#include <iostream>

void algorithm_displacement_computation::set_input(std::shared_ptr<const algorithm_displacement_creation> displacement,
    std::shared_ptr<const algorithm_smoothing> smoothing, cuda::displacement::method_t method, cuda::displacement::parameter_t displacement_parameters)
{
    this->displacement = displacement;
    this->smoothing = smoothing;
    this->method = method;
    this->displacement_parameters = displacement_parameters;
}

std::uint32_t algorithm_displacement_computation::calculate_hash() const
{
    if (!(this->displacement->is_valid() && this->smoothing->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->smoothing->get_hash(), this->method, this->displacement_parameters);
}

bool algorithm_displacement_computation::run_computation()
{
    std::cout << "  calculating new positions on the GPU" << std::endl;

    this->displacement->get_results().displacements->displace(this->method, this->displacement_parameters,
        this->smoothing->get_results().positions, this->smoothing->get_results().displacements);

    this->results.displacements = this->displacement->get_results().displacements;

    return true;
}

const algorithm_displacement_computation::results_t& algorithm_displacement_computation::get_results() const
{
    return this->results;
}
