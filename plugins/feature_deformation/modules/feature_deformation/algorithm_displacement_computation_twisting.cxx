#include "algorithm_displacement_computation_twisting.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_twisting.h"
#include "displacement.h"
#include "hash.h"

#include <iostream>

void algorithm_displacement_computation_twisting::set_input(const std::shared_ptr<const algorithm_displacement_computation> displacement,
    const std::shared_ptr<const algorithm_twisting> twisting, const cuda::displacement::method_t method,
    const cuda::displacement::parameter_t displacement_parameters)
{
    this->displacement = displacement;
    this->twisting = twisting;
    this->method = method;
    this->displacement_parameters = displacement_parameters;
}

std::uint32_t algorithm_displacement_computation_twisting::calculate_hash() const
{
    if (!(this->displacement->is_valid() && this->twisting->is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->twisting->get_hash(), this->method, this->displacement_parameters);
}

bool algorithm_displacement_computation_twisting::run_computation()
{
    if (!this->is_quiet()) std::cout << "  calculating new positions on the GPU" << std::endl;

    //this->displacement->get_results().displacements->displace_twisting(this->method, this->displacement_parameters,
    //    this->twisting->get_results().rotations);

    this->results.displacements = this->displacement->get_results().displacements;

    return true;
}

const algorithm_displacement_computation_twisting::results_t& algorithm_displacement_computation_twisting::get_results() const
{
    return this->results;
}
