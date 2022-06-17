#include "algorithm_displacement_computation_winding.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_smoothing.h"
#include "algorithm_twisting.h"
#include "displacement.h"
#include "hash.h"

#include <iostream>

void algorithm_displacement_computation_winding::set_input(const std::shared_ptr<const algorithm_displacement_computation> displacement,
    const std::shared_ptr<const algorithm_smoothing> smoothing, const cuda::displacement::method_t method,
    const cuda::displacement::parameter_t displacement_parameters, const bool active)
{
    this->displacement = displacement;
    this->smoothing = smoothing;
    this->method = method;
    this->displacement_parameters = displacement_parameters;
    this->active = active;
}

std::uint32_t algorithm_displacement_computation_winding::calculate_hash() const
{
    if (!(this->displacement->is_valid() && this->smoothing->is_valid() && this->active &&
        this->method == cuda::displacement::method_t::b_spline_joints))
    {
        return -1;
    }

    return jenkins_hash(this->displacement->get_hash(), this->smoothing->get_hash(),
        this->method, this->displacement_parameters.b_spline, this->active);
}

bool algorithm_displacement_computation_winding::run_computation()
{
    if (!this->is_quiet()) std::cout << "  calculating new positions on the GPU" << std::endl;

    this->displacement->get_results().displacements->displace_winding(this->method, this->displacement_parameters,
        this->smoothing->get_results().positions, this->smoothing->get_results().displacements);

    this->results.displacements = this->displacement->get_results().displacements;

    return true;
}

const algorithm_displacement_computation_winding::results_t& algorithm_displacement_computation_winding::get_results() const
{
    return this->results;
}
