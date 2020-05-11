#include "algorithm_displacement_precomputation.h"

#include "algorithm_displacement_creation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "hash.h"

#include <iostream>

void algorithm_displacement_precomputation::set_input(const algorithm_displacement_creation& displacement,
    const algorithm_smoothing& smoothing, const algorithm_line_input& input_lines, cuda::displacement::method_t method,
    cuda::displacement::parameter_t displacement_parameters, cuda::displacement::b_spline_parameters_t bspline_parameters)
{
    this->displacement = displacement;
    this->smoothing = smoothing;
    this->input_lines = input_lines;
    this->method = method;
    this->displacement_parameters = displacement_parameters;
    this->bspline_parameters = bspline_parameters;
}

std::uint32_t algorithm_displacement_precomputation::calculate_hash() const
{
    if (!(this->displacement.get().is_valid() && this->smoothing.get().is_valid() && this->input_lines.get().is_valid()))
    {
        return -1;
    }

    return jenkins_hash(this->displacement.get().get_hash(), this->input_lines.get().get_hash(), this->method, this->bspline_parameters.degree);
}

bool algorithm_displacement_precomputation::run_computation()
{
    if (this->method == cuda::displacement::method_t::b_spline || this->method == cuda::displacement::method_t::b_spline_joints)
    {
        std::cout << "  precomputing B-spline mapping on the GPU" << std::endl;

        this->displacement.get().get_results().displacements->precompute(this->displacement_parameters, this->smoothing.get().get_results().positions);

        return true;
    }

    return false;
}
