#include "algorithm_displacement_precomputation.h"

#include "algorithm_displacement_creation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "hash.h"

#include <iostream>

void algorithm_displacement_precomputation::set_input(std::shared_ptr<const algorithm_displacement_creation> displacement,
    std::shared_ptr<const algorithm_smoothing> smoothing, std::shared_ptr<const algorithm_line_input> input_lines, cuda::displacement::method_t method,
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
    if (!(this->displacement->is_valid() && this->smoothing->is_valid() && this->input_lines->is_valid()))
    {
        return -1;
    }

    if (this->method == cuda::displacement::method_t::b_spline || this->method == cuda::displacement::method_t::b_spline_joints)
    {
        return jenkins_hash(this->displacement->get_hash(), this->input_lines->get_hash(), this->method, this->bspline_parameters.degree);
    }

    return this->get_hash();
}

bool algorithm_displacement_precomputation::run_computation()
{
    if (this->method == cuda::displacement::method_t::b_spline || this->method == cuda::displacement::method_t::b_spline_joints)
    {
        if (!this->is_quiet()) std::cout << "  precomputing B-spline mapping on the GPU" << std::endl;

        this->displacement->get_results().displacements->precompute(this->displacement_parameters, this->smoothing->get_results().positions);
    }

    return true;
}
