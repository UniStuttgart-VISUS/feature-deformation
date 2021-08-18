#include "algorithm_displacement_assessment.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "hash.h"

#include <iostream>

void algorithm_displacement_assessment::set_input(const std::shared_ptr<const algorithm_displacement_computation> displacement,
    const std::shared_ptr<const algorithm_smoothing> smoothing, const cuda::displacement::method_t method,
    const cuda::displacement::parameter_t displacement_parameters, const cuda::displacement::b_spline_parameters_t bspline_parameters,
    const bool assess_mapping)
{
    this->displacement = displacement;
    this->smoothing = smoothing;
    this->method = method;
    this->displacement_parameters = displacement_parameters;
    this->bspline_parameters = bspline_parameters;
    this->assess_mapping = assess_mapping;
}

std::uint32_t algorithm_displacement_assessment::calculate_hash() const
{
    if (!(this->displacement->is_valid()))
    {
        return -1;
    }

    if ((this->method == cuda::displacement::method_t::b_spline || this->method == cuda::displacement::method_t::b_spline_joints)
        && this->assess_mapping)
    {
        return jenkins_hash(this->displacement->get_hash(), this->method, this->bspline_parameters.degree);
    }

    return this->get_hash();
}

bool algorithm_displacement_assessment::run_computation()
{
    if ((this->method == cuda::displacement::method_t::b_spline || this->method == cuda::displacement::method_t::b_spline_joints)
        && this->assess_mapping)
    {
        if (!this->is_quiet()) std::cout << "  assessing B-spline mapping on the GPU" << std::endl;

        this->displacement->get_results().displacements->assess_quality(this->displacement_parameters,
            this->smoothing->get_results().positions, this->smoothing->get_results().displacements);
    }

    return true;
}
