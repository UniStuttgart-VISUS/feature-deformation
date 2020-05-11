#pragma once

#include "algorithm.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"

#include <type_traits>

class algorithm_displacement_precomputation : public algorithm<const algorithm_displacement_creation&, const algorithm_smoothing&,
    const algorithm_line_input&, cuda::displacement::method_t, cuda::displacement::parameter_t, cuda::displacement::b_spline_parameters_t>
{
public:
    /// Default constructor
    algorithm_displacement_precomputation() = default;

protected:
    /// Set input
    virtual void set_input(
        const algorithm_displacement_creation& displacement,
        const algorithm_smoothing& smoothing,
        const algorithm_line_input& input_lines,
        cuda::displacement::method_t method,
        cuda::displacement::parameter_t displacement_parameters,
        cuda::displacement::b_spline_parameters_t bspline_parameters
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::reference_wrapper<const algorithm_displacement_creation> displacement;
    std::reference_wrapper<const algorithm_smoothing> smoothing;

    /// Parameters
    std::reference_wrapper<const algorithm_line_input> input_lines;
    cuda::displacement::method_t method;
    cuda::displacement::parameter_t displacement_parameters;
    cuda::displacement::b_spline_parameters_t bspline_parameters;
};
