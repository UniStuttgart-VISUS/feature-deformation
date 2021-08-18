#pragma once

#include "algorithm.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"

#include <memory>

class algorithm_displacement_assessment : public algorithm<std::shared_ptr<const algorithm_displacement_computation>,
    std::shared_ptr<const algorithm_smoothing>, cuda::displacement::method_t,
    cuda::displacement::parameter_t, cuda::displacement::b_spline_parameters_t, bool>
{
public:
    /// Default constructor
    algorithm_displacement_assessment() = default;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_displacement_computation> displacement,
        std::shared_ptr<const algorithm_smoothing> smoothing,
        cuda::displacement::method_t method,
        cuda::displacement::parameter_t displacement_parameters,
        cuda::displacement::b_spline_parameters_t bspline_parameters,
        bool assess_mapping
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_displacement_computation> displacement;
    std::shared_ptr<const algorithm_smoothing> smoothing;

    /// Parameters
    cuda::displacement::method_t method;
    cuda::displacement::parameter_t displacement_parameters;
    cuda::displacement::b_spline_parameters_t bspline_parameters;

    bool assess_mapping;
};
