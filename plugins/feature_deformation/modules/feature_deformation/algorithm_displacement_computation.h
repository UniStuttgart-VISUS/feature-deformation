#pragma once

#include "algorithm.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"

#include <array>
#include <memory>
#include <vector>

class algorithm_displacement_computation : public algorithm<std::shared_ptr<const algorithm_displacement_creation>,
    std::shared_ptr<const algorithm_smoothing>, cuda::displacement::method_t, cuda::displacement::parameter_t>
{
public:
    /// Default constructor
    algorithm_displacement_computation() = default;

    /// Get results
    struct results_t
    {
        std::shared_ptr<cuda::displacement> displacements;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_displacement_creation> displacement,
        std::shared_ptr<const algorithm_smoothing> smoothing,
        cuda::displacement::method_t method,
        cuda::displacement::parameter_t displacement_parameters
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_displacement_creation> displacement;
    std::shared_ptr<const algorithm_smoothing> smoothing;

    /// Parameters
    cuda::displacement::method_t method;
    cuda::displacement::parameter_t displacement_parameters;

    /// Results
    results_t results;
};
