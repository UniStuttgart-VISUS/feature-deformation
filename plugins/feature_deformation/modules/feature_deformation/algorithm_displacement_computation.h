#pragma once

#include "algorithm.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"

#include <array>
#include <type_traits>
#include <vector>

class algorithm_displacement_computation : public algorithm<const algorithm_displacement_creation&, const algorithm_smoothing&,
    cuda::displacement::method_t, cuda::displacement::parameter_t>
{
public:
    /// Default constructor
    algorithm_displacement_computation() = default;

    /// Get results
    struct results_t
    {
        std::vector<std::array<float, 4>> positions;
        std::vector<std::array<float, 4>> displacements;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        const algorithm_displacement_creation& displacement,
        const algorithm_smoothing& smoothing,
        cuda::displacement::method_t method,
        cuda::displacement::parameter_t displacement_parameters
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
    cuda::displacement::method_t method;
    cuda::displacement::parameter_t displacement_parameters;

    /// Results
    results_t results;
};
