#pragma once

#include "algorithm.h"
#include "algorithm_input.h"

#include "displacement.h"

#include <memory>
#include <type_traits>

class algorithm_displacement_creation : public algorithm<const algorithm_input&>
{
public:
    /// Default constructor
    algorithm_displacement_creation() = default;

    /// Get results
    struct results_t
    {
        std::shared_ptr<cuda::displacement> displacements;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        const algorithm_input& input
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::reference_wrapper<const algorithm_input> input;

    /// Results
    results_t results;
};
