#pragma once

#include "algorithm.h"
#include "algorithm_line_input.h"
#include "smoothing.h"

#include "Eigen/Dense"

#include <array>
#include <memory>
#include <vector>

class algorithm_smoothing : public algorithm<std::shared_ptr<const algorithm_line_input>, smoothing::method_t, smoothing::variant_t, float, int>
{
public:
    /// Default constructor
    algorithm_smoothing() = default;

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
        std::shared_ptr<const algorithm_line_input> line_input,
        smoothing::method_t method,
        smoothing::variant_t variant,
        float lambda,
        int num_iterations
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    std::shared_ptr<const algorithm_line_input> line_input;

    /// Parameters
    smoothing::method_t method;
    smoothing::variant_t variant;
    float lambda;
    int num_iterations;

    /// Results
    results_t results;
};
