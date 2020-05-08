#pragma once

#include "algorithm.h"

#include "vtkImageData.h"

#include "Eigen/Dense"

#include <array>

class algorithm_grid_input : public algorithm<vtkImageData*>
{
public:
    /// Default constructor
    algorithm_grid_input() = default;

    /// Get results
    struct results_t
    {
        vtkImageData* grid;

        std::array<int, 6> extent;
        std::array<int, 3> dimension;
        Eigen::Vector3f origin;
        Eigen::Vector3f spacing;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        vtkImageData* grid
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    vtkImageData* input_grid;

    /// Results
    results_t results;
};
