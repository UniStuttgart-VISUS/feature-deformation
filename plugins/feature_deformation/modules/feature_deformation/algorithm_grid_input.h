#pragma once

#include "algorithm.h"
#include "algorithm_input.h"

#include "vtkImageData.h"

#include "Eigen/Dense"

#include <array>
#include <vector>

class algorithm_grid_input : public algorithm<vtkImageData*>, public algorithm_input
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

        std::vector<std::array<float, 3>> points;
    };

    const results_t& get_results() const;

    /// Get points
    virtual algorithm_input::points_t get_points() const override;

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
