#pragma once

#include "algorithm.h"
#include "algorithm_input.h"

#include "vtkPointSet.h"

#include "Eigen/Dense"

#include <array>
#include <vector>

class algorithm_geometry_input : public algorithm<std::vector<vtkPointSet*>>, public algorithm_input
{
public:
    /// Default constructor
    algorithm_geometry_input() = default;

    /// Get results
    struct results_t
    {
        std::vector<vtkPointSet*> geometry;
        std::vector<std::array<float, 3>> points;
    };

    const results_t& get_results() const;

    /// Get points
    virtual algorithm_input::points_t get_points() const override;

protected:
    /// Set input
    virtual void set_input(
        std::vector<vtkPointSet*> geometry
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    std::vector<vtkPointSet*> input_geometry;

    /// Results
    results_t results;
};
