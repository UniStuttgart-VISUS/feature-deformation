#pragma once

#include "algorithm.h"

#include "vtkPolyData.h"

#include "Eigen/Dense"

#include <array>
#include <vector>

class algorithm_line_input : public algorithm<vtkPolyData*, int>
{
public:
    /// Default constructor
    algorithm_line_input() = default;

    /// Get results
    struct results_t
    {
        std::vector<Eigen::Vector3f> selected_line;
        std::vector<std::array<float, 3>> lines;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        vtkPolyData* lines,
        int selected_line_id
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    vtkPolyData* input_lines;

    /// Parameters
    int selected_line_id;

    /// Results
    results_t results;
};
