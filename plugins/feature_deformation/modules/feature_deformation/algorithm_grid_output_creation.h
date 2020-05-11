#pragma once

#include "algorithm.h"
#include "algorithm_grid_input.h"

#include "vtkMultiBlockDataSet.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_grid_output_creation : public algorithm<std::shared_ptr<const algorithm_grid_input>, bool>
{
public:
    /// Default constructor
    algorithm_grid_output_creation() = default;

    /// Get results
    struct results_t
    {
        vtkSmartPointer<vtkMultiBlockDataSet> grid;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_grid_input> input_grid,
        bool remove_cells
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_grid_input> input_grid;

    /// Parameters
    bool remove_cells;

    /// Results
    results_t results;
};
