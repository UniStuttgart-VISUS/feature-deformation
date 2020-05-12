#pragma once

#include "algorithm.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_grid_input.h"
#include "algorithm_grid_output_update.h"
#include "algorithm_vectorfield_input.h"

#include "vtkMultiBlockDataSet.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_grid_output_vectorfield : public algorithm<std::shared_ptr<const algorithm_grid_input>,
    std::shared_ptr<const algorithm_grid_output_update>, std::shared_ptr<const algorithm_vectorfield_input>>
{
public:
    /// Default constructor
    algorithm_grid_output_vectorfield() = default;

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
        std::shared_ptr<const algorithm_grid_output_update> output_grid,
        std::shared_ptr<const algorithm_vectorfield_input> vector_field
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    std::shared_ptr<const algorithm_grid_input> input_grid;
    std::shared_ptr<const algorithm_grid_output_update> output_grid;
    std::shared_ptr<const algorithm_vectorfield_input> vector_field;

    /// Results
    results_t results;
};
