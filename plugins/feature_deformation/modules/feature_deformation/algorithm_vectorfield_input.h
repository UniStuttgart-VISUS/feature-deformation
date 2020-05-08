#pragma once

#include "algorithm.h"
#include "algorithm_grid_input.h"

#include "vtkDataArray.h"

#include <type_traits>

class algorithm_vectorfield_input : public algorithm<const algorithm_grid_input&, const std::string&>
{
public:
    /// Default constructor
    algorithm_vectorfield_input() = default;

    /// Get results
    struct results_t
    {
        vtkDataArray* vector_field;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        const algorithm_grid_input& input_grid,
        const std::string& array_name
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    std::reference_wrapper<const algorithm_grid_input> input_grid;
    std::string array_name;

    /// Results
    results_t results;
};
