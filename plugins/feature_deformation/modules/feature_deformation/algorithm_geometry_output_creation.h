#pragma once

#include "algorithm.h"
#include "algorithm_geometry_input.h"

#include "vtkMultiBlockDataSet.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_geometry_output_creation : public algorithm<std::shared_ptr<const algorithm_geometry_input>>
{
public:
    /// Default constructor
    algorithm_geometry_output_creation() = default;

    /// Get results
    struct results_t
    {
        vtkSmartPointer<vtkMultiBlockDataSet> geometry;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_geometry_input> input_geometry
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_geometry_input> input_geometry;

    /// Results
    results_t results;
};
