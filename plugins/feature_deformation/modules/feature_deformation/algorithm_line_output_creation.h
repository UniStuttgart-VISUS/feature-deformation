#pragma once

#include "algorithm.h"
#include "algorithm_line_input.h"

#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_line_output_creation : public algorithm<std::shared_ptr<const algorithm_line_input>>
{
public:
    /// Default constructor
    algorithm_line_output_creation() = default;

    /// Get results
    struct results_t
    {
        vtkSmartPointer<vtkPolyData> lines;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_line_input> input_lines
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_line_input> input_lines;

    /// Results
    results_t results;
};
