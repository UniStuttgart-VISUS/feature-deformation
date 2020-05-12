#pragma once

#include "algorithm.h"
#include "algorithm_grid_output_update.h"

#include "vtkInformation.h"

#include <memory>

class algorithm_grid_output_set : public algorithm<std::shared_ptr<const algorithm_grid_output_update>, vtkInformation*, double>
{
public:
    /// Default constructor
    algorithm_grid_output_set() = default;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_grid_output_update> output_grid,
        vtkInformation* output_information,
        double data_time
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_grid_output_update> output_grid;
    vtkInformation* output_information;

    /// Parameters
    double data_time;
};
