#pragma once

#include "algorithm.h"
#include "algorithm_compute_tearing.h"
#include "algorithm_displacement_assessment.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_computation_twisting.h"
#include "algorithm_grid_input.h"
#include "algorithm_grid_output_creation.h"
#include "displacement.h"

#include "vtkMultiBlockDataSet.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_grid_output_update : public algorithm<std::shared_ptr<const algorithm_grid_input>,
    std::shared_ptr<const algorithm_grid_output_creation>, std::shared_ptr<const algorithm_displacement_computation>,
    std::shared_ptr<const algorithm_displacement_computation_twisting>, std::shared_ptr<const algorithm_displacement_assessment>,
    std::shared_ptr<const algorithm_compute_tearing>, bool, float, bool>
{
public:
    /// Default constructor
    algorithm_grid_output_update() = default;

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
        std::shared_ptr<const algorithm_grid_output_creation> output_grid,
        std::shared_ptr<const algorithm_displacement_computation> displacement,
        std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting,
        std::shared_ptr<const algorithm_displacement_assessment> assessment,
        std::shared_ptr<const algorithm_compute_tearing> tearing,
        bool remove_cells,
        float remove_cells_scalar,
        bool minimal_output
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
    std::shared_ptr<const algorithm_grid_output_creation> output_grid;
    std::shared_ptr<const algorithm_displacement_computation> displacement;
    std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting;
    std::shared_ptr<const algorithm_displacement_assessment> assessment;
    std::shared_ptr<const algorithm_compute_tearing> tearing;

    /// Parameters
    bool remove_cells;
    float remove_cells_scalar;
    bool minimal_output;

    /// Results
    results_t results;
};
