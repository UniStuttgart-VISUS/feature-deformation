#pragma once

#include "algorithm.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_grid_input.h"
#include "algorithm_smoothing.h"
#include "algorithm_vectorfield_input.h"
#include "twisting.h"

#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>
#include <memory>
#include <vector>

class algorithm_twisting : public algorithm<std::shared_ptr<const algorithm_vectorfield_input>, std::shared_ptr<const algorithm_grid_input>,
    std::shared_ptr<const algorithm_smoothing>, std::shared_ptr<const algorithm_displacement_computation>>
{
public:
    /// Default constructor
    algorithm_twisting();

    /// Get results
    struct results_t
    {
        std::vector<std::array<float, 4>> rotations;
        vtkSmartPointer<vtkDoubleArray> coordinate_systems;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_vectorfield_input> vector_field,
        std::shared_ptr<const algorithm_grid_input> grid,
        std::shared_ptr<const algorithm_smoothing> straight_feature_line,
        std::shared_ptr<const algorithm_displacement_computation> displacement
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    std::shared_ptr<const algorithm_vectorfield_input> vector_field;
    std::shared_ptr<const algorithm_grid_input> grid;
    std::shared_ptr<const algorithm_smoothing> straight_feature_line;
    std::shared_ptr<const algorithm_displacement_computation> displacement;

    /// Results
    results_t results;

    /// Twister
    std::unique_ptr<twisting> twister;
    uint32_t twister_hash;
};
