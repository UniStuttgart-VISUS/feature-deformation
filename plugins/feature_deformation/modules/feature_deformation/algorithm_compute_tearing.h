#pragma once

#include "algorithm.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_displacement_precomputation.h"
#include "algorithm_grid_input.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"
#include "displacement.h"
#include "smoothing.h"

#include "Eigen/Dense"

#include <memory>

class algorithm_compute_tearing : public algorithm<std::shared_ptr<const algorithm_line_input>,
    std::shared_ptr<const algorithm_grid_input>, smoothing::method_t, smoothing::variant_t, float, int,
    cuda::displacement::method_t, cuda::displacement::parameter_t, cuda::displacement::b_spline_parameters_t, float>
{
public:
    /// Default constructor
    algorithm_compute_tearing() = default;

    /// Get results
    struct results_t
    {
        vtkSmartPointer<vtkIdTypeArray> tearing_cells;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_line_input> line_input,
        std::shared_ptr<const algorithm_grid_input> grid_input,
        smoothing::method_t smoothing_method,
        smoothing::variant_t smoothing_variant,
        float lambda,
        int num_iterations,
        cuda::displacement::method_t displacement_method,
        cuda::displacement::parameter_t displacement_parameters,
        cuda::displacement::b_spline_parameters_t bspline_parameters,
        float remove_cells_scalar
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

private:
    /// Input
    std::shared_ptr<const algorithm_line_input> line_input;
    std::shared_ptr<const algorithm_grid_input> grid_input;

    /// Parameters
    smoothing::method_t smoothing_method;
    smoothing::variant_t smoothing_variant;
    float lambda;
    int num_iterations;
    cuda::displacement::method_t displacement_method;
    cuda::displacement::parameter_t displacement_parameters;
    cuda::displacement::b_spline_parameters_t bspline_parameters;
    float remove_cells_scalar;

    /// Results
    results_t results;

    /// State
    std::shared_ptr<algorithm_smoothing> alg_smoothing;
    std::shared_ptr<algorithm_displacement_creation> alg_displacement_creation;
    std::shared_ptr<algorithm_displacement_precomputation> alg_displacement_precomputation;
    std::shared_ptr<algorithm_displacement_computation> alg_displacement_computation;
};
