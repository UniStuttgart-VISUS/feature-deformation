#pragma once

#include "algorithm.h"
#include "algorithm_displacement_assessment.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_computation_twisting.h"
#include "algorithm_displacement_computation_winding.h"
#include "algorithm_geometry_output_creation.h"
#include "displacement.h"

#include "vtkMultiBlockDataSet.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_geometry_output_update : public algorithm<std::shared_ptr<const algorithm_geometry_output_creation>,
    std::shared_ptr<const algorithm_displacement_computation>, std::shared_ptr<const algorithm_displacement_computation_winding>,
    std::shared_ptr<const algorithm_displacement_computation_twisting>, std::shared_ptr<const algorithm_displacement_assessment>,
    cuda::displacement::method_t, bool, bool>
{
public:
    /// Default constructor
    algorithm_geometry_output_update() = default;

    /// Get results
    struct results_t
    {
        vtkSmartPointer<vtkMultiBlockDataSet> geometry;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_geometry_output_creation> output_geometry,
        std::shared_ptr<const algorithm_displacement_computation> displacement,
        std::shared_ptr<const algorithm_displacement_computation_winding> displacement_winding,
        std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting,
        std::shared_ptr<const algorithm_displacement_assessment> assessment,
        cuda::displacement::method_t displacement_method,
        bool minimal_output,
        bool output_bspline_distance
    ) override;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const override;

    /// Run computation
    virtual bool run_computation() override;

    /// Print cache load message
    virtual void cache_load() const override;

private:
    /// Input
    std::shared_ptr<const algorithm_geometry_output_creation> output_geometry;
    std::shared_ptr<const algorithm_displacement_computation> displacement;
    std::shared_ptr<const algorithm_displacement_computation_winding> displacement_winding;
    std::shared_ptr<const algorithm_displacement_computation_twisting> displacement_twisting;
    std::shared_ptr<const algorithm_displacement_assessment> assessment;

    /// Parameters
    cuda::displacement::method_t displacement_method;
    bool minimal_output;
    bool output_bspline_distance;

    /// Results
    results_t results;
};
