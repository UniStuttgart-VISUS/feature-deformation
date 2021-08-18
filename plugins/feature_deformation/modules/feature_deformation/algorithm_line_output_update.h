#pragma once

#include "algorithm.h"
#include "algorithm_displacement_assessment.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_line_output_creation.h"
#include "displacement.h"

#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include <memory>

class algorithm_line_output_update : public algorithm<std::shared_ptr<const algorithm_line_output_creation>,
    std::shared_ptr<const algorithm_displacement_computation>, std::shared_ptr<const algorithm_displacement_assessment>,
    cuda::displacement::method_t, bool>
{
public:
    /// Default constructor
    algorithm_line_output_update() = default;

    /// Get results
    struct results_t
    {
        vtkSmartPointer<vtkPolyData> lines;
    };

    const results_t& get_results() const;

protected:
    /// Set input
    virtual void set_input(
        std::shared_ptr<const algorithm_line_output_creation> output_lines,
        std::shared_ptr<const algorithm_displacement_computation> displacement,
        std::shared_ptr<const algorithm_displacement_assessment> assessment,
        cuda::displacement::method_t displacement_method,
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
    std::shared_ptr<const algorithm_line_output_creation> output_lines;
    std::shared_ptr<const algorithm_displacement_computation> displacement;
    std::shared_ptr<const algorithm_displacement_assessment> assessment;

    /// Parameters
    cuda::displacement::method_t displacement_method;
    bool output_bspline_distance;

    /// Results
    results_t results;
};
