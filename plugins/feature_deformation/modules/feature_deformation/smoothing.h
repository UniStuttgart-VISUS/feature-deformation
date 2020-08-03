#pragma once

#include "Eigen/Dense"

#include <utility>
#include <vector>

class smoothing
{
public:
    /// Methods
    enum class method_t
    {
        direct,
        smoothing,
        time_local
    };

    /// Variant
    enum class variant_t
    {
        fixed_endpoints,
        growing
    };

    /// Set method and variant, as well as parameters for Gaussian smoothing
    smoothing(std::vector<Eigen::Vector3f> line, method_t method, variant_t variant, float lambda, std::size_t num_iterations);

    /// Perform one step for the chosen method and variant
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> next_step();

    /// Check if there is another step available, i.e. the line is not straight yet
    bool has_step() const;

    /// Get displacement after all performed steps
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& get_displacement() const;

private:
    /// Resample line, such that each of its points coincides with a time step
    void resample_line();

    /// Use PCA to calculate the endpoints of a straight line with the same arc length
    std::pair<Eigen::Vector3f, Eigen::Vector3f> approx_line_pca() const;

    /// Gaussian line smoothing
    std::vector<Eigen::Vector3f> gaussian_line_smoothing(std::size_t offset = 1) const;

    /// Line for straightening
    std::vector<Eigen::Vector3f> line;

    /// Displacement after all performed steps
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> displacement;

    /// Tracking of performed steps
    std::size_t num_performed_steps;
    std::size_t num_steps;

    enum class state_t
    {
        growing, shrinking
    } state;

    float max_distance;

    /// Parameters
    const method_t method;
    const variant_t variant;

    const float lambda;
};
