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
        fixed_arclength
    };

    /// Set method and variant, as well as parameters for Gaussian/Taubin smoothing
    smoothing(std::vector<Eigen::Vector3f> line, method_t method, variant_t variant, float lambda, float mu, std::size_t num_iterations);

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

    /// Gaussian smoothing
    Eigen::Vector3f gaussian_smoothing(const std::vector<Eigen::Vector3f>& points, std::size_t index, float weight) const;

    /// Gaussian line smoothing
    std::vector<Eigen::Vector3f> gaussian_line_smoothing() const;

    /// Taubin line smoothing
    std::vector<Eigen::Vector3f> taubin_line_smoothing() const;

    /// Calculate arc length of a line
    float calculate_arc_length(const std::vector<Eigen::Vector3f>& line) const;

    /// Line for straightening
    std::vector<Eigen::Vector3f> line;

    /// Original arc length
    const float arc_length;

    /// Displacement after all performed steps
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> displacement;

    /// Tracking of performed steps
    std::size_t num_performed_steps;
    std::size_t num_steps;

    /// Parameters
    const method_t method;
    const variant_t variant;

    const float lambda;
    const float mu;
};
