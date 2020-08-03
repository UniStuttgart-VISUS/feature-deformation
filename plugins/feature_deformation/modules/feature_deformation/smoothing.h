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
        smoothing
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
    void next_step();

    /// Check if there is another step available, i.e. the line is not straight yet
    bool has_step() const;

    /// Get displacement after all performed steps
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> get_displacement() const;

private:
    /// Use PCA to calculate the endpoints of a straight line with the same arc length
    std::pair<Eigen::Vector3f, Eigen::Vector3f> approx_line_pca() const;

    /// Gaussian line smoothing
    void gaussian_line_smoothing(bool fixed);

    /// Line for straightening
    std::vector<Eigen::Vector3f> line;
    Eigen::MatrixXf vertices;

    /// Matrices for implicit smoothing calculation
    Eigen::MatrixXf A_fixed;
    Eigen::MatrixXf A_moving;

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
