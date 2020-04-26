#include "smoothing.h"

#include "Eigen/Dense"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

smoothing::smoothing(std::vector<Eigen::Vector3f> line, const method_t method, const variant_t variant, const float lambda, const std::size_t num_iterations) :
    line(line),
    num_performed_steps(0),
    method(method),
    variant(method == method_t::time_local ? variant_t::fixed_endpoints : variant),
    lambda(lambda),
    num_steps(num_iterations),
    state(state_t::growing),
    max_distance((line.back() - line.front()).norm())
{
    // Resample line if necessary
    if (this->method == method_t::time_local)
    {
        resample_line();
    }

    // Create initial zero-displacement
    this->displacement.resize(this->line.size());

    for (std::size_t i = 0; i < this->line.size(); ++i)
    {
        this->displacement[i].first = this->line[i];
        this->displacement[i].second.setZero();
    }
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> smoothing::next_step()
{
    // Depending on method and variant, deform line
    std::vector<Eigen::Vector3f> target_line;
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> displacement;

    switch (this->method)
    {
    case method_t::time_local:
    case method_t::direct:
    {
        // Compute target line endpoints
        Eigen::Vector3f line_start, line_end;

        switch (this->variant)
        {
        case variant_t::fixed_endpoints:
            // Set target line end points to the input line endpoints
            line_start = this->line.front();
            line_end = this->line.back();

            break;
        case variant_t::growing:
            // Compute target line endpoints using PCA
            std::tie(line_start, line_end) = approx_line_pca();

            break;
        }

        // Uniformly subdivide the target line
        target_line.resize(this->line.size());
        target_line.front() = line_start;
        target_line.back() = line_end;

        const auto direction = (target_line.back() - target_line.front()) / static_cast<float>(target_line.size() - 1);

        for (std::size_t i = 1; i < target_line.size() - 1; ++i)
        {
            target_line[i] = target_line.front() + i * direction;
        }
    }

        break;
    case method_t::smoothing:
        // Apply smoothing step to the line
        switch (this->variant)
        {
        case variant_t::fixed_endpoints:
            // Apply Gaussian smoothing step
            target_line = gaussian_line_smoothing();

            break;
        case variant_t::growing:
            // Apply Gaussian smoothing step, fixing the endpoints only after letting them grow apart
            if (this->state == state_t::growing)
            {
                auto new_line = gaussian_line_smoothing(0);

                const auto distance = (new_line.front() - new_line.back()).norm();

                if (distance < 0.9 * this->max_distance)
                {
                    this->state = state_t::shrinking;
                }

                this->max_distance = std::max(this->max_distance, distance);

                target_line = new_line;
            }
            else
            {
                target_line = gaussian_line_smoothing();
            }

            break;
        }

        break;
    }

    // Calculate displacement
    displacement.resize(target_line.size());

    for (std::size_t i = 0; i < target_line.size(); ++i)
    {
        displacement[i].first = this->line[i];
        displacement[i].second = target_line[i] - this->line[i];
    }

    // Accumulate displacements
    for (std::size_t i = 0; i < this->displacement.size(); ++i)
    {
        // Only add displacement vector (.second), retaining the original position (.first)
        this->displacement[i].second += displacement[i].second;
    }

    // Prepare for the next iteration
    std::swap(this->line, target_line);

    ++this->num_performed_steps;

    // Return displacement calculated for this step
    return displacement;
}

bool smoothing::has_step() const
{
    switch (this->method)
    {
    case method_t::direct:
    case method_t::time_local:
        return this->num_performed_steps == 0;

        break;
    case method_t::smoothing:
        return this->num_performed_steps < this->num_steps;

        break;
    }

    return false;
}

const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& smoothing::get_displacement() const
{
    return this->displacement;
}

void smoothing::resample_line()
{
    // Get line starts and ends
    const auto inverse = this->line.front()[2] > this->line.back()[2];

    const auto line_start = this->line.front();
    const auto line_target = this->line.back();
    const auto line_direction = (line_target - line_start) / (this->line.size() - 1);

    // Calculate sample z-positions
    std::vector<float> time_points(this->line.size());

    for (std::size_t i = 0; i < this->line.size(); ++i)
    {
        time_points[i] = line_start[2] + i * line_direction[2];
    }

    // (Re-)sample original line
    std::vector<Eigen::Vector3f> resampled_line = this->line;

    std::size_t time_index = 1;

    for (std::size_t i = 0; i < this->line.size() - 1; ++i)
    {
        // Does the first line segment's point coincide with a time step?
        if (i != 0 && time_points[time_index] == this->line[i][2])
        {
            resampled_line[time_index++] = this->line[i];
        }

        // Does a time step lie within the segment? -> Interpolate
        auto between = time_points[time_index] > this->line[i][2] && time_points[time_index] < this->line[i + 1][2];
        auto between_inverse = time_points[time_index] < this->line[i][2] && time_points[time_index] > this->line[i + 1][2];

        while (time_index < (this->line.size() - 1) && ((between && !inverse) || (between_inverse && inverse)))
        {
            const auto lambda = (time_points[time_index] - this->line[i][2]) / (this->line[i + 1][2] - this->line[i][2]);
            resampled_line[time_index++] = this->line[i] + lambda * (this->line[i + 1] - this->line[i]);

            between = time_points[time_index] > this->line[i][2] && time_points[time_index] < this->line[i + 1][2];
            between_inverse = time_points[time_index] < this->line[i][2] && time_points[time_index] > this->line[i + 1][2];
        }
    }

    // Set resampled line as input line
    std::swap(this->line, resampled_line);
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> smoothing::approx_line_pca() const
{
    // Calculate center of mass
    Eigen::Vector3f center{ 0.0f, 0.0f, 0.0f };

    for (const auto& point : this->line)
    {
        center += point;
    }

    center /= static_cast<float>(this->line.size());

    // Calculate line length
    float length = 0.0f;

    for (std::size_t i = 0; i < this->line.size() - 1; ++i)
    {
        length += (this->line[i + 1] - this->line[i]).norm();
    }

    // Using PCA, approximate straight line through the polyline
    Eigen::Matrix<float, Eigen::Dynamic, 3> pca_matrix;
    pca_matrix.resize(this->line.size(), Eigen::NoChange);

    for (std::size_t i = 0; i < this->line.size(); ++i)
    {
        pca_matrix.block(i, 0, 1, 3) = (this->line[i] - center).transpose();
    }

    const Eigen::Matrix3f pca_matrix_sqr = pca_matrix.transpose() * pca_matrix;

    const Eigen::EigenSolver<Eigen::Matrix3f> eigen(pca_matrix_sqr, true);
    const auto eigenvalues = eigen.eigenvalues();
    const auto eigenvectors = eigen.eigenvectors();

    const auto max_index = (eigenvalues[0].real() > eigenvalues[1].real() && eigenvalues[0].real() > eigenvalues[2].real()) ? 0
        : ((eigenvalues[1].real() > eigenvalues[2].real()) ? 1 : 2);

    const Eigen::Vector3f approx_line = eigenvectors.col(max_index).real();

    // Set line endpoints as equidistant to the center of mass
    const Eigen::Vector3f endpoint_1 = center + 0.5 * length * approx_line;
    const Eigen::Vector3f endpoint_2 = center - 0.5 * length * approx_line;

    if ((endpoint_1 - this->line.front()).norm() < (endpoint_1 - this->line.back()).norm())
    {
        return std::make_pair(endpoint_1, endpoint_2);
    }
    else
    {
        return std::make_pair(endpoint_2, endpoint_1);
    }
}

Eigen::Vector3f smoothing::gaussian_smoothing(const std::vector<Eigen::Vector3f>& points, const std::size_t index, const float weight) const
{
    Eigen::Vector3f point = points[index];

    if (index == 0)
    {
        point = point + weight * (points[index + 1] - point);
    }
    else if (index == points.size() - 1)
    {
        point = point + weight * (points[index - 1] - point);
    }
    else
    {
        point = point + weight * (
            0.5f * (points[index - 1] - point) +
            0.5f * (points[index + 1] - point));
    }

    return point;
}

std::vector<Eigen::Vector3f> smoothing::gaussian_line_smoothing(const std::size_t offset) const
{
    auto deformed_line = this->line;

    for (std::size_t i = offset; i < this->line.size() - offset; ++i)
    {
        deformed_line[i] = gaussian_smoothing(this->line, i, this->lambda);
    }

    return deformed_line;
}
