#include "smoothing.h"

#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/Sparse"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

smoothing::smoothing(std::vector<Eigen::Vector3f> line, const method_t method, const variant_t variant, const float lambda) :
    line(line),
    vertices(line.size(), 3),
    num_performed_steps(0),
    method(method),
    variant(variant),
    lambda(lambda),
    state(state_t::growing),
    max_distance((line.back() - line.front()).norm())
{
    // Create matrix representing the points of the polyline
    const auto n = this->line.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        this->vertices.row(i) = this->line[i].transpose();
    }

    // Create weight matrix for fixed end points
    Eigen::SparseMatrix<float> L_fixed(n, n);

    for (std::size_t j = 1; j < n - 1; ++j)
    {
        const auto weight_left = 1.0f / (this->line[j] - this->line[j - 1]).norm();
        const auto weight_right = 1.0f / (this->line[j] - this->line[j - 1]).norm();
        const auto weight_sum = weight_left + weight_right;

        L_fixed.insert(j, j - 1) = weight_left / weight_sum;
        L_fixed.insert(j, j) = -1.0f;
        L_fixed.insert(j, j + 1) = weight_right / weight_sum;
    }

    // Create weight matrix for moving end points
    auto L_moving = L_fixed;

    L_moving.insert(0, 0) = -1.0f;
    L_moving.insert(0, 1) = 1.0f;
    L_moving.insert(n - 1, n - 2) = 1.0f;
    L_moving.insert(n - 1, n - 1) = -1.0f;

    // Create matrices
    Eigen::SparseMatrix<float> I(n, n);
    I.setIdentity();

    this->A_fixed = (I - this->lambda * L_fixed);
    this->A_moving = (I - this->lambda * L_moving);
}

void smoothing::next_step()
{
    if (this->method == method_t::smoothing)
    {
        // Apply smoothing step to the line
        switch (this->variant)
        {
        case variant_t::fixed_endpoints:
            // Apply Gaussian smoothing step
            gaussian_line_smoothing(true);

            break;
        case variant_t::growing:
            // Apply Gaussian smoothing step, fixing the endpoints only after letting them grow apart
            if (this->state == state_t::growing)
            {
                gaussian_line_smoothing(false);

                const auto distance = (this->vertices.row(0) - this->vertices.row(this->line.size() - 1)).norm();

                if (distance < 0.9 * this->max_distance)
                {
                    this->state = state_t::shrinking;
                }

                this->max_distance = std::max(this->max_distance, distance);
            }
            else
            {
                gaussian_line_smoothing(true);
            }

            break;
        }
    }

    ++this->num_performed_steps;
}

bool smoothing::has_step() const
{
    switch (this->method)
    {
    case method_t::direct:
        return this->num_performed_steps == 0;

        break;
    case method_t::smoothing:
        return true;

        break;
    }

    return false;
}

const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> smoothing::get_displacement() const
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> displacement(this->line.size());

    // Define target line directly and calculate displacement
    if (this->method == method_t::direct)
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
        const auto direction = (line_end - line_start) / static_cast<float>(this->line.size() - 1);

        for (std::size_t i = 0; i < this->line.size(); ++i)
        {
            const auto target = line_start + i * direction;

            displacement[i].first = this->line[i];
            displacement[i].second = target - this->line[i];
        }
    }

    // Extract displacement from previous smoothing
    else
    {
        for (std::size_t i = 0; i < this->line.size(); ++i)
        {
            displacement[i].first = this->line[i];
            displacement[i].second = this->vertices.row(i).transpose() - this->line[i];
        }
    }

    return displacement;
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

void smoothing::gaussian_line_smoothing(const bool fixed)
{
    const Eigen::BiCGSTAB<Eigen::SparseMatrix<float>, Eigen::IncompleteLUT<float>> solver(fixed ? this->A_fixed : this->A_moving);

    const Eigen::VectorXf x1 = solver.solve(this->vertices.col(0));
    const Eigen::VectorXf x2 = solver.solve(this->vertices.col(1));
    const Eigen::VectorXf x3 = solver.solve(this->vertices.col(2));

    this->vertices << x1, x2, x3;
}
