#include "algorithm_smoothing.h"

#include "hash.h"
#include "smoothing.h"

#include "Eigen/Dense"

#include <iostream>

algorithm_smoothing::algorithm_smoothing() : smoother(nullptr), smoother_hash(-1)
{
}

void algorithm_smoothing::set_input(const std::shared_ptr<const algorithm_line_input> line_input,
    const smoothing::method_t method, const smoothing::variant_t variant, const float lambda, const int num_iterations)
{
    this->line_input = line_input;
    this->method = method;
    this->variant = variant;
    this->lambda = lambda;
    this->num_iterations = num_iterations;
}

std::uint32_t algorithm_smoothing::calculate_hash() const
{
    if (!this->line_input->is_valid())
    {
        return -1;
    }

    return jenkins_hash(this->line_input->get_hash(), this->method, this->variant, this->lambda, this->num_iterations);
}

bool algorithm_smoothing::run_computation()
{
    if (!this->is_quiet()) std::cout << "Smoothing line" << std::endl;

    // Create smoother if necessary
    const auto new_smoother_hash = jenkins_hash(this->line_input->get_hash(), this->method, this->variant, this->lambda);

    if (this->smoother == nullptr || this->smoother_hash != new_smoother_hash || this->num_smoothing_steps > this->num_iterations)
    {
        this->smoother = std::make_unique<smoothing>(this->line_input->get_results().selected_line, this->method, this->variant, this->lambda);

        this->smoother_hash = new_smoother_hash;
        this->num_smoothing_steps = 0;
    }

    // Smooth line
    while (this->smoother->has_step() && this->num_smoothing_steps < this->num_iterations)
    {
        this->smoother->next_step();

        ++this->num_smoothing_steps;
    }

    const auto smoothing_results = this->smoother->get_displacement();

    // Convert results
    this->results.positions.resize(smoothing_results.size());
    this->results.displacements.resize(smoothing_results.size());

    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(results.displacements.size()); ++i)
    {
        this->results.positions[i] = { smoothing_results[i].first[0], smoothing_results[i].first[1], smoothing_results[i].first[2], 1.0f };
        this->results.displacements[i] = { smoothing_results[i].second[0], smoothing_results[i].second[1], smoothing_results[i].second[2], 0.0f };
    }

    return true;
}

void algorithm_smoothing::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading smoothed line from cache" << std::endl;
}

const algorithm_smoothing::results_t& algorithm_smoothing::get_results() const
{
    return this->results;
}
