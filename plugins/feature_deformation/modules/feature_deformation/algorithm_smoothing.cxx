#include "algorithm_smoothing.h"

#include "hash.h"
#include "smoothing.h"

#include "Eigen/Dense"

#include <iostream>

void algorithm_smoothing::set_input(const algorithm_line_input& line_input,
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
    if (!this->line_input.get().is_valid())
    {
        return -1;
    }

    return jenkins_hash(this->line_input.get().get_hash(), this->method, this->variant, this->lambda, this->num_iterations);
}

bool algorithm_smoothing::run_computation()
{
    std::cout << "Smoothing line" << std::endl;

    // Smooth line
    smoothing smoother(this->line_input.get().get_results().selected_line, this->method, this->variant, this->lambda, this->num_iterations);

    while (smoother.has_step())
    {
        smoother.next_step();
    }

    const auto smoothing_results = smoother.get_displacement();

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
    std::cout << "Loading smoothed line from cache" << std::endl;
}

const algorithm_smoothing::results_t& algorithm_smoothing::get_results() const
{
    return this->results;
}
