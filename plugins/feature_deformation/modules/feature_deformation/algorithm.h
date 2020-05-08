#pragma once

#include <cstdint>

template <typename... input_t>
class algorithm
{
public:
    /// Run computation
    virtual bool run(input_t...) final;

    /// Get hash of the computation
    std::uint32_t get_hash() const;

    /// Get validity
    bool is_valid() const;

protected:
    /// Set default hash and initialize as invalid
    algorithm();

    /// Set input
    virtual void set_input(input_t...) = 0;

    /// Calculate hash
    virtual std::uint32_t calculate_hash() const = 0;

    /// Run computation
    virtual bool run_computation() = 0;

    /// Allow algorithm to do something although there is no need for computation
    virtual void cache_load() const;

private:
    /// Store hash of input, assuming deterministic behavior of the algorithm
    std::uint32_t hash;

    /// Validity state
    bool valid;
};

#include "algorithm.inl"
