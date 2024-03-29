#include "algorithm.h"

template <typename... input_t>
inline algorithm<input_t...>::algorithm() : hash(-1), valid(false), quiet(false)
{
}

template <typename... input_t>
inline bool algorithm<input_t...>::run(input_t... input)
{
    // Set input
    set_input(input...);

    // Check hash...
    const auto new_hash = calculate_hash();

    // ... invalid input
    if (new_hash == -1)
    {
        this->hash = -1;
        this->valid = false;
        return false;
    }

    // ... cache hit
    if (new_hash == this->hash)
    {
        cache_load();

        this->valid = true;
        return true;
    }

    // ... cache miss
    if (run_computation())
    {
        this->hash = new_hash;

        this->valid = true;
        return true;
    }

    // Computation error
    this->hash = -1;
    this->valid = false;
    return false;
}

template <typename... input_t>
inline std::uint32_t algorithm<input_t...>::get_hash() const
{
    return this->hash;
}

template <typename... input_t>
inline bool algorithm<input_t...>::is_valid() const
{
    return this->valid;
}

template <typename... input_t>
inline void algorithm<input_t...>::cache_load() const
{
}

template <typename... input_t>
inline void algorithm<input_t...>::be_quiet(const bool quiet)
{
    this->quiet = quiet;
}

template <typename... input_t>
inline bool algorithm<input_t...>::is_quiet() const
{
    return this->quiet;
}
