#pragma once

#include <cstdint>

/// <summary>Jenkins one-at-a-time hash function for one or two input values</summary>
/// <typeparam name="T">Type of first</typeparam>
/// <typeparam name="U">Type of second</typeparam>
/// <param name="first">First value to hash</param>
/// <param name="second">Second value to hash</param>
/// <returns>Hash</returns>
template <typename T, typename U = void*>
uint32_t jenkins_hash(const T& first, const U second = nullptr)
{
    const char* first_ptr = reinterpret_cast<const char*>(&first);
    const char* second_ptr = reinterpret_cast<const char*>(&second);

    uint32_t hash = 0;

    for (size_t i = 0; i < sizeof(first); ++i)
    {
        hash += first_ptr[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    if (typeid(U) != typeid(void*))
    {
        for (size_t i = 0; i < sizeof(second); ++i)
        {
            hash += second_ptr[i];
            hash += (hash << 10);
            hash ^= (hash >> 6);
        }
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}

/// <summary>Jenkins one-at-a-time hash function for multiple input values</summary>
/// <typeparam name="T">Type of first</typeparam>
/// <typeparam name="U">Type of second</typeparam>
/// <typeparam name="Targs">Type(s) of additional parameters</typeparam>
/// <param name="first">First value to hash</param>
/// <param name="second">Second value to hash</param>
/// <param name="values">Additional values to hash</param>
/// <returns>Hash</returns>
template <typename T, typename U, typename... Targs>
uint32_t jenkins_hash(const T& first, const U& second, const Targs&... values)
{
    return jenkins_hash(first, jenkins_hash(second, values...));
}
