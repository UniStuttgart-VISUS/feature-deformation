#pragma once

#include <cstdint>
#include <cstddef>

/// <summary>
/// Jenkins one-at-a-time hash function for one or two input values
/// </summary>
/// <remarks>Implementation of pseuco-code on Wikipedia: 'https://en.wikipedia.org/wiki/Jenkins_hash_function'</remarks>
/// <template name="first_t">Type of first value</template>
/// <template name="second_t">Type of second value</template>
/// <param name="first">First value</param>
/// <param name="second">Second value</param>
/// <returns>Computed hash</returns>
template <typename first_t, typename second_t = void*>
std::uint32_t joaat_hash(const first_t& first, const second_t second = nullptr)
{
    // Implementation of pseuco-code on Wikipedia: https://en.wikipedia.org/wiki/Jenkins_hash_function
    const char* first_ptr = (const char*)&first;
    const char* second_ptr = (const char*)&second;

    std::uint32_t hash = 0;

    for (std::size_t i = 0; i < sizeof(first); ++i)
    {
        hash += first_ptr[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    if (typeid(second_t) != typeid(void*))
    {
        for (std::size_t i = 0; i < sizeof(second); ++i)
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

/// <summary>
/// Jenkins one-at-a-time hash function for multiple input values
/// </summary>
/// <remarks>Implementation of pseuco-code on Wikipedia: 'https://en.wikipedia.org/wiki/Jenkins_hash_function'</remarks>
/// <template name="first_t">Type of first value</template>
/// <template name="second_t">Type of second value</template>
/// <template name="other_t">Type(s) of additional parameters</template>
/// <param name="first">First value</param>
/// <param name="second">Second value</param>
/// <param name="values">Additional values</param>
/// <returns>Computed hash</returns>
template <typename first_t, typename second_t, typename... other_t>
std::uint32_t joaat_hash(const first_t& first, const second_t& second, const other_t&... values)
{
    return joaat_hash(first, joaat_hash(second, values...));
}
