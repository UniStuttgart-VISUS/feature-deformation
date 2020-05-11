#pragma once

#include <array>
#include <type_traits>
#include <vector>

class algorithm_input
{
public:
    /// Get points and hash
    struct points_t
    {
        std::reference_wrapper<const std::vector<std::array<float, 3>>> points;

        std::uint32_t hash;
        bool valid;
    };

    virtual points_t get_points() const = 0;
};
