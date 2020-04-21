#pragma once

#include <array>
#include <utility>
#include <vector>

class b_spline
{
public:
    /// Set input line and degree of the B-Spline
    b_spline(std::vector<std::array<float, 4>> line, std::size_t degree);

    /// Calculate position
    std::array<float, 4> calculate_position(float arc_position) const;

    /// Get knot vector
    const std::vector<float>& get_knot_vector() const;

private:
    /// B-Spline basis function
    float basis_function(const std::vector<float>& knot_vector, std::size_t de_boor_index, std::size_t degree, float u) const;

    /// Input line
    const std::vector<std::array<float, 4>> line;

    /// Degree of the B-Spline
    const std::size_t degree;

    /// Knot vector
    std::vector<float> knot_vector;
};
