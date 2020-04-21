#include "b-spline.h"

#include <array>
#include <numeric>
#include <utility>
#include <vector>

b_spline::b_spline(std::vector<std::array<float, 4>> line, const std::size_t degree) :
    line(line),
    degree(degree)
{
    // Create knot vector
    this->knot_vector.resize(line.size() + degree + 1);

    auto first = this->knot_vector.begin();
    auto last = this->knot_vector.end();

    // Use multiplicity of the first and last de Boor point
    for (std::size_t i = 0; i < degree; ++i)
    {
        this->knot_vector[i] = 0.0f;
        this->knot_vector[this->knot_vector.size() - 1 - i] = static_cast<float>(line.size() - degree);
    }

    // Adjust iterators
    first += degree;
    last -= degree;

    // Set increasing knot values
    std::iota(first, last, 0.0f);
}

std::array<float, 4> b_spline::calculate_position(float arc_position) const
{
    std::array<float, 4> position{ 0.0, 0.0f, 0.0f, 1.0f };

    // Adjust arc position
    const auto last_position = *(this->knot_vector.cend() - this->degree);

    if (arc_position >= last_position)
    {
        arc_position = last_position - 0.00001f;
    }

    // Calculate position with the basis function
    for (std::size_t i = 0; i < this->line.size(); ++i)
    {
        const auto N = basis_function(this->knot_vector, i, this->degree, arc_position);

        position[0] += N * this->line[i][0];
        position[1] += N * this->line[i][1];
        position[2] += N * this->line[i][2];
    }

    return position;
}

const std::vector<float>& b_spline::get_knot_vector() const
{
    return this->knot_vector;
}

float b_spline::basis_function(const std::vector<float>& knot_vector, const std::size_t de_boor_index, const std::size_t degree, const float u) const
{
    // 1 if u_i <= u < u_i+1, 0 otherwise
    if (degree == 0)
    {
        return (knot_vector[de_boor_index] <= u && u < knot_vector[de_boor_index + 1]) ? 1.0f : 0.0f;
    }

    // Calculate recursively
    const auto Ni = basis_function(knot_vector, de_boor_index, degree - 1, u);
    const auto Nip1 = basis_function(knot_vector, de_boor_index + 1, degree - 1, u);

    const auto part_1 = (knot_vector[de_boor_index + degree] - knot_vector[de_boor_index] == 0.0f) ? 0.0f :
        ((u - knot_vector[de_boor_index]) / (knot_vector[de_boor_index + degree] - knot_vector[de_boor_index]));
    const auto part_2 = (knot_vector[de_boor_index + degree + 1] - knot_vector[de_boor_index + 1] == 0.0f) ? 0.0f :
        ((knot_vector[de_boor_index + degree + 1] - u) / (knot_vector[de_boor_index + degree + 1] - knot_vector[de_boor_index + 1]));

    return part_1 * Ni + part_2 * Nip1;
}
