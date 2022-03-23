#include "jacobian.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkPoints.h"

#include "Eigen/Dense"

#include <array>

int calc_index_point(const std::array<int, 3>& dimension, int x, int y, int z)
{
    return (z * dimension[1] + y) * dimension[0] + x;
}

double calc_jacobian(vtkDataArray* field, const int center,
    const int index, const int max, const int component, double h_l, double h_r, const int offset)
{
    double left_diff = 0.0;
    double right_diff = 0.0;
    int num = 0;

    if (center != 0) // Backward difference
    {
        const auto left = field->GetComponent(static_cast<long long>(index) - offset, component);
        const auto right = field->GetComponent(index, component);

        left_diff = (right - left) / h_l;
        ++num;
    }
    if (center != max) // Forward difference
    {
        const auto left = field->GetComponent(index, component);
        const auto right = field->GetComponent(static_cast<long long>(index) + offset, component);

        right_diff = (right - left) / h_r;
        ++num;
    }

    return (left_diff + right_diff) / num;
}

double calc_jacobian(vtkPoints* field, const int center,
    const int index, const int max, const int component, double h_l, double h_r, const int offset)
{
    double left_diff = 0.0;
    double right_diff = 0.0;
    int num = 0;

    std::array<double, 3> point_l{}, point_r{};

    if (center != 0) // Backward difference
    {
        field->GetPoint(static_cast<std::size_t>(index) - offset, point_l.data());
        field->GetPoint(index, point_r.data());

        left_diff = (point_r[component] - point_l[component]) / h_l;
        ++num;
    }
    if (center != max) // Forward difference
    {
        field->GetPoint(index, point_l.data());
        field->GetPoint(static_cast<std::size_t>(index) + offset, point_r.data());

        right_diff = (point_r[component] - point_l[component]) / h_r;
        ++num;
    }

    return (left_diff + right_diff) / num;
}

double calc_jacobian_irregular(vtkDataArray* field, const int center,
    const int index, const int max, const int component, vtkDoubleArray* h, const int offset)
{
    double h_l = 0.0;
    double h_r = 0.0;

    if (center != 0) // Backward difference
    {
        Eigen::Vector3d left, right;
        h->GetTuple(static_cast<long long>(index) - offset, left.data());
        h->GetTuple(index, right.data());

        h_l = (right - left).norm();
    }
    if (center != max) // Forward difference
    {
        Eigen::Vector3d left, right;
        h->GetTuple(index, left.data());
        h->GetTuple(static_cast<long long>(index) + offset, right.data());

        h_r = (right - left).norm();
    }

    return calc_jacobian(field, center, index, max, component, h_l, h_r, offset);
}
