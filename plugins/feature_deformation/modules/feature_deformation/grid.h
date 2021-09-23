#pragma once

#include "vtkDataArray.h"

#include "Eigen/Dense"

#include <array>

class grid
{
public:
    grid(std::array<int, 3> dimension, const Eigen::Vector3d& spacing, vtkDataArray* data);
    grid(std::array<int, 3> dimension, vtkDataArray* positions, vtkDataArray* data);

    Eigen::Vector3d h_plus(const std::array<int, 3>& coords) const;
    Eigen::Vector3d h_minus(const std::array<int, 3>& coords) const;

    Eigen::Matrix<double, Eigen::Dynamic, 1> value(const std::array<int, 3>& coords) const;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix(const std::array<int, 3>& coords) const;

    int components() const;

    std::array<int, 3> left(const std::array<int, 3>& coords) const;
    std::array<int, 3> right(const std::array<int, 3>& coords) const;
    std::array<int, 3> bottom(const std::array<int, 3>& coords) const;
    std::array<int, 3> top(const std::array<int, 3>& coords) const;
    std::array<int, 3> back(const std::array<int, 3>& coords) const;
    std::array<int, 3> front(const std::array<int, 3>& coords) const;

private:
    std::array<int, 3> dimension;
    vtkDataArray* data;

    bool deformed;

    union
    {
        Eigen::Vector3d spacing;
        vtkDataArray* positions;
    };

    std::size_t index(const std::array<int, 3>& coords) const;
};
