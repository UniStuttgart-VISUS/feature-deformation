#pragma once

#include "grid.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix3d unit();

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient(const grid& data,
    const std::array<int, 3>& coords, const Eigen::Matrix3d& jacobian);
