#pragma once

#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix3d unit();

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient(const grid& data,
    const std::array<int, 3>& coords, const Eigen::Matrix3d& jacobian);

vtkSmartPointer<vtkDoubleArray> gradient_field(const grid& data, vtkDataArray* jacobian_field);
