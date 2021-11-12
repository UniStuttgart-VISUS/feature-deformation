#pragma once

#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>

enum class gradient_method_t
{
    differences, least_squares
};

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_differences(const grid& data,
    const std::array<int, 3>& coords);

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_least_squares(const grid& data,
    const std::array<int, 3>& coords, int kernel_size);

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient(const grid& data,
    const std::array<int, 3>& coords, gradient_method_t method, int kernel_size);

vtkSmartPointer<vtkDoubleArray> gradient_field(const grid& data, gradient_method_t method, int kernel_size = 1);
