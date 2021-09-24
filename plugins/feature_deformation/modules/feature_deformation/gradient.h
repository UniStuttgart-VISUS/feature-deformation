#pragma once

#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient(const grid& data,
    const std::array<int, 3>& coords);

vtkSmartPointer<vtkDoubleArray> gradient_field(const grid& data);
