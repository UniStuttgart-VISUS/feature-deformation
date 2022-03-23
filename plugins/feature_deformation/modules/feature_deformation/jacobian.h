#pragma once

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkPoints.h"

#include <array>

int calc_index_point(const std::array<int, 3>& dimension, int x, int y, int z);

double calc_jacobian(vtkDataArray* field, const int center,
    const int index, const int max, const int component, double h_l, double h_r, const int offset);

double calc_jacobian(vtkPoints* field, const int center,
    const int index, const int max, const int component, double h_l, double h_r, const int offset);

double calc_jacobian_irregular(vtkDataArray* field, const int center,
    const int index, const int max, const int component, vtkDoubleArray* h, const int offset);
