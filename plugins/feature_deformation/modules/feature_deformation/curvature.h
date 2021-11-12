#pragma once

#include "gradient.h"
#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include <utility>

struct curvature_and_torsion_t
{
    vtkSmartPointer<vtkDoubleArray> first_derivative;
    vtkSmartPointer<vtkDoubleArray> second_derivative;
    vtkSmartPointer<vtkDoubleArray> curvature;
    vtkSmartPointer<vtkDoubleArray> curvature_vector;
    vtkSmartPointer<vtkDoubleArray> curvature_gradient;
    vtkSmartPointer<vtkDoubleArray> torsion;
    vtkSmartPointer<vtkDoubleArray> torsion_vector;
    vtkSmartPointer<vtkDoubleArray> torsion_gradient;
};

curvature_and_torsion_t curvature_and_torsion(const grid& vector_field, gradient_method_t method, int kernel_size = 1);
