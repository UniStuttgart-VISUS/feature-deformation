#pragma once

#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include <utility>

struct curvature_and_torsion_t
{
    vtkSmartPointer<vtkDoubleArray> curvature;
    vtkSmartPointer<vtkDoubleArray> curvature_vector;
    vtkSmartPointer<vtkDoubleArray> curvature_gradient;
    vtkSmartPointer<vtkDoubleArray> torsion;
    vtkSmartPointer<vtkDoubleArray> torsion_vector;
    vtkSmartPointer<vtkDoubleArray> torsion_gradient;
};

curvature_and_torsion_t curvature_and_torsion(const grid& vector_field);