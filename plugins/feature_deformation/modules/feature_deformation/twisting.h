#pragma once

#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include "Eigen/Dense"

#include <array>
#include <memory>
#include <utility>
#include <vector>

class twisting
{
public:
    /// Set input for twisting
    twisting(std::vector<Eigen::Vector3f> line, vtkSmartPointer<vtkStructuredGrid> vector_field);

    /// Perform twisting algorithm
    void run();

    /// Get rotations from twisting
    const std::pair<std::vector<Eigen::Vector3f>, vtkSmartPointer<vtkDoubleArray>> get_rotations() const;

private:
    /// Line for twisting
    const std::vector<Eigen::Vector3f> line;

    /// Grid and vector field
    vtkSmartPointer<vtkStructuredGrid> vector_field;

    /// Resulting rotations
    std::vector<Eigen::Vector3f> rotations;
    vtkSmartPointer<vtkDoubleArray> coordinate_systems;
};
