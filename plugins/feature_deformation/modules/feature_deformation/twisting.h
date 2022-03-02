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
    twisting(std::vector<Eigen::Vector3d> line, vtkSmartPointer<vtkStructuredGrid> vector_field, int selected_eigenvector);

    /// Perform twisting algorithm
    bool run();

    /// Get rotations from twisting
    const std::pair<std::vector<float>, std::vector<Eigen::Matrix3d>> get_rotations() const;

private:
    /// Line for twisting
    const std::vector<Eigen::Vector3d> line;

    /// Grid and vector field
    vtkSmartPointer<vtkStructuredGrid> vector_field;

    /// Parameter
    int selected_eigenvector;

    /// Resulting rotations
    std::vector<float> rotations;
    std::vector<Eigen::Matrix3d> coordinate_systems;
};
