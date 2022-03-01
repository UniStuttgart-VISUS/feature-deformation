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
    twisting(std::vector<Eigen::Vector3d> line, vtkSmartPointer<vtkStructuredGrid> vector_field);

    /// Perform twisting algorithm
    bool run();

    /// Get rotations from twisting
    const std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Matrix3d>> get_rotations() const;

private:
    /// Line for twisting
    const std::vector<Eigen::Vector3d> line;

    /// Grid and vector field
    vtkSmartPointer<vtkStructuredGrid> vector_field;

    /// Resulting rotations
    std::vector<Eigen::Vector3d> rotations;
    std::vector<Eigen::Matrix3d> coordinate_systems;
};
