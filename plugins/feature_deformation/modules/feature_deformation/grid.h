#pragma once

#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkStructuredGrid.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix3d unit();

class grid
{
public:
    grid(std::array<int, 3> dimension, const Eigen::Vector3d& spacing, const vtkDataArray* data, const vtkDataArray* jacobians = nullptr);
    grid(std::array<int, 3> dimension, const vtkDataArray* positions, const vtkDataArray* data, const vtkDataArray* jacobians = nullptr);
    grid(const vtkImageData* vtk_grid, const vtkDataArray* data, const vtkDataArray* jacobians = nullptr);
    grid(const vtkStructuredGrid* vtk_grid, const vtkDataArray* data, const vtkDataArray* jacobians = nullptr);
    grid(const grid& grid, const vtkDataArray* data);

    virtual ~grid() noexcept;

    Eigen::Vector3d h_plus(const std::array<int, 3>& coords) const;
    Eigen::Vector3d h_minus(const std::array<int, 3>& coords) const;

    Eigen::Vector3d offset(const std::array<int, 3>& source, const std::array<int, 3>& target) const;

    Eigen::Matrix<double, Eigen::Dynamic, 1> value(const std::array<int, 3>& coords) const;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix(const std::array<int, 3>& coords) const;

    Eigen::Matrix3d jacobian(const std::array<int, 3>& coords) const;

    const std::array<int, 3>& dimensions() const;
    const Eigen::Vector3d& get_spacing() const;
    int components() const;

    std::array<int, 3> left(const std::array<int, 3>& coords) const;
    std::array<int, 3> right(const std::array<int, 3>& coords) const;
    std::array<int, 3> bottom(const std::array<int, 3>& coords) const;
    std::array<int, 3> top(const std::array<int, 3>& coords) const;
    std::array<int, 3> back(const std::array<int, 3>& coords) const;
    std::array<int, 3> front(const std::array<int, 3>& coords) const;

private:
    std::array<int, 3> dimension;
    const vtkDataArray* data;

    bool deformed;

    union
    {
        Eigen::Vector3d spacing;
        const vtkDataArray* positions;
    };

    const vtkDataArray* jacobians;

    bool own_positions;

    std::size_t index(const std::array<int, 3>& coords) const;
};
