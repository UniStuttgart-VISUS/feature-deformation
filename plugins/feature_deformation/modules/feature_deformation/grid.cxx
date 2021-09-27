#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkStructuredGrid.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix3d unit()
{
    Eigen::Matrix3d mat;
    mat.setZero();
    mat.diagonal().setOnes();

    return mat;
}

grid::grid(std::array<int, 3> dimension, const Eigen::Vector3d& spacing, vtkDataArray* data, vtkDataArray* jacobians)
    : dimension(dimension), data(data), deformed(false), spacing(spacing), jacobians(jacobians), own_positions(false){ }

grid::grid(std::array<int, 3> dimension, vtkDataArray* positions, vtkDataArray* data, vtkDataArray* jacobians)
    : dimension(dimension), data(data), deformed(true), positions(positions), jacobians(jacobians), own_positions(false) { }

grid::grid(vtkStructuredGrid* vtk_grid, vtkDataArray* data, vtkDataArray* jacobians)
    : data(data), jacobians(jacobians), own_positions(true)
{
    vtk_grid->GetDimensions(dimension.data());

    // Get positions
    auto positions = vtkDoubleArray::New();
    positions->SetNumberOfComponents(3);
    positions->SetNumberOfTuples(vtk_grid->GetNumberOfPoints());

    std::array<double, 3> point{};

    for (vtkIdType i = 0; i < vtk_grid->GetNumberOfPoints(); ++i)
    {
        vtk_grid->GetPoint(i, point.data());
        positions->SetTuple(i, point.data());
    }

    this->deformed = true;
    this->positions = positions;
}

grid::grid(const grid& grid, vtkDataArray* data)
    : dimension(grid.dimension), data(data), deformed(grid.deformed), jacobians(grid.jacobians), own_positions(false)
{
    if (this->deformed)
    {
        this->positions = grid.positions;
    }
    else
    {
        this->spacing = grid.spacing;
    }
}

grid::~grid() noexcept
{
    if (this->own_positions)
    {
        positions->Delete();
    }
}

Eigen::Vector3d grid::h_plus(const std::array<int, 3>& coords) const
{
    if (this->deformed)
    {
        Eigen::Vector3d pos_current, pos_x, pos_y, pos_z;
        this->positions->GetTuple(index(coords), pos_current.data());
        this->positions->GetTuple(index(right(coords)), pos_x.data());
        this->positions->GetTuple(index(top(coords)), pos_y.data());
        this->positions->GetTuple(index(front(coords)), pos_z.data());

        const auto h_x = (pos_x - pos_current).norm();
        const auto h_y = (pos_y - pos_current).norm();
        const auto h_z = (pos_z - pos_current).norm();

        return Eigen::Vector3d(h_x, h_y, h_z);
    }
    else
    {
        const auto h_x = (coords[0] < this->dimension[0] - 1) ? this->spacing[0] : 0.0;
        const auto h_y = (coords[1] < this->dimension[1] - 1) ? this->spacing[1] : 0.0;
        const auto h_z = (coords[2] < this->dimension[2] - 1) ? this->spacing[2] : 0.0;

        return Eigen::Vector3d(h_x, h_y, h_z);
    }
}

Eigen::Vector3d grid::h_minus(const std::array<int, 3>& coords) const
{
    if (this->deformed)
    {
        Eigen::Vector3d pos_current, pos_x, pos_y, pos_z;
        this->positions->GetTuple(index(coords), pos_current.data());
        this->positions->GetTuple(index(left(coords)), pos_x.data());
        this->positions->GetTuple(index(bottom(coords)), pos_y.data());
        this->positions->GetTuple(index(back(coords)), pos_z.data());

        const auto h_x = (pos_current - pos_x).norm();
        const auto h_y = (pos_current - pos_y).norm();
        const auto h_z = (pos_current - pos_z).norm();

        return Eigen::Vector3d(h_x, h_y, h_z);
    }
    else
    {
        const auto h_x = (coords[0] > 0) ? this->spacing[0] : 0.0;
        const auto h_y = (coords[1] > 0) ? this->spacing[1] : 0.0;
        const auto h_z = (coords[2] > 0) ? this->spacing[2] : 0.0;

        return Eigen::Vector3d(h_x, h_y, h_z);
    }
}

Eigen::Matrix<double, Eigen::Dynamic, 1> grid::value(const std::array<int, 3>& coords) const
{
    Eigen::VectorXd val(this->data->GetNumberOfComponents());

    this->data->GetTuple(index(coords), val.data());

    return val;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> grid::matrix(const std::array<int, 3>& coords) const
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat.resize(static_cast<int>(std::sqrt(components())), static_cast<int>(std::sqrt(components())));

    this->data->GetTuple(index(coords), mat.data());

    return mat;
}

Eigen::Matrix3d grid::jacobian(const std::array<int, 3>& coords) const
{
    Eigen::Matrix3d jac = unit();

    if (this->jacobians != nullptr)
    {
        this->jacobians->GetTuple(index(coords), jac.data());
    }

    return jac;
}

const std::array<int, 3>& grid::dimensions() const
{
    return this->dimension;
}

int grid::components() const
{
    return this->data->GetNumberOfComponents();
}

std::array<int, 3> grid::left(const std::array<int, 3>& coords) const
{
    if (coords[0] > 0)
    {
        return std::array<int, 3>{ coords[0] - 1, coords[1], coords[2] };
    }

    return coords;
}

std::array<int, 3> grid::right(const std::array<int, 3>& coords) const
{
    if (coords[0] < this->dimension[0] - 1)
    {
        return std::array<int, 3>{ coords[0] + 1, coords[1], coords[2] };
    }

    return coords;
}

std::array<int, 3> grid::bottom(const std::array<int, 3>& coords) const
{
    if (coords[1] > 0)
    {
        return std::array<int, 3>{ coords[0], coords[1] - 1, coords[2] };
    }

    return coords;
}

std::array<int, 3> grid::top(const std::array<int, 3>& coords) const
{
    if (coords[1] < this->dimension[1] - 1)
    {
        return std::array<int, 3>{ coords[0], coords[1] + 1, coords[2] };
    }

    return coords;
}

std::array<int, 3> grid::back(const std::array<int, 3>& coords) const
{
    if (coords[2] > 0)
    {
        return std::array<int, 3>{ coords[0], coords[1], coords[2] - 1 };
    }

    return coords;
}

std::array<int, 3> grid::front(const std::array<int, 3>& coords) const
{
    if (coords[2] < this->dimension[2] - 1)
    {
        return std::array<int, 3>{ coords[0], coords[1], coords[2] + 1 };
    }

    return coords;
}

std::size_t grid::index(const std::array<int, 3>& coords) const
{
    return (coords[2] * this->dimension[1] + coords[1]) * this->dimension[0] + coords[0];
}
