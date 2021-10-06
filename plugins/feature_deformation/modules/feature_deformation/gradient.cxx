#include "gradient.h"

#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_differences(const grid& data,
    const std::array<int, 3>& coords)
{
    const auto jacobian = data.jacobian(coords);
    const auto metric_tensor = jacobian.transpose() * jacobian;
    const auto inverse_metric = metric_tensor.inverse();

    // Forward derivative
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_fwd, gradient_bck;

    const auto h_fwd = data.h_plus(coords);
    const auto h_bck = data.h_minus(coords);

    Eigen::VectorXd diff_right = (data.value(data.right(coords)) - data.value(coords)) / h_fwd[0];
    Eigen::VectorXd diff_left = (data.value(coords) - data.value(data.left(coords))) / h_bck[0];

    Eigen::VectorXd diff_top = (data.value(data.top(coords)) - data.value(coords)) / h_fwd[1];
    Eigen::VectorXd diff_bottom = (data.value(coords) - data.value(data.bottom(coords))) / h_bck[1];

    if (h_fwd[0] == 0.0) diff_right = diff_left;
    if (h_bck[0] == 0.0) diff_left = diff_right;
    if (h_fwd[1] == 0.0) diff_top = diff_bottom;
    if (h_bck[1] == 0.0) diff_bottom = diff_top;

    if (h_fwd[2] == 0 && h_bck[2] == 0) // 2D case
    {
        gradient_fwd.resize((data.components() == 1) ? 1 : 2, 2);
        gradient_bck.resize((data.components() == 1) ? 1 : 2, 2);

        gradient_fwd.setZero();
        gradient_bck.setZero();

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                const auto basis_vec = jacobian.col(i).head(2);

                if (data.components() == 1)
                {
                    gradient_fwd += inverse_metric(i, j) * ((j == 0) ? diff_right : diff_top) * basis_vec.norm() * basis_vec.transpose();
                    gradient_bck += inverse_metric(i, j) * ((j == 0) ? diff_left : diff_bottom) * basis_vec.norm() * basis_vec.transpose();
                }
                else
                {
                    gradient_fwd += inverse_metric(i, j) * ((j == 0) ? diff_right.head(2) : diff_top.head(2)) * basis_vec.norm() * basis_vec.transpose();
                    gradient_bck += inverse_metric(i, j) * ((j == 0) ? diff_left.head(2) : diff_bottom.head(2)) * basis_vec.norm() * basis_vec.transpose();
                }
            }
        }
    }
    else // 3D case
    {
        Eigen::VectorXd diff_front = (data.value(data.front(coords)) - data.value(coords)) / h_fwd[2];
        Eigen::VectorXd diff_back = (data.value(coords) - data.value(data.back(coords))) / h_bck[2];

        if (h_fwd[2] == 0.0) diff_front = diff_back;
        if (h_bck[2] == 0.0) diff_back = diff_front;

        gradient_fwd.resize((data.components() == 1) ? 1 : 3, 3);
        gradient_bck.resize((data.components() == 1) ? 1 : 3, 3);

        gradient_fwd.setZero();
        gradient_bck.setZero();

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                const auto basis_vec = jacobian.col(i);

                gradient_fwd += inverse_metric(i, j) * ((j == 0) ? diff_right : ((j == 1) ? diff_top : diff_front)) * basis_vec.norm() * basis_vec.transpose();
                gradient_bck += inverse_metric(i, j) * ((j == 0) ? diff_left : ((j == 1) ? diff_bottom : diff_back)) * basis_vec.norm() * basis_vec.transpose();
            }
        }
    }

    // Combine forward and backward into "central difference"
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient = 0.5 * (gradient_fwd + gradient_bck);

    if (h_fwd[2] == 0 && h_bck[2] == 0) // 2D case
    {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_3D;
        gradient_3D.resize(3, (data.components() == 1) ? 1 : 3);
        gradient_3D.setZero();

        if (data.components() == 1)
        {
            gradient_3D.block(0, 0, 2, 1) = gradient.transpose();
        }
        else
        {
            gradient_3D.block(0, 0, 2, 2) = gradient;
            gradient_3D(2, 2) = 1.0;
        }

        return gradient_3D;
    }
    else
    {
        if (data.components() == 1)
        {
            return gradient.transpose();
        }
        else
        {
            return gradient;
        }
    }
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_least_squares(const grid& data,
    const std::array<int, 3>& coords, const int kernel_size)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient;

    if (data.components() == 1)
    {
        gradient.resize(3, 1);
    }
    else
    {
        gradient.resize(data.components(), 3);
    }

    const auto twoD = data.h_plus(coords)[2] == 0.0 && data.h_minus(coords)[2] == 0.0;

    // Create matrix A, weight matrix W and right-hand side b
    for (int c = 0; c < data.components(); ++c)
    {
        Eigen::Matrix3d A;
        Eigen::Vector3d b;

        A.setZero();
        b.setZero();

        for (int zz = -kernel_size; zz <= kernel_size; ++zz)
        {
            for (int yy = -kernel_size; yy <= kernel_size; ++yy)
            {
                for (int xx = -kernel_size; xx <= kernel_size; ++xx)
                {
                    const std::array<int, 3> kernel_coords{ coords[0] + xx, coords[1] + yy, coords[2] + (twoD ? 0 : zz) };

                    if ((xx == 0 && yy == 0 && zz == 0) || kernel_coords[0] < 0 || kernel_coords[1] < 0 || kernel_coords[2] < 0 ||
                        kernel_coords[0] >= data.dimensions()[0] || kernel_coords[1] >= data.dimensions()[1] || kernel_coords[2] >= data.dimensions()[2]) continue;

                    const auto neighbor_offset = data.offset(coords, kernel_coords) + Eigen::Vector3d(0, 0, (twoD ? zz : 0));

                    A(0, 0) += neighbor_offset[0] * neighbor_offset[0];
                    A(0, 1) += neighbor_offset[0] * neighbor_offset[1];
                    A(0, 2) += neighbor_offset[0] * neighbor_offset[2];

                    A(1, 0) += neighbor_offset[1] * neighbor_offset[0];
                    A(1, 1) += neighbor_offset[1] * neighbor_offset[1];
                    A(1, 2) += neighbor_offset[1] * neighbor_offset[2];

                    A(2, 0) += neighbor_offset[2] * neighbor_offset[0];
                    A(2, 1) += neighbor_offset[2] * neighbor_offset[1];
                    A(2, 2) += neighbor_offset[2] * neighbor_offset[2];

                    const auto f = (data.value(kernel_coords) - data.value(coords))[c];

                    b[0] += f * neighbor_offset[0];
                    b[1] += f * neighbor_offset[1];
                    b[2] += f * neighbor_offset[2];
                }
            }
        }

        auto det = A.determinant();
        det = det == 0.0 ? 1.0 : det;

        Eigen::Matrix3d gradient_1;
        gradient_1.col(0) = b;
        gradient_1.col(1) = A.row(1);
        gradient_1.col(2) = A.row(2);

        Eigen::Matrix3d gradient_2;
        gradient_2.col(0) = A.row(0);
        gradient_2.col(1) = b;
        gradient_2.col(2) = A.row(2);

        Eigen::Matrix3d gradient_3;
        gradient_3.col(0) = A.row(0);
        gradient_3.col(1) = A.row(1);
        gradient_3.col(2) = b;

        Eigen::Vector3d c_gradient(
            gradient_1.determinant() / det,
            gradient_2.determinant() / det,
            gradient_3.determinant() / det
        );

        if (data.components() == 1)
        {
            gradient = c_gradient;
        }
        else
        {
            gradient.row(c) = c_gradient;
        }
    }

    if (twoD)
    {
        if (data.components() == 1)
        {
            gradient(2) = 0.0;
        }
        else
        {
            gradient(2, 0) = 0.0;
            gradient(2, 1) = 0.0;
            gradient(0, 2) = 0.0;
            gradient(1, 2) = 0.0;
            gradient(2, 2) = 1.0;
        }
    }

    return gradient;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient(const grid& data,
    const std::array<int, 3>& coords, const gradient_method_t method, const int kernel_size)
{
    switch (method)
    {
    case gradient_method_t::differences:
        return gradient_differences(data, coords);

        break;
    case gradient_method_t::least_squares:
        return gradient_least_squares(data, coords, kernel_size);

        break;
    default:
        std::cerr << "Unknown gradient computation method." << std::endl;
        return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>();
    }
}

vtkSmartPointer<vtkDoubleArray> gradient_field(const grid& data, const gradient_method_t method, const int kernel_size)
{
    const auto dim_x = data.dimensions()[0];
    const auto dim_y = data.dimensions()[1];
    const auto dim_z = data.dimensions()[2];

    const auto dim = static_cast<vtkIdType>(dim_x) * dim_y * dim_z;

    std::size_t index = 0;

    // First derivative
    auto field = vtkSmartPointer<vtkDoubleArray>::New();
    field->SetNumberOfComponents(data.components() == 1 ? 3 : 9);
    field->SetNumberOfTuples(dim);

    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                const Eigen::MatrixXd derivative = gradient(data, { x, y, z }, method, kernel_size);

                field->SetTuple(index, derivative.data());

                ++index;
            }
        }
    }

    return field;
}
