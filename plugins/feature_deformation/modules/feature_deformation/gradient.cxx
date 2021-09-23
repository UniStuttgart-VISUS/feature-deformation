#include "gradient.h"

#include "grid.h"

#include "Eigen/Dense"

#include <array>

Eigen::Matrix3d unit()
{
    Eigen::Matrix3d mat;
    mat.setZero();
    mat.diagonal().setOnes();

    return mat;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient(const grid& data,
    const std::array<int, 3>& coords, const Eigen::Matrix3d& jacobian)
{
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

                gradient_fwd += inverse_metric(i, j) * ((j == 0) ? diff_right : diff_top) * basis_vec.norm() * basis_vec.transpose();
                gradient_bck += inverse_metric(i, j) * ((j == 0) ? diff_left : diff_bottom) * basis_vec.norm() * basis_vec.transpose();
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

    if (data.components() == 1)
    {
        gradient_fwd.transpose();
        gradient_bck.transpose();
    }

    return 0.5 * (gradient_fwd + gradient_bck);
}
