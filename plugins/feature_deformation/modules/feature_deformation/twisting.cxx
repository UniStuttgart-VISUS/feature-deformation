#include "twisting.h"

#include "gradient.h"
#include "grid.h"

#include "vtkCell.h"
#include "vtkDataArray.h"
#include "vtkIdList.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include "Eigen/Dense"

#include <array>
#include <cmath>
#include <utility>
#include <vector>

twisting::twisting(std::vector<Eigen::Vector3d> line, vtkSmartPointer<vtkStructuredGrid> vector_field) :
    line(line), vector_field(vector_field)
{
}

bool twisting::run()
{
    // Calculate for every point of the polyline the rotation necessary to adjust the spanned coordinate
    // system by the eigenvectors on the orthogonal plane to the feature line. To this end, use the
    // spanned coordinate system of the first point on the line as reference.

    // Check straightness of the line and output results as info/warning
    const auto& start = this->line.front();
    const auto& end = this->line.back();

    const auto direction = (end - start).normalized();

    // TODO: sanity check

    const auto twoD = this->vector_field->GetDimensions()[2] == 1;

    this->rotations.resize(this->line.size());
    this->coordinate_systems.resize(this->line.size());

    // For each point, calculate the eigenvectors on the orthogonal plane
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> coordinate_systems(this->line.size());

    auto jacobians = gradient_field(grid(this->vector_field,
        this->vector_field->GetPointData()->GetArray("velocity")), gradient_method_t::least_squares);

    auto jacobian_field = vtkSmartPointer<vtkStructuredGrid>::New();
    jacobian_field->CopyStructure(this->vector_field);
    jacobian_field->GetPointData()->AddArray(jacobians);

    std::array<double, 3> temp_point{}, p_coords{};
    std::array<double, 8> weights{};
    int sub_id{};

    std::size_t index = 0;
    vtkCell* previous_cell = nullptr;
    vtkCell* cell = nullptr;

    bool is_twisted = true;

    for (const auto& point : this->line)
    {
        temp_point = { static_cast<double>(point[0]), static_cast<double>(point[1]), static_cast<double>(point[2]) };

        // Interpolate Jacobian
        cell = jacobian_field->FindAndGetCell(temp_point.data(), previous_cell, 0, 0.0, sub_id, p_coords.data(), weights.data());
        auto point_ids = (cell != nullptr ? cell : previous_cell)->GetPointIds(); // TODO: handling of outside cells
        previous_cell = (cell != nullptr ? cell : previous_cell);

        if (cell == nullptr && previous_cell == nullptr)
        {
            return false;
        }

        Eigen::Matrix3d jacobian, summed_jacobian;
        summed_jacobian.setZero();

        double summed_weight = 0.0;

        for (vtkIdType i = 0; i < point_ids->GetNumberOfIds(); ++i)
        {
            jacobians->GetTuple(i, jacobian.data());

            summed_jacobian += weights[i] * jacobian;
            summed_weight += weights[i];
        }

        jacobian = summed_jacobian / summed_weight;

        // Extract eigenvectors
        Eigen::EigenSolver<Eigen::Matrix3d> eigensolver(jacobian, true);

        std::array<Eigen::Vector3d, 3> eigenvectors;
        std::array<double, 3> eigenvalues{};
        std::array<bool, 3> is_real{};
        std::size_t num_real = 0;

        for (std::size_t i = 0; i < (twoD ? 2 : 3); ++i)
        {
            eigenvectors[i] = eigensolver.eigenvectors().real().col(i);
            eigenvalues[i] = eigensolver.eigenvalues().real()[i];
            is_real[i] = eigensolver.eigenvalues().imag()[i] == 0.0;

            if (is_real[i])
            {
                this->coordinate_systems[index](i * 3 + 0) = eigenvectors[i][0];
                this->coordinate_systems[index](i * 3 + 1) = eigenvectors[i][1];
                this->coordinate_systems[index](i * 3 + 2) = eigenvectors[i][2];

                ++num_real;
            }
            else
            {
                this->coordinate_systems[index](i * 3 + 0) = 0.0;
                this->coordinate_systems[index](i * 3 + 1) = 0.0;
                this->coordinate_systems[index](i * 3 + 2) = 0.0;
            }
        }

        // Filter eigenvectors...
        if (num_real > 1)
        {
            // ... only use real eigenvalues
            if (!is_real[0])
            {
                eigenvectors[0] = eigenvectors[2];
                eigenvalues[0] = eigenvalues[2];
                is_real[0] = is_real[2];
            }
            else if (!is_real[1])
            {
                eigenvectors[1] = eigenvectors[2];
                eigenvalues[1] = eigenvalues[2];
                is_real[1] = is_real[2];
            }

            // ... ignore eigenvector parallel to feature line
            if (is_real[2])
            {
                const auto angle_1 = std::abs(direction.dot(eigenvectors[0].normalized()));
                const auto angle_2 = std::abs(direction.dot(eigenvectors[1].normalized()));
                const auto angle_3 = std::abs(direction.dot(eigenvectors[2].normalized()));

                if (angle_1 < angle_2 && angle_1 < angle_3)
                {
                    eigenvectors[0] = eigenvectors[2];
                    eigenvalues[0] = eigenvalues[2];
                }
                else if (angle_2 < angle_3)
                {
                    eigenvectors[1] = eigenvectors[2];
                    eigenvalues[1] = eigenvalues[2];
                }
            }

            coordinate_systems[index].first = eigenvectors[0];
            coordinate_systems[index].second = eigenvectors[1];
        }
        else
        {
            is_twisted = false;
        }

        ++index;
    }

    // Beginning at the end of the line and moving forward, adjust the back-most coordinate systems
    // to match the one to their front
    if (is_twisted)
    {
        // TODO
    }

    return true;
}

const std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Matrix3d>> twisting::get_rotations() const
{
    return std::make_pair(this->rotations, this->coordinate_systems);
}
