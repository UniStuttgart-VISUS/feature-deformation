#include "twisting.h"

#include "gradient.h"
#include "grid.h"

#include "common/math.h"

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

    const auto twoD = this->vector_field->GetDimensions()[2] == 1;

    this->rotations.resize(this->line.size());
    this->coordinate_systems.resize(this->line.size());

    // For each point, calculate the eigenvectors
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> coordinate_systems(this->line.size());

    const auto velocities = this->vector_field->GetPointData()->GetArray("velocity");
    const auto deformation = this->vector_field->GetPointData()->GetArray("jacobian");

    const grid velocity_grid(this->vector_field, velocities, deformation);

    auto jacobians = gradient_field(velocity_grid, gradient_method_t::least_squares);

    std::array<double, 3> temp_point{}, p_coords{};
    std::array<double, 8> weights{};
    int sub_id{};

    std::size_t index = 0;
    vtkCell* previous_cell = nullptr;
    vtkCell* cell = nullptr;

    std::size_t non_twisted = 0;

    for (const auto& point : this->line)
    {
        temp_point = { static_cast<double>(point[0]), static_cast<double>(point[1]), static_cast<double>(point[2]) };

        // Interpolate Jacobian
        cell = this->vector_field->FindAndGetCell(temp_point.data(), previous_cell, 0, 0.0, sub_id, p_coords.data(), weights.data());
        auto point_ids = (cell != nullptr ? cell : previous_cell)->GetPointIds(); // TODO: handling of outside cells
        previous_cell = (cell != nullptr ? cell : previous_cell);

        if (cell == nullptr && previous_cell == nullptr)
        {
            return false;
        }

        Eigen::Matrix3d jacobian, summed_jacobian;
        summed_jacobian.setZero();

        for (vtkIdType i = 0; i < point_ids->GetNumberOfIds(); ++i)
        {
            jacobians->GetTuple(point_ids->GetId(i), jacobian.data());

            summed_jacobian += weights[i] * jacobian;
        }

        jacobian = summed_jacobian;

        // Extract eigenvectors
        Eigen::EigenSolver<Eigen::Matrix3d> eigensolver(jacobian, true);

        std::array<Eigen::Vector3d, 3> eigenvectors;
        std::array<double, 3> eigenvalues{};
        std::array<bool, 3> is_real{};
        std::size_t num_real = 0;

        for (std::size_t i = 0; i < (twoD ? 2 : 3); ++i)
        {
            eigenvectors[i] = eigensolver.eigenvectors().real().col(i).normalized();
            eigenvalues[i] = eigensolver.eigenvalues().real()[i];
            is_real[i] = eigensolver.eigenvalues().imag()[i] == 0.0;

            if (is_real[i])
            {
                ++num_real;
            }
        }

        if (num_real > 1)
        {
            // Ignore eigenvector (most) parallel to feature line
            if (is_real[2])
            {
                const auto inv_90_angle_1 = std::abs(direction.dot(eigenvectors[0].normalized()));
                const auto inv_90_angle_2 = std::abs(direction.dot(eigenvectors[1].normalized()));
                const auto inv_90_angle_3 = std::abs(direction.dot(eigenvectors[2].normalized()));

                if (inv_90_angle_1 > inv_90_angle_2 && inv_90_angle_1 > inv_90_angle_3)
                {
                    std::swap(eigenvectors[0], eigenvectors[2]);
                    std::swap(eigenvalues[0], eigenvalues[2]);
                }
                else if (inv_90_angle_2 > inv_90_angle_3)
                {
                    std::swap(eigenvectors[1], eigenvectors[2]);
                    std::swap(eigenvalues[1], eigenvalues[2]);
                }
            }

            // Project eigenvectors onto the plane orthogonal to the feature line tangent
            coordinate_systems[index].first = eigenvectors[0] - eigenvectors[0].dot(direction) * direction;
            coordinate_systems[index].second = eigenvectors[1] - eigenvectors[1].dot(direction) * direction;

            coordinate_systems[index].first.normalize();
            coordinate_systems[index].second.normalize();
        }
        else if (index > 0)
        {
            coordinate_systems[index].first = coordinate_systems[index - 1].first;
            coordinate_systems[index].second = coordinate_systems[index - 1].second;

            ++non_twisted;
        }
        else
        {
            std::cerr << "ERROR: First set of eigenvectors are non-real." << std::endl;
            non_twisted = coordinate_systems.size(); // TODO
        }

        // Set debug output
        this->coordinate_systems[index].setZero();

        ++index;
    }

    std::cout << "Number of non-real eigenvector pairs: " << non_twisted << " / " << coordinate_systems.size() << std::endl;

    if (true /*non_twisted < 0.1 * coordinate_systems.size()*/) // TODO
    {
        // Sort eigenvectors such that the change is minimal
        for (std::size_t i = 0; i < coordinate_systems.size() - 1; ++i)
        {
            const auto inv_90_angle_1 = std::abs(coordinate_systems[i].first.dot(coordinate_systems[i + 1].first));
            const auto inv_90_angle_2 = std::abs(coordinate_systems[i].first.dot(coordinate_systems[i + 1].second));

            if (inv_90_angle_1 < inv_90_angle_2)
            {
                std::swap(coordinate_systems[i + 1].first, coordinate_systems[i + 1].second);
            }
        }

        // Invert eigenvectors such that the change is minimal
        for (std::size_t i = 0; i < coordinate_systems.size() - 1; ++i)
        {
            const auto inv_180_angle_1 = coordinate_systems[i].first.dot(coordinate_systems[i + 1].first);
            const auto inv_180_angle_2 = coordinate_systems[i].second.dot(coordinate_systems[i + 1].second);

            if (inv_180_angle_1 < 0)
            {
                coordinate_systems[i + 1].first *= -1.0;
            }
            if (inv_180_angle_2 < 0)
            {
                coordinate_systems[i + 1].second *= -1.0;
            }
        }

        // Set debug output
        /*for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            this->coordinate_systems[i].col(0) = coordinate_systems[i].first;
            this->coordinate_systems[i].col(1) = coordinate_systems[i].second;
            this->coordinate_systems[i].col(2) = direction;
        }*/

        // Beginning at the end of the line and moving forward, adjust the back-most coordinate systems
        // to match the one to their front
        const auto system_index = 0; // TODO: parameter

        auto rotate = [&direction](Eigen::Vector3d vector, float angle) -> Eigen::Vector3d
        {
            return vector * std::cos(angle) + direction.cross(vector) * std::sin(angle) + direction * direction.dot(vector) * (1.0 - std::cos(angle));
        };

        for (std::size_t i = coordinate_systems.size() - 1; i > 0; --i)
        {
            const auto& current = system_index == 0 ? coordinate_systems[i].first : coordinate_systems[i].second;
            const auto& comparison = system_index == 0 ? coordinate_systems[i - 1].first : coordinate_systems[i - 1].second;

            auto angle = std::acos(current.dot(comparison));

            // Adjust winding direction
            const auto positive = rotate(current, angle);
            const auto negative = rotate(current, -angle);

            const auto inv_180_angle_positive = positive.dot(comparison);
            const auto inv_180_angle_negative = negative.dot(comparison);

            if (inv_180_angle_positive < inv_180_angle_negative)
            {
                angle *= -1.0;
            }

            // Set rotation
            this->rotations[i] = static_cast<float>(angle);

            for (std::size_t j = i + 1; j < coordinate_systems.size(); ++j)
            {
                this->rotations[j] += static_cast<float>(angle);
            }
        }

        this->rotations[0] = 0.0f;

        for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            this->rotations[i] = fmodf(this->rotations[i], static_cast<float>(2.0 * pi));
        }

        // Set debug output
        for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            this->coordinate_systems[i].col(0) = rotate(coordinate_systems[i].first, this->rotations[i]);
            this->coordinate_systems[i].col(1) = rotate(coordinate_systems[i].second, this->rotations[i]);
            this->coordinate_systems[i].col(2) = direction;
        }
    }

    return true;
}

const std::pair<std::vector<float>, std::vector<Eigen::Matrix3d>> twisting::get_rotations() const
{
    return std::make_pair(this->rotations, this->coordinate_systems);
}
