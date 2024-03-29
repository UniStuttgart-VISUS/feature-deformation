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
#include <map>
#include <utility>
#include <vector>

twisting::twisting(std::vector<Eigen::Vector3d> line, vtkSmartPointer<vtkStructuredGrid> vector_field, const int selected_eigenvector) :
    line(line), vector_field(vector_field), selected_eigenvector(selected_eigenvector)
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

    const auto& dimension = velocity_grid.dimensions();
    const auto factor_x = 1uLL;
    const auto factor_y = factor_x * dimension[0];
    const auto factor_z = factor_y * dimension[1];

    std::map<std::size_t, Eigen::Matrix3d> jacobians;

    std::array<double, 3> temp_point{}, p_coords{};
    std::array<double, 8> weights{};
    int sub_id{};

    std::size_t index = 0;
    vtkCell* previous_cell = nullptr;
    vtkCell* cell = nullptr;

    std::vector<std::size_t> non_twisted;

    for (const auto& point : this->line)
    {
        // Interpolate Jacobian
        temp_point = { static_cast<double>(point[0]), static_cast<double>(point[1]), static_cast<double>(point[2]) };

        cell = this->vector_field->FindAndGetCell(temp_point.data(), previous_cell, 0, 0.0, sub_id, p_coords.data(), weights.data());

        if (cell != nullptr)
        {
            auto point_ids = cell->GetPointIds();

            Eigen::Matrix3d jacobian, summed_jacobian;
            summed_jacobian.setZero();

            for (vtkIdType i = 0; i < point_ids->GetNumberOfIds(); ++i)
            {
                const auto id = point_ids->GetId(i);

                if (jacobians.find(id) != jacobians.end())
                {
                    jacobian = jacobians.at(id);
                }
                else
                {
                    const auto z = static_cast<int>(id / factor_z);
                    const auto rest_z = id % factor_z;

                    const auto y = static_cast<int>(rest_z / factor_y);
                    const auto rest_y = rest_z % factor_y;

                    const auto x = static_cast<int>(rest_y / factor_x);

                    jacobian = jacobians[id] = gradient_least_squares(velocity_grid, { x, y, z });
                }

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
            else
            {
                non_twisted.push_back(index);
            }
        }
        else
        {
            non_twisted.push_back(index);
        }

        // Set debug output
        this->coordinate_systems[index].setZero();
        this->rotations[index] = 0.0f;

        previous_cell = cell;
        ++index;
    }

    if (non_twisted.size() > 0)
    {
        std::cout << "Number of non-real eigenvector pairs: " << non_twisted.size() << " / " << coordinate_systems.size() << std::endl;
    }

    if (non_twisted.size() < 0.3 * coordinate_systems.size())
    {
        // Interpolate at non-twisted line points...
        std::size_t begin = 0uLL, end = non_twisted.size() - 1;

        // ... using same value at the beginning
        if (non_twisted.front() == 0uLL)
        {
            auto index = non_twisted.front();

            while (non_twisted[index + 1] - non_twisted[index] == 1)
            {
                ++index;
            }

            for (std::size_t i = non_twisted.front(); i <= index; ++i)
            {
                coordinate_systems[i].first = coordinate_systems[index + 1].first;
                coordinate_systems[i].second = coordinate_systems[index + 1].second;
            }

            begin = index + 1;
        }

        // ... using same value at the end
        if (non_twisted.back() == coordinate_systems.size() - 1)
        {
            auto index = non_twisted.back();
            auto index_arr = non_twisted.size() - 1;

            while (non_twisted[index_arr] - non_twisted[index_arr - 1] == 1)
            {
                --index;
                --index_arr;
            }

            for (std::size_t i = non_twisted.back(); i >= index; --i)
            {
                coordinate_systems[i].first = coordinate_systems[index - 1].first;
                coordinate_systems[i].second = coordinate_systems[index - 1].second;
            }

            end = index_arr - 1;
        }

        // ... linear interpolating in between
        for (auto index_arr = begin; index_arr <= end; ++index_arr)
        {
            const auto index_left = non_twisted[index_arr] - 1;

            while (non_twisted[index_arr + 1] - non_twisted[index_arr] == 1)
            {
                ++index_arr;
            }

            const auto index_right = non_twisted[index_arr] + 1;

            const auto distance = index_right - index_left;

            for (auto index = index_left + 1; index < index_right; ++index)
            {
                const auto weight = (index - index_left) / static_cast<double>(distance);

                coordinate_systems[index].first = (weight * coordinate_systems[index_right].first
                    + (1.0 - weight) * coordinate_systems[index_left].first).normalized();
                coordinate_systems[index].second = (weight * coordinate_systems[index_right].second
                    + (1.0 - weight) * coordinate_systems[index_left].second).normalized();
            }
        }

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
        for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            this->coordinate_systems[i].col(0) = coordinate_systems[i].first;
            this->coordinate_systems[i].col(1) = coordinate_systems[i].second;
        }

        // Calculate a representative vector, to which all others should be aligned
        Eigen::Vector3d comparison;
        comparison.setZero();

        for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            comparison += this->selected_eigenvector == 0 ? coordinate_systems[i].first : coordinate_systems[i].second;
        }

        if (comparison.isZero())
        {
            comparison = this->selected_eigenvector == 0 ? coordinate_systems[0].first : coordinate_systems[0].second;
        }
        else
        {
            comparison.normalize();
        }

        // Calculate rotation necessary to match the previously calculated representative vector
        auto rotate = [&direction](Eigen::Vector3d vector, float angle) -> Eigen::Vector3d
        {
            return vector * std::cos(angle) + direction.cross(vector) * std::sin(angle) + direction * direction.dot(vector) * (1.0 - std::cos(angle));
        };

        for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            const auto& current = this->selected_eigenvector == 0 ? coordinate_systems[i].first : coordinate_systems[i].second;

            auto angle = std::acos(std::max(std::min(current.dot(comparison), 1.0), -1.0));

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
        }

        // Set debug output
        for (std::size_t i = 0; i < coordinate_systems.size(); ++i)
        {
            this->coordinate_systems[i].col(2) = rotate(this->selected_eigenvector == 0
                ? coordinate_systems[i].first : coordinate_systems[i].second, this->rotations[i]);
        }
    }

    return true;
}

const std::pair<std::vector<float>, std::vector<Eigen::Matrix3d>> twisting::get_rotations() const
{
    return std::make_pair(this->rotations, this->coordinate_systems);
}
