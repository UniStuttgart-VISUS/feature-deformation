#include "curvature.h"

#include "gradient.h"
#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <cmath>
#include <utility>

curvature_and_torsion_t curvature_and_torsion(const grid& vector_field,
    const gradient_method_t method, const int kernel_size, vtkDataArray* directions)
{
    const auto dim_x = vector_field.dimensions()[0];
    const auto dim_y = vector_field.dimensions()[1];
    const auto dim_z = vector_field.dimensions()[2];

    const auto dim = static_cast<vtkIdType>(dim_x) * dim_y * dim_z;

    std::size_t index = 0;

    // First derivative
    auto first_derivatives = vtkSmartPointer<vtkDoubleArray>::New();
    first_derivatives->SetName("First Derivative");
    first_derivatives->SetNumberOfComponents(3);
    first_derivatives->SetNumberOfTuples(dim);

    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                const Eigen::Matrix3d derivative = gradient(vector_field, { x, y, z }, method, kernel_size);
                const Eigen::Vector3d vector = vector_field.value({ x, y, z });

                const Eigen::Vector3d first_derivative = derivative * vector;

                first_derivatives->SetTuple(index, first_derivative.data());

                ++index;
            }
        }
    }

    // Second derivative
    const grid derivative_field(vector_field, first_derivatives);

    index = 0;

    auto second_derivatives = vtkSmartPointer<vtkDoubleArray>::New();
    second_derivatives->SetName("Second Derivative");
    second_derivatives->SetNumberOfComponents(3);
    second_derivatives->SetNumberOfTuples(dim);

    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                const Eigen::Matrix3d derivative = gradient(derivative_field, { x, y, z }, method, kernel_size);
                const Eigen::Vector3d vector = vector_field.value({ x, y, z });

                const Eigen::Vector3d second_derivative = derivative * vector;

                second_derivatives->SetTuple(index, second_derivative.data());

                ++index;
            }
        }
    }

    // Curvature and torsion
    index = 0;

    auto curvature = vtkSmartPointer<vtkDoubleArray>::New();
    curvature->SetName("Curvature");
    curvature->SetNumberOfComponents(1);
    curvature->SetNumberOfTuples(dim);

    auto curvature_vector = vtkSmartPointer<vtkDoubleArray>::New();
    curvature_vector->SetName("Curvature Vector");
    curvature_vector->SetNumberOfComponents(3);
    curvature_vector->SetNumberOfTuples(dim);

    auto torsion = vtkSmartPointer<vtkDoubleArray>::New();
    torsion->SetName("Torsion");
    torsion->SetNumberOfComponents(1);
    torsion->SetNumberOfTuples(dim);

    auto torsion_vector = vtkSmartPointer<vtkDoubleArray>::New();
    torsion_vector->SetName("Torsion Vector");
    torsion_vector->SetNumberOfComponents(3);
    torsion_vector->SetNumberOfTuples(dim);

    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                Eigen::Vector3d vector, first_derivative, second_derivative;

                vector = vector_field.value({ x, y, z });
                first_derivatives->GetTuple(index, first_derivative.data());
                second_derivatives->GetTuple(index, second_derivative.data());

                if (dim_z == 1)
                {
                    const double curv = vector.cross(first_derivative)[2] / std::pow(vector.norm(), 3.0);
                    const Eigen::Vector3d curv_vec = curv * Eigen::Vector3d(-vector[1], vector[0], 0.0).normalized();

                    curvature->SetValue(index, curv);
                    curvature_vector->SetTuple(index, curv_vec.data());

                    torsion->SetValue(index, 0.0);
                    torsion_vector->SetTuple3(index, 0.0, 0.0, 0.0);
                }
                else
                {
                    const auto cross = vector.cross(first_derivative);

                    const double curv = cross.norm() / std::pow(vector.norm(), 3.0);
                    const Eigen::Vector3d curv_vec = curv * first_derivative.cross(second_derivative.cross(first_derivative)).normalized();

                    curvature->SetValue(index, curv);
                    curvature_vector->SetTuple(index, curv_vec.data());

                    const double tors = (curv == 0.0) ? 0.0 : (cross.dot(second_derivative) / cross.squaredNorm());
                    const Eigen::Vector3d tors_vec = tors * cross.normalized();

                    torsion->SetValue(index, tors);
                    torsion_vector->SetTuple(index, tors_vec.data());
                }

                ++index;
            }
        }
    }

    // Compute curvature and torsion gradients
    const grid curvature_grid(vector_field, curvature);
    const grid curvature_vector_grid(vector_field, curvature_vector);
    const grid torsion_grid(vector_field, torsion);
    const grid torsion_vector_grid(vector_field, torsion_vector);

    auto curvature_gradient = gradient_field(curvature_grid, method, kernel_size);
    auto curvature_vector_gradient = gradient_field(curvature_vector_grid, method, kernel_size);
    auto torsion_gradient = gradient_field(torsion_grid, method, kernel_size);
    auto torsion_vector_gradient = gradient_field(torsion_vector_grid, method, kernel_size);

    curvature_gradient->SetName("Curvature Gradient");
    curvature_vector_gradient->SetName("Curvature Vector Gradient");
    torsion_gradient->SetName("Torsion Gradient");
    torsion_vector_gradient->SetName("Torsion Vector Gradient");

    // Compute directional curvature and torsion gradients
    vtkSmartPointer<vtkDoubleArray> curvature_directional_gradient = nullptr;
    vtkSmartPointer<vtkDoubleArray> torsion_directional_gradient = nullptr;

    if (directions != nullptr)
    {
        curvature_directional_gradient = vtkSmartPointer<vtkDoubleArray>::New();
        curvature_directional_gradient->SetName("Directional Curvature Gradient");
        curvature_directional_gradient->SetNumberOfComponents(3);
        curvature_directional_gradient->SetNumberOfTuples(dim);

        torsion_directional_gradient = vtkSmartPointer<vtkDoubleArray>::New();
        torsion_directional_gradient->SetName("Directional Torsion Gradient");
        torsion_directional_gradient->SetNumberOfComponents(3);
        torsion_directional_gradient->SetNumberOfTuples(dim);

        index = 0;

        for (int z = 0; z < dim_z; ++z)
        {
            for (int y = 0; y < dim_y; ++y)
            {
                for (int x = 0; x < dim_x; ++x)
                {
                    Eigen::Matrix3d curvature_derivative, torsion_derivative;
                    Eigen::Vector3d direction;

                    curvature_vector_gradient->GetTuple(index, curvature_derivative.data());
                    torsion_vector_gradient->GetTuple(index, torsion_derivative.data());
                    directions->GetTuple(index, direction.data());

                    direction.normalize();

                    const Eigen::Vector3d directional_curvature_derivative = curvature_derivative * direction;
                    const Eigen::Vector3d directional_torsion_derivative = torsion_derivative * direction;

                    curvature_directional_gradient->SetTuple(index, directional_curvature_derivative.data());
                    torsion_directional_gradient->SetTuple(index, directional_torsion_derivative.data());

                    ++index;
                }
            }
        }
    }

    // Return computed fields
    curvature_and_torsion_t ret;

    ret.first_derivative = first_derivatives;
    ret.second_derivative = second_derivatives;
    ret.curvature = curvature;
    ret.curvature_vector = curvature_vector;
    ret.curvature_gradient = curvature_gradient;
    ret.curvature_vector_gradient = curvature_vector_gradient;
    ret.curvature_directional_gradient = curvature_directional_gradient;
    ret.torsion = torsion;
    ret.torsion_vector = torsion_vector;
    ret.torsion_gradient = torsion_gradient;
    ret.torsion_vector_gradient = torsion_vector_gradient;
    ret.torsion_directional_gradient = torsion_directional_gradient;

    return ret;
}
