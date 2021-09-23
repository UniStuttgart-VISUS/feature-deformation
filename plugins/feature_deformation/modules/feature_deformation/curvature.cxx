#include "curvature.h"

#include "gradient.h"
#include "grid.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

#include "Eigen/Dense"

#include <cmath>
#include <utility>

curvature_and_torsion_t curvature_and_torsion(const grid& vector_field, vtkDataArray* jacobians)
{
    const auto dim_x = vector_field.dimensions()[0];
    const auto dim_y = vector_field.dimensions()[1];
    const auto dim_z = vector_field.dimensions()[2];

    Eigen::Matrix3d jacobian;
    std::size_t index = 0;

    // First derivative
    auto first_derivatives = vtkSmartPointer<vtkDoubleArray>::New();
    first_derivatives->SetNumberOfComponents(3);
    first_derivatives->SetNumberOfTuples(dim_x * dim_y * dim_z);

    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                jacobians->GetTuple(index, jacobian.data());

                const Eigen::Matrix3d derivative = gradient(vector_field, { x, y, z }, jacobian);
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
    second_derivatives->SetNumberOfComponents(3);
    second_derivatives->SetNumberOfTuples(dim_x * dim_y * dim_z);

    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                jacobians->GetTuple(index, jacobian.data());

                const Eigen::Matrix3d derivative = gradient(derivative_field, { x, y, z }, jacobian);
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
    curvature->SetNumberOfTuples(dim_x * dim_y * dim_z);

    auto curvature_vector = vtkSmartPointer<vtkDoubleArray>::New();
    curvature_vector->SetName("Curvature Vector");
    curvature_vector->SetNumberOfComponents(3);
    curvature_vector->SetNumberOfTuples(dim_x * dim_y * dim_z);

    auto torsion = vtkSmartPointer<vtkDoubleArray>::New();
    torsion->SetName("Torsion");
    torsion->SetNumberOfComponents(1);
    torsion->SetNumberOfTuples(dim_x * dim_y * dim_z);

    auto torsion_vector = vtkSmartPointer<vtkDoubleArray>::New();
    torsion_vector->SetName("Torsion Vector");
    torsion_vector->SetNumberOfComponents(3);
    torsion_vector->SetNumberOfTuples(dim_x * dim_y * dim_z);

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

                const Eigen::Vector3d curv = vector.cross(first_derivative);
                const Eigen::Vector3d curv_vec = (curv.norm() / std::pow(vector.norm(), 3.0)) * curv.cross(vector).normalized();

                const double tors = curv.dot(second_derivative) / curv.squaredNorm();
                const Eigen::Vector3d tors_vec = tors * curv.normalized();

                curvature->SetValue(index, curv.norm() / std::pow(vector.norm(), 3.0));
                curvature_vector->SetTuple(index, curv_vec.data());

                torsion->SetValue(index, tors);
                torsion_vector->SetTuple(index, tors_vec.data());

                ++index;
            }
        }
    }

    return curvature_and_torsion_t{ curvature, curvature_vector, torsion, torsion_vector };
}
