#include "feature_flow_separation.h"

#include "vtkAlgorithm.h"
#include "vtkDoubleArray.h"
#include "vtkIdList.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkUnstructuredGrid.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>

vtkStandardNewMacro(feature_flow_separation);

feature_flow_separation::feature_flow_separation()
{
    this->SetNumberOfInputPorts(2);
    this->SetNumberOfOutputPorts(1);
}

feature_flow_separation::~feature_flow_separation()
{
}

int feature_flow_separation::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    return 1;
}

int feature_flow_separation::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGrid");
        return 1;
    }

    return 0;
}

int feature_flow_separation::RequestData(vtkInformation*, vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Get input feature line and grid containting the vector field
    auto input_feature = vtkPolyData::SafeDownCast(input_vector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
    auto input_grid = vtkUnstructuredGrid::SafeDownCast(input_vector[1]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

    auto vector_field = GetInputArrayToProcess(0, input_grid);

    if (vector_field == nullptr)
    {
        std::cerr << "No input vector field provided." << std::endl;
        return 0;
    }

    // Get output
    auto output_grid = vtkUnstructuredGrid::SafeDownCast(output_vector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
    output_grid->DeepCopy(input_grid);

    // Separate vector field...
    // ... get first line and take its endpoints for reconstruction of the straight line
    auto point_ids = vtkSmartPointer<vtkIdList>::New();

    input_feature->BuildCells();
    input_feature->GetCellPoints(0, point_ids);

    Eigen::Vector3d feature_start, feature_end;

    input_feature->GetPoint(point_ids->GetId(0), feature_start.data());
    input_feature->GetPoint(point_ids->GetId(point_ids->GetNumberOfIds() - 1), feature_end.data());

    // ... define transformation into "feature space"
    const Eigen::Vector3d translation = -feature_start;
    const Eigen::Vector3d feature_direction = (feature_end - feature_start).normalized();

    Eigen::Vector3d first_basis, second_basis;

    if (feature_direction[0] == 0.0)
    {
        first_basis[0] = 0.0;
        first_basis[1] = -feature_direction[2];
        first_basis[2] = feature_direction[1];
    }
    else if (feature_direction[1] == 0.0)
    {
        first_basis[0] = -feature_direction[2];
        first_basis[1] = 0.0;
        first_basis[2] = feature_direction[0];
    }
    else
    {
        first_basis[0] = -feature_direction[1];
        first_basis[1] = feature_direction[0];
        first_basis[2] = 0.0;
    }

    first_basis.normalize();
    second_basis = feature_direction.cross(first_basis).normalized();

    Eigen::Matrix3d rotation;
    rotation << first_basis, second_basis, feature_direction;

    // ... separate into divergent and rotational part
    auto vector_field_part = vtkSmartPointer<vtkDoubleArray>::New();
    vector_field_part->DeepCopy(vector_field);
    vector_field_part->SetName("Vector Field Part");

    Eigen::Vector3d vector, position;
    vector.setZero();

    for (vtkIdType i = 0; i < vector_field->GetNumberOfTuples(); ++i)
    {
        vector_field->GetTuple(i, vector.data());
        vector = rotation * vector;

        input_grid->GetPoint(i, position.data());
        position += translation;
        position = rotation * position;

        const Eigen::Vector2d divergent_direction = position.head(2).normalized();

        const Eigen::Vector2d divergent_part = vector.head(2).dot(divergent_direction) * divergent_direction;
        const Eigen::Vector2d rotational_part = vector.head(2) - divergent_part;

        if (this->VectorPart == 0)
        {
            vector = Eigen::Vector3d(rotational_part[0], rotational_part[1], vector[2]);
        }
        else
        {
            vector = Eigen::Vector3d(divergent_part[0], divergent_part[1], vector[2]);
        }

        if (this->Transformed == 0)
        {
            vector = rotation.inverse() * vector;
        }

        vector_field_part->SetTuple(i, vector.data());
    }

    output_grid->GetPointData()->AddArray(vector_field_part);

    // ... if requested, keep grid transformed
    if (this->Transformed == 1)
    {
        for (vtkIdType i = 0; i < input_grid->GetNumberOfPoints(); ++i)
        {
            input_grid->GetPoint(i, position.data());

            position += translation;
            position = rotation * position;

            output_grid->GetPoints()->SetPoint(i, position.data());
        }
    }

    return 1;
}
