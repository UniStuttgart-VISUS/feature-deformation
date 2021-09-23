#include "curvature.h"

#include "../feature_deformation/curvature.h"
#include "../feature_deformation/grid.h"

#include "vtkAlgorithm.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include <array>
#include <iostream>

vtkStandardNewMacro(curvature);

curvature::curvature()
{
    this->SetNumberOfInputPorts(1);
    this->SetNumberOfOutputPorts(1);
}

curvature::~curvature()
{
}

int curvature::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    return 1;
}

int curvature::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkStructuredGrid");
        return 1;
    }

    return 0;
}

int curvature::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Get input grid
    auto vtk_grid = vtkStructuredGrid::SafeDownCast(input_vector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

    auto vector_field = GetInputArrayToProcess(0, vtk_grid);
    auto jacobian_field = GetInputArrayToProcess(1, vtk_grid);

    if (vector_field == nullptr)
    {
        std::cerr << "No input vector field provided." << std::endl;
        return 0;
    }

    if (jacobian_field == nullptr)
    {
        std::cout << "No Jacobian field provided, assuming no deformation." << std::endl;

        jacobian_field = vtkDoubleArray::New();
        jacobian_field->SetNumberOfComponents(9);
        jacobian_field->SetNumberOfTuples(vector_field->GetNumberOfTuples());

        const std::array<double, 9> unit_matrix{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

        for (vtkIdType i = 0; i < jacobian_field->GetNumberOfTuples(); ++i)
        {
            jacobian_field->SetTuple(i, unit_matrix.data());
        }
    }

    std::array<int, 3> dimensions;
    vtk_grid->GetDimensions(dimensions.data());

    // Get positions
    auto positions = vtkSmartPointer<vtkDoubleArray>::New();
    positions->SetNumberOfComponents(3);
    positions->SetNumberOfTuples(vtk_grid->GetNumberOfPoints());

    std::array<double, 3> point;

    for (vtkIdType i = 0; i < vtk_grid->GetNumberOfPoints(); ++i)
    {
        vtk_grid->GetPoint(i, point.data());
        positions->SetTuple(i, point.data());
    }

    // Compute
    const grid vector_grid(dimensions, positions, vector_field);

    const auto curvature = curvature_and_torsion(vector_grid, jacobian_field);

    // Set output
    auto output_grid = vtkStructuredGrid::SafeDownCast(output_vector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
    output_grid->ShallowCopy(vtk_grid);
    output_grid->GetPointData()->AddArray(curvature.curvature);
    output_grid->GetPointData()->AddArray(curvature.curvature_vector);
    output_grid->GetPointData()->AddArray(curvature.torsion);

    // TODO: Destroy jacobian_field if necessary :X

    return 1;
}
