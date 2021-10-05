#include "curvature.h"

#include "../feature_deformation/curvature.h"
#include "../feature_deformation/gradient.h"
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

    // Compute
    const grid vector_grid(vtk_grid, vector_field, jacobian_field);

    const auto curvature = curvature_and_torsion(vector_grid, gradient_method_t::least_squares, 1);

    // Set output
    auto output_grid = vtkStructuredGrid::SafeDownCast(output_vector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
    output_grid->ShallowCopy(vtk_grid);
    output_grid->GetPointData()->AddArray(curvature.curvature);
    output_grid->GetPointData()->AddArray(curvature.curvature_vector);
    output_grid->GetPointData()->AddArray(curvature.curvature_gradient);
    output_grid->GetPointData()->AddArray(curvature.torsion);
    output_grid->GetPointData()->AddArray(curvature.torsion_vector);
    output_grid->GetPointData()->AddArray(curvature.torsion_gradient);

    return 1;
}
