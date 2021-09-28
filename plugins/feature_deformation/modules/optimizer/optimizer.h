#pragma once

#include "vtkStructuredGridAlgorithm.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include <vector>

class VTK_EXPORT optimizer : public vtkStructuredGridAlgorithm
{
public:
    static optimizer* New();
    vtkTypeMacro(optimizer, vtkStructuredGridAlgorithm);

    vtkGetMacro(NumSteps, int);
    vtkSetMacro(NumSteps, int);

    vtkGetMacro(StepSize, double);
    vtkSetMacro(StepSize, double);

    vtkGetMacro(Error, double);
    vtkSetMacro(Error, double);

protected:
    optimizer();
    ~optimizer();

    virtual int FillInputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    optimizer(const optimizer&);
    void operator=(const optimizer&);

    void compute(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
        vtkDataArray* vector_field_original, vtkSmartPointer<vtkDoubleArray> vector_field_deformed,
        vtkSmartPointer<vtkDoubleArray> jacobian_field);

    int NumSteps;
    double StepSize;
    double Error;

    std::uint32_t hash;
    std::vector<vtkSmartPointer<vtkStructuredGrid>> results;
};
