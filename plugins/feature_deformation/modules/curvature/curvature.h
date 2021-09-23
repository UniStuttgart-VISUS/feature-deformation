#pragma once

#include "vtkStructuredGridAlgorithm.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"

class VTK_EXPORT curvature : public vtkStructuredGridAlgorithm
{
public:
    static curvature* New();
    vtkTypeMacro(curvature, vtkStructuredGridAlgorithm);

protected:
    curvature();
    ~curvature();

    virtual int FillInputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    curvature(const curvature&);
    void operator=(const curvature&);
};
