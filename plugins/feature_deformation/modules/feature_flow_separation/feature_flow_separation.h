#pragma once

#include "vtkUnstructuredGridAlgorithm.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"

class VTK_EXPORT feature_flow_separation : public vtkUnstructuredGridAlgorithm
{
public:
    static feature_flow_separation* New();
    vtkTypeMacro(feature_flow_separation, vtkUnstructuredGridAlgorithm);

    vtkGetMacro(VectorPart, int);
    vtkSetMacro(VectorPart, int);

    vtkGetMacro(Transformed, int);
    vtkSetMacro(Transformed, int);

protected:
    feature_flow_separation();
    ~feature_flow_separation();

    virtual int FillInputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    feature_flow_separation(const feature_flow_separation&);
    void operator=(const feature_flow_separation&);

    int VectorPart;
    int Transformed;
};
