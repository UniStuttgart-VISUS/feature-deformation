#pragma once

#include "vtkImageAlgorithm.h"

#include "../feature_deformation/curvature.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"

#include <array>
#include <vector>

class VTK_EXPORT fix_curvature_gradient : public vtkImageAlgorithm
{
public:
    static fix_curvature_gradient* New();
    vtkTypeMacro(fix_curvature_gradient, vtkImageAlgorithm);

    vtkGetMacro(NumSteps, int);
    vtkSetMacro(NumSteps, int);

    vtkGetMacro(StepSize, double);
    vtkSetMacro(StepSize, double);

    vtkGetMacro(Error, double);
    vtkSetMacro(Error, double);

protected:
    fix_curvature_gradient();
    ~fix_curvature_gradient();

    virtual int FillInputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    fix_curvature_gradient(const fix_curvature_gradient&);
    void operator=(const fix_curvature_gradient&);

    void compute_finite_differences(vtkImageData* original_grid, vtkDataArray* vector_field_original, vtkDataArray* vector_field_deformed);

    int solve(const grid& original_vector_field, vtkSmartPointer<vtkDoubleArray> vector_field,
        const curvature_and_torsion_t& original_curvature, curvature_and_torsion_t deformed_curvature, double error_max);

    double calculate_error(int index, int index_block, const curvature_and_torsion_t& original_curvature,
        const curvature_and_torsion_t& deformed_curvature) const;

    std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> calculate_error_field(
        const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature) const;

    inline void output_copy(vtkImageData* grid, vtkDataArray* field) const
    {
        auto field_out = vtkSmartPointer<vtkDoubleArray>::New();
        field_out->DeepCopy(field);

        grid->GetPointData()->AddArray(field_out);
    }

    template <typename... T>
    inline void output_copy(vtkImageData* grid, vtkDataArray* field, T... fields) const
    {
        output_copy(grid, field);
        output_copy(grid, fields...);
    }

    int NumSteps;
    double StepSize;
    double Error;

    std::uint32_t hash;
    std::vector<vtkSmartPointer<vtkImageData>> results;
};
