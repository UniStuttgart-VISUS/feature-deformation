#pragma once

#include "vtkStructuredGridAlgorithm.h"

#include "../feature_deformation/curvature.h"

#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include <array>
#include <tuple>
#include <utility>
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

    vtkSmartPointer<vtkDoubleArray> compute_gradient_descent(const std::array<int, 3>& dimension,
        const vtkStructuredGrid* original_grid, const vtkDataArray* vector_field_original, const vtkDataArray* positions,
        const vtkDataArray* gradient_difference, const curvature_and_torsion_t& original_curvature) const;

    std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> apply_gradient_descent(
        const std::array<int, 3>& dimension, double step_size, const vtkDataArray* positions,
        const vtkDataArray* gradient_difference, const vtkDataArray* gradient_descent) const;

    std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> calculate_gradient_difference(
        const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature,
        const vtkDataArray* jacobian_field) const;

    vtkSmartPointer<vtkStructuredGrid> create_output(const std::array<int, 3>& dimension, const vtkDoubleArray* positions) const;

    inline void output_copy(vtkStructuredGrid* grid, vtkSmartPointer<vtkDoubleArray>& field) const
    {
        auto field_out = vtkSmartPointer<vtkDoubleArray>::New();
        field_out->DeepCopy(field);

        grid->GetPointData()->AddArray(field_out);
    }

    template <typename... T>
    inline void output_copy(vtkStructuredGrid* grid, vtkSmartPointer<vtkDoubleArray>& field, T... fields) const
    {
        output_copy(grid, field);
        output_copy(grid, fields...);
    }

    int NumSteps;
    double StepSize;
    double Error;

    std::uint32_t hash;
    std::vector<vtkSmartPointer<vtkStructuredGrid>> results;
};
