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

    vtkGetMacro(ErrorDefinition, int);
    vtkSetMacro(ErrorDefinition, int);

    vtkGetMacro(NumSteps, int);
    vtkSetMacro(NumSteps, int);

    vtkGetMacro(StepSize, double);
    vtkSetMacro(StepSize, double);

    vtkGetMacro(StepSizeMethod, int);
    vtkSetMacro(StepSizeMethod, int);

    vtkGetMacro(StepSizeControl, int);
    vtkSetMacro(StepSizeControl, int);

    vtkGetMacro(Error, double);
    vtkSetMacro(Error, double);

    vtkGetMacro(Adjustment, double);
    vtkSetMacro(Adjustment, double);

    vtkGetMacro(MaxAdjustments, int);
    vtkSetMacro(MaxAdjustments, int);

    vtkGetMacro(Threshold, double);
    vtkSetMacro(Threshold, double);

    vtkGetMacro(Increase, int);
    vtkSetMacro(Increase, int);

    vtkGetMacro(Stop, int);
    vtkSetMacro(Stop, int);

    vtkGetMacro(GradientMethod, int);
    vtkSetMacro(GradientMethod, int);

    vtkGetMacro(GradientKernel, int);
    vtkSetMacro(GradientKernel, int);

protected:
    optimizer();
    ~optimizer();

    virtual int FillInputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    optimizer(const optimizer&);
    void operator=(const optimizer&);

    enum class error_definition_t
    {
        vector_difference, angle, length_difference
    };

    enum class step_size_method_t
    {
        normalized, norm, error
    };

    enum class step_size_control_t
    {
        dynamic, fixed
    };

    void compute(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
        vtkDataArray* vector_field_original);

    vtkSmartPointer<vtkDoubleArray> compute_gradient_descent(const std::array<int, 3>& dimension,
        const vtkStructuredGrid* original_grid, const vtkDataArray* vector_field_original, const vtkDataArray* positions,
        const vtkDataArray* errors, const curvature_and_torsion_t& original_curvature) const;

    std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> apply_gradient_descent(
        const std::array<int, 3>& dimension, double step_size, const vtkDataArray* positions,
        const vtkDataArray* errors, const vtkDataArray* gradient_descent) const;

    double calculate_error(int index, int index_block, const curvature_and_torsion_t& original_curvature,
        const curvature_and_torsion_t& deformed_curvature, const vtkDataArray* jacobian_field,
        error_definition_t error_definition) const;

    std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> calculate_error_field(
        const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature,
        const vtkDataArray* jacobian_field, error_definition_t error_definition) const;

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

    int ErrorDefinition;

    int NumSteps;
    double StepSize;
    int StepSizeMethod;
    int StepSizeControl;
    double Error;

    double Adjustment;
    int MaxAdjustments;
    double Threshold;
    int Increase;
    int Stop;

    int GradientMethod;
    int GradientKernel;

    std::uint32_t hash;
    std::vector<vtkSmartPointer<vtkStructuredGrid>> results;
};
