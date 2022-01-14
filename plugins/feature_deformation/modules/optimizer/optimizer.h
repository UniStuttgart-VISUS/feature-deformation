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

    vtkGetMacro(IgnoreBorder, int);
    vtkSetMacro(IgnoreBorder, int);

    vtkGetMacro(NumSteps, int);
    vtkSetMacro(NumSteps, int);

    vtkGetMacro(StepSize, double);
    vtkSetMacro(StepSize, double);

    vtkGetMacro(StepSizeMethod, int);
    vtkSetMacro(StepSizeMethod, int);

    vtkGetMacro(Error, double);
    vtkSetMacro(Error, double);

    vtkGetMacro(AbortWhenGrowing, int);
    vtkSetMacro(AbortWhenGrowing, int);

    vtkGetMacro(GradientMethod, int);
    vtkSetMacro(GradientMethod, int);

    vtkGetMacro(GradientKernel, int);
    vtkSetMacro(GradientKernel, int);

    vtkGetMacro(GradientStep, double);
    vtkSetMacro(GradientStep, double);

    vtkGetMacro(CSVOutput, int);
    vtkSetMacro(CSVOutput, int);

protected:
    optimizer();
    ~optimizer();

    virtual int FillInputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    optimizer(const optimizer&);
    void operator=(const optimizer&);

    enum class step_size_method_t
    {
        normalized, norm, error
    };

    enum class step_size_control_t
    {
        dynamic, fixed
    };

    struct step_size_t
    {
        double step_size, error_avg, error_max;

        bool operator==(const step_size_t& rhs) const { return this->step_size == rhs.step_size; }
    };

    void compute_gradient_descent(vtkStructuredGrid* original_grid, vtkStructuredGrid* deformed_grid,
        vtkDataArray* vector_field_original, vtkDataArray* original_feature_mapping, vtkDataArray* feature_mapping);

    std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> compute_descent(
        const std::array<int, 3>& dimension, const vtkDataArray* vector_field,
        const curvature_and_torsion_t& original_curvature, const vtkDataArray* jacobian_field,
        const vtkDataArray* positions, const vtkDataArray* errors, vtkDoubleArray* derivative_direction) const;

    std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> apply_descent(
        const std::array<int, 3>& dimension, double step_size, const vtkDataArray* positions,
        const vtkDataArray* errors, const vtkDataArray* gradient_descent) const;

    curvature_and_torsion_t blockwise_curvature(const std::array<int, 3>& dimension,
        double rotation, const vtkDataArray* positions, const vtkDataArray* vector_field,
        const vtkDataArray* jacobian_field, const vtkDataArray* derivative_direction) const;

    double calculate_error(int index, int index_block, const curvature_and_torsion_t& original_curvature,
        const curvature_and_torsion_t& deformed_curvature, const vtkDataArray* jacobian_field) const;

    std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> calculate_error_field(
        const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature,
        const vtkDataArray* jacobian_field) const;

    vtkSmartPointer<vtkStructuredGrid> create_output(const std::array<int, 3>& dimension, const vtkDoubleArray* positions) const;

    void output_info(double original_error_avg, double original_error_max, double error_avg, double error_max,
        double min_error_avg, double min_error_max, int min_error_avg_step, int min_error_max_step,
        const std::vector<double>& errors_avg, const std::vector<double>& errors_max) const;

    inline void output_copy(vtkStructuredGrid* grid, vtkDataArray* field) const
    {
        if (field != nullptr)
        {
            auto field_out = vtkSmartPointer<vtkDoubleArray>::New();
            field_out->DeepCopy(field);

            grid->GetPointData()->AddArray(field_out);
        }
    }

    inline void output_copy(vtkStructuredGrid* grid, const curvature_and_torsion_t& field) const
    {
        output_copy(grid, field.first_derivative);
        output_copy(grid, field.second_derivative);
        output_copy(grid, field.curvature);
        output_copy(grid, field.curvature_vector);
        output_copy(grid, field.curvature_gradient);
        output_copy(grid, field.curvature_vector_gradient);
        output_copy(grid, field.curvature_directional_gradient);
        output_copy(grid, field.torsion);
        output_copy(grid, field.torsion_vector);
        output_copy(grid, field.torsion_gradient);
        output_copy(grid, field.torsion_vector_gradient);
        output_copy(grid, field.torsion_directional_gradient);
    }

    template <typename... T>
    inline void output_copy(vtkStructuredGrid* grid, vtkDataArray* field, T... fields) const
    {
        output_copy(grid, field);
        output_copy(grid, fields...);
    }

    template <typename... T>
    inline void output_copy(vtkStructuredGrid* grid, const curvature_and_torsion_t& field, T... fields) const
    {
        output_copy(grid, field);
        output_copy(grid, fields...);
    }

    int IgnoreBorder;

    int NumSteps;
    double StepSize;
    int StepSizeMethod;
    double Error;
    int AbortWhenGrowing;

    int GradientMethod;
    int GradientKernel;
    double GradientStep;

    int CSVOutput;

    std::uint32_t hash;
    std::vector<vtkSmartPointer<vtkStructuredGrid>> results;
};
