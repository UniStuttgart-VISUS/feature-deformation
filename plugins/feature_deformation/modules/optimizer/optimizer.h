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

    vtkGetMacro(Method, int);
    vtkSetMacro(Method, int);

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

    vtkGetMacro(StepSizeMin, double);
    vtkSetMacro(StepSizeMin, double);

    vtkGetMacro(StepSizeMax, double);
    vtkSetMacro(StepSizeMax, double);

    vtkGetMacro(LineSearchSteps, double);
    vtkSetMacro(LineSearchSteps, double);

    vtkGetMacro(GradientMethod, int);
    vtkSetMacro(GradientMethod, int);

    vtkGetMacro(GradientKernel, int);
    vtkSetMacro(GradientKernel, int);

    vtkGetMacro(GradientStep, double);
    vtkSetMacro(GradientStep, double);

    vtkGetMacro(CheckWolfe, int);
    vtkSetMacro(CheckWolfe, int);

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

    enum class error_definition_t
    {
        vector_difference, angle, length_difference
    };

    enum class method_t
    {
        gradient, nonlinear_conjugate
    };

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
        vtkDataArray* vector_field_original);

    std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> compute_descent(
        const std::array<int, 3>& dimension, const vtkStructuredGrid* original_grid,
        const vtkDataArray* vector_field_original, const vtkDataArray* positions,
        const vtkDataArray* errors, const curvature_and_torsion_t& original_curvature,
        const vtkDataArray* previous_gradient_descent) const;

    std::pair<vtkSmartPointer<vtkDoubleArray>, vtkSmartPointer<vtkDoubleArray>> apply_descent(
        const std::array<int, 3>& dimension, double step_size, const vtkDataArray* positions,
        const vtkDataArray* errors, const vtkDataArray* gradient_descent) const;

    double calculate_error(int index, int index_block, const curvature_and_torsion_t& original_curvature,
        const curvature_and_torsion_t& deformed_curvature, const vtkDataArray* jacobian_field,
        error_definition_t error_definition) const;

    std::tuple<vtkSmartPointer<vtkDoubleArray>, double, double> calculate_error_field(
        const curvature_and_torsion_t& original_curvature, const curvature_and_torsion_t& deformed_curvature,
        const vtkDataArray* jacobian_field, error_definition_t error_definition) const;

    bool satisfies_armijo(double old_value, double new_value, const Eigen::VectorXd& direction,
        const Eigen::VectorXd& gradient, double step_size, double constant) const;

    bool satisfies_wolfe_curv(const Eigen::VectorXd& direction, const Eigen::VectorXd& old_gradient,
        const Eigen::VectorXd& new_gradient, double constant) const;

    vtkSmartPointer<vtkStructuredGrid> create_output(const std::array<int, 3>& dimension, const vtkDoubleArray* positions) const;

    inline void output_copy(vtkStructuredGrid* grid, vtkDataArray* field) const
    {
        auto field_out = vtkSmartPointer<vtkDoubleArray>::New();
        field_out->DeepCopy(field);

        grid->GetPointData()->AddArray(field_out);
    }

    template <typename... T>
    inline void output_copy(vtkStructuredGrid* grid, vtkDataArray* field, T... fields) const
    {
        output_copy(grid, field);
        output_copy(grid, fields...);
    }

    int ErrorDefinition;
    int Method;

    int NumSteps;
    double StepSize;
    int StepSizeMethod;
    int StepSizeControl;
    double Error;

    double StepSizeMin;
    double StepSizeMax;
    int LineSearchSteps;

    int GradientMethod;
    int GradientKernel;
    double GradientStep;
    int CheckWolfe;

    int CSVOutput;

    std::uint32_t hash;
    std::vector<vtkSmartPointer<vtkStructuredGrid>> results;
};
