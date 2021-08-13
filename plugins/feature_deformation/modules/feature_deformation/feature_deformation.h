#pragma once

#include "vtkDataObjectAlgorithm.h"

#include "algorithm_compute_gauss.h"
#include "algorithm_compute_tearing.h"
#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_displacement_precomputation.h"
#include "algorithm_geometry_input.h"
#include "algorithm_grid_input.h"
#include "algorithm_grid_output_creation.h"
#include "algorithm_grid_output_set.h"
#include "algorithm_grid_output_update.h"
#include "algorithm_grid_output_vectorfield.h"
#include "algorithm_line_input.h"
#include "algorithm_geometry_output_creation.h"
#include "algorithm_geometry_output_set.h"
#include "algorithm_geometry_output_update.h"
#include "algorithm_line_output_creation.h"
#include "algorithm_line_output_set.h"
#include "algorithm_line_output_update.h"
#include "algorithm_smoothing.h"
#include "algorithm_vectorfield_input.h"

#include "displacement.h"
#include "smoothing.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"

#include <memory>

class VTK_EXPORT feature_deformation : public vtkDataObjectAlgorithm
{
public:
    static feature_deformation* New();
    vtkTypeMacro(feature_deformation, vtkDataObjectAlgorithm);

    vtkGetMacro(LineID, int);
    vtkSetMacro(LineID, int);

    vtkGetMacro(Method, int);
    vtkSetMacro(Method, int);

    vtkGetMacro(Variant, int);
    vtkSetMacro(Variant, int);

    vtkGetMacro(MaxNumIterations, int);
    vtkSetMacro(MaxNumIterations, int);

    vtkGetMacro(Lambda, double);
    vtkSetMacro(Lambda, double);

    vtkGetMacro(Interpolator, int);
    vtkSetMacro(Interpolator, int);

    vtkGetMacro(InterpolatorThreshold, double);
    vtkSetMacro(InterpolatorThreshold, double);

    vtkGetMacro(Exponent, double);
    vtkSetMacro(Exponent, double);

    vtkGetMacro(Duration, int);
    vtkSetMacro(Duration, int);

    vtkGetMacro(InterpolateSmoothingFactor, int);
    vtkSetMacro(InterpolateSmoothingFactor, int);

    vtkGetMacro(Inverse, int);
    vtkSetMacro(Inverse, int);

    vtkGetMacro(Weight, int);
    vtkSetMacro(Weight, int);

    vtkGetMacro(EpsilonScalar, double);
    vtkSetMacro(EpsilonScalar, double);

    vtkGetMacro(VoronoiDistance, int);
    vtkSetMacro(VoronoiDistance, int);

    vtkGetMacro(SplineDegree, int);
    vtkSetMacro(SplineDegree, int);

    vtkGetMacro(GaussParameter, double);
    vtkSetMacro(GaussParameter, double);

    vtkGetMacro(Subdivisions, int);
    vtkSetMacro(Subdivisions, int);

    vtkGetMacro(ComputeGauss, int);
    vtkSetMacro(ComputeGauss, int);

    vtkGetMacro(CheckHandedness, int);
    vtkSetMacro(CheckHandedness, int);

    vtkGetMacro(CheckConvexity, int);
    vtkSetMacro(CheckConvexity, int);

    vtkGetMacro(CheckVolume, int);
    vtkSetMacro(CheckVolume, int);

    vtkGetMacro(VolumePercentage, double);
    vtkSetMacro(VolumePercentage, double);

    vtkGetMacro(GaussSubdivisions, int);
    vtkSetMacro(GaussSubdivisions, int);

    vtkGetMacro(ComputeTearing, int);
    vtkSetMacro(ComputeTearing, int);

    vtkGetMacro(OutputBSplineDistance, int);
    vtkSetMacro(OutputBSplineDistance, int);

    vtkGetMacro(OutputDeformedGrid, int);
    vtkSetMacro(OutputDeformedGrid, int);

    vtkGetMacro(OutputVectorField, int);
    vtkSetMacro(OutputVectorField, int);

    vtkGetMacro(RemoveCells, int);
    vtkSetMacro(RemoveCells, int);

    vtkGetMacro(RemoveCellsScalar, double);
    vtkSetMacro(RemoveCellsScalar, double);

    vtkGetStringMacro(ParameterLog);
    vtkSetStringMacro(ParameterLog);

    vtkGetStringMacro(PerformanceLog);
    vtkSetStringMacro(PerformanceLog);

    vtkGetMacro(Quiet, int);
    vtkSetMacro(Quiet, int);

    void RemoveAllGeometryInputs();

protected:
    feature_deformation();
    ~feature_deformation();

    virtual int FillInputPortInformation(int, vtkInformation*) override;
    virtual int FillOutputPortInformation(int, vtkInformation*) override;

    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    feature_deformation(const feature_deformation&);
    void operator=(const feature_deformation&);

    struct cache_output_lines_t;
    struct cache_output_geometry_t;

    /// Process parameters
    void process_parameters(double time);

    /// ID of the polyline which defines the grid deformation
    int LineID;

    /// Method for deforming the polyline and corresponding parameters
    int Method;
    int Variant;
    int MaxNumIterations;
    double Lambda;

    /// Animation parameters
    int Interpolator;
    double InterpolatorThreshold;
    double Exponent;
    int Duration;
    int InterpolateSmoothingFactor;
    int Inverse;

    /// Weighting function for the displacement and corresponding parameters
    int Weight;
    double EpsilonScalar;
    int VoronoiDistance;
    int SplineDegree;
    double GaussParameter;
    int Subdivisions;

    /// Pre-computation options
    int ComputeGauss;
    int CheckHandedness;
    int CheckConvexity;
    int CheckVolume;
    double VolumePercentage;
    int GaussSubdivisions;
    int ComputeTearing;

    /// Output options
    int OutputBSplineDistance;
    int OutputDeformedGrid;
    int OutputVectorField;
    int RemoveCells;
    double RemoveCellsScalar;

    /// Parameters and performance log
    char* ParameterLog;
    char* PerformanceLog;

    /// Logging options
    int Quiet;

    // Processed parameters
    struct parameter_t
    {
        int selected_line_id;

        smoothing::method_t smoothing_method;
        smoothing::variant_t variant;
        float lambda;
        int num_iterations;
        int max_num_iterations;

        cuda::displacement::method_t displacement_method;
        cuda::displacement::parameter_t displacement_parameters;
        cuda::displacement::inverse_distance_weighting_parameters_t idw_parameters;
        cuda::displacement::projection_parameters_t projection_parameters;
        cuda::displacement::b_spline_parameters_t bspline_parameters;

        bool compute_gauss;
        bool check_handedness;
        bool check_convexity;
        bool check_volume;
        double volume_percentage;
        int num_subdivisions;
        bool compute_tearing;

        bool output_bspline_distance;
        bool output_deformed_grid;
        bool output_vector_field;
        bool remove_cells;
        float remove_cells_scalar;

    } parameters;

    // Input algorithms
    std::shared_ptr<algorithm_grid_input> alg_grid_input;
    std::shared_ptr<algorithm_line_input> alg_line_input;
    std::shared_ptr<algorithm_geometry_input> alg_geometry_input;
    std::shared_ptr<algorithm_vectorfield_input> alg_vectorfield_input;

    // Pre-computation algorithms
    std::shared_ptr<algorithm_compute_gauss> alg_compute_gauss;
    std::shared_ptr<algorithm_compute_tearing> alg_compute_tearing;

    // Computation algorithms
    std::shared_ptr<algorithm_smoothing> alg_smoothing;

    std::shared_ptr<algorithm_displacement_creation> alg_displacement_creation_lines,
        alg_displacement_creation_grid, alg_displacement_creation_geometry;
    std::shared_ptr<algorithm_displacement_precomputation> alg_displacement_precomputation_lines,
        alg_displacement_precomputation_grid, alg_displacement_precomputation_geometry;
    std::shared_ptr<algorithm_displacement_computation> alg_displacement_computation_lines,
        alg_displacement_computation_grid, alg_displacement_computation_geometry;

    // Output algorithms
    std::shared_ptr<algorithm_line_output_creation> alg_line_output_creation;
    std::shared_ptr<algorithm_line_output_update> alg_line_output_update;
    std::shared_ptr<algorithm_line_output_set> alg_line_output_set;

    std::shared_ptr<algorithm_grid_output_creation> alg_grid_output_creation;
    std::shared_ptr<algorithm_grid_output_set> alg_grid_output_set;
    std::shared_ptr<algorithm_grid_output_update> alg_grid_output_update;
    std::shared_ptr<algorithm_grid_output_vectorfield> alg_grid_output_vectorfield;

    std::shared_ptr<algorithm_geometry_output_creation> alg_geometry_output_creation;
    std::shared_ptr<algorithm_geometry_output_update> alg_geometry_output_update;
    std::shared_ptr<algorithm_geometry_output_set> alg_geometry_output_set;

    // Variable for counting the number of output frames
    int frames;
};
