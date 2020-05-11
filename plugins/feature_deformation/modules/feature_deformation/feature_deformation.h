#pragma once

#include "vtkAlgorithm.h"

#include "algorithm_displacement_computation.h"
#include "algorithm_displacement_creation.h"
#include "algorithm_displacement_precomputation.h"
#include "algorithm_geometry_input.h"
#include "algorithm_grid_input.h"
#include "algorithm_line_input.h"
#include "algorithm_smoothing.h"

#include "displacement.h"
#include "smoothing.h"

#include "vtkDataArray.h"
#include "vtkIdTypeArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkPointSet.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkUnstructuredGrid.h"

#include "Eigen/Dense"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class VTK_EXPORT feature_deformation : public vtkAlgorithm
{
public:
    static feature_deformation* New();
    vtkTypeMacro(feature_deformation, vtkAlgorithm);

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

    vtkGetMacro(Duration, int);
    vtkSetMacro(Duration, int);

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

    vtkGetMacro(OutputResampledGrid, int);
    vtkSetMacro(OutputResampledGrid, int);

    vtkGetMacro(RemoveCells, int);
    vtkSetMacro(RemoveCells, int);

    vtkGetMacro(RemoveCellsScalar, double);
    vtkSetMacro(RemoveCellsScalar, double);

    void RemoveAllGeometryInputs();

protected:
    feature_deformation();
    ~feature_deformation();

    virtual int FillInputPortInformation(int, vtkInformation*) override;
    virtual int FillOutputPortInformation(int, vtkInformation*) override;

    virtual int ProcessRequest(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

    virtual int RequestDataObject(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
    virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
    virtual int RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
    virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

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
    int Duration;
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
    int OutputResampledGrid;
    int RemoveCells;
    double RemoveCellsScalar;

    // Processed parameters
    struct parameter_t
    {
        int selected_line_id;

        smoothing::method_t smoothing_method;
        smoothing::variant_t variant;
        float lambda;
        int num_iterations;

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

        bool output_deformed_grid;
        bool output_vector_field;
        bool output_resampled_grid;
        bool remove_cells;
        float remove_cells_scalar;

    } parameters;

    // Input algorithms
    std::shared_ptr<algorithm_grid_input> alg_grid_input;
    std::shared_ptr<algorithm_line_input> alg_line_input;
    std::shared_ptr<algorithm_geometry_input> alg_geometry_input;

    // Computation algorithms
    std::shared_ptr<algorithm_smoothing> alg_smoothing;

    std::shared_ptr<algorithm_displacement_creation> alg_displacement_creation_lines,
        alg_displacement_creation_grid, alg_displacement_creation_geometry;
    std::shared_ptr<algorithm_displacement_precomputation> alg_displacement_precomputation_lines,
        alg_displacement_precomputation_grid, alg_displacement_precomputation_geometry;
    std::shared_ptr<algorithm_displacement_computation> alg_displacement_computation_lines,
        alg_displacement_computation_grid, alg_displacement_computation_geometry;

    // Output algorithms





    // Variable for counting the number of output frames
    int frames;







    //// Precomputation caches
    //struct cache_precompute_tearing_t : public cache_t
    //{
    //    vtkSmartPointer<vtkIdTypeArray> removed_cells;

    //} precompute_tearing;

    //// Output caches
    //struct cache_output_lines_t : public cache_t
    //{
    //    vtkSmartPointer<vtkPolyData> data;

    //} output_lines;

    //struct cache_output_geometry_t : public cache_t
    //{
    //    vtkSmartPointer<vtkMultiBlockDataSet> data;

    //} output_geometry;



    /*/// Create and manipulate grid
    void create_undeformed_grid(vtkPointSet* output_deformed_grid, const std::array<int, 6>& extent,
        const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const;

    void create_cells(vtkUnstructuredGrid* output_deformed_grid, vtkUnstructuredGrid* output_deformed_grid_removed,
        const std::array<int, 3>& dimension, const Eigen::Vector3f& spacing) const;

    /// Set output
    void set_output_deformed_grid(vtkPointSet* output_deformed_grid, const cuda::displacement& grid_displacement) const;

    void set_output_deformed_lines(vtkPolyData* input_lines, vtkPolyData* output_deformed_lines, const cuda::displacement& line_displacement,
        bool modified, cache_output_lines_t& output_lines) const;

    void set_output_deformed_geometry(const std::vector<vtkPointSet*>& input_geometry, vtkMultiBlockDataSet* output_deformed_geometry,
        const cuda::displacement& geometry_displacement, bool modified, cache_output_geometry_t& output_geometry) const;

    /// Deform velocities using a displacement map
    void create_displacement_field(vtkPointSet* output_deformed_grid) const;

    void deform_velocities(vtkPointSet* output_deformed_grid, vtkDataArray* data_array,
        const std::array<int, 3>& dimension, const Eigen::Vector3f& spacing) const;

    /// Resample the deformed grid on the original one
    void resample_grid(vtkPointSet* output_deformed_grid, vtkImageData* output_resampled_grid, const std::string& velocity_name,
        const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const;*/
};
