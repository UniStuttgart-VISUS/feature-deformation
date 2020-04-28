#pragma once

#include "vtkAlgorithm.h"

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
    void cache_parameter_lines();
    void cache_parameter_smoothing(double time);
    void cache_parameter_displacement();
    void cache_parameter_precompute();
    void cache_parameter_output_grid();

    bool parameter_checks() const;

    /// Get input
    void cache_input_grid(vtkInformationVector* input_grid_vector);
    void cache_input_lines(vtkInformationVector* input_lines_vector);
    void cache_input_geometry(vtkInformationVector* input_geometry_vector);

    /// Get positions and displacements
    std::pair<std::vector<std::array<float, 4>>, std::vector<std::array<float, 4>>> get_displacements(
        const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& displacement) const;

    /// Create and manipulate grid
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
        const std::array<int, 3>& dimension, const Eigen::Vector3f& origin, const Eigen::Vector3f& spacing) const;

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
    int GaussSubdivisions;
    int ComputeTearing;

    /// Output options
    int OutputBSplineDistance;
    int OutputDeformedGrid;
    int OutputVectorField;
    int OutputResampledGrid;
    int RemoveCells;
    double RemoveCellsScalar;

    // Caching
    struct cache_t
    {
        cache_t() : modified(true), hash(-1), valid(false) {};

        bool modified;
        uint32_t hash;

        bool valid;
    };

    // Parameter cache
    struct cache_parameter_line_t : public cache_t
    {
        int selected_line_id;

        smoothing::method_t method;

    } parameter_lines;

    struct cache_parameter_smoothing_t : public cache_t
    {
        smoothing::variant_t variant;

        float lambda;
        float mu;

        int num_iterations;
        int max_num_iterations;

        bool modified_time;

    } parameter_smoothing;

    struct cache_parameter_displacement_t : public cache_t
    {
        cuda::displacement::method_t method;
        cuda::displacement::parameter_t parameters;

        cuda::displacement::inverse_distance_weighting_parameters_t idw_parameters;
        cuda::displacement::projection_parameters_t projection_parameters;
        cuda::displacement::b_spline_parameters_t bspline_parameters;

    } parameter_displacement;

    struct cache_parameter_precompute_t : public cache_t
    {
        bool compute_gauss;
        bool compute_tearing;

    } parameter_precompute;

    struct cache_parameter_output_grid_t : public cache_t
    {
        bool output_deformed_grid;
        bool output_vector_field;
        bool output_resampled_grid;

        bool remove_cells;
        float remove_cells_scalar;

    } parameter_output_grid;

    // Input caches
    struct cache_input_grid_t : public cache_t
    {
        vtkImageData* grid;

        struct cache_input_data_t : public cache_t
        {
            vtkDataArray* data;

        } input_data;

        std::array<int, 6> extent;
        std::array<int, 3> dimension;
        Eigen::Vector3f origin;
        Eigen::Vector3f spacing;

    } input_grid;

    struct cache_input_lines_t : public cache_t
    {
        std::vector<std::array<float, 3>> lines;
        std::vector<Eigen::Vector3f> selected_line;

    } input_lines;

    struct cache_input_geometry_t : public cache_t
    {
        std::vector<std::array<float, 3>> geometry;

    } input_geometry;

    // Precomputation caches
    struct cache_precompute_tearing_t : public cache_t
    {
        vtkSmartPointer<vtkIdTypeArray> removed_cells;

    } precompute_tearing;

    // Results caches
    struct cache_results_smoothing_t : public cache_t
    {
        std::vector<std::array<float, 4>> positions;
        std::vector<std::array<float, 4>> displacements;

    } results_smoothing;

    struct cache_results_displacement_t : public cache_t
    {
        std::shared_ptr<cuda::displacement> displacement;

    } results_grid_displacement, results_line_displacement, results_geometry_displacement;

    // Output caches
    struct cache_output_lines_t : public cache_t
    {
        vtkSmartPointer<vtkPolyData> data;

    } output_lines;

    struct cache_output_geometry_t : public cache_t
    {
        vtkSmartPointer<vtkMultiBlockDataSet> data;

    } output_geometry;

    // Variable for counting the number of output frames
    int frames;
};
