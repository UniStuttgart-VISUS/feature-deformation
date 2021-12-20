#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <tuple>
#include <vector>

namespace cuda
{
    class displacement
    {
    public:
        /// Method
        enum class method_t
        {
            greedy = 0,
            greedy_joints = 5,
            voronoi = 1,
            projection = 2,
            b_spline = 3,
            b_spline_joints = 4
        };

        /// Different sets of parameters, depending on the method
        struct inverse_distance_weighting_parameters_t
        {
            float exponent;
            int neighborhood;
        };
        struct projection_parameters_t
        {
            float gauss_parameter;
        };
        struct b_spline_parameters_t
        {
            int degree;
            float gauss_parameter;
            int iterations;
        };

        union parameter_t
        {
            inverse_distance_weighting_parameters_t inverse_distance_weighting;
            projection_parameters_t projection;
            b_spline_parameters_t b_spline;
        };

        /// Get original points
        displacement(std::vector<std::array<float, 3>> points);

        /// Free resources
        virtual ~displacement();

        /// Perform precomputation for B-Splines
        void precompute(parameter_t parameters, const std::vector<std::array<float, 4>>& positions);

        /// Assess the quality of the results for B-Splines
        void assess_quality(parameter_t parameters, const std::vector<std::array<float, 4>>& positions,
            const std::vector<std::array<float, 4>>& displacements);

        /// Displace all points using interpolation
        void displace(method_t method, parameter_t parameters, const std::vector<std::array<float, 4>>& positions,
            const std::vector<std::array<float, 4>>& displacements);

        /// Return displaced results
        const std::vector<std::array<float, 3>>& get_results() const;

        /// Return displacement IDs
        const std::tuple<std::vector<float4>, std::vector<float3>, std::vector<float3>> get_displacement_info() const;

    private:
        /// Upload points to the GPU
        void upload_points();

        /// Create knot vector
        std::vector<float> create_knot_vector(int degree, const std::vector<std::array<float, 4>>& positions) const;

        /// Displace positions
        std::vector<std::array<float, 4>> displace_positions(const std::vector<std::array<float, 4>>& positions,
            const std::vector<std::array<float, 4>>& displacements) const;

        /// Compute derivative of the B-spline
        std::vector<std::array<float, 4>> compute_derivative(const std::vector<std::array<float, 4>>& positions,
            typename std::vector<float>::const_iterator knot_begin, int degree) const;

        /// (Displaced) points
        mutable std::vector<std::array<float, 3>> points;

        /// ID of the nearest displacement position
        mutable std::vector<float4> displacement_info;
        mutable std::vector<float3> mapping, mapping_orig;

        /// CUDA resources
        float3* cuda_res_input_points;
        float3* cuda_res_output_points;
        float4* cuda_res_info;
        float3* cuda_res_mapping_point;
        float3* cuda_res_mapping_tangent;
        float3* cuda_res_mapping_direction;
        float3* cuda_res_mapping_direction_orig;
        float* cuda_res_mapping_arc_position;
        float* cuda_res_mapping_arc_position_displaced;
    };
}
