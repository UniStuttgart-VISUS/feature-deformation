#include "displacement.h"

#include "b-spline.h"

#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <vector_types.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#define __pi 3.1415926535897932384626433833

/// Textures:
/// 0: positions (de Boor points)
#define __positions__ 0
/// 1: first B-spline derivative
#define __first_derivative__ 1
/// 2: angles for twisting
#define __twisting_angle__ 2
/// 3: displacements
#define __displacements__ 3
/// 4: knot vector
#define __knot_vector__ 4
/// 5: displaced positions (displaced de Boor points)
#define __displaced_positions__ 5
/// 6: first displaced B-spline derivative
#define __displaced_derivative__ 6
/// 7: second B-spline derivative
#define __second_derivative__ 7
__constant__ cudaTextureObject_t textures[8];


#define __kernel__parameters__ <<< num_blocks, num_threads >>>
#define __get_kernel__parameters__ const int tid = threadIdx.x; const int gid = blockIdx.x * blockDim.x + tid;


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CUDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


/// Device math and functionality
namespace
{
    inline __host__ __device__ int maxi(int a, int b)
    {
        return (a > b) ? a : b;
    }

    inline __host__ __device__ int mini(int a, int b)
    {
        return (a < b) ? a : b;
    }

    inline __host__ __device__ float3 operator+(float3 v, float3 w)
    {
        return float3{ v.x + w.x, v.y + w.y, v.z + w.z };
    }

    inline __host__ __device__ float3 operator-(float3 v, float3 w)
    {
        return float3{ v.x - w.x, v.y - w.y, v.z - w.z };
    }

    inline __host__ __device__ float3 operator*(float a, float3 v)
    {
        return float3{ a * v.x, a * v.y, a * v.z };
    }

    inline __host__ __device__ float3 operator/(float3 v, float a)
    {
        return float3{ v.x / a, v.y / a, v.z / a };
    }

    inline __host__ __device__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline __host__ __device__ float3 cross(float3 a, float3 b)
    {
        return float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    inline __host__ __device__ float length(float3 v)
    {
        return sqrt(dot(v, v));
    }

    inline __host__ __device__ float3 normalize(float3 v)
    {
        return v / length(v);
    }

    inline __host__ __device__ float project_point_onto_line(const float3& point, const float3& lp_1, const float3& lp_2)
    {
        const auto numerator =
            ((point.x - lp_1.x) * (lp_2.x - lp_1.x)) +
            ((point.y - lp_1.y) * (lp_2.y - lp_1.y)) +
            ((point.z - lp_1.z) * (lp_2.z - lp_1.z));

        const auto denominator =
            powf(lp_2.x - lp_1.x, 2.0f) +
            powf(lp_2.y - lp_1.y, 2.0f) +
            powf(lp_2.z - lp_1.z, 2.0f);

        return numerator / denominator;
    }

    inline __device__ float3 fetch(cudaTextureObject_t texture, int index)
    {
        const auto fetched = tex1Dfetch<float4>(texture, index);
        return float3{ fetched.x, fetched.y, fetched.z };
    }

    inline __device__ float fetch1D(cudaTextureObject_t texture, int index)
    {
        return tex1Dfetch<float>(texture, index);
    }
}

/// B-Spline basis function
__device__
float basis_function(const float u, int degree, int de_boor_index, const int derivative)
{
    // Get knot vector
    const auto knot_vector_0 = fetch1D(textures[__knot_vector__], derivative + de_boor_index);
    const auto knot_vector_1 = fetch1D(textures[__knot_vector__], derivative + de_boor_index + 1);
    const auto knot_vector_degree_0 = fetch1D(textures[__knot_vector__], derivative + de_boor_index + degree);
    const auto knot_vector_degree_1 = fetch1D(textures[__knot_vector__], derivative + de_boor_index + degree + 1);

    // 1 if u_i <= u < u_i+1, 0 otherwise
    if (degree == 0)
    {
        return (knot_vector_0 <= u && u < knot_vector_1) ? 1.0f : 0.0f;
    }

    // Calculate recursively
    const auto Ni = basis_function(u, degree - 1, de_boor_index, derivative);
    const auto Nip1 = basis_function(u, degree - 1, de_boor_index + 1, derivative);

    const auto part_1 = (knot_vector_degree_0 - knot_vector_0 == 0.0f) ? 0.0f : ((u - knot_vector_0) / (knot_vector_degree_0 - knot_vector_0));
    const auto part_2 = (knot_vector_degree_1 - knot_vector_1 == 0.0f) ? 0.0f : ((knot_vector_degree_1 - u) / (knot_vector_degree_1 - knot_vector_1));

    return part_1 * Ni + part_2 * Nip1;
}

/// Compute point on the B-Spline
__device__
float3 compute_point(float u, int degree, const int num_de_boor_points, const int texture_index, const int derivative = 0)
{
    float3 point{ 0.0f, 0.0f, 0.0f };

    // Handle right boundary
    const auto u_max = fetch1D(textures[__knot_vector__], num_de_boor_points);

    if (u >= u_max)
    {
        u = u_max - 0.00001f;
    }

    // Compute point
    degree -= derivative;

    for (std::size_t j = floorf(u); j <= floorf(u) + degree; ++j)
    {
        const auto N = basis_function(u, degree, j, derivative);
        const auto de_boor_point = fetch(textures[texture_index], j);

        point = point + N * de_boor_point;
    }

    return point;
}

/// Compute rotation
__device__
float3 rotate(float3 point, float3 axis, float angle)
{
    return dot(axis, point) * axis + cross(cosf(angle) * cross(axis, point), axis) + sinf(angle) * cross(axis, point);
}

__device__
float4 compute_rotation(float3 source_vec, float3 target_vec, const bool adjust_sign = true)
{
    if (length(source_vec - target_vec) < 0.0001f)
    {
        return float4{ 1.0f, 0.0f, 0.0f, 0.0f };
    }

    const auto axis = normalize(cross(source_vec, target_vec));
    const auto angle = acosf(fmaxf(fminf(dot(source_vec, target_vec), 1.0f), -1.0f));

    if (adjust_sign)
    {
        const auto diff_pos_angle = length(rotate(source_vec, axis, angle) - target_vec);
        const auto diff_neg_angle = length(rotate(source_vec, axis, -angle) - target_vec);

        if (diff_pos_angle < diff_neg_angle)
        {
            return float4{ axis.x, axis.y, axis.z, angle };
        }
        else
        {
            return float4{ axis.x, axis.y, axis.z, -angle };
        }
    }
    else
    {
        return float4{ axis.x, axis.y, axis.z, angle };
    }
}

__device__
float3 compute_rotation(float3 point, int index)
{
    const auto origin_start = fetch(textures[__positions__], index);
    const auto origin_end = fetch(textures[__positions__], index + 1);
    const auto origin_vector = normalize(origin_end - origin_start);

    const auto target_start = origin_start + fetch(textures[__displacements__], index);
    const auto target_end = origin_end + fetch(textures[__displacements__], index + 1);
    const auto target_vector = normalize(target_end - target_start);

    const auto axis = cross(origin_vector, target_vector);
    const auto angle = acosf(dot(origin_vector, target_vector));

    const auto position = point - 0.5f * (origin_start + origin_end);
    const auto rotated = cosf(angle) * position + sinf(angle) * cross(axis, position) + (1.0f - cosf(angle)) * dot(axis, position) * axis;

    return rotated - position;
}

/**
* Precompute the mapping of points onto the B-Spline
*
* @param in_points              Point positions
* @param point_mapping          Corresponding point on the B-Spline
* @param tangent_mapping        Corresponding tangent on the B-Spline
* @param arc_position_mapping   Corresponding arc position on the B-Spline
* @param num_points             Number of points
* @param num_displacements      Number of de Boor points
* @param iterations             Number of iterations to find the corresponding point on the B-spline
* @param degree                 B-Spline degree
*/
__global__
void precompute_mapping_kernel(const float3* in_points, float3* point_mapping, float3* tangent_mapping, float* arc_position_mapping,
    const int num_points, const int num_displacements, const int iterations, const int degree)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get point
        float3 point = in_points[gid];

        // Find parameter for which the distance from the point to the B-Spline, defined by the input positions, is minimal
        // Try to find a local minimum by subdividing the segments
        float min_arc_position = 0.0f;
        float min_distance = CUDART_NORM_HUGE_F;

        const float delta = fetch1D(textures[__knot_vector__], degree + 1) - fetch1D(textures[__knot_vector__], degree);

        for (int index = degree; index < num_displacements; ++index)
        {
            // Start at the center of the segment
            auto u_left = fetch1D(textures[__knot_vector__], index);
            auto u_right = u_left + delta;
            auto u = u_left + 0.5f * delta;

            // Loop a few times
            bool good_match = false;

            for (int i = 0; i < iterations && !good_match; ++i)
            {
                // The subdivision plane is defined by the position and the tangent at u
                const auto position = compute_point(u, degree, num_displacements, __positions__);
                const auto tangent = compute_point(u, degree, num_displacements, __first_derivative__, 1);

                const auto direction = dot(tangent, point - position);

                if (fabsf(direction) < 0.00001f)
                {
                    good_match = true;
                }
                else if (direction > 0.0f)
                {
                    // In front of the plane
                    u_left = u;
                }
                else
                {
                    // Behind the plane
                    u_right = u;
                }

                u = 0.5f * (u_left + u_right);
            }

            // Calculate distance and set new if its is smaller
            const auto position = compute_point(u, degree, num_displacements, __positions__);
            const auto distance = length(point - position);

            if (distance < min_distance)
            {
                min_arc_position = u;
                min_distance = distance;
            }
        }

        // Get point and tangent of original B-spline at the arc position
        const auto origin = compute_point(min_arc_position, degree, num_displacements, __positions__);
        const auto tangent = normalize(compute_point(min_arc_position, degree, num_displacements, __first_derivative__, 1));

        // Store results
        point_mapping[gid] = origin;
        tangent_mapping[gid] = tangent;
        arc_position_mapping[gid] = min_arc_position;
    }
}

/**
* Assess quality by mapping of displaced points onto the straightened B-Spline
*
* @param in_points              Point positions
* @param infos                  Output arc position on the B-Spline to infos
* @param num_points             Number of points
* @param num_displacements      Number of de Boor points
* @param iterations             Number of iterations to find the corresponding point on the B-spline
* @param degree                 B-Spline degree
*/
__global__
void assess_mapping_kernel(const float3* in_points, float4* infos, const int num_points,
    const int num_displacements, const int iterations, const int degree)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get point
        float3 point = in_points[gid];
        float4 info = infos[gid];

        // Find parameter for which the distance from the point to the B-Spline, defined by the input positions, is minimal
        // Try to find a local minimum by subdividing the segments
        float min_arc_position = 0.0f;
        float min_distance = CUDART_NORM_HUGE_F;

        const float delta = fetch1D(textures[__knot_vector__], degree + 1) - fetch1D(textures[__knot_vector__], degree);

        for (int index = degree; index < num_displacements; ++index)
        {
            // Start at the center of the segment
            auto u_left = fetch1D(textures[__knot_vector__], index);
            auto u_right = u_left + delta;
            auto u = u_left + 0.5f * delta;

            // Loop a few times
            bool good_match = false;

            for (int i = 0; i < iterations && !good_match; ++i)
            {
                // The subdivision plane is defined by the position and the tangent at u
                const auto position = compute_point(u, degree, num_displacements, __positions__);
                const auto tangent = compute_point(u, degree, num_displacements, __first_derivative__, 1);

                const auto direction = dot(tangent, point - position);

                if (fabsf(direction) < 0.00001f)
                {
                    good_match = true;
                }
                else if (direction > 0.0f)
                {
                    // In front of the plane
                    u_left = u;
                }
                else
                {
                    // Behind the plane
                    u_right = u;
                }

                u = 0.5f * (u_left + u_right);
            }

            // Calculate distance and set new if its is smaller
            const auto position = compute_point(u, degree, num_displacements, __positions__);
            const auto distance = length(point - position);

            if (distance < min_distance)
            {
                min_arc_position = u;
                min_distance = distance;
            }
        }

        info.x = min_arc_position;
        info.y = fabsf(info.x - info.w);

        // Store results
        infos[gid] = info;
    }
}

/**
* Interpolate displacement using inverse distance weighting and move points accordingly
*
* @param in_points              Points to displace
* @param out_points             Displaced points
* @param infos                  Information about the displacement
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
* @param idw_exponent           Exponent for inverse distance weighting
* @param neighborhood_kernel    Voronoi neighborhood
*/
__global__
void displacement_kernel_idw(const float3* in_points, float3* out_points, float4* infos,
    const int num_points, const int num_displacements, const float idw_exponent, int neighborhood_kernel)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get point
        float3 point = in_points[gid];
        float4 info = infos[gid];

        // Get nearest displacement position
        int nearest_index = 0;
        float nearest_distance = length(point - fetch(textures[__positions__], 0));

        for (int i = 1; i < num_displacements; ++i)
        {
            const float distance = length(point - fetch(textures[__positions__], i));

            if (distance < nearest_distance)
            {
                nearest_index = i;
                nearest_distance = distance;
            }
        }

        // Count changes of Voronoi cells and store current one
        if (info.x != nearest_index)
        {
            info.y += 1.0f;
        }

        info.x = nearest_index;

        // Initialize displacement and weight
        float3 displacement = { 0.0f, 0.0f, 0.0f };
        float weights = 0.0f;

        // Check if points coincide
        if (nearest_distance < 0.0001f)
        {
            displacement = fetch(textures[__displacements__], nearest_index);
            weights = 1.0f;
        }
        else
        {
            // Add weighted displacement of the adjacent neighbors, if they exist
            const int lower_bound = maxi(0, nearest_index - neighborhood_kernel);
            const int upper_bound = mini(num_displacements - 1, nearest_index + neighborhood_kernel);

            for (int i = lower_bound; i <= upper_bound; ++i)
            {
                const auto position = fetch(textures[__positions__], i);
                const auto vector = fetch(textures[__displacements__], i);

                const auto weight_neighbor = 1.0f / powf(length(point - position), idw_exponent);

                displacement = displacement + weight_neighbor * vector;
                weights += weight_neighbor;
            }

            info.z = length(displacement);
            info.w = weights;
        }

        // Normalize
        if (weights > 0.0001f)
        {
            point = point + (1.0f / weights) * displacement;
        }

        // Store result
        out_points[gid] = point;
        infos[gid] = info;
    }
}

/**
* Interpolate displacement using inverse distance weighting for handles and joints and move points accordingly
*
* @param in_points              Points to displace
* @param out_points             Displaced points
* @param infos                  Information about the displacement
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
* @param idw_exponent           Exponent for inverse distance weighting
*/
__global__
void displacement_kernel_idw_joints(const float3* in_points, float3* out_points, float4* infos,
    const int num_points, const int num_displacements, const float idw_exponent)
{
    __get_kernel__parameters__

        if (gid < num_points)
        {
            // Get point
            float3 point = in_points[gid];
            float4 info = infos[gid];

            // Get nearest displacement position
            int nearest_index = 0;
            float nearest_distance = length(point - fetch(textures[__positions__], 0));

            for (int i = 1; i < num_displacements; ++i)
            {
                const float distance = length(point - fetch(textures[__positions__], i));

                if (distance < nearest_distance)
                {
                    nearest_index = i;
                    nearest_distance = distance;
                }
            }

            // Calculate translation
            float3 displacement = { 0.0f, 0.0f, 0.0f };
            float displacement_weights = 0.0f;

            if (nearest_distance < 0.0001f)
            {
                displacement = fetch(textures[__displacements__], nearest_index);
                displacement_weights = 1.0f;
            }
            else
            {
                // Add weighted displacement of the adjacent neighbors, if they exist
                for (int i = 0; i < num_displacements; ++i)
                {
                    const auto position = fetch(textures[__positions__], i);
                    const auto vector = fetch(textures[__displacements__], i);

                    const auto weight_neighbor = 1.0f / powf(length(point - position), idw_exponent);

                    displacement = displacement + weight_neighbor * vector;
                    displacement_weights += weight_neighbor;
                }

                info.z = length(displacement);
                info.w = displacement_weights;
            }

            // Normalize translation
            if (displacement_weights > 0.0001f)
            {
                point = point + (displacement / displacement_weights);
            }

            // Get nearest rotation position
            int nearest_joint_index = 0;
            float nearest_joint_distance = length(point - 0.5f * (fetch(textures[__positions__], 0) + fetch(textures[__positions__], 1)));

            for (int i = 1; i < num_displacements - 1; ++i)
            {
                const float distance = length(point - 0.5f * (fetch(textures[__positions__], i) + fetch(textures[__positions__], i + 1)));

                if (distance < nearest_joint_distance)
                {
                    nearest_joint_index = i;
                    nearest_joint_distance = distance;
                }
            }

            // Calculate rotation
            float3 rotation = { 0.0f, 0.0f, 0.0f };
            float rotation_weights = 0.0f;

            if (nearest_joint_distance < 0.0001f)
            {
                rotation = compute_rotation(point, nearest_joint_index);
                rotation_weights = 1.0f;
            }
            else
            {
                // Add weighted rotation
                for (int i = 0; i < num_displacements - 1; ++i)
                {
                    const auto position_start = fetch(textures[__positions__], i);
                    const auto position_end = fetch(textures[__positions__], i + 1);

                    const auto weight_neighbor = 1.0f / powf(length(point - 0.5f * (position_start + position_end)), idw_exponent);

                    rotation = rotation + weight_neighbor * compute_rotation(point, i);
                    rotation_weights += weight_neighbor;
                }
            }

            // Normalize rotation
            if (rotation_weights > 0.0001f)
            {
                point = point + (rotation / rotation_weights);
            }

            // Store result
            out_points[gid] = point;
            infos[gid] = info;
        }
}

/**
* Interpolate displacement projecting points onto the polyline and move points accordingly
*
* @param in_points              Points to displace
* @param out_points             Displaced points
* @param infos                  Information about the displacement
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
*/
__global__
void displacement_kernel_projection(const float3* in_points, float3* out_points, float4* infos,
    const int num_points, const int num_displacements)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get point
        float3 point = in_points[gid];
        float4 info = infos[gid];

        // Get nearest displacement position
        int nearest_index = 0;
        float nearest_distance = length(point - fetch(textures[__positions__], 0));

        for (int i = 1; i < num_displacements; ++i)
        {
            const float distance = length(point - fetch(textures[__positions__], i));

            if (distance < nearest_distance)
            {
                nearest_index = i;
                nearest_distance = distance;
            }
        }

        // Initialize displacement
        float3 displacement = { 0.0f, 0.0f, 0.0f };

        // Projection onto the line and apply linear interpolation
        const auto position_previous = fetch(textures[__positions__], maxi(0, nearest_index - 1));
        const auto position_current = fetch(textures[__positions__], nearest_index);
        const auto position_next = fetch(textures[__positions__], mini(num_displacements - 1, nearest_index + 1));

        const auto vector_previous = fetch(textures[__displacements__], maxi(0, nearest_index - 1));
        const auto vector_current = fetch(textures[__displacements__], nearest_index);
        const auto vector_next = fetch(textures[__displacements__], mini(num_displacements - 1, nearest_index + 1));

        if (nearest_index == 0)
        {
            const auto t = project_point_onto_line(point, position_current, position_next);

            if (t >= 0.0f && t <= 1.0f)
            {
                const auto vector = (1.0f - t) * vector_current + t * vector_next;
                const auto position = position_current + t * (position_next - position_current);

                displacement = vector;

                info.w = t;
                info.z += length(vector);
            }
            else
            {
                displacement = vector_current;

                info.w = -0.5;
                info.z += length(vector_current);
            }
        }
        else if (nearest_index == num_displacements - 1)
        {
            const auto t = project_point_onto_line(point, position_previous, position_current);

            if (t >= 0.0f && t <= 1.0f)
            {
                const auto vector = (1.0f - t) * vector_previous + t * vector_current;
                const auto position = position_previous + t * (position_current - position_previous);

                displacement = vector;

                info.w = t;
                info.z += length(vector);
            }
            else
            {
                displacement = vector_current;

                info.w = 1.5f;
                info.z += length(vector_current);
            }
        }
        else
        {
            const auto t1 = project_point_onto_line(point, position_previous, position_current);
            const auto t2 = project_point_onto_line(point, position_current, position_next);

            if (t1 >= 0.0f && t1 <= 1.0f && t2 <= 0.0f)
            {
                const auto vector = (1.0f - t1) * vector_previous + t1 * vector_current;
                const auto position = position_previous + t1 * (position_current - position_previous);

                displacement = vector;

                info.w = t1;
                info.z += length(vector);
            }
            else if (t2 >= 0.0f && t2 <= 1.0f && t1 >= 1.0f)
            {
                const auto vector = (1.0f - t2) * vector_current + t2 * vector_next;
                const auto position = position_current + t2 * (position_next - position_current);

                displacement = vector;

                info.w = t2;
                info.z += length(vector);
            }
            else if (t1 >= 1.0f && t2 <= 0.0f)
            {
                displacement = vector_current;

                info.w = 1.5f;
                info.z += length(vector_current);
            }
            else
            {
                const auto position_1 = position_previous + t1 * (position_current - position_previous);
                const auto position_2 = position_current + t2 * (position_next - position_current);

                if (length(point - position_1) < length(point - position_2))
                {
                    const auto vector = (1.0f - t1) * vector_previous + t1 * vector_current;

                    displacement = vector;

                    info.w = t1;
                    info.z += length(vector);
                }
                else
                {
                    const auto vector = (1.0f - t2) * vector_current + t2 * vector_next;

                    displacement = vector;

                    info.w = t2;
                    info.z += length(vector);
                }
            }
        }

        // Apply displacement
        point = point + displacement;

        // Store result
        out_points[gid] = point;
        infos[gid] = info;
    }
}

/**
* Interpolate displacement at handles using B-splines and move points accordingly
*
* @param in_points              Points to displace
* @param point_mapping          Corresponding point on the B-Spline
* @param arc_position_mapping   Corresponding arc position on the B-Spline
* @param out_points             Displaced points
* @param infos                  Information about the displacement
* @param mapping_direction      Direction from point to position on deformed B-Spline
* @param mapping_direction_orig Direction from point to position on original B-Spline
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
* @param degree                 B-Spline degree
*/
__global__
void displacement_kernel_spline_handles(const float3* in_points, const float3* point_mapping, const float* arc_position_mapping,
    float3* out_points, float4* infos, float3* mapping_direction, float3* mapping_direction_orig,
    const int num_points, const int num_displacements, const int degree)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get point
        float3 point = in_points[gid];
        float4 info = infos[gid];

        // Interpolate the displacement on the B-Spline at the position given by the arc position
        // and use a Gauss function to lessen the effect for points further away
        const auto distance = length(point - point_mapping[gid]);

        const float3 displacement = compute_point(arc_position_mapping[gid], degree, num_displacements, __displacements__);

        const auto u = arc_position_mapping[gid];

        info.w = u;
        info.z = distance;

        // Get original mapping, using orthogonal direction for nodes mapped to the end points
        const auto u_max = fetch1D(textures[__knot_vector__], num_displacements);

        if (u > u_max - 0.001 || u < 0.001)
        {
            mapping_direction_orig[gid] = rotate(compute_point(u, degree, num_displacements, __first_derivative__, 1), float3{ 0.0, 0.0, 1.0 }, __pi / 2.0);
        }
        else
        {
            mapping_direction_orig[gid] = point_mapping[gid] - point;
        }

        // Apply displacement
        point = point + displacement;

        // Get deformed mapping, using orthogonal direction for nodes mapped to the end points
        if (u > u_max - 0.001 || u < 0.001)
        {
            mapping_direction[gid] = rotate(compute_point(u, degree, num_displacements, __displaced_derivative__, 1), float3{ 0.0, 0.0, 1.0 }, __pi / 2.0);
        }
        else
        {
            mapping_direction[gid] = compute_point(u, degree, num_displacements, __displaced_positions__) - point;
        }

        // Store result
        out_points[gid] = point;
        infos[gid] = info;
    }
}

/**
* Interpolate displacement at joints using B-splines and move points accordingly
*
* @param in_points              Points to displace
* @param point_mapping          Corresponding point on the B-Spline
* @param tangent_mapping        Corresponding tangent on the B-Spline
* @param arc_position_mapping   Corresponding arc position on the B-Spline
* @param out_points             Displaced points
* @param infos                  Information about the displacement
* @param mapping_direction      Direction from point to position on deformed B-Spline
* @param mapping_direction_orig Direction from point to position on original B-Spline
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
* @param degree                 B-Spline degree
*/
__global__
void displacement_kernel_spline_joints(const float3* in_points, const float3* point_mapping, const float3* tangent_mapping,
    const float* arc_position_mapping, float3* out_points, float4* infos, float3* mapping_direction,
    float3* mapping_direction_orig, const int num_points, const int num_displacements, const int degree)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get point
        float3 point = in_points[gid];
        float4 info = infos[gid];

        //// Interpolate the displacement on the B-Spline at the position given by the arc position
        //// and use a Gauss function to lessen the effect for points further away

        // Get point and tangent of original and deformed B-spline at the arc position
        const auto u = arc_position_mapping[gid];

        const auto origin = point_mapping[gid];
        const auto tangent = tangent_mapping[gid];

        const auto deformed_origin = compute_point(u, degree, num_displacements, __displaced_positions__);
        const auto deformed_tangent = normalize(compute_point(u, degree, num_displacements, __displaced_derivative__, 1));

        const auto distance = length(point - origin);

        // Create rotation, rotating the original tangent onto the deformed one
        const auto rotation = compute_rotation(tangent, deformed_tangent);

        const float3 axis{ rotation.x, rotation.y, rotation.z };
        const auto angle = rotation.w;

        const auto point_relative_to_origin = point - origin;
        const auto point_relative_to_origin_rotated = rotate(point_relative_to_origin, axis, angle);
        const auto point_rotated = point_relative_to_origin_rotated + origin;
        const auto point_rotated_and_translated = point_rotated - origin + deformed_origin;

        info.w = u;
        info.z = distance;

        // Get original mapping, using orthogonal direction for nodes mapped to the end points
        const auto u_max = fetch1D(textures[__knot_vector__], num_displacements);

        if (u > u_max - 0.001 || u < 0.001)
        {
            mapping_direction_orig[gid] = rotate(tangent, float3{ 0.0, 0.0, 1.0 }, __pi / 2.0);
        }
        else
        {
            mapping_direction_orig[gid] = origin - point;
        }

        // Apply displacement
        point = point_rotated_and_translated;

        // Get deformed mapping, using orthogonal direction for nodes mapped to the end points
        if (u > u_max - 0.001 || u < 0.001)
        {
            mapping_direction[gid] = rotate(deformed_tangent, float3{ 0.0, 0.0, 1.0 }, __pi / 2.0);
        }
        else
        {
            mapping_direction[gid] = deformed_origin - point;
        }

        // Store result
        out_points[gid] = point;
        infos[gid] = info;
    }
}

/**
* Interpolate winding preserving displacement at joints using B-splines and move points accordingly
*
* @param in_points              Points to displace
* @param out_points             Displaced points
* @param arc_position_mapping   Corresponding arc position on the B-Spline
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
* @param degree                 B-Spline degree
*/
__global__
void displacement_kernel_winding(const float3* in_points, float3* out_points, float4* out_infos, const float* arc_position_mapping,
    const int num_points, const int num_displacements, const int degree)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get information for rotating
        const auto u = arc_position_mapping[gid];

        const auto center_of_rotation = compute_point(u, degree, num_displacements, __displaced_positions__);
        const auto axis_of_rotation = normalize(compute_point(1.0, degree, num_displacements, __displaced_positions__)
            - compute_point(0.0, degree, num_displacements, __displaced_positions__));

        const auto tangent = normalize(compute_point(u, degree, num_displacements, __first_derivative__, 1));
        const auto second_derivative = compute_point(u, degree, num_displacements, __second_derivative__, 2);
        const auto binormal = normalize(cross(tangent, second_derivative));
        const auto normal = cross(tangent, binormal);

        // Create rotation, rotating the normal onto the deformed B-spline
        const auto deformed_tangent = normalize(compute_point(u, degree, num_displacements, __displaced_derivative__, 1));

        const auto rotation = compute_rotation(tangent, deformed_tangent);

        const float3 axis{ rotation.x, rotation.y, rotation.z };
        const auto angle = rotation.w;

        const auto rotated_normal = normalize(rotate(normal, axis, angle));

        // Calculate rotated normal at the center of the B-spline as reference
        float3 winding_axis;
        float winding_angle;

        {
            const auto u_max = fetch1D(textures[__knot_vector__], num_displacements);

            const auto u = u_max / 2.0f;

            const auto tangent = normalize(compute_point(u, degree, num_displacements, __first_derivative__, 1));
            const auto second_derivative = compute_point(u, degree, num_displacements, __second_derivative__, 2);
            const auto binormal = normalize(cross(tangent, second_derivative));
            const auto normal = cross(tangent, binormal);

            // Create rotation, rotating the normal onto the deformed B-spline
            const auto deformed_tangent = normalize(compute_point(u, degree, num_displacements, __displaced_derivative__, 1));

            const auto rotation = compute_rotation(tangent, deformed_tangent);

            const float3 axis{ rotation.x, rotation.y, rotation.z };
            const auto angle = rotation.w;

            const auto reference_normal = normalize(rotate(normal, axis, angle));

            const auto winding_rotation = compute_rotation(rotated_normal, reference_normal);

            winding_axis = float3{ winding_rotation.x, winding_rotation.y, winding_rotation.z };
            winding_angle = winding_rotation.w;
        }

        // Rotate point around the (assumedly) straight feature line
        out_points[gid] = center_of_rotation + rotate(in_points[gid] - center_of_rotation, winding_axis, winding_angle);
        out_infos[gid] = float4{ winding_axis.x, winding_axis.y, winding_axis.z, winding_angle };
    }
}

/**
* Interpolate twisting displacement at joints using B-splines and move points accordingly
*
* @param in_points              Points to displace
* @param out_points             Displaced points
* @param arc_position_mapping   Corresponding arc position on the B-Spline
* @param num_points             Number of points
* @param num_displacements      Number of displacement vectors and positions
* @param degree                 B-Spline degree
*/
__global__
void displacement_kernel_twisting(const float3* in_points, float3* out_points, const float* arc_position_mapping,
    const int num_points, const int num_displacements, const int degree)
{
    __get_kernel__parameters__

    if (gid < num_points)
    {
        // Get information for rotating
        const auto u = arc_position_mapping[gid];

        const auto center_of_rotation = compute_point(u, degree, num_displacements, __displaced_positions__);
        const auto angle_of_rotation = compute_point(u, degree, num_displacements, __twisting_angle__);
        const auto axis_of_rotation = normalize(compute_point(1.0, degree, num_displacements, __displaced_positions__)
            - compute_point(0.0, degree, num_displacements, __displaced_positions__));

        // Rotate point around the (assumedly) straight feature line
        out_points[gid] = center_of_rotation + rotate(in_points[gid] - center_of_rotation, axis_of_rotation, angle_of_rotation.x);
    }
}

/// Functionality for resource management
namespace
{
    void initialize_texture(const void* h_data, const int num_elements, const int c0,
        const int c1, const int c2, const int c3, cudaTextureObject_t* texture, void** d_data)
    {
        const int num_bytes = c0 + c1 + c2 + c3;

        cudaError_t err;
        std::stringstream ss;

        err = cudaMalloc((void**)d_data, num_elements * num_bytes);
        if (err)
        {
            ss << "Error allocating memory using cudaMalloc for displacements" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }

        err = cudaMemcpy(*d_data, h_data, num_elements * num_bytes, cudaMemcpyHostToDevice);
        if (err)
        {
            ss << "Error copying memory using cudaMemcpy for displacements" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }

        cudaChannelFormatDesc cfdesc = cudaCreateChannelDesc(c0 * 8, c1 * 8, c2 * 8, c3 * 8, cudaChannelFormatKindFloat);

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = *d_data;
        resDesc.res.linear.desc = cfdesc;
        resDesc.res.linear.sizeInBytes = num_elements * num_bytes;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(texture, &resDesc, &texDesc, nullptr);
    }
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CUDA END ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


/// Basic vector math
namespace
{
    std::array<float, 4> add(const std::array<float, 4>& lhs, const std::array<float, 4>& rhs)
    {
        return std::array<float, 4> {
            lhs[0] + rhs[0],
            lhs[1] + rhs[1],
            lhs[2] + rhs[2],
            lhs[3] + rhs[3]
        };
    };

    std::array<float, 4> sub(const std::array<float, 4>& lhs, const std::array<float, 4>& rhs)
    {
        return std::array<float, 4> {
            lhs[0] - rhs[0],
            lhs[1] - rhs[1],
            lhs[2] - rhs[2],
            lhs[3] - rhs[3]
        };
    };

    std::array<float, 4> mul(const float lhs, const std::array<float, 4>& rhs)
    {
        return std::array<float, 4> {
            lhs * rhs[0],
            lhs * rhs[1],
            lhs * rhs[2],
            lhs * rhs[3]
        };
    };
}

cuda::displacement::displacement(std::vector<std::array<float, 3>> points) :
    points(std::move(points)), cuda_res_input_points(nullptr), cuda_res_output_points(nullptr),
    cuda_res_output_twisting_points(nullptr), cuda_res_info(nullptr), cuda_res_mapping_point(nullptr),
    cuda_res_mapping_tangent(nullptr), cuda_res_mapping_direction(nullptr),
    cuda_res_mapping_direction_orig(nullptr), cuda_res_mapping_arc_position(nullptr),
    cuda_res_mapping_arc_position_displaced(nullptr)
{
    upload_points();
}

cuda::displacement::~displacement()
{
    // Free CUDA resources
    if (this->cuda_res_input_points != nullptr) cudaFree(this->cuda_res_input_points);
    if (this->cuda_res_output_points != nullptr) cudaFree(this->cuda_res_output_points);
    if (this->cuda_res_output_twisting_points != nullptr) cudaFree(this->cuda_res_output_twisting_points);
    if (this->cuda_res_info != nullptr) cudaFree(this->cuda_res_info);
    if (this->cuda_res_mapping_point != nullptr) cudaFree(this->cuda_res_mapping_point);
    if (this->cuda_res_mapping_tangent != nullptr) cudaFree(this->cuda_res_mapping_tangent);
    if (this->cuda_res_mapping_direction != nullptr) cudaFree(this->cuda_res_mapping_direction);
    if (this->cuda_res_mapping_direction_orig != nullptr) cudaFree(this->cuda_res_mapping_direction_orig);
    if (this->cuda_res_mapping_arc_position != nullptr) cudaFree(this->cuda_res_mapping_arc_position);
    if (this->cuda_res_mapping_arc_position_displaced != nullptr) cudaFree(this->cuda_res_mapping_arc_position_displaced);
}

void cuda::displacement::upload_points()
{
    // Create CUDA image and upload input points to the GPU
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_input_points, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for input points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    {
        const auto err = cudaMemcpy(this->cuda_res_input_points, this->points.data(), this->points.size() * sizeof(float3), cudaMemcpyHostToDevice);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying to GPU memory using cudaMemcpy for input points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    // Create writeable CUDA image and upload output points to the GPU
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_output_points, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for output points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    {
        const auto err = cudaMemcpy(this->cuda_res_output_points, this->points.data(), this->points.size() * sizeof(float3), cudaMemcpyHostToDevice);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying to GPU memory using cudaMemcpy for output points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    // Create writeable CUDA image and upload output points to the GPU
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_output_twisting_points, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for output (twisting) points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    {
        const auto err = cudaMemcpy(this->cuda_res_output_twisting_points, this->points.data(), this->points.size() * sizeof(float3), cudaMemcpyHostToDevice);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying to GPU memory using cudaMemcpy for output (twisting) points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    // Create writeable CUDA image to store additional information
    this->displacement_info.resize(this->points.size(), float4{ -1.0f, 0.0f, 0.0f, 0.0f });

    {
        const auto err = cudaMalloc((void**)&this->cuda_res_info, this->displacement_info.size() * sizeof(float4));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for displacement information" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    {
        const auto err = cudaMemcpy(this->cuda_res_info, this->displacement_info.data(), this->displacement_info.size() * sizeof(float4), cudaMemcpyHostToDevice);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying to GPU memory using cudaMemcpy for displacement information" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    // Create writeable CUDA images to store a precomputed mapping
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_mapping_point, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for points of precomputed mapping" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_mapping_tangent, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for tangents of precomputed mapping" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_mapping_direction, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for direction of precomputed mapping" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_mapping_direction_orig, this->points.size() * sizeof(float3));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for direction of precomputed original mapping" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_mapping_arc_position, this->points.size() * sizeof(float));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for arc positions of precomputed mapping" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }
    {
        const auto err = cudaMalloc((void**)&this->cuda_res_mapping_arc_position_displaced, this->points.size() * sizeof(float));

        if (err)
        {
            std::stringstream ss;
            ss << "Error allocating memory using cudaMalloc for arc positions of assessed mapping" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }
}

void cuda::displacement::precompute(const parameter_t parameters, const std::vector<std::array<float, 4>>& positions)
{
    if (positions.empty()) return;

    // CUDA kernel parameters
    int num_threads = 128;
    int num_blocks = static_cast<int>(this->points.size()) / num_threads
        + (static_cast<int>(this->points.size()) % num_threads == 0 ? 0 : 1);

    // Upload positions to GPU
    cudaTextureObject_t cuda_tex_positions;
    float4* cuda_res_positions;

    initialize_texture((void*)positions.data(), static_cast<int>(positions.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_positions, (void**)&cuda_res_positions);

    cudaMemcpyToSymbol(textures, &cuda_tex_positions, sizeof(cudaTextureObject_t), __positions__ * sizeof(cudaTextureObject_t));

    // Create knot vector, compute B-Spline derivative, and upload them to the GPU
    const auto knot_vector = create_knot_vector(parameters.b_spline.degree, positions);
    const auto positions_first_derivative = compute_derivative(positions, knot_vector.cbegin(), parameters.b_spline.degree);

    cudaTextureObject_t cuda_tex_first_derivative;
    cudaTextureObject_t cuda_tex_knot_vector;

    float4* cuda_res_first_derivative;
    float* cuda_res_knot_vector;

    initialize_texture((void*)positions_first_derivative.data(), static_cast<int>(positions_first_derivative.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_first_derivative, (void**)&cuda_res_first_derivative);

    initialize_texture((void*)knot_vector.data(), static_cast<int>(knot_vector.size()),
        sizeof(float), 0, 0, 0, &cuda_tex_knot_vector, (void**)&cuda_res_knot_vector);

    cudaMemcpyToSymbol(textures, &cuda_tex_first_derivative, sizeof(cudaTextureObject_t), __first_derivative__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_knot_vector, sizeof(cudaTextureObject_t), __knot_vector__ * sizeof(cudaTextureObject_t));

    // Run precomputation
    precompute_mapping_kernel __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_mapping_point, this->cuda_res_mapping_tangent,
        this->cuda_res_mapping_arc_position, static_cast<int>(this->points.size()), static_cast<int>(positions.size()),
        parameters.b_spline.iterations, parameters.b_spline.degree);

    // Destroy resources
    cudaDestroyTextureObject(cuda_tex_first_derivative);
    cudaFree(cuda_res_first_derivative);

    cudaDestroyTextureObject(cuda_tex_knot_vector);
    cudaFree(cuda_res_knot_vector);

    cudaDestroyTextureObject(cuda_tex_positions);
    cudaFree(cuda_res_positions);
}

void cuda::displacement::assess_quality(const parameter_t parameters, const std::vector<std::array<float, 4>>& _positions,
    const std::vector<std::array<float, 4>>& displacements)
{
    if (_positions.empty()) return;

    std::vector<std::array<float, 4>> positions(_positions.begin(), _positions.end());

    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        positions[i][0] += displacements[i][0];
        positions[i][1] += displacements[i][1];
        positions[i][2] += displacements[i][2];
        positions[i][3] += displacements[i][3];
    }

    // CUDA kernel parameters
    int num_threads = 128;
    int num_blocks = static_cast<int>(this->points.size()) / num_threads
        + (static_cast<int>(this->points.size()) % num_threads == 0 ? 0 : 1);

    // Upload positions to GPU
    cudaTextureObject_t cuda_tex_positions;
    float4* cuda_res_positions;

    initialize_texture((void*)positions.data(), static_cast<int>(positions.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_positions, (void**)&cuda_res_positions);

    cudaMemcpyToSymbol(textures, &cuda_tex_positions, sizeof(cudaTextureObject_t), __positions__ * sizeof(cudaTextureObject_t));

    // Create knot vector, compute B-Spline derivative, and upload them to the GPU
    const auto knot_vector = create_knot_vector(parameters.b_spline.degree, positions);
    const auto positions_first_derivative = compute_derivative(positions, knot_vector.cbegin(), parameters.b_spline.degree);

    cudaTextureObject_t cuda_tex_first_derivative;
    cudaTextureObject_t cuda_tex_knot_vector;

    float4* cuda_res_first_derivative;
    float* cuda_res_knot_vector;

    initialize_texture((void*)positions_first_derivative.data(), static_cast<int>(positions_first_derivative.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_first_derivative, (void**)&cuda_res_first_derivative);

    initialize_texture((void*)knot_vector.data(), static_cast<int>(knot_vector.size()),
        sizeof(float), 0, 0, 0, &cuda_tex_knot_vector, (void**)&cuda_res_knot_vector);

    cudaMemcpyToSymbol(textures, &cuda_tex_first_derivative, sizeof(cudaTextureObject_t), __first_derivative__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_knot_vector, sizeof(cudaTextureObject_t), __knot_vector__ * sizeof(cudaTextureObject_t));

    // Run precomputation
    assess_mapping_kernel __kernel__parameters__(this->cuda_res_output_points, this->cuda_res_info,
        static_cast<int>(this->points.size()), static_cast<int>(positions.size()),
        parameters.b_spline.iterations, parameters.b_spline.degree);

    // Destroy resources
    cudaDestroyTextureObject(cuda_tex_first_derivative);
    cudaFree(cuda_res_first_derivative);

    cudaDestroyTextureObject(cuda_tex_knot_vector);
    cudaFree(cuda_res_knot_vector);

    cudaDestroyTextureObject(cuda_tex_positions);
    cudaFree(cuda_res_positions);
}

void cuda::displacement::displace(const method_t method, const parameter_t parameters,
    const std::vector<std::array<float, 4>>& positions, const std::vector<std::array<float, 4>>& displacements)
{
    // Sanity checks and early termination
    if (positions.size() != displacements.size())
    {
        throw std::runtime_error("Number of positions and displacements does not match");
    }

    if (positions.empty()) return;

    // CUDA kernel parameters
    int num_threads = 128;
    int num_blocks = static_cast<int>(this->points.size()) / num_threads
        + (static_cast<int>(this->points.size()) % num_threads == 0 ? 0 : 1);

    // Upload positions and displacements to GPU (needed by all methods)
    cudaTextureObject_t cuda_tex_positions, cuda_tex_displacements;
    float4* cuda_res_positions, * cuda_res_displacements;

    initialize_texture((void*)positions.data(), static_cast<int>(positions.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_positions, (void**)&cuda_res_positions);

    initialize_texture((void*)displacements.data(), static_cast<int>(displacements.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_displacements, (void**)&cuda_res_displacements);

    cudaMemcpyToSymbol(textures, &cuda_tex_positions, sizeof(cudaTextureObject_t), __positions__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_displacements, sizeof(cudaTextureObject_t), __displacements__ * sizeof(cudaTextureObject_t));

    // Depending on method, upload further information to the GPU and start CUDA computation
    switch (method)
    {
    case method_t::greedy:
        // Run computation
        displacement_kernel_idw __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_output_points, this->cuda_res_info,
            static_cast<int>(this->points.size()), static_cast<int>(positions.size()),
            parameters.inverse_distance_weighting.exponent, static_cast<int>(positions.size()));

        break;
    case method_t::greedy_joints:
        displacement_kernel_idw_joints __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_output_points, this->cuda_res_info,
            static_cast<int>(this->points.size()), static_cast<int>(positions.size()), parameters.inverse_distance_weighting.exponent);

        break;
    case method_t::voronoi:
        // Run computation
        displacement_kernel_idw __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_output_points, this->cuda_res_info,
            static_cast<int>(this->points.size()), static_cast<int>(positions.size()),
            parameters.inverse_distance_weighting.exponent, parameters.inverse_distance_weighting.neighborhood);

        break;
    case method_t::projection:
        // Run computation
        displacement_kernel_projection __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_output_points, this->cuda_res_info,
            static_cast<int>(this->points.size()), static_cast<int>(positions.size()));

        break;
    case method_t::b_spline:
    {
        // Create and upload knot vector
        const auto knot_vector = create_knot_vector(parameters.b_spline.degree, positions);

        cudaTextureObject_t cuda_tex_knot_vector;
        float* cuda_res_knot_vector;

        initialize_texture((void*)knot_vector.data(), static_cast<int>(knot_vector.size()),
            sizeof(float), 0, 0, 0, &cuda_tex_knot_vector, (void**)&cuda_res_knot_vector);

        cudaMemcpyToSymbol(textures, &cuda_tex_knot_vector, sizeof(cudaTextureObject_t), __knot_vector__ * sizeof(cudaTextureObject_t));

        // Run computation
        displacement_kernel_spline_handles __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_mapping_point,
            this->cuda_res_mapping_arc_position, this->cuda_res_output_points, this->cuda_res_info, this->cuda_res_mapping_direction,
            this->cuda_res_mapping_direction_orig, static_cast<int>(this->points.size()), static_cast<int>(positions.size()),
            parameters.b_spline.degree);

        // Destroy resources
        cudaDestroyTextureObject(cuda_tex_knot_vector);
        cudaFree(cuda_res_knot_vector);

        break;
    }
    case method_t::b_spline_joints:
    {
        // Create knot vector, compute displaced positions, and compute B-Spline derivatives
        const auto knot_vector = create_knot_vector(parameters.b_spline.degree, positions);

        const auto positions_first_derivative = compute_derivative(positions, knot_vector.cbegin(), parameters.b_spline.degree);

        const auto displaced_positions = displace_positions(positions, displacements);
        const auto displaced_positions_first_derivative = compute_derivative(displaced_positions, knot_vector.cbegin(), parameters.b_spline.degree);

        // Upload additional information to the GPU
        cudaTextureObject_t cuda_tex_first_derivative, cuda_tex_knot_vector, cuda_tex_displaced_positions, cuda_tex_displaced_first_derivative;

        float4* cuda_res_first_derivative;
        float* cuda_res_knot_vector;
        float4* cuda_res_displaced_positions, * cuda_res_displaced_first_derivative;

        initialize_texture((void*)positions_first_derivative.data(), static_cast<int>(positions_first_derivative.size()),
            sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_first_derivative, (void**)&cuda_res_first_derivative);

        initialize_texture((void*)knot_vector.data(), static_cast<int>(knot_vector.size()),
            sizeof(float), 0, 0, 0, &cuda_tex_knot_vector, (void**)&cuda_res_knot_vector);

        initialize_texture((void*)displaced_positions.data(), static_cast<int>(displaced_positions.size()),
            sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_displaced_positions, (void**)&cuda_res_displaced_positions);

        initialize_texture((void*)displaced_positions_first_derivative.data(), static_cast<int>(displaced_positions_first_derivative.size()),
            sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_displaced_first_derivative, (void**)&cuda_res_displaced_first_derivative);

        cudaMemcpyToSymbol(textures, &cuda_tex_first_derivative, sizeof(cudaTextureObject_t), __first_derivative__ * sizeof(cudaTextureObject_t));
        cudaMemcpyToSymbol(textures, &cuda_tex_knot_vector, sizeof(cudaTextureObject_t), __knot_vector__ * sizeof(cudaTextureObject_t));
        cudaMemcpyToSymbol(textures, &cuda_tex_displaced_positions, sizeof(cudaTextureObject_t), __displaced_positions__ * sizeof(cudaTextureObject_t));
        cudaMemcpyToSymbol(textures, &cuda_tex_displaced_first_derivative, sizeof(cudaTextureObject_t), __displaced_derivative__ * sizeof(cudaTextureObject_t));

        displacement_kernel_spline_joints __kernel__parameters__(this->cuda_res_input_points, this->cuda_res_mapping_point, this->cuda_res_mapping_tangent,
            this->cuda_res_mapping_arc_position, this->cuda_res_output_points, this->cuda_res_info, this->cuda_res_mapping_direction,
            this->cuda_res_mapping_direction_orig, static_cast<int>(this->points.size()), static_cast<int>(positions.size()),
            parameters.b_spline.degree);

        // Destroy resources
        cudaDestroyTextureObject(cuda_tex_first_derivative);
        cudaFree(cuda_res_first_derivative);

        cudaDestroyTextureObject(cuda_tex_knot_vector);
        cudaFree(cuda_res_knot_vector);

        cudaDestroyTextureObject(cuda_tex_displaced_positions);
        cudaFree(cuda_res_displaced_positions);

        cudaDestroyTextureObject(cuda_tex_displaced_first_derivative);
        cudaFree(cuda_res_displaced_first_derivative);

        break;
    }
    }

    // Destroy resources
    cudaDestroyTextureObject(cuda_tex_positions);
    cudaFree(cuda_res_positions);

    cudaDestroyTextureObject(cuda_tex_displacements);
    cudaFree(cuda_res_displacements);
}

void cuda::displacement::displace_winding(const method_t method, const parameter_t parameters,
    const std::vector<std::array<float, 4>>& positions, const std::vector<std::array<float, 4>>& displacements)
{
    // CUDA kernel parameters
    int num_threads = 128;
    int num_blocks = static_cast<int>(this->points.size()) / num_threads
        + (static_cast<int>(this->points.size()) % num_threads == 0 ? 0 : 1);

    // Upload B-spline information to GPU
    const auto knot_vector = create_knot_vector(parameters.b_spline.degree, positions);

    const auto positions_first_derivative = compute_derivative(positions, knot_vector.cbegin(), parameters.b_spline.degree);
    const auto positions_second_derivative = compute_derivative(positions_first_derivative, knot_vector.cbegin() + 1, parameters.b_spline.degree - 1); // TODO: is this correct?

    const auto displaced_positions = displace_positions(positions, displacements);
    const auto displaced_positions_first_derivative = compute_derivative(displaced_positions, knot_vector.cbegin(), parameters.b_spline.degree);

    cudaTextureObject_t cuda_tex_knot_vector, cuda_tex_first_derivative,
        cuda_tex_second_derivative, cuda_tex_displaced_positions, cuda_tex_displaced_first_derivative;

    float* cuda_res_knot_vector;
    float4* cuda_res_first_derivative, * cuda_res_second_derivative, * cuda_res_displaced_positions, * cuda_res_displaced_first_derivative;

    initialize_texture((void*)knot_vector.data(), static_cast<int>(knot_vector.size()),
        sizeof(float), 0, 0, 0, &cuda_tex_knot_vector, (void**)&cuda_res_knot_vector);

    initialize_texture((void*)positions_first_derivative.data(), static_cast<int>(positions_first_derivative.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_first_derivative, (void**)&cuda_res_first_derivative);

    initialize_texture((void*)positions_second_derivative.data(), static_cast<int>(positions_second_derivative.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_second_derivative, (void**)&cuda_res_second_derivative);

    initialize_texture((void*)displaced_positions.data(), static_cast<int>(displaced_positions.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_displaced_positions, (void**)&cuda_res_displaced_positions);

    initialize_texture((void*)displaced_positions_first_derivative.data(), static_cast<int>(displaced_positions_first_derivative.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_displaced_first_derivative, (void**)&cuda_res_displaced_first_derivative);

    cudaMemcpyToSymbol(textures, &cuda_tex_knot_vector, sizeof(cudaTextureObject_t), __knot_vector__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_first_derivative, sizeof(cudaTextureObject_t), __first_derivative__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_second_derivative, sizeof(cudaTextureObject_t), __second_derivative__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_displaced_positions, sizeof(cudaTextureObject_t), __displaced_positions__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_displaced_first_derivative, sizeof(cudaTextureObject_t), __displaced_derivative__ * sizeof(cudaTextureObject_t));

    displacement_kernel_winding __kernel__parameters__(this->cuda_res_output_points, this->cuda_res_output_twisting_points,
        this->cuda_res_info, this->cuda_res_mapping_arc_position, static_cast<int>(this->points.size()),
        static_cast<int>(positions.size()), parameters.b_spline.degree);

    // Destroy resources
    cudaDestroyTextureObject(cuda_tex_knot_vector);
    cudaFree(cuda_res_knot_vector);

    cudaDestroyTextureObject(cuda_tex_first_derivative);
    cudaFree(cuda_res_first_derivative);

    cudaDestroyTextureObject(cuda_tex_second_derivative);
    cudaFree(cuda_res_second_derivative);

    cudaDestroyTextureObject(cuda_tex_displaced_positions);
    cudaFree(cuda_res_displaced_positions);

    cudaDestroyTextureObject(cuda_tex_displaced_first_derivative);
    cudaFree(cuda_res_displaced_first_derivative);
}

void cuda::displacement::displace_twisting(const method_t method, const parameter_t parameters,
    const std::vector<std::array<float, 4>>& positions, const std::vector<std::array<float, 4>>& displacements,
    const std::vector<std::array<float, 4>>& rotations)
{
    // Sanity checks and early termination
    if (positions.size() != rotations.size())
    {
        throw std::runtime_error("Number of positions and rotations does not match");
    }

    if (rotations.empty()) return;

    // CUDA kernel parameters
    int num_threads = 128;
    int num_blocks = static_cast<int>(this->points.size()) / num_threads
        + (static_cast<int>(this->points.size()) % num_threads == 0 ? 0 : 1);

    // Upload positions and rotations to GPU
    const auto displaced_positions = displace_positions(positions, displacements);

    cudaTextureObject_t cuda_tex_positions, cuda_tex_rotations;
    float4* cuda_res_positions, * cuda_res_rotations;

    initialize_texture((void*)displaced_positions.data(), static_cast<int>(displaced_positions.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_positions, (void**)&cuda_res_positions);

    initialize_texture((void*)rotations.data(), static_cast<int>(rotations.size()),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), &cuda_tex_rotations, (void**)&cuda_res_rotations);

    cudaMemcpyToSymbol(textures, &cuda_tex_positions, sizeof(cudaTextureObject_t), __displaced_positions__ * sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(textures, &cuda_tex_rotations, sizeof(cudaTextureObject_t), __twisting_angle__ * sizeof(cudaTextureObject_t));

    displacement_kernel_twisting __kernel__parameters__(this->cuda_res_output_points, this->cuda_res_output_twisting_points,
        this->cuda_res_mapping_arc_position, static_cast<int>(this->points.size()),
        static_cast<int>(rotations.size()), parameters.b_spline.degree);

    // Destroy resources
    cudaDestroyTextureObject(cuda_tex_positions);
    cudaFree(cuda_res_positions);

    cudaDestroyTextureObject(cuda_tex_rotations);
    cudaFree(cuda_res_rotations);
}

std::vector<float> cuda::displacement::create_knot_vector(int degree, const std::vector<std::array<float, 4>>& positions) const
{
    return b_spline(positions, degree).get_knot_vector();
}

std::vector<std::array<float, 4>> cuda::displacement::displace_positions(const std::vector<std::array<float, 4>>& positions,
    const std::vector<std::array<float, 4>>& displacements) const
{
    std::vector<std::array<float, 4>> displaced_positions(positions.size());

    for (std::size_t index = 0; index < displaced_positions.size(); ++index)
    {
        displaced_positions[index] = add(positions[index], displacements[index]);
    }

    return displaced_positions;
}

std::vector<std::array<float, 4>> cuda::displacement::compute_derivative(const std::vector<std::array<float, 4>>& positions,
    const typename std::vector<float>::const_iterator knot_begin, const int degree) const
{
    std::vector<std::array<float, 4>> derivative(positions.size() - 1);

    for (std::size_t index = 0; index < derivative.size(); ++index)
    {
        const auto scalar = degree / (*(knot_begin + index + degree + 1) - *(knot_begin + index + 1));

        derivative[index] = mul(scalar, sub(positions[index + 1], positions[index]));
    }

    return derivative;
}

const std::vector<std::array<float, 3>>& cuda::displacement::get_results() const
{
    // Download displaced points from the GPU
    const auto err = cudaMemcpy(this->points.data(), this->cuda_res_output_points, this->points.size() * sizeof(float3), cudaMemcpyDeviceToHost);

    if (err)
    {
        std::stringstream ss;
        ss << "Error copying from GPU memory using cudaMemcpy for output points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
        throw std::runtime_error(ss.str());
    }

    return this->points;
}

const std::vector<std::array<float, 3>>& cuda::displacement::get_results_twisting() const
{
    // Download displaced points from the GPU
    const auto err = cudaMemcpy(this->points.data(), this->cuda_res_output_twisting_points, this->points.size() * sizeof(float3), cudaMemcpyDeviceToHost);

    if (err)
    {
        std::stringstream ss;
        ss << "Error copying from GPU memory using cudaMemcpy for output (twisting) points" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
        throw std::runtime_error(ss.str());
    }

    return this->points;
}

const std::tuple<std::vector<float4>, std::vector<float3>, std::vector<float3>> cuda::displacement::get_displacement_info() const
{
    // Download displacement information, as well as mapping from the GPU
    {
        const auto err = cudaMemcpy(this->displacement_info.data(), this->cuda_res_info, this->displacement_info.size() * sizeof(float4), cudaMemcpyDeviceToHost);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying from GPU memory using cudaMemcpy for displacement information" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    {
        this->mapping.resize(this->displacement_info.size());

        const auto err = cudaMemcpy(this->mapping.data(), this->cuda_res_mapping_direction, this->mapping.size() * sizeof(float3), cudaMemcpyDeviceToHost);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying from GPU memory using cudaMemcpy for mapping information" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    {
        this->mapping_orig.resize(this->displacement_info.size());

        const auto err = cudaMemcpy(this->mapping_orig.data(), this->cuda_res_mapping_direction_orig, this->mapping_orig.size() * sizeof(float3), cudaMemcpyDeviceToHost);

        if (err)
        {
            std::stringstream ss;
            ss << "Error copying from GPU memory using cudaMemcpy for original mapping information" << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    return std::make_tuple(this->displacement_info, this->mapping, this->mapping_orig);
}
