cmake_minimum_required(VERSION 3.12)

# Create VTK module
set(sources
  b-spline.h b-spline.cxx
  curvature.h curvature.cxx
  displacement.h displacement.cu
  gradient.h gradient.cxx
  grid.h grid.cxx
  jacobian.h jacobian.cxx
  hash.h
  smoothing.h smoothing.cxx
  twisting.h twisting.cxx

  performance.h performance.cxx

  algorithm.h algorithm.inl

  algorithm_input.h

  algorithm_compute_tearing.h algorithm_compute_tearing.cxx

  algorithm_displacement_assessment.h algorithm_displacement_assessment.cxx
  algorithm_displacement_computation.h algorithm_displacement_computation.cxx
  algorithm_displacement_computation_twisting.h algorithm_displacement_computation_twisting.cxx
  algorithm_displacement_creation.h algorithm_displacement_creation.cxx
  algorithm_displacement_precomputation.h algorithm_displacement_precomputation.cxx

  algorithm_geometry_input.h algorithm_geometry_input.cxx
  algorithm_geometry_output_creation.h algorithm_geometry_output_creation.cxx
  algorithm_geometry_output_set.h algorithm_geometry_output_set.cxx
  algorithm_geometry_output_update.h algorithm_geometry_output_update.cxx

  algorithm_grid_input.h algorithm_grid_input.cxx
  algorithm_grid_output_creation.h algorithm_grid_output_creation.cxx
  algorithm_grid_output_set.h algorithm_grid_output_set.cxx
  algorithm_grid_output_update.h algorithm_grid_output_update.cxx
  algorithm_grid_output_vectorfield.h algorithm_grid_output_vectorfield.cxx

  algorithm_line_input.h algorithm_line_input.cxx
  algorithm_line_output_creation.h algorithm_line_output_creation.cxx
  algorithm_line_output_set.h algorithm_line_output_set.cxx
  algorithm_line_output_update.h algorithm_line_output_update.cxx

  algorithm_smoothing.h algorithm_smoothing.cxx
  algorithm_twisting.h algorithm_twisting.cxx

  algorithm_vectorfield_input.h algorithm_vectorfield_input.cxx)

pv_module(feature_deformation ${PROJECT_NAME} "${sources}" feature_deformation_target)

# Find and link OpenMP
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${feature_deformation_target} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find and link Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Target for eigen3 not found")
endif()

target_link_libraries(${feature_deformation_target} PUBLIC Eigen3::Eigen)

# Find and link CGAL
find_package(CGAL REQUIRED)

if(NOT TARGET CGAL::CGAL)
  message(FATAL_ERROR "Target for CGAL not found")
endif()

set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_MINSIZEREL Release)
set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

target_link_libraries(${feature_deformation_target} PUBLIC CGAL::CGAL)

# Configure CUDA
set_target_properties(${feature_deformation_target} PROPERTIES CUDA_STANDARD 17)
target_include_directories(${feature_deformation_target} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
