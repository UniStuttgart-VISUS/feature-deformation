cmake_minimum_required(VERSION 3.12)

# Create VTK module
set(sources
  ../feature_deformation/curvature.h ../feature_deformation/curvature.cxx
  ../feature_deformation/gradient.h ../feature_deformation/gradient.cxx
  ../feature_deformation/grid.h ../feature_deformation/grid.cxx)

pv_module(curvature ${PROJECT_NAME} "${sources}" curvature_target)

# Find and link Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Target for eigen3 not found")
endif()

target_link_libraries(${curvature_target} PUBLIC Eigen3::Eigen)
