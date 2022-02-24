cmake_minimum_required(VERSION 3.12)

# Create VTK module
pv_module(feature_flow_separation ${PROJECT_NAME} "" feature_flow_separation_target)

# Find and link Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Target for eigen3 not found")
endif()

target_link_libraries(${feature_flow_separation_target} PUBLIC Eigen3::Eigen)
