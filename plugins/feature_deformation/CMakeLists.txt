cmake_minimum_required(VERSION 3.12)

# Create project for feature deformation plugins
pv_project(VISUSfeature_deformation OFF)

if (NOT ENABLE_CUDA AND PARAVIEW_PLUGIN_ENABLE_VISUSfeature_deformation)
  message(FATAL_ERROR "The feature deformation plugin requires CUDA to be enabled.")
endif()
