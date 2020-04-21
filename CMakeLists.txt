cmake_minimum_required(VERSION 3.12)

# Top-level options
option(ENABLE_CUDA "Enable CUDA, which might be necessary for some of the plugins." ON)

set(languages CXX)
if (ENABLE_CUDA)
  list(APPEND languages CUDA)
endif()

# Create top-level project
project(VISUS_ParaView_Plugins ${languages})

# Add plugins
add_subdirectory(plugins)
