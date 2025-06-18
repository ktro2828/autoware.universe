cmake_minimum_required(VERSION 3.14)
project(autoware_mtr)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- find dependencies ---
find_package(autoware_cmake REQUIRED)
find_package(glog REQUIRED)
autoware_package()

# --- find CUDA/CUDNN/TensorRT dependencies ---
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
find_package(TENSORRT REQUIRED)

# --- check CUDA tools and TensorRT version ---
if(NOT (CUDAToolkit_FOUND AND CUDNN_FOUND AND TENSORRT_FOUND))
  message(FATAL_ERROR "cuda, cudnn, tensorrt libraries are not found")
endif()

if(TENSORRT_VERSION VERSION_LESS 8.5)
  message(
    FATAL_ERROR
      "Unsupported version TensorRT ${TENSORRT_VERSION} detected. This package requires TensorRT 8.5 or later."
  )
endif()

# --- link targets ---
file(GLOB_RECURSE MTR_SOURCES src/*.cpp)
ament_auto_add_library(${PROJECT_NAME} SHARED ${MTR_SOURCES})
target_link_libraries(${PROJECT_NAME} glog::glog)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN "autoware::mtr::MTRNode" EXECUTABLE
  ${PROJECT_NAME}_exe)

if(BUILD_TESTING)
  ament_add_ros_isolated_gtest(test_mtr
    # archetype
    test/test_agent.cpp
    test/test_exception.cpp
    test/test_fixed_queue.cpp
    test/test_map.cpp
    test/test_polyline.cpp
    test/test_result.cpp
    test/test_tensor.cpp)
  target_link_libraries(test_mtr ${PROJECT_NAME})
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
