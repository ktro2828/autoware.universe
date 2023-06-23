cmake_minimum_required(VERSION 3.8)
project(patchworkpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-nonportable-include-path)
endif()

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
ament_auto_find_build_dependencies()

include_directories(include SYSTEM ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

ament_auto_add_library(patchwork_pp_node SHARED src/patchworkpp.cpp src/params.cpp)

target_link_libraries(patchwork_pp_node ${PCL_LIBRARIES} ${EIGEN3_LIBRARIES})

rclcpp_components_register_node(
  patchwork_pp_node PLUGIN "patchwork_pp::PatchWorkPP" EXECUTABLE patchwork_pp)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
