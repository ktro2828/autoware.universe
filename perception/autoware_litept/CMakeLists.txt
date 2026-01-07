cmake_minimum_required(VERSION 3.14)
project(autoware_litept LANGUAGES CXX CUDA)

find_package(autoware_cmake REQUIRED)
autoware_package()

# --- find CUDA/TensorRT dependencies ---
find_package(CUDA)
find_package(CUDAToolkit)
find_package(TENSORRT)

# --- build libraries ---
if(CUDA_FOUND AND TENSORRT_FOUND)
    # --- CUDA kernel libraries ---
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr -diag-suppress 1675 --extended-lambda")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_87,code=sm_87")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=sm_89")
    # NOTE(ktro2828): PTX support for newer GPUs until we can compile directly
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=compute_89")
    # TODO(ktro2828): enable when the driver supports it list(APPEND
    # CUDA_NVCC_FLAGS "-gencode arch=compute_120,code=sm_120")

    cuda_add_library(${PROJECT_NAME}_cuda_lib
        SHARED
        src/postprocess/postprocessor.cu
        src/preprocess/preprocessor.cu)
    target_include_directories(${PROJECT_NAME}_cuda_lib
        SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
        ${CUDA_INCLUDE_DIRS}
        ${autoware_cuda_utils_INCLUDE_DIRS}
        ${autoware_point_types_INCLUDE_DIRS})

    # --- TensorRT libraries ---
    ament_auto_add_library(${PROJECT_NAME}_lib SHARED src/litept_trt.cpp)
    target_compile_definitions(
    ${PROJECT_NAME}_lib
    PRIVATE TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR})
    target_include_directories(${PROJECT_NAME}_lib
        SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
        ${TENSORRT_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${autoware_cuda_utils_INCLUDE_DIRS}
        ${autoware_point_types_INCLUDE_DIRS})
    target_link_libraries(
        ${PROJECT_NAME}_lib
        ${NVINFER}
        ${NVONNXPARSER}
        ${CUDA_LIBRARIES}
        ${CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${PROJECT_NAME}_cuda_lib)

    # --- ROS node components ---
    ament_auto_add_library(${PROJECT_NAME}_component SHARED src/litept_node.cpp)
    target_include_directories(${PROJECT_NAME}_component
        SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
    target_link_libraries(${PROJECT_NAME}_component ${PROJECT_NAME}_lib)
    rclcpp_components_register_node(${PROJECT_NAME}_component
        PLUGIN "autoware::litept::LitePTNode"
        EXECUTABLE ${PROJECT_NAME}_node)

    # --- install libraries ---
    install(TARGETS ${PROJECT_NAME}_cuda_lib DESTINATION lib)
    ament_auto_package(INSTALL_TO_SHARE launch config)
else()
    ament_auto_package(INSTALL_TO_SHARE launch config)
endif()

if(BUILD_TESTING)

endif()
