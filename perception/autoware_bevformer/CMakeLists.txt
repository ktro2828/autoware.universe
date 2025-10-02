cmake_minimum_required(VERSION 3.14)
project(autoware_bevformer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- find dependencies ---
find_package(autoware_cmake REQUIRED)
autoware_package()

# --- find CUDA/CUDNN/TensorRT dependenices ---
find_package(CUDA)
find_package(CUDAToolkit)
find_package(CUDNN)
find_package(TENSORRT)

# --- Check CUDA tools and TensorRT version ---
if(CUDAToolkit_FOUND AND CUDNN_FOUND AND TENSORRT_FOUND)
    set(CUDA_AVAILABLE TRUE)
    if(TENSORRT_VERSION VERSION_LESS 8.5)
        set(TENSORRT_AVAILABLE FALSE)
        message(WARNING
        "Unsupported version TensorRT ${TENSORRT_VERSION} detected. "
        "This package requires TensorRT 8.5 or later.")
    else()
        set(TENSORRT_AVAILABLE TRUE)
    endif()
else()
    set(CUDA_AVAILABLE FALSE)
    set(TENSORRT_AVAILABLE FALSE)
    message(WARNING "CUDA, CUDNN and TensorRT libraries are not found.")
endif()

if(CUDA_AVAILABLE AND TENSORRT_AVAILABLE)
    # Set CUDA architecture flags for proper half-precision support
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extended-lambda")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment -Wno-deprecated-declarations -Wno-unused-parameter")

    file(GLOB_RECURSE TENSORRT_SOURCES src/tensorrt/*.cpp src/tensorrt/*.cu)
    cuda_add_library(${PROJECT_NAME}_OPS SHARED ${TENSORRT_SOURCES})
    target_include_directories(${PROJECT_NAME}_OPS PRIVATE
        src/tensorrt/common
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME}_OPS ${CUDNN_LIBRARY} ${TENSORRT_LIBRARIES})

    # Suppress warnings for TensorRT plugins
    # TODO(ktro2828): Remove the following options
    target_compile_options(${PROJECT_NAME}_OPS PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter -Wno-deprecated-declarations -Wno-comment -Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter,-Wno-deprecated-declarations,-Wno-comment,-Wno-unused-variable>
    )
endif()

ament_auto_add_library(${PROJECT_NAME} SHARED
    "src/node.cpp"
    "src/camera_buffer.cpp"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:src>
)

rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN "autoware::bevformer::BEVFormerNode"
    EXECUTABLE ${PROJECT_NAME}_exe
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
