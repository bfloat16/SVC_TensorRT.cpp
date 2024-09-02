cmake_minimum_required(VERSION 3.8)

# Set the CMP0141 policy to NEW if it exists
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

project(SVC_TensorRT CXX)

find_package(CUDAToolkit REQUIRED)

include_directories(
    "${CMAKE_SOURCE_DIR}/TensorRT/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)
add_executable(TensorRT_Loader SVC_TensorRT.cpp/TensorRT_Loader.cpp)
target_link_libraries(TensorRT_Loader nvinfer_10 nvinfer_plugin_10 nvonnxparser_10 cudart ${CUDAToolkit_LIBRARIES})