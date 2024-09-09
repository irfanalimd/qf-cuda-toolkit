cmake_minimum_required(VERSION 3.10)
project(CUDAImageProcessing CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

include_directories(include)

file(GLOB SOURCES "src/*.cu" "src/*.cpp" "src/utils/*.cu" "src/filters/*.cu")
# file(GLOB CPP_SOURCES "src/*.cpp" "src/filters/*.cpp")

add_executable(image_processor ${SOURCES})

target_link_libraries(image_processor)

set_target_properties(image_processor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
