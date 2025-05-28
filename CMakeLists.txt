cmake_minimum_required(VERSION 3.10)
project(CUDAImageProcessing CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(OpenCV QUIET)

include_directories(include)

file(GLOB CORE_SOURCES 
    "src/*.cu" 
    "src/*.cpp" 
    "src/utils/*.cu" 
    "src/filters/*.cu"
    "src/filters/*.cpp")

list(REMOVE_ITEM CORE_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/optical_flow_demo.cpp")

add_executable(image_processor ${CORE_SOURCES})
target_link_libraries(image_processor)
set_target_properties(image_processor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(OpenCV_FOUND)
    add_executable(optical_flow_demo 
        src/optical_flow_demo.cpp
        src/filters/optical_flow.cu
        src/filters/optical_flow_cpu.cpp
        src/image_io.cu
        src/utils/cuda_utils.cu
        src/logger.cpp)
    
    target_link_libraries(optical_flow_demo ${OpenCV_LIBS})
    set_target_properties(optical_flow_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    message(STATUS "OpenCV found - optical_flow_demo will be built")
else()
    message(STATUS "OpenCV not found - optical_flow_demo will not be built")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/MD")

add_custom_target(benchmark
    COMMAND image_processor benchmark ${CMAKE_SOURCE_DIR}/data/input/bird.jpg
    DEPENDS image_processor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)