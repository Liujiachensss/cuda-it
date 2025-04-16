cmake_minimum_required(VERSION 3.14)

project(ProgrammingMassivelyParallelProcessorsLANGUAGES CXX CUDA)

# find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDAToolKit_ROOT /usr/local/cuda)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++) #${CMAKE_CXX_COMPILER}
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CUDA_COMPILE /usr/local/cuda-12.3/bin/nvcc)

# set(CUDA_NVCC_FLAGS "-DNDEBUG; -O3; -rdc=true; -Xcompiler; -fPIC; -allow-unsupported-compiler;-fmad=false; --default-stream=per-thread")
set(CMAKE_CUDA_FLAGS_DEBUG ${CUDA_NVCC_FLAGS} "-g -G")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

include_directories(
    ~/learning/cuda-samples/Common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")
set(CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# add_executable(8-stencil 8-stencil.cu)
add_subdirectory(stencil)
add_subdirectory(gemm)
