cmake_minimum_required(VERSION 3.11)

set(name melgan)
project(${name} CXX CUDA)

# Environment
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Test
enable_testing()
add_subdirectory(googletest)

# Code
set(headers src/model.h)
set(sources src/model.cu)
add_library(${name} STATIC ${sources} ${headers})

set_target_properties(${name} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)
