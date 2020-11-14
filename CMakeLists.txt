cmake_minimum_required(VERSION 3.11)

# Compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)

# Project
set(name melgan)
project(${name} CXX CUDA)

# Environment
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

# Library
set(headers src/block.hpp
            src/convert.hpp
            src/cuda.hpp
            src/kernel.hpp
            src/layer.hpp
            src/load.hpp
            src/model.hpp
            src/save.hpp)
set(sources src/block.cu
            src/convert.cpp
            src/cuda.cu
            src/kernel.cu
            src/layer.cu
            src/load.cpp
            src/melgan.cpp
            src/model.cu
            src/save.cpp)
add_library(${name} STATIC ${sources} ${headers})
target_include_directories(${name} PUBLIC include)
target_include_directories(${name} PUBLIC src)
set_target_properties(${name} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${name} PRIVATE
                               $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
target_compile_options(${name} PRIVATE
                               $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

# Test
enable_testing()
add_subdirectory(googletest)
add_subdirectory(test)

# Link
add_executable(infer src/main.cpp)
target_link_libraries(infer ${name} Boost::program_options)
