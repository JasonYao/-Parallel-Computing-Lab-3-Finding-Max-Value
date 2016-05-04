cmake_minimum_required(VERSION 3.3)
project(lab3)

find_package(CUDA QUIET REQUIRED)


# Specify binary name and source file to build it from
cuda_add_executable(
        maxgpu
        maxgpu.cu)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set(SOURCE_FILES maxseq.c maxgpu.cu)
add_executable(lab3 ${SOURCE_FILES})