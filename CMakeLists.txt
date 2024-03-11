cmake_minimum_required(VERSION 3.27)
project(integral_image_parallel CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(integral_image_parallel main.cu)

set_target_properties(integral_image_parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
