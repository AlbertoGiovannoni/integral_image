cmake_minimum_required(VERSION 3.27)
project(integral_image_parallel CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

# Aggiungi tutti i file sorgente CUDA al progetto
set(CUDA_SOURCE_FILES
        parallel.cu
)

# Aggiungi tutti i file sorgente C++ al progetto
set(CPP_SOURCE_FILES
        sequential.cpp
)

# Aggiungi l'eseguibile con i file sorgente CUDA e C++
add_executable(integral_image_parallel parallel.cu)
add_executable(integral_image_sequential sequential.cpp)


# Imposta la compilazione separabile per il codice CUDA
set_target_properties(integral_image_parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
