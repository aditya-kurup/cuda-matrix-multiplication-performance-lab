cmake_minimum_required(VERSION 3.18)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cuda_matrix_multiplication_performance_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul_lab src/main.cu)
target_include_directories(matmul_lab PRIVATE include)
target_compile_options(matmul_lab PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME gpu_correctness COMMAND matmul_lab --test)
endif()
