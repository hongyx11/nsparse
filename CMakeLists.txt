cmake_minimum_required(VERSION 3.25)
# define project
project(nsparse LANGUAGES C CXX CUDA VERSION 0.0.1)
# set c++ standard and find packages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(spgemm cuda-cpp/sample/spgemm/spgemm_cu_csr.cu)
target_include_directories(spgemm PUBLIC cuda-cpp/inc)
string(APPEND CMAKE_CUDA_FLAGS " -I${CMAKE_CURRENT_SOURCE_DIR}/cuda-cpp/inc ")

# CUDA
option(USE_CUDA "USE_GPU" ON)
if(USE_CUDA)
find_package(CUDAToolkit REQUIRED)
target_compile_definitions(spgemm PUBLIC USE_CUDA)
target_link_libraries(spgemm PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
endif()
