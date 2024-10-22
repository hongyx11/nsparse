cmake_minimum_required(VERSION 3.25)
# define project
project(nsparse LANGUAGES C CXX CUDA VERSION 0.0.1)
# set c++ standard and find packages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
find_package(CUDAToolkit REQUIRED)

set(binary_names 
    spgemm-cusparse 
    spgemm-hash
)
set(source_files 
    cuda-cpp/sample/spgemm/spgemm_cu_csr.cu 
    cuda-cpp/sample/spgemm/spgemm_hash.cu 
)
list(LENGTH source_files source_files_len)
list(LENGTH binary_names binary_names_len)
if(NOT binary_names_len EQUAL source_files_len)
    message(FATAL_ERROR "binary_names and source_files must have the same length!")
endif()

math(EXPR last_index "${binary_names_len} - 1")

# Iterate over the indices and add each executable
foreach(i RANGE 0 ${last_index})
    message(STATUS "${i}")
    list(GET binary_names ${i} bin_name)
    list(GET source_files ${i} src_file)
    add_executable(${bin_name} ${src_file})
    target_include_directories(${bin_name} PUBLIC cuda-cpp/inc)
    target_link_libraries(${bin_name} PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
endforeach()

# add include path to cuda flags
string(APPEND CMAKE_CUDA_FLAGS " -I${CMAKE_CURRENT_SOURCE_DIR}/cuda-cpp/inc ")



