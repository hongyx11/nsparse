cmake_minimum_required(VERSION 3.20) 
project(nsparse LANGUAGES C CXX CUDA)

#set c++ standard and find packages
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80 90)
#Define the header - only library target
add_library(nsparse 
    cuda-cpp/src/BIN.cu 
    cuda-cpp/src/nsparse.cu 
    cuda-cpp/src/SpGEMM.cu 
    cuda-cpp/src/nsparse_asm.cu 
    cuda-cpp/src/SpGEMM.cu 
    cuda-cpp/src/HashSpGEMM.cu)

#Set the include directories for the INTERFACE target
target_include_directories(nsparse PUBLIC ${CMAKE_SOURCE_DIR}/cuda-cpp/include)
target_link_libraries(nsparse PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
option(ORIGIN_HASH "Use the original hash function" ON)
if(ORIGIN_HASH)
    target_compile_definitions(nsparse PUBLIC -DNSPARSE_ORIGIN_HASH)
endif(ORIGIN_HASH)

target_compile_definitions(nsparse PUBLIC USE_CUDA)
# add_executable(spgemm - csr example / spgemm - csr.cu)
# target_link_libraries(spgemm - csr nsparse)

# add_executable(spmv - csr example / spmv - csr.cu)
# target_link_libraries(spmv - csr nsparse)


add_executable(spgemm-hash cuda-cpp/sample/spgemm/spgemm_hash.cu)
target_link_libraries(spgemm-hash PUBLIC nsparse)

set(TARGETNAME spgemm-hash)

# openmp
find_package(OpenMP)
if(TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
  target_compile_definitions(${TARGETNAME} PUBLIC THREADED)
  target_link_libraries(${TARGETNAME} PUBLIC OpenMP::OpenMP_CXX)
elseif(OPENMP_FOUND)
  target_compile_definitions(${TARGETNAME} PUBLIC THREADED)
  target_compile_options(${TARGETNAME} PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(${TARGETNAME} PUBLIC "${OpenMP_CXX_FLAGS}")
endif()

