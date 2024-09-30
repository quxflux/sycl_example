cmake_minimum_required(VERSION 3.20)  
  
project(test_application)  

include(cmake/CPM.cmake)

add_library(sycl_target INTERFACE)

option(SYCL_STARTER_USE_CUDA_BACKEND "enable CUDA backend" OFF)
option(SYCL_STARTER_USE_FMA "enable FMA, might cause deviations for FP operations" OFF)

set(SYCL_STARTER_SYCL_TARGETS spir64)
if (SYCL_STARTER_USE_CUDA_BACKEND)
    list(PREPEND SYCL_STARTER_SYCL_TARGETS nvptx64-nvidia-cuda)
endif()

list(JOIN SYCL_STARTER_SYCL_TARGETS "," SYCL_STARTER_SYCL_TARGETS_CMD)

if (SYCL_STARTER_USE_FMA)
    set(SYCL_STARTER_FMA -fma)
endif()

set(SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_STARTER_SYCL_TARGETS_CMD} ${SYCL_STARTER_FMA})
message("SYCL flags: ${SYCL_FLAGS}")
target_compile_features(sycl_target INTERFACE cxx_std_23)
target_compile_options(sycl_target INTERFACE ${SYCL_FLAGS})
target_link_options(sycl_target INTERFACE ${SYCL_FLAGS})

add_subdirectory(src)
