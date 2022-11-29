cmake_minimum_required(VERSION 3.10)

if(GINN_ENABLE_GPU)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  project(ginn LANGUAGES CUDA CXX)
else()
  project(ginn LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(GINN_ENABLE_GPU)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)

  set(CMAKE_CUDA_ARCHITECTURES 70)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --x cu -w -Wno-deprecated-gpu-targets -DGINN_ENABLE_GPU")
  set(GPU_OPTIM_OPTS -O3 --ptxas-options=-O3 --compiler-options=-Ofast)
endif()

include_directories("${PROJECT_SOURCE_DIR}/")
include_directories("${PROJECT_SOURCE_DIR}/extern/eigen/")
include_directories("${PROJECT_SOURCE_DIR}/extern/fmt/include/")
include_directories("${PROJECT_SOURCE_DIR}/extern/Catch2/single_include/catch2/")
include_directories("${PROJECT_SOURCE_DIR}/extern/pybind11/include/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-unused-but-set-parameter")
endif()

add_subdirectory("test" test)
add_subdirectory("examples" examples)

add_subdirectory(extern/pybind11)
add_subdirectory("ginn-py" ginn-py)
