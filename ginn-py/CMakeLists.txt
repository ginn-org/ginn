set(PY_SOURCES ginn-py.cu.cpp
        dev-py.cu.cpp
        tensor-py.cu.cpp
        node-py.cu.cpp
        node/affine-py.cu.cpp
        node/common-py.cu.cpp
        node/compare-py.cu.cpp
        node/nonlin-py.cu.cpp
        node/layout-py.cu.cpp
        node/pick-py.cu.cpp
        node/prod-py.cu.cpp
        node/reduce-py.cu.cpp
        node/select-py.cu.cpp
        node/weight-py.cu.cpp
        init-py.cu.cpp
        update-py.cu.cpp)

if(GINN_ENABLE_GPU)
  foreach(src ${PY_SOURCES})
    set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

pybind11_add_module(ginn ${PY_SOURCES})

if(GINN_ENABLE_GPU)
  target_compile_options(ginn PRIVATE ${GPU_OPTIM_OPTS})
  target_link_libraries(ginn PUBLIC CUDA::curand CUDA::cublas)
else()
  if(APPLE)
    target_compile_options(ginn PRIVATE -Ofast)
  else()
    target_compile_options(ginn PRIVATE -Ofast -march=native -mtune=native)
  endif()
endif()
