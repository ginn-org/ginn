set(PY_SOURCES ginn.cpp
        dev-py.cpp
        tensor-py.cpp
        node-py.cpp
        node/affine-py.cpp
        node/common-py.cpp
        node/compare-py.cpp
        node/nonlin-py.cpp
        node/layout-py.cpp
        node/pick-py.cpp
        node/prod-py.cpp
        node/reduce-py.cpp
        node/select-py.cpp
        node/weight-py.cpp
        init-py.cpp
        update-py.cpp)

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
