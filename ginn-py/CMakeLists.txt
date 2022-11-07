set(PY_SOURCES ginn.cpp
        dev-py.cpp
        tensor-py.cpp
        node-py.cpp
        node/common-py.cpp
        node/layout-py.cpp )

if(GINN_ENABLE_GPU)
  foreach(src ${PY_SOURCES})
    set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

pybind11_add_module(ginn ${PY_SOURCES})

if(GINN_ENABLE_GPU)
  target_compile_options(ginn PRIVATE ${GPU_OPTIM_OPTS})
  target_link_libraries(ginn PUBLIC CUDA::curand CUDA::cublas)
endif()
