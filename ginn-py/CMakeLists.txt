if(GINN_ENABLE_GPU)
  set_source_files_properties(ginn.cpp PROPERTIES LANGUAGE CUDA)
endif()
pybind11_add_module(ginn ginn.cpp)

if(GINN_ENABLE_GPU)
  target_compile_options(ginn PRIVATE ${GPU_OPTIM_OPTS})
  target_link_options(ginn PRIVATE ${GPU_LINK_OPTS})
endif()
