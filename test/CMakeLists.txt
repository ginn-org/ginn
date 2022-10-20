if(GINN_ENABLE_GPU)
  add_executable(cudatensortest tensor.cu.cpp)
  set_source_files_properties(tensor.cu.cpp PROPERTIES LANGUAGE CUDA)

  add_executable(cudanodestest nodes.cu.cpp)
  set_source_files_properties(nodes.cu.cpp PROPERTIES LANGUAGE CUDA)

  add_executable(cudaconvnodestest convnodes.cu.cpp)
  set_source_files_properties(convnodes.cu.cpp PROPERTIES LANGUAGE CUDA)

  add_executable(cudadropoutnodestest dropoutnodes.cu.cpp)
  set_source_files_properties(dropoutnodes.cu.cpp PROPERTIES LANGUAGE CUDA)

  add_executable(cudadevtest dev.cu.cpp)
  set_source_files_properties(dev.cu.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(includetest include.cpp)
add_executable(nodestest nodes.cu.cpp)
add_executable(convnodestest convnodes.cu.cpp)
add_executable(dropoutnodestest dropoutnodes.cu.cpp)
add_executable(tensortest tensor.cu.cpp)
add_executable(autobatchtest autobatch.cpp)
add_executable(graphtest graph.cpp)
add_executable(modeltest model.cpp)
add_executable(sampletest sample.cpp)
add_executable(utiltest util.cpp)
add_executable(devtest dev.cu.cpp)

set(CPU_TESTS includetest
              nodestest
              convnodestest
              dropoutnodestest
              tensortest
              autobatchtest
              graphtest
              modeltest
              sampletest
              utiltest
              devtest)

set(DEBUG_OPTS -O0 -g -fprofile-arcs -ftest-coverage)

if(GINN_ENABLE_GPU)
  set(GPU_TESTS cudanodestest
                cudaconvnodestest
                cudadropoutnodestest
                cudatensortest
                cudadevtest)
  set(GPU_DEBUG_OPTS -O0 -g)
endif()

foreach(target ${CPU_TESTS})
  target_compile_options(${target} PRIVATE ${DEBUG_OPTS})
  if(APPLE)
    target_link_options(${target} PRIVATE "-coverage")
  else()
    target_link_libraries(${target} gcov)
  endif()
  string(REGEX REPLACE "test$" "" exec_name ${target})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exec_name})
endforeach()

add_custom_target(tests DEPENDS ${CPU_TESTS})

if(GINN_ENABLE_GPU)
  foreach(target ${GPU_TESTS})
    target_compile_options(${target} PRIVATE ${GPU_DEBUG_OPTS})
    target_link_libraries(${target} CUDA::curand CUDA::cublas)
    string(REGEX REPLACE "test$" "" exec_name ${target})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exec_name})
  endforeach()

  add_custom_target(cudatests DEPENDS ${GPU_TESTS})
endif()
