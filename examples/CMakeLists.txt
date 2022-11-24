set(THREADS_PREFER_PTHREAD_FLAG ON)
if(APPLE)
  #https://stackoverflow.com/a/54606606
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
else()
  find_package(Threads REQUIRED)
endif()

if(GINN_ENABLE_GPU)
  add_executable(mnist-cu mnist.cu.cpp)
  set_source_files_properties(mnist.cu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(mnist-conv-cu mnist-conv.cu.cpp)
  set_source_files_properties(mnist-conv.cu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(mnist-layers-cu mnist-layers.cu.cpp)
  set_source_files_properties(mnist-layers.cu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(mnist-dp-cu mnist-dp.cu)
  set_source_files_properties(mnist-dp.cu PROPERTIES LANGUAGE CUDA)
  add_executable(sum-lstm-cu sum-lstm.cu.cpp)
  set_source_files_properties(sum-lstm.cu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(sstb-treelstm-cu sstb-treelstm.cu.cpp)
  set_source_files_properties(sstb-treelstm.cu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(benchmark-cu benchmark.cu.cpp)
  set_source_files_properties(benchmark.cu.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(min-gpt-cu min-gpt.cu.cpp)
  set_source_files_properties(min-gpt.cu.cpp PROPERTIES LANGUAGE CUDA)
else()
  add_executable(mnist mnist.cu.cpp)
  add_executable(mnist-conv mnist-conv.cu.cpp)
  add_executable(mnist-layers mnist-layers.cu.cpp)
  add_executable(sum-lstm sum-lstm.cu.cpp)
  add_executable(sstb-treelstm sstb-treelstm.cu.cpp)
  add_executable(benchmark benchmark.cu.cpp)
  add_executable(min-gpt min-gpt.cu.cpp)
endif()
add_executable(lstm-tag lstm-tag.cpp)

if (GINN_ENABLE_GPU)
  set(GPU_EXAMPLES mnist-cu
                   mnist-conv-cu
                   mnist-layers-cu
                   mnist-dp-cu
                   sum-lstm-cu
                   sstb-treelstm-cu
                   benchmark-cu
                   min-gpt-cu)
  set(CPU_EXAMPLES lstm-tag)
else()
  set(CPU_EXAMPLES mnist
                   mnist-conv
                   mnist-layers
                   sum-lstm
                   sstb-treelstm
                   lstm-tag
                   benchmark
                   min-gpt)
  # Can build min-gpt example in grad checking mode by setting EXAMPLES_GRADCHECK flag
  if (EXAMPLES_GRADCHECK)
    target_compile_options(min-gpt PRIVATE -DGRADCHECK -O0 -g)
  endif()
endif()

if(APPLE)
  set(EXAMPLE_OPTS -Ofast)
else()
  set(EXAMPLE_OPTS -Ofast -march=native -mtune=native)
endif()

if (GINN_ENABLE_GPU)
  set(GPU_EXAMPLE_OPTS ${GPU_OPTIM_OPTS})
endif()

foreach(target ${CPU_EXAMPLES})
  target_compile_options(${target} PRIVATE ${EXAMPLE_OPTS})
  if(NOT APPLE)
    target_link_libraries(${target} PRIVATE Threads::Threads)
  endif()
endforeach()

add_custom_target(examples DEPENDS ${CPU_EXAMPLES})

if(GINN_ENABLE_GPU)
  foreach(target ${GPU_EXAMPLES})
    target_compile_options(${target} PRIVATE ${GPU_EXAMPLE_OPTS})
    target_link_libraries(${target} CUDA::curand CUDA::cublas)
  endforeach()

  add_custom_target(cudaexamples DEPENDS ${GPU_EXAMPLES})
endif()
