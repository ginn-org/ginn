test_sources = [
  'autobatch.cpp',
  'convnodes.cu.cpp',
  'dev.cu.cpp',
  'dropoutnodes.cu.cpp',
  'graph.cpp',
  'include.cpp',
  'model.cpp',
  'nodes.cu.cpp',
  'sample.cpp',
  'tensor.cu.cpp',
  'util.cpp',
]

cuda_test_sources = [
  'convnodes.cu.cpp',
  'dev.cu.cpp',
  'dropoutnodes.cu.cpp',
  'nodes.cu.cpp',
  'tensor.cu.cpp',
]

foreach src : test_sources
  exec_name = src.replace('test/', '').replace('.cu.cpp', '').replace('.cpp', '') + 'test'
  executable(exec_name,
             src,
             include_directories : incdir,
             dependencies : test_deps)
endforeach

if gpu
  fs = import('fs')
  foreach src : cuda_test_sources
    exec_name = 'cuda' + src.replace('test/', '').replace('.cu.cpp', '').replace('.cu', '') + 'test'
    copy_name = src.replace('.cu.cpp', '.copy.cu')
    copied_src = fs.copyfile(src, copy_name)

    executable(exec_name,
               copied_src,
               include_directories : incdir,
               dependencies : test_deps + [cuda_dep],
               cuda_args : cuda_args + ['-DGINN_ENABLE_GPU'])
  endforeach
endif
