sources = [
  'mnist-conv.cu.cpp',
  'mnist-layers.cu.cpp',
  'sstb-treelstm.cu.cpp',
  'bench.cu.cpp',
  'lstm-tag.cpp',
  'min-gpt.cu.cpp',
  'mnist.cu.cpp',
  'sum-lstm.cu.cpp',
]

cuda_sources = [
  'mnist-conv.cu.cpp',
  'mnist-layers.cu.cpp',
  'sstb-treelstm.cu.cpp',
  'bench.cu.cpp',
  'min-gpt.cu.cpp',
  'mnist.cu.cpp',
  'sum-lstm.cu.cpp',
]

examples = []

foreach src : sources
  exec_name = src.replace('.cu.cpp', '').replace('.cpp', '')
  exe = executable(exec_name,
                   src,
                   include_directories : incdir,
                   dependencies : example_deps)
  examples += [exe]
endforeach

if gpu
  fs = import('fs')
  foreach src : cuda_sources
    exec_name = 'cuda' + src.replace('.cu.cpp', '').replace('.cu', '')
    copy_name = src.replace('.cu.cpp', '.copy.cu')
    copied_src = fs.copyfile(src, copy_name)

    exe = executable(exec_name,
                     copied_src,
                     include_directories : incdir,
                     dependencies : example_deps + [cuda_dep],
                     cuda_args : cuda_args + ['-DGINN_ENABLE_GPU'])
    examples += [exe]
  endforeach
endif

alias_target('examples', examples)
