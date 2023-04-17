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

examples_gradcheck = get_option('examples-gradcheck')

examples = []

foreach src : sources
  exec_name = src.replace('.cu.cpp', '').replace('.cpp', '')
  if examples_gradcheck and exec_name == 'min-gpt'
    exe = executable(exec_name,
                     src,
                     include_directories : incdir,
                     dependencies : test_deps,
                     cpp_args: ['-DGRADCHECK'])
  else
    exe = executable(exec_name,
                     src,
                     include_directories : incdir,
                     dependencies : example_deps)
  endif
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
