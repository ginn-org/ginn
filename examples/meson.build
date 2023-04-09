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

foreach src : sources
  exec_name = src.replace('.cu.cpp', '').replace('.cpp', '')
  executable(exec_name,
             src,
             include_directories : incdir,
             dependencies : example_deps)
endforeach
