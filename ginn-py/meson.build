pymod = import('python')
py = pymod.find_installation('python3.9')

pybind11 = subproject('pybind11').get_variable('pybind11_dep')

sources = [
  'ginn-py.cu.cpp',
  'dev-py.cu.cpp',
  'tensor-py.cu.cpp',
  'node-py.cu.cpp',
  'node/affine-py.cu.cpp',
  'node/common-py.cu.cpp',
  'node/compare-py.cu.cpp',
  'node/nonlin-py.cu.cpp',
  'node/layout-py.cu.cpp',
  'node/pick-py.cu.cpp',
  'node/prod-py.cu.cpp',
  'node/reduce-py.cu.cpp',
  'node/select-py.cu.cpp',
  'node/weight-py.cu.cpp',
  'init-py.cu.cpp',
  'update-py.cu.cpp',
]


if gpu
  fs = import('fs')
  copied_sources = []
  foreach src : sources
    exec_name = 'cuda' + src.replace('.cu.cpp', '').replace('.cu', '')
    copy_name = src.replace('.cu.cpp', '.copy.cu').replace('/', '__')
    copied_src = fs.copyfile(src, copy_name)
    copied_sources += [copied_src]
  endforeach

  tgt = py.extension_module('ginn',
                            sources : copied_sources,
                            include_directories : incdir,
                            dependencies : deps + [pybind11, cuda_dep],
                            cuda_args : cuda_args + ['-DGINN_ENABLE_GPU'])
else
  tgt = py.extension_module('ginn',
                            sources : sources,
                            include_directories : incdir,
                            dependencies : deps + [pybind11])
endif

message('Python target name: ' + tgt.name())
alias_target('python', tgt)
