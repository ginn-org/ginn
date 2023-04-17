project('ginn', 'cpp', default_options : ['cpp_std=c++17', 'cuda_std=c++17'])

gpu = get_option('gpu')

if gpu
  add_languages('cuda', required : true)
endif

incdir = include_directories('./')

eigen        = subproject('eigen').get_variable('eigen_dep')
catch        = subproject('catch2').get_variable('catch2_dep')
# why do i still observe some build steps from fmt despite using header-only dep?
fmt          = subproject('fmt').get_variable('fmt_header_only_dep')
cppitertools = subproject('cppitertools').get_variable('cppitertools_dep')
tblr         = subproject('tblr').get_variable('tblr_dep')
threads      = dependency('threads')

deps         = [eigen, fmt, cppitertools, tblr]
test_deps    = deps + [catch]
example_deps = deps + [threads]


if gpu
  cuda = import('unstable-cuda')
  cuda_args = cuda.nvcc_arch_flags('11.1', ['7.0'])
  cuda_dep = dependency('cuda',
                        version : '>=11.1',
                        modules : ['curand', 'cublas', 'cublasLt'])
endif

subdir('test')
subdir('examples')
subdir('ginn-py')
