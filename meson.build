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

cpp_args = ['-ftemplate-backtrace-limit=0']


if gpu
  cuda = import('unstable-cuda')
  cuda_args = cuda.nvcc_arch_flags('11.1', ['7.0']) + [
                '-ftemplate-backtrace-limit=0',
                '-Xcudafe', '--display_error_number',
                # the following disabled warnings are similar to:
                #   warning: __host__ annotation is ignored on a function that is explicitly defaulted on its first declaration
                #   warning: calling a __host__ function from a __host__ __device__ function is not allowed
                # which also appear in eigen unit tests, so i assume they are benign
                '-Xcudafe', '--diag_suppress=3123',
                '-Xcudafe', '--diag_suppress=3124',
                '-Xcudafe', '--diag_suppress=3125',
                '-Xcudafe', '--diag_suppress=3126',
                '-Xcudafe', '--diag_suppress=3127',
                '-Xcudafe', '--diag_suppress=20013',
                '-Xcudafe', '--diag_suppress=20014',
                '-Xcudafe', '--diag_suppress=20015',
                # 445 appears as:
                #   ./ginn/tensor.h(178): warning #445-D: constant "Rank" is not used in declaring the parameter types of function template "ginn::Tensor<ScalarType>::Tensor<Rank>(ginn::DevPtr, ginn::NestedInitList<Rank, ScalarType>)"
                #   ./ginn/tensor.h(184): warning #445-D: constant "Rank" is not used in declaring the parameter types of function template "ginn::Tensor<ScalarType>::Tensor<Rank>(ginn::NestedInitList<Rank, ScalarType>)"
                # which looks clearly wrong? is it a nvcc issue?
                '-Xcudafe', '--diag_suppress=445',
              ]
  cuda_dep = dependency('cuda',
                        version : '>=11.1',
                        modules : ['curand', 'cublas', 'cublasLt'])
endif

subdir('test')
subdir('examples')
subdir('ginn-py')
