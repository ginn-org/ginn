project('ginn', 'cpp', default_options : ['cpp_std=c++17', 'cuda_std=c++17'])

gpu = get_option('gpu')

if gpu
  add_languages('cuda', required : true)
endif

# TODO: move these to subprojects. cppitertools needs patch
incdir = include_directories('./',
                             './extern/',
                             './extern/tblr/tblr/',
                             './extern/pybind11/include/')

eigen = subproject('eigen').get_variable('eigen_dep')
catch = subproject('catch2').get_variable('catch2_dep')
# why do i still observe some build steps from fmt despite using header-only dep?
fmt = subproject('fmt').get_variable('fmt_header_only_dep')

test_deps = [eigen, catch, fmt]


if gpu
  cuda = import('unstable-cuda')
  cuda_args = cuda.nvcc_arch_flags('11.1', ['7.0'])
  cuda_dep = dependency('cuda', version : '>=11.1', modules : ['curand', 'cublas'])
endif

subdir('test')
