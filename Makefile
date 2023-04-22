CXX ?= g++
CUDA_CXX = $(CXX)
NVCC = nvcc

BUILD_PATH ?= ./build
EXAMPLES_PATH = examples
TESTS_PATH = test

INCLUDES = -I./ \
     -I./extern/eigen/ \
     -I./extern/tblr/ \
     -I./extern/fmt/include/ \
     -I./extern/cppitertools/ \
     -I./extern/Catch2/single_include/

CUDA_INCLUDES = 
CUDA_LINKS = -lcurand -lcublas -lcublasLt

CXXFLAGS = -std=c++17 -Wall -Wno-unused-but-set-parameter -ftemplate-backtrace-limit=0 -Wno-deprecated-declarations

CUDAFLAGS = -std=c++17 -Xcompiler=-Wall,-Winvalid-pch,-Wextra -DGINN_ENABLE_GPU --x cu \
            -gencode arch=compute_70,code=sm_70 \
            -ftemplate-backtrace-limit=0 \
            -Xcudafe --display_error_number \
            -Xcudafe --diag_suppress=3123 \
            -Xcudafe --diag_suppress=3124 \
            -Xcudafe --diag_suppress=3125 \
            -Xcudafe --diag_suppress=3126 \
            -Xcudafe --diag_suppress=3127 \
            -Xcudafe --diag_suppress=20013 \
            -Xcudafe --diag_suppress=20014 \
            -Xcudafe --diag_suppress=20015 \
						-Xcudafe --diag_suppress=445

# disabling 445 for now because the following _must_ be a nvcc bug? "Rank" _is_ used?
#
# ./ginn/tensor.h(178): warning #445-D: constant "Rank" is not used in declaring the parameter types of function template "ginn::Tensor<ScalarType>::Tensor<Rank>(ginn::DevPtr, ginn::NestedInitList<Rank, ScalarType>)"
#
# ./ginn/tensor.h(184): warning #445-D: constant "Rank" is not used in declaring the parameter types of function template "ginn::Tensor<ScalarType>::Tensor<Rank>(ginn::NestedInitList<Rank, ScalarType>)"
#


OPTFLAGS = -Ofast -march=native -mtune=native -pthread
CUOPTFLAGS = -O3 -Xptxas -O3 -Xcompiler -O3

DEBUGFLAGS = -g -O0 -fprofile-arcs -ftest-coverage
CUDEBUGFLAGS = -g -O0

# ______________ Create paths for builds __________________
tests_path:
	@mkdir -p $(BUILD_PATH)/$(TESTS_PATH)

examples_path:
	@mkdir -p $(BUILD_PATH)/$(EXAMPLES_PATH)

py_build_path:
	@mkdir -p $(BUILD_PATH)/$(PY_PATH)

# _____________________ Tests _____________________________

tests: autobatchtest \
	     convnodestest \
       devtest \
       dropoutnodestest \
       graphtest \
       includetest \
       modeltest \
       nodestest \
       sampletest \
       tensortest \
       utiltest

cudatests: cudaconvnodestest \
           cudadevtest \
           cudadropoutnodestest \
           cudanodestest \
           cudatensortest

%test : $(TESTS_PATH)/%.cu.cpp tests_path
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(DEBUGFLAGS) -o $(BUILD_PATH)/$(TESTS_PATH)/$@

%test : $(TESTS_PATH)/%.cpp tests_path
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(DEBUGFLAGS) -o $(BUILD_PATH)/$(TESTS_PATH)/$@

cuda%test : $(TESTS_PATH)/%.cu.cpp tests_path
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) $< $(CUDEBUGFLAGS) -o $(BUILD_PATH)/$(TESTS_PATH)/$@

cuda%test : $(TESTS_PATH)/%.cpp tests_path
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) $< $(CUDEBUGFLAGS) -o $(BUILD_PATH)/$(TESTS_PATH)/$@

# _____________________ Examples __________________________
examples: mnist \
          mnist-conv \
          mnist-layers \
          sum-lstm \
          sstb-treelstm \
          lstm-tag \
          bench \
          min-gpt

cudaexamples: mnist-cu \
              mnist-conv-cu \
              mnist-layers-cu \
              mnist-dp-cu \
              sum-lstm-cu \
              sstb-treelstm-cu \
              bench-cu \
              min-gpt-cu

% : $(EXAMPLES_PATH)/%.cu.cpp examples_path
	$(CXX) $< $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) -o $(BUILD_PATH)/$(EXAMPLES_PATH)/$@

% : $(EXAMPLES_PATH)/%.cpp examples_path
	$(CXX) $< $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) -o $(BUILD_PATH)/$(EXAMPLES_PATH)/$@

%-cu : $(EXAMPLES_PATH)/%.cu.cpp examples_path
	$(NVCC) $< $(CUDAFLAGS) $(CUOPTFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) -o $(BUILD_PATH)/$(EXAMPLES_PATH)/$@

%-cu : $(EXAMPLES_PATH)/%.cu examples_path
	$(NVCC) $< $(CUDAFLAGS) $(CUOPTFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) -o $(BUILD_PATH)/$(EXAMPLES_PATH)/$@

clean:
	rm -rf $(BUILD_PATH) *.gcda *.gcno
