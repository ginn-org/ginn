CXX ?= g++
CUDA_CXX = $(CXX)
NVCC = nvcc

BUILD_PATH ?= ./build
EXAMPLES_PATH = examples
TESTS_PATH = test

INCLUDES = -I./ \
	   -I./subprojects/eigen/ \
	   -I./extern/tblr/ \
	   -I./subprojects/fmt-9.1.0/include/ \
	   -I./extern/ \
		 -I./subprojects/Catch2-2.13.8/single_include/

CUDA_INCLUDES = 
CUDA_LINKS = -lcurand -lcublas -lcublasLt

CXXFLAGS = -std=c++17 -Wall -Wno-unused-but-set-parameter -ftemplate-backtrace-limit=0 -Wno-deprecated-declarations


CUDAFLAGS = -std=c++17 -DGINN_ENABLE_GPU --x cu \
						-gencode arch=compute_70,code=sm_70 \
            -w

OPTFLAGS = -Ofast -march=native -mtune=native -pthread
CUOPTFLAGS = -O3 -Xptxas -O3 -Xcompiler -O3

DEBUGFLAGS = -g -O0 -fprofile-arcs -ftest-coverage
CUDEBUGFLAGS = -g -O0

PYTHON_CXXFLAGS = -Wno-unused-parameter -Wno-missing-field-initializers

# ______________ Create paths for builds __________________
tests_path:
	@mkdir -p $(BUILD_PATH)/$(TESTS_PATH)

examples_path:
	@mkdir -p $(BUILD_PATH)/$(EXAMPLES_PATH)

py_build_path:
	@mkdir -p $(BUILD_PATH)/$(PY_PATH)

# _____________________ Tests _____________________________

tests: includetest \
       nodestest \
       convnodestest \
       dropoutnodestest \
       tensortest \
       autobatchtest \
       graphtest \
       modeltest \
       sampletest \
       utiltest

cudatests: cudanodestest cudaconvnodestest cudadropoutnodestest cudatensortest


%test : $(TESTS_PATH)/%.cu.cpp tests_path
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(DEBUGFLAGS) -o $(BUILD_PATH)/$(TESTS_PATH)/$*

%test : $(TESTS_PATH)/%.cpp tests_path
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(DEBUGFLAGS) -o $(BUILD_PATH)/$(TESTS_PATH)/$*

cuda%test : $(TESTS_PATH)/%.cu.cpp tests_path
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) $< $(CUDEBUGFLAGS) \
		-o $(BUILD_PATH)/$(TESTS_PATH)/cuda$*

cuda%test : $(TESTS_PATH)/%.cpp tests_path
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) $< $(CUDEBUGFLAGS) \
		-o $(BUILD_PATH)/$(TESTS_PATH)/cuda$*

%test : $(TESTS_PATH)/%.cu tests_path
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) $< $(CUDEBUGFLAGS) \
		-o $(BUILD_PATH)/$(TESTS_PATH)/$*

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
	$(NVCC) $< $(CUDAFLAGS) $(CUOPTFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) \
		-o $(BUILD_PATH)/$(EXAMPLES_PATH)/$@

%-cu : $(EXAMPLES_PATH)/%.cu examples_path
	$(NVCC) $< $(CUDAFLAGS) $(CUOPTFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(CUDA_LINKS) \
		-o $(BUILD_PATH)/$(EXAMPLES_PATH)/$@

python: $(PY_PATH)/*.i py_build_path
	$(SWIG) -c++ -python -py3 -I../ -outdir $(BUILD_PATH)/$(PY_PATH) -o $(BUILD_PATH)/$(PY_PATH)/ginn_wrap.cxx $(PY_PATH)/ginn.i
	$(CXX) $(CXXFLAGS) $(PYTHON_CXXFLAGS) $(OPTFLAGS) -c -fpic $(BUILD_PATH)/$(PY_PATH)/ginn_wrap.cxx $(INCLUDES) $(PYTHON_INCLUDES) -o $(BUILD_PATH)/$(PY_PATH)/ginn_wrap.o
	$(CXX) $(CXXFLAGS) $(PYTHON_CXXFLAGS) $(OPTFLAGS) -shared $(BUILD_PATH)/$(PY_PATH)/ginn_wrap.o $(INCLUDES) -o $(BUILD_PATH)/$(PY_PATH)/_ginn.so

clean:
	rm -rf $(BUILD_PATH) *.gcda *.gcno
