// Copyright 2022 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GINN_CUDEF_H
#define GINN_CUDEF_H

#include <ginn/except.h>
#include <iostream>
#include <memory>

#define GINN_CUDA_CALL(x)                                                      \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA failure in " << #x << std::endl                       \
                << cudaGetErrorString(err) << std::endl;                       \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

#define GINN_CURAND_CALL(x)                                                    \
  do {                                                                         \
    auto err = x;                                                              \
    if (err != CURAND_STATUS_SUCCESS) {                                        \
      std::cerr << "CURAND failure in " << #x << std::endl                     \
                << err << std::endl                                            \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

#define GINN_CUBLAS_CALL(x)                                                    \
  do {                                                                         \
    cublasStatus_t stat = x;                                                   \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "CUBLAS failure in " << #x << std::endl                     \
                << stat << std::endl                                           \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

namespace ginn {

class CurandGenerator {
 private:
  std::unique_ptr<curandGenerator_t> gen;
  const int dev_; // device index
  void set_device() { GINN_CUDA_CALL(cudaSetDevice(dev_)); }

 public:
  CurandGenerator(int dev = 0) : dev_(dev) {
    set_device();
    gen = std::make_unique<curandGenerator_t>();
    GINN_CURAND_CALL(curandCreateGenerator(&*gen, CURAND_RNG_PSEUDO_DEFAULT));
    // TODO: set seeds properly
    GINN_CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*gen, 1234ULL + dev_));
  }
  CurandGenerator(CurandGenerator&&) = default;
  ~CurandGenerator() {
    set_device();
    if (gen) { GINN_CURAND_CALL(curandDestroyGenerator(*gen)); }
  }
  void uniform(Real* data, size_t size) {
    set_device();
    GINN_CURAND_CALL(curandGenerateUniform(*gen, data, size));
  }
  void uniform(Int* data, size_t size) {
    GINN_THROW("Random generation for Int GPU tensors is not implemented yet!");
  }
};

} // namespace ginn

#endif
