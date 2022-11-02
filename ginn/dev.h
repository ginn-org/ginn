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

#ifndef GINN_DEV_H
#define GINN_DEV_H

#include <ginn/def.h>

#ifdef GINN_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <ginn/cudef.h>
#endif
#include <ginn/except.h>

namespace ginn {

inline int gpus() {
  int num_gpus = -1;
#ifdef GINN_ENABLE_GPU
  try {
    GINN_CUDA_CALL(cudaGetDeviceCount(&num_gpus));
  } catch (const CudaError&) {
    return 0;
  }
#endif
  return num_gpus;
}

inline Eigen::DefaultDevice& cpu_device() {
  static Eigen::DefaultDevice dev;
  return dev;
};

enum DeviceType { CPU, GPU, NULL_DEV };

struct DeviceId { // to distinguish multiple gpus
  const DeviceType type;
  const size_t idx = 0;
  bool operator==(const DeviceId& other) const {
    return type == other.type and idx == other.idx;
  }
};

class Device {
 public:
  virtual void* alloc(size_t size) = 0;
  virtual void* realloc(void* data, size_t size) = 0;
  virtual void free(void* data) = 0;
  virtual DeviceType type() const = 0;
  virtual DeviceId id() const = 0;
  virtual short precedence() const { return 0; }

  void copy(const Device& other, void* data, void* other_data, size_t size);

  virtual ~Device() = default;
};

using DevPtr = std::shared_ptr<Device>;

// For nodes that don't allocate anything and not use any device
// TODO: Reevaluate if this is really needed or not.
class NullDevice : public Device {
 public:
  void* alloc(size_t /*size*/) override { return nullptr; }
  void* realloc(void* /*data*/, size_t /*size*/) override { return nullptr; }
  void free(void* /*data*/) override {}
  DeviceType type() const override { return NULL_DEV; }
  DeviceId id() const override { return {NULL_DEV, 0}; }
  short precedence() const override { return -1; }
};

inline auto null_dev() {
  static auto dev = std::make_shared<NullDevice>();
  return dev;
}

class CpuDevice : public Device {
 public:
  void* alloc(size_t size) override { return malloc(size); }
  void* realloc(void* data, size_t size) override {
    return ::realloc(data, size);
  }
  void free(void* data) override { ::free(data); }
  DeviceType type() const override { return CPU; }
  DeviceId id() const override { return {CPU, 0}; }
};

inline auto Cpu() { return std::make_shared<CpuDevice>(); }

inline auto& cpu() {
  static auto dev = Cpu();
  return dev;
}

class PreallocCpuDevice : public Device {
 private:
  using Byte = std::byte;
  static_assert(sizeof(Byte) == 1);

  std::vector<Byte> storage_;
  Byte* offset_ = nullptr;

 public:
  PreallocCpuDevice(size_t size) : storage_(size), offset_(storage_.data()) {}

  void* alloc(size_t size) override {
    GINN_ASSERT((offset_ + size) < (storage_.data() + storage_.size()),
                "Prealloc device out of memory! Used: " +
                    std::to_string(offset_ - storage_.data()) +
                    " Attempted to allocate: " + std::to_string(size));
    void* rval = offset_;
    offset_ += size;
    return rval;
  }
  void* realloc(void* /*data*/, size_t /*size*/) override { return nullptr; }
  void free(void* /*data*/) override {}
  DeviceType type() const override { return CPU; }
  DeviceId id() const override { return {CPU, 0}; }
  short precedence() const override { return 1; }
  void reset() { offset_ = storage_.data(); }
  size_t used() const { return offset_ - storage_.data(); }
  size_t size() const { return storage_.size(); }
};

inline auto PreallocCpu(size_t size) {
  return std::make_shared<PreallocCpuDevice>(size);
}

#ifdef GINN_ENABLE_GPU
class GpuDevice : public Device {
 private:
  const size_t id_;
  CurandGenerator gen_;
  std::unique_ptr<cudaStream_t> stream_;
  std::unique_ptr<cublasHandle_t> handle_;
  std::unique_ptr<Eigen::GpuStreamDevice> gsd_;
  std::unique_ptr<Eigen::GpuDevice> gd_;

  void set_device() { GINN_CUDA_CALL(cudaSetDevice(id_)); }

 public:
  void* alloc(size_t size) override {
    set_device();
    void* p = nullptr;
    GINN_CUDA_CALL(cudaMalloc(&p, size));
    assert(p); // TODO: See if this is needed or if the above throws
    return p;
  }
  void* realloc(void* data, size_t size) override {
    free(data);
    return alloc(size);
  }
  void free(void* data) override {
    set_device();
    GINN_CUDA_CALL(cudaFree(data));
  }
  DeviceType type() const override { return GPU; }
  DeviceId id() const override { return {GPU, id_}; }
  auto& stream() { return *stream_; }
  auto& handle() { return *handle_; }
  auto& eigen_gpu_device() { return *gd_; }
  auto& gen() { return gen_; }

  GpuDevice(size_t id = 0) : id_(id), gen_(id) {
    set_device();
    GINN_CUDA_CALL(cudaDeviceSynchronize());
    stream_ = std::make_unique<cudaStream_t>();
    GINN_CUDA_CALL(cudaStreamCreate(&*stream_));
    handle_ = std::make_unique<cublasHandle_t>();
    GINN_CUBLAS_CALL(cublasCreate(&*handle_));
    gsd_ = std::make_unique<Eigen::GpuStreamDevice>(&*stream_, id_);
    gd_ = std::make_unique<Eigen::GpuDevice>(&*gsd_);
  }
  GpuDevice(GpuDevice&&) = default;
  ~GpuDevice() {
    if (handle_) { GINN_CUBLAS_CALL(cublasDestroy(*handle_)); }
    if (stream_) { GINN_CUDA_CALL(cudaStreamDestroy(*stream_)); }
  }
};

inline auto Gpu(size_t id) { return std::make_shared<GpuDevice>(id); }

inline auto& gpu(int idx = 0) {
  using Ptr = std::shared_ptr<GpuDevice>;
  auto helper = []() {
    std::vector<Ptr> devs;
    for (size_t i = 0; i < gpus(); i++) {
      devs.push_back(Gpu(i));
    }
    return devs;
  };
  static std::vector<Ptr> devs = helper();
  return devs.at(idx);
}

class PreallocGpuDevice : public Device {
 private:
  const size_t id_ = 0;
  thrust::device_vector<std::byte> storage_;
  thrust::device_ptr<std::byte> offset_ = nullptr;
  size_t size_ = 0;

  void set_device() { GINN_CUDA_CALL(cudaSetDevice(id_)); }

 public:
  PreallocGpuDevice(size_t id, size_t size) : id_(id) {
    set_device();
    GINN_CUDA_CALL(cudaDeviceSynchronize());

    size_ = size;
    storage_ = thrust::device_vector<std::byte>(size_);
    offset_ = storage_.data();
  }
  PreallocGpuDevice(size_t size) : PreallocGpuDevice(0, size) {}
  PreallocGpuDevice(const PreallocGpuDevice&) = delete;
  PreallocGpuDevice(PreallocGpuDevice&& other) = default;
  ~PreallocGpuDevice() = default;

  void* alloc(size_t size) override {
    GINN_ASSERT(
        (offset_ + size) < (storage_.data() + size_),
        "Prealloc device out of memory! Used: " + std::to_string(used()) +
            " Attempted to allocate: " + std::to_string(size));
    void* rval = thrust::raw_pointer_cast(offset_);
    offset_ = offset_ + size;
    return rval;
  }
  void* realloc(void* data, size_t size) override { return nullptr; }
  void free(void* data) override {}
  DeviceType type() const override { return GPU; }
  DeviceId id() const override { return {GPU, id_}; }
  short precedence() const override { return 1; }
  void reset() { offset_ = storage_.data(); }
  size_t used() const { return offset_ - storage_.data(); }
  size_t size() const { return storage_.size(); }
};

inline auto PreallocGpu(size_t id, size_t size) {
  return std::make_shared<PreallocGpuDevice>(id, size);
}
inline auto PreallocGpu(size_t size) {
  return std::make_shared<PreallocGpuDevice>(size);
}

inline CurandGenerator& curand_gen(int idx = 0) { return gpu(idx)->gen(); }
inline cublasHandle_t& cublas_handle(int idx = 0) { return gpu(idx)->handle(); }
inline Eigen::GpuDevice& gpu_device(int idx = 0) {
  return gpu(idx)->eigen_gpu_device();
}

class PeerAccess {
 private:
  std::vector<std::vector<bool>> peer_access_;

 public:
  PeerAccess() : peer_access_(gpus(), std::vector<bool>(gpus())) {
    for (int i = 0; i < gpus(); i++) {
      for (int j = 0; j < gpus(); j++) {
        if (i == j) {
          peer_access_[i][j] = true;
        } else {
          int access;
          GINN_CUDA_CALL(cudaDeviceCanAccessPeer(&access, i, j));
          if (access == 1) {
            // device i can access device j's memory
            GINN_CUDA_CALL(cudaSetDevice(i));
            GINN_CUDA_CALL(cudaDeviceEnablePeerAccess(j, 0));
            peer_access_[i][j] = true;
          }
        }
      }
    }
  }

  bool enabled(int i, int j) { return peer_access_[i][j]; }
};

#endif

inline void
Device::copy(const Device& other, void* data, void* other_data, size_t size) {
  if (type() == CPU and other.type() == CPU) {
    memcpy(data, other_data, size);
#ifdef GINN_ENABLE_GPU
  } else if (type() == CPU and other.type() == GPU) {
    GINN_CUDA_CALL(cudaSetDevice(other.id().idx));
    GINN_CUDA_CALL(cudaMemcpy(data, other_data, size, cudaMemcpyDeviceToHost));
  } else if (type() == GPU and other.type() == CPU) {
    auto& stream = gpu(id().idx)->stream();
    GINN_CUDA_CALL(cudaSetDevice(id().idx));
    GINN_CUDA_CALL(cudaMemcpyAsync(
        data, other_data, size, cudaMemcpyHostToDevice, stream));
  } else if (type() == GPU and other.type() == GPU) {
    if (id().idx == other.id().idx) {
      auto& stream = gpu(id().idx)->stream();
      GINN_CUDA_CALL(cudaSetDevice(id().idx));
      GINN_CUDA_CALL(cudaStreamSynchronize(gpu(other.id().idx)->stream()));
      GINN_CUDA_CALL(cudaMemcpyAsync(
          data, other_data, size, cudaMemcpyDeviceToDevice, stream));
    } else { // copy across devices
      static PeerAccess acc;
      if (acc.enabled(id().idx, other.id().idx)) {
        // Peer access to other gpu from this gpu is enabled
        // Make a direct copy across devices
        GINN_CUDA_CALL(cudaStreamSynchronize(gpu(other.id().idx)->stream()));
        GINN_CUDA_CALL(cudaSetDevice(id().idx));
        auto& stream = gpu(id().idx)->stream();
        GINN_CUDA_CALL(
            cudaMemcpyAsync(data, other_data, size, cudaMemcpyDefault, stream));
        // TODO: random deadlock here if i dont synchronize :< why?
        GINN_CUDA_CALL(cudaDeviceSynchronize());
      } else {
        // Naive copy through host
        std::vector<std::byte> tmp(size);
        GINN_CUDA_CALL(cudaSetDevice(other.id().idx));
        GINN_CUDA_CALL(cudaStreamSynchronize(gpu(other.id().idx)->stream()));
        GINN_CUDA_CALL(cudaMemcpy(
            (void*)tmp.data(), other_data, size, cudaMemcpyDeviceToHost));

        auto& stream = gpu(id().idx)->stream();
        GINN_CUDA_CALL(cudaSetDevice(id().idx));
        GINN_CUDA_CALL(cudaMemcpyAsync(
            data, (void*)tmp.data(), size, cudaMemcpyHostToDevice, stream));
      }
    }
#endif
  } else {
    GINN_THROW("Unexpected device in Device::copy()!");
  }
}

template <typename NodeType>
auto best_dev(const std::vector<NodeType>& ins) {
  // Inspect devices of all the inputs, adopt the one with the highest
  // precedence.
  GINN_ASSERT(not ins.empty());
  auto max = std::max_element(ins.begin(), ins.end(), [&](auto& i, auto& j) {
    return i->dev()->precedence() < j->dev()->precedence();
  });
  return (*max)->dev();
}

} // namespace ginn

#endif
