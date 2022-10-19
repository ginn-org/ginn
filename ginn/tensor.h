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

#ifndef GINN_TENSOR_H
#define GINN_TENSOR_H

#include <ginn/def.h>
#include <ginn/dev.h>
#include <ginn/eigenop/helpers.h>
#include <ginn/except.h>
#include <ginn/util/util.h>
#include <utility>
#include <vector>

namespace ginn {

// Tensor class seems to be a necessary evil because Eigen does not
// manage its own memory when GPU devices are used.

using Shape = std::vector<Size>;
template <Size N>
using Index = Eigen::array<Size, N>;

template <Size N>
Shape IndexToShape(const Index<N>& index) {
  Shape shape(N);
  for (Size i = 0; i < N; i++) { shape[i] = index[i]; }
  return shape;
}

template <Size N>
Index<N> ShapeToIndex(const Shape& shape) {
  GINN_ASSERT(N == shape.size());
  Index<N> rval;
  for (Size i = 0; i < N; i++) { rval[i] = shape[i]; }
  return rval;
}

// Forward declare helper types. These are invoked when a Tensor is on the
// lefthand side of a statement.

template <typename InnerExpr>
class LhsExpr;

template <typename InnerExpr, Size N>
class SliceExpr;

template <typename InnerExpr, Size N>
class ChipExpr;

// Core Tensor class, specialized as Tensor and IntTensor on ScalarType Real and
// Int, respectively.
template <typename ScalarType = Real>
class Tensor {
 public:
  using Scalar = ScalarType;

 private:
  DevPtr dev_ = nullptr;
  Shape shape_ = {0};
  Scalar* data_ = nullptr; // owned in most cases
  bool owns_mem_ = true;   // whether this Tensor owns data_

 public:
  DevPtr dev() const { return dev_; }
  Shape shape() const { return shape_; }
  Scalar* data() { return data_; }

  static Size size(const Shape& shape) {
    Size i = 1;
    for (auto j : shape) { i *= j; }
    return i;
  }

  Size size() const { return size(shape()); }

  // Move this tensor to another device
  auto& move_to(const DevPtr& to) {
    GINN_ASSERT(owns_mem_);
    if (dev_ == to) { return *this; }
    Tensor<Scalar> from(dev_);
    from.data_ = data_;
    from.shape_ = shape_; // to make sure destructor deallocates this

    dev_ = to;
    data_ = nullptr;

    if (size() > 0) {
      allocate(size());
      copy(from.dev(), from.data_, size());
    }

    return *this;
  }

  // Copy this tensor to another device
  auto copy_to(const DevPtr& to) const { return Tensor<Scalar>(to, *this); }

  // If tensor not on specified device, copy to it. Otherwise return a shallow
  // copy based on Tensor::map() which uses the same storage
  auto maybe_copy_to(const DevPtr& to) {
    if (dev_->id() == to->id()) {
      return Tensor<Scalar>().map(*this);
    } else {
      return copy_to(to);
    }
  }

  Scalar item() const {
    GINN_ASSERT(size() == 1, "item() can be invoked on size 1 tensors!");
    return copy_to(cpu()).v()[0];
  }

 private:
  // Manage memory
  void allocate(Size size) {
    GINN_ASSERT(data_ == nullptr);
    GINN_ASSERT(owns_mem_);
    if (size > 0) { data_ = (Scalar*)dev_->alloc(size * sizeof(Scalar)); }
  }

  void copy(const DevPtr& from, Scalar* other_data, Size size) {
    dev_->copy(*from, data_, other_data, size * sizeof(Scalar));
  }

  void reallocate(Size size) {
    GINN_ASSERT(owns_mem_);
    if (size > 0) {
      data_ = (Scalar*)dev_->realloc((void*)data_, size * sizeof(Scalar));
    } else {
      free();
    }
  }

  void free() {
    GINN_ASSERT(owns_mem_);
    if (size() > 0) { dev_->free(data_); }
    data_ = nullptr;
  }

 public:
  // Construct
  Tensor(DevPtr dev = cpu()) : dev_(std::move(dev)) {}
  Tensor(DevPtr dev, Shape shape) : dev_(std::move(dev)), shape_(std::move(shape)) {
    allocate(size());
  }
  Tensor(DevPtr dev, Shape shape, Scalar val)
      : dev_(std::move(dev)), shape_(std::move(shape)) {
    allocate(size());
    fill(val);
  }
  Tensor(const Shape& shape, const std::vector<Scalar>& val)
      : dev_(cpu()), shape_(shape) {
    GINN_ASSERT(size() == (Size)val.size(),
                "Size of Shape (" + std::to_string(size()) +
                    ") does not match size of values (" +
                    std::to_string(val.size()) + ")!");
    allocate(size());
    auto vmap = v();
    for (size_t i = 0; i < val.size(); i++) { vmap[i] = val[i]; }
  }
  Tensor(DevPtr dev, Shape shape, std::vector<Scalar> val)
      : Tensor<Scalar>(std::move(shape), std::move(val)) {
    move_to(dev);
  }

  template <int Rank>
  Tensor(DevPtr dev, NestedInitList<Rank, Scalar> val)
      : dev_(std::move(dev)), shape_(shape_of<Size, Rank, Scalar>(val)) {
    set<Rank>(val);
  }

  template <int Rank>
  Tensor(NestedInitList<Rank, Scalar> val)
      : dev_(cpu()), shape_(shape_of<Size, Rank, Scalar>(val)) {
    set<Rank>(val);
  }

  // Copy & Move construct
  Tensor(const Tensor<Scalar>& other) : dev_(other.dev_), shape_(other.shape_) {
    allocate(size());
    copy(other.dev_, other.data_, size());
  }
  Tensor(Tensor<Scalar>&& other)
      : dev_(other.dev_),
        shape_(other.shape_),
        data_(other.data_),
        owns_mem_(other.owns_mem_) {
    other.data_ = nullptr;
    other.shape_ = {0};
  }

  // Copy & Move assign
  // TODO: Consider operator= copying the device as well, to avoid confusion.
  // Maybe only having the special ctor and `move_to()` as well as
  // `maybe_copy_to()` is enough to transfer tensor across devices. This just
  // seems confusing.
  auto& operator=(const Tensor<Scalar>& other) {
    if (this == &other) { return *this; }
    resize(other.shape_);
    copy(other.dev_, other.data_, size());
    return *this;
  }
  auto& operator=(Tensor<Scalar>&& other) {
    if (dev_ == other.dev_) {
      free();
      shape_ = other.shape_;
      data_ = other.data_;
      dev_ = other.dev_;
      other.data_ = nullptr;
      other.shape_ = {0};
    } else {
      // operator= is defined to keep the device, therefore move is not possible
      *this = other;
    }
    return *this;
  }
  auto& operator+=(const Tensor<Scalar>& other) {
    *this += other.t();
    return *this;
  }

  // Construct by copying across devices
  Tensor(DevPtr dev, const Tensor<Scalar>& other)
      : dev_(std::move(dev)), shape_(other.shape_) {
    allocate(size());
    copy(other.dev_, other.data_, size());
  }

  // Destroy
  ~Tensor() {
    if (owns_mem_) { free(); }
  }

  // Resize
  void resize(const Shape& shape) {
    if (size() != size(shape)) {
      if (size() == 0) {
        allocate(size(shape));
      } else {
        reallocate(size(shape));
      }
    }
    shape_ = shape;
  }

  template <typename OtherScalar>
  Tensor<OtherScalar> cast() const {
    Tensor<OtherScalar> other(dev(), shape());
    other = t().template cast<OtherScalar>();
    return other;
  }

  // Views on Tensors

  // Reduce (by increasing or decreasing rank) shape to a given rank
  static Shape reduce(const Shape& shape, Size dim) {
    if (dim < (Size)shape.size()) {
      Size last = shape[dim - 1];
      for (Size i = shape.size() - 1; i >= dim; i--) { last *= shape[i]; }
      Shape new_shape(dim);
      for (Size i = 0; i < dim; i++) {
        new_shape[i] = (i == (dim - 1)) ? last : shape[i];
      }
      return new_shape;
    } else {
      Shape new_shape(shape);
      while ((Size)new_shape.size() < dim) { new_shape.push_back(1); }
      return new_shape;
    }
  }

  // Make this tensor a (possibly reshaped) non-memory-owning shallow copy
  // of the other
  auto& map(Tensor<Scalar>& other, const Shape& shape) {
    GINN_ASSERT(size(shape) == size(other.shape()));
    if (owns_mem_) { free(); }
    dev_ = other.dev_;
    data_ = other.data_;
    shape_ = shape;
    owns_mem_ = false;
    return *this;
  }

  // Make this tensor a (possibly reshaped) non-memory-owning shallow copy
  // of the other
  auto& map(Tensor<Scalar>& other) { return map(other, other.shape()); }

  // Make this tensor a non-memory-owning shallow copy of a subtensor
  // of the other
  auto& map(Tensor<Scalar>& other, Shape shape, Size offset) {
    GINN_ASSERT((size(shape) + offset) <= size(other.shape()));
    if (owns_mem_) { free(); }
    dev_ = other.dev_;
    data_ = other.data_ + offset;
    shape_ = shape;
    owns_mem_ = false;
    return *this;
  }

  // Return a possibly reshaped map (shallow, non-memory-owning copy) to this
  // tensor
  auto reshaped(const Shape& shape) {
    Tensor<Scalar> t;
    t.map(*this, shape);
    return t;
  }

  // View as classical (CPU) Eigen matrix
  auto m() {
    GINN_ASSERT(dev()->type() == CPU, "m() can only be invoked on Cpu tensors!");
    auto dims = reduce(shape_, 2);
    return MatrixMap<Scalar>(data_, dims[0], dims[1]);
  }
  // TODO: should there be a Map type to const?
  auto m() const {
    GINN_ASSERT(dev()->type() == CPU, "m() can only be invoked on Cpu tensors!");
    auto dims = reduce(shape_, 2);
    return MatrixMap<Scalar>(data_, dims[0], dims[1]);
  }

  auto v() {
    GINN_ASSERT(dev()->type() == CPU, "v() can only be invoked on Cpu tensors!");
    auto dims = reduce(shape_, 1);
    return VectorMap<Scalar>(data_, dims[0]);
  }
  auto v() const {
    GINN_ASSERT(dev()->type() == CPU, "v() can only be invoked on Cpu tensors!");
    auto dims = reduce(shape_, 1);
    return VectorMap<Scalar>(data_, dims[0]);
  }

  // begin() and end() help with feeding tensors into generic algorithms
  const Scalar* begin() const {
    GINN_ASSERT(dev()->type() == CPU,
                "begin() can only be invoked on Cpu tensors!");
    return data_;
  }

  const Scalar* end() const {
    GINN_ASSERT(dev()->type() == CPU,
                "end() can only be invoked on Cpu tensors!");
    return data_ + size();
  }

  // wrap a Rank-0 (single element) Tensor around a scalar entry of this tensor
  auto operator()(Size i) {
    Tensor<Scalar> t;
    t.dev_ = dev_;
    t.shape_ = {}; // has to be rank 0
    t.owns_mem_ = false;
    t.data_ = data_ + i;
    return t;
  }

  auto operator()(Size i, Size j) {
    Tensor<Scalar> t;
    t.dev_ = dev_;
    t.shape_ = {}; // has to be rank 0
    t.owns_mem_ = false;
    t.data_ = data_ + i + j * rows();
    return t;
  }

  // View as tensor (Eigen / unsupported) with given rank
  template <size_t Rank>
  TensorMap<Scalar, Rank> view();
  template <size_t Rank>
  const TensorMap<Scalar, Rank> view() const;

  // View as rank 2 tensor
  TensorMap<Scalar, 2> t();
  TensorMap<Scalar, 2> const t() const;

  // This helper method is used to simplify device (CPU/GPU) based dispatching
  // of evaluators, such as :
  //     tensor.lhs() = ...   or
  //     tensor.lhs() += ...
  template <unsigned long R = 2>
  auto lhs() {
    return LhsExpr<decltype(view<R>())>(view<R>(), dev());
  }

  template <unsigned long N>
  auto slice(const Index<N>& offsets, const Index<N>& sizes) {
    using LhsExprType = decltype(view<N>());
    return SliceExpr<LhsExprType, N>(dev(), view<N>(), offsets, sizes);
  }

  template <Size N>
  auto chip(Size offset, Size dim) {
    using LhsExprType = decltype(view<N>());
    return ChipExpr<LhsExprType, N>(dev(), view<N>(), offset, dim);
  }

  // Operator overloads for Eigen expressions to avoid having to use .lhs()
  // when possible
  template <typename RhsExpr>
  void operator=(RhsExpr e) {
    lhs<eigen::ndims<RhsExpr>()>() = e;
  }

  template <typename RhsExpr>
  void operator+=(RhsExpr e) {
    lhs<eigen::ndims<RhsExpr>()>() += e;
  }

  template <typename RhsExpr>
  void operator-=(RhsExpr e) {
    lhs<eigen::ndims<RhsExpr>()>() -= e;
  }

  Size rows() const { return reduce(shape_, 2)[0]; }
  Size cols() const { return reduce(shape_, 2)[1]; }
  Shape shape2() const { return reduce(shape_, 2); }
  void fill(Scalar c) { lhs() = t().constant(c); }
  void set_zero() { fill(Scalar{0}); }
  void set_ones() { fill(Scalar{1}); }

  void set_random() {
    // TODO: making a copy here for now, get rid of this
    if constexpr (std::is_same_v<Scalar, Half>) {
      Tensor<Real> tmp(dev(), shape());
      tmp.set_random();
      *this = tmp.cast<Scalar>();
    } else {
      if (dev_->type() == CPU) {
        m().setRandom();
      }
#ifdef GINN_ENABLE_GPU
      else if (dev_->type() == GPU) {
        curand_gen(dev_->id().idx).uniform(data_, size());
        lhs() = -1 + (2 * t());
      }
#endif
      else {
        GINN_ASSERT(false);
      }
    }
  }

  void set(const std::vector<Scalar>& val) {
    if (dev_->type() == CPU) {
      auto v_ = v();
      auto s = std::min((size_t)v_.size(), val.size());
      for (size_t i = 0; i < s; i++) { v_[i] = val[i]; }
#ifdef GINN_ENABLE_GPU
    } else if (dev_->type() == GPU) {
      Tensor<Scalar> tmp(cpu(), shape());
      tmp = *this;
      tmp.set(val);
      *this = tmp;
#endif
    } else {
      GINN_THROW("Unexpected device type!");
    }
  }

  template <typename... Args>
  void set(const Args&... args) {
    set(std::vector<Scalar>{Scalar(args)...});
  }

  template <int Rank>
  void set(NestedInitList<Rank, Scalar> val) {
    resize(shape_of<Size, Rank, Scalar>(val));
    if (dev_->type() == CPU) {
      assign<Rank, Scalar>(view<Rank>(), val);
#ifdef GINN_ENABLE_GPU
    } else if (dev_->type() == GPU) {
      Tensor<Scalar> tmp(cpu(), shape());
      tmp.set<Rank>(val);
      *this = tmp;
#endif
    } else {
      GINN_THROW("Unexpected device type!");
    }
  }

  bool operator==(const Tensor<Scalar>& other) const {
    if (dev()->type() == CPU) {
      if (other.dev()->type() == CPU) {
        return shape_ == other.shape_ and v() == other.v();
      }
      Tensor<Scalar> other_cp(cpu());
      other_cp = other;
      return operator==(other_cp);
    }
    // TODO: compare gpu tensors without moving both to cpu
    Tensor<Scalar> cp(cpu());
    cp = *this;
    return cp == other;
  }

  void save(std::ostream& out) const {
    if (dev()->type() == CPU) {
      out << shape_.size() << std::endl;
      for (Size dim : shape_) { out << dim << " "; }
      out << std::endl;
      for (Size i = 0; i < size(); i++) {
        if (i > 0) { out << " "; }
        out << data_[i];
      }
      out << std::endl;
    } else {
      Tensor<Scalar> cp(cpu());
      cp = *this;
      cp.save(out);
    }
  }

  void load(std::istream& in) {
    if (dev()->type() == CPU) {
      Size r;
      in >> r;
      Shape new_shape(r);
      for (Size& dim : new_shape) { in >> dim; }
      resize(new_shape);
      for (Size i = 0; i < size(); i++) {
        double val;
        in >> val;
        data_[i] = Scalar(val);
      }
      char end_of_line;
      in.get(end_of_line);
      GINN_ASSERT(end_of_line == '\n');
    } else {
      Tensor<Scalar> cp(cpu());
      cp.load(in);
      *this = cp;
    }
  }
};

template <typename Scalar, size_t... Indices>
auto view_impl(Scalar* data,
               const Shape& shape,
               std::index_sequence<Indices...>) {
  if constexpr (sizeof...(Indices) == 0) {
    return TensorMap<Scalar, 0>(data);
  } else {
    auto dims = Tensor<Scalar>::reduce(shape, sizeof...(Indices));
    return TensorMap<Scalar, sizeof...(Indices)>(data, (dims[Indices])...);
  }
}

template <typename Scalar>
template <size_t Rank>
TensorMap<Scalar, Rank> Tensor<Scalar>::view() {
  return view_impl(data_, shape_, std::make_index_sequence<Rank>());
}

template <typename Scalar>
template <size_t Rank>
const TensorMap<Scalar, Rank> Tensor<Scalar>::view() const {
  return view_impl(data_, shape_, std::make_index_sequence<Rank>());
}

template <typename Scalar>
inline TensorMap<Scalar, 2> Tensor<Scalar>::t() {
  return view<2>();
}

template <typename Scalar>
inline const TensorMap<Scalar, 2> Tensor<Scalar>::t() const {
  return view<2>();
}

// Lefthandside expressions for Eigen expressions
//
// This helper class is used to simplify device (CPU/GPU) based dispatching
// of evaluators, such as :
//     Lhs(CPU, SomeEigenExpr) = OtherEigenExpr
//     Lhs(GPU, SomeEigenExpr) += OtherEigenExpr
// Lefthandside Expressions
template <typename InnerExpr>
class LhsExpr {
 public:
  InnerExpr e;
  DevPtr dev;
  LhsExpr(InnerExpr a_e, DevPtr a_dev) : e(a_e), dev(std::move(a_dev)) {}

#ifdef GINN_ENABLE_GPU
#define LHSEXPR_IMPLEMENT(op)                                                  \
  template <typename RhsExpr>                                                  \
  void op(RhsExpr rhs) {                                                       \
    if (dev->type() == CPU) {                                                   \
      e.device(cpu_device()).op(rhs);                                          \
    } else if (dev->type() == GPU) {                                            \
      auto& gd = gpu_device(dev->id().idx);                                     \
      GINN_CUDA_CALL(cudaSetDevice(dev->id().idx));                             \
      e.device(gd).op(rhs);                                                    \
    } else {                                                                   \
      GINN_THROW("Unexpected device!");                                        \
    }                                                                          \
  }
#else
#define LHSEXPR_IMPLEMENT(op)                                                  \
  template <typename RhsExpr>                                                  \
  void op(RhsExpr rhs) {                                                       \
    if (dev->type() == CPU) {                                                   \
      e.device(cpu_device()).op(rhs);                                          \
    } else {                                                                   \
      GINN_THROW("Unexpected device!");                                        \
    }                                                                          \
  }
#endif

  LHSEXPR_IMPLEMENT(operator=)
  LHSEXPR_IMPLEMENT(operator+=)
  LHSEXPR_IMPLEMENT(operator-=)
};

template <typename InnerExpr>
auto Lhs(DevPtr dev, InnerExpr e) {
  return LhsExpr<InnerExpr>(e, std::move(dev));
}

template <typename LhsExpr, Size N>
class SliceExpr {
 private:
  DevPtr dev_;
  LhsExpr lhs_;
  Index<N> offsets_, sizes_;

 public:
  SliceExpr(DevPtr dev, LhsExpr lhs, Index<N> offsets, Index<N> sizes)
      : dev_(std::move(dev)),
        lhs_(std::move(lhs)),
        offsets_(std::move(offsets)),
        sizes_(std::move(sizes)) {}
  template <typename RhsExpr>
  void operator=(RhsExpr rhs) {
    Lhs(dev_, lhs_.slice(offsets_, sizes_)) = rhs;
  }
  template <typename RhsExpr>
  void operator+=(RhsExpr rhs) {
    Lhs(dev_, lhs_.slice(offsets_, sizes_)) += rhs;
  }
};

template <typename LhsExpr, Size N>
class ChipExpr {
 private:
  DevPtr dev_;
  LhsExpr lhs_;
  Size offset_, dim_;

 public:
  ChipExpr(DevPtr dev, LhsExpr lhs, Size offset, Size dim)
      : dev_(std::move(dev)), lhs_(std::move(lhs)), offset_(offset), dim_(dim) {}
  template <typename RhsExpr>
  void operator=(RhsExpr rhs) {
    Lhs(dev_, lhs_.chip(offset_, dim_)) = rhs;
  }
  template <typename RhsExpr>
  void operator+=(RhsExpr rhs) {
    Lhs(dev_, lhs_.chip(offset_, dim_)) += rhs;
  }
};

} // end namespace ginn

#endif
