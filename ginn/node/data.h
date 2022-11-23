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

#ifndef GINN_NODE_DATA_H
#define GINN_NODE_DATA_H

#include <ginn/node.h>
#include <ginn/util/util.h>

namespace ginn {

// Base type for the most common use case where an expression has a value, and
// the corresponding gradient with respect to that. Since many operations fit
// this bill, most nodes will derive from this type and the constructors
// provided here are for convenience purposes.
template <typename Scalar = Real>
class BaseDataNode : public Node<Scalar> {
 private:
  Tensor<Scalar> fx_, dfx_;

 public:
  // TODO: These used to be protected but nvcc breaks. Why?
  // Protected constructors for helping derived nodes
  BaseDataNode(DevPtr dev, const std::vector<BaseNodePtr>& ins)
      : Node<Scalar>(ins), fx_(dev), dfx_(dev) {}
  BaseDataNode(DevPtr dev,
               const Shape& shape,
               const std::vector<BaseNodePtr>& ins)
      : Node<Scalar>(ins), fx_(dev, shape), dfx_(dev) {}

  template <typename NodeType>
  BaseDataNode(const std::vector<NodeType>& ins)
      : Node<Scalar>(ins), fx_(best_dev(ins)), dfx_(fx_.dev()) {}

  BaseDataNode(DevPtr dev = cpu()) : fx_(dev), dfx_(dev) {}
  BaseDataNode(const Shape& shape) : fx_(cpu(), shape), dfx_(cpu()) {}
  BaseDataNode(DevPtr dev, const Shape& shape) : fx_(dev, shape), dfx_(dev) {}

  BaseDataNode(const std::initializer_list<BaseNodePtr>& ins)
      : BaseDataNode(std::vector<BaseNodePtr>(ins)) {}
  BaseDataNode(DevPtr dev, const std::initializer_list<BaseNodePtr>& ins)
      : BaseDataNode(dev, std::vector<BaseNodePtr>(ins)) {}

 private:
  bool has_grad_ = true;

 public:
  const Tensor<Scalar>& value() const override { return fx_; }
  const Tensor<Scalar>& grad() const override { return dfx_; }
  using Node<Scalar>::value;
  using Node<Scalar>::grad;

  bool has_grad() const override { return has_grad_; }
  virtual void has_grad(bool hg) { has_grad_ = hg; }

  // Do not override to keep the class abstract
  std::string name() const override = 0;
};

template <typename Scalar = Real>
class DataNode : public BaseDataNode<Scalar> {
 public:
  DataNode(DevPtr dev = cpu()) : BaseDataNode<Scalar>(dev) {
    this->forwarded = true;
  }
  DataNode(Shape shape) : BaseDataNode<Scalar>(std::move(shape)) {
    this->forwarded = true;
  }
  DataNode(DevPtr dev, Shape shape)
      : BaseDataNode<Scalar>(std::move(dev), std::move(shape)) {
    this->forwarded = true;
  }

  using Node<Scalar>::value;

  void move_to(const DevPtr& to) {
    this->value().move_to(to);
    this->grad().move_to(to);
  }

  void fill(Scalar val) { value().fill(val); }
  void set_zero() { value().set_zero(); }
  void set_ones() { value().set_ones(); }
  void set_random() { value().set_random(); }

  void reset_forwarded() override {}

  template <typename OtherScalar>
  auto cast() const {
    auto other = make_ref<DataNode<OtherScalar>>(this->dev(), this->shape());
    other->has_grad(this->has_grad());
    other->forwarded = this->forwarded;
    other->value() = this->value().template cast<OtherScalar>();
    other->grad() = this->grad().template cast<OtherScalar>();
    return other;
  }

  Shape shape() const override { return value().shape(); }

  std::string name() const override { return "DataNode"; }
};

template <typename Scalar = Real>
using DataPtr = Ptr<DataNode<Scalar>>;

template <typename Scalar = Real>
class ConstantLikeNode : public BaseDataNode<Scalar> {
 private:
  Scalar val_;

  void forward_() override {
    this->value().resize(this->in()->shape());
    this->value().fill(val_);
  }

 public:
  ConstantLikeNode(NodePtr<Scalar> in, Real val)
      : BaseDataNode<Scalar>({in}), val_(val) {}

  std::string name() const override { return "ConstantLike"; }
};

// Factory functions

GINN_MAKE_TEMPLATE_FACTORY(Data);

template <typename Scalar = Real>
auto Data(DevPtr dev, std::initializer_list<Size> shape) {
  return Data<Scalar>(dev, Shape(shape));
}
template <typename Scalar = Real>
auto FixedData(DevPtr dev, std::initializer_list<Size> shape) {
  return FixedData<Scalar>(dev, Shape(shape));
}

template <typename Scalar>
auto Constant(DevPtr dev, const Shape& shape, Scalar val) {
  auto x = FixedData<Scalar>(dev, shape);
  x->value().fill(val);
  return x;
}

template <typename Scalar, typename InScalar>
auto ConstantLike(NodePtr<InScalar> x, Scalar val) {
  return make_ref<ConstantLikeNode<Scalar>>(x, val);
}

template <typename Scalar = Real>
auto Zero(DevPtr dev, const Shape& s) {
  return Constant<Scalar>(dev, s, 0);
}

template <typename Scalar>
auto ZeroLike(NodePtr<Scalar> x) {
  return ConstantLike<Scalar>(x, 0);
}

template <typename Scalar = Real>
auto Ones(DevPtr dev, const Shape& s) {
  return Constant<Scalar>(dev, s, 1);
}

template <typename Scalar>
auto OnesLike(NodePtr<Scalar> x) {
  return ConstantLike<Scalar>(x, 1);
}

template <typename Scalar = Real>
auto Random(DevPtr dev, const Shape& shape) {
  auto x = Data<Scalar>(dev, shape);
  x->value().set_random();
  return x;
}

template <typename Scalar = Real>
auto Random(const Shape& shape) {
  return Random<Scalar>(cpu(), shape);
}

// Temporary? workaround for lack of "uniform" impl for Half
template <>
inline auto Random<Half>(DevPtr dev, const Shape& shape) {
  return Random(dev, shape)->cast<Half>();
}

template <>
inline auto Random<Half>(const Shape& shape) {
  return Random(shape)->cast<Half>();
}

template <typename Scalar = Real>
auto FixedRandom(DevPtr dev, const Shape& shape) {
  auto x = FixedData<Scalar>(dev, shape);
  x->value().set_random();
  return x;
}

template <int Rank, typename Scalar = Real>
auto Values(DevPtr dev, NestedInitList<Rank, Scalar> val) {
  auto x = Data<Scalar>(std::move(dev), shape_of<Size, Rank, Scalar>(val));
  x->value().template set<Rank>(val);
  return x;
}

template <int Rank, typename Scalar = Real>
auto Values(NestedInitList<Rank, Scalar> val) {
  return Values<Rank, Scalar>(cpu(), val);
}

// template <int Rank>
// auto Values(DevPtr dev, Real val) {
//  static_assert(Rank == 0);
//  auto x = Data(dev, {});
//  x->value().set({val});
//  return x;
//}
//
// template <int Rank>
// auto Values(Real val) {
//  return Values<Rank>(cpu(), val);
//}

// TODO: FixedValues? Overload Constant()?

} // namespace ginn

#endif
