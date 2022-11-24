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

#ifndef GINN_WEIGHT_H
#define GINN_WEIGHT_H

#include <atomic>
#include <ginn/node.h>
#include <mutex>
#include <string>

namespace ginn {

enum class Copy { Tied, Deep };

template <typename Scalar = Real>
class WeightNode : public Node<Scalar> {
 private:
  static size_t next_id() {
    static std::atomic<size_t> id{0};
    return id++;
  }

  size_t id_;
  Tensor<Scalar> dfx_;
  std::shared_ptr<Tensor<Scalar>> fx_; // to be shared across threads
  std::shared_ptr<std::mutex> access_;

  bool has_grad_ = true;

 public:
  std::string label;

  Tensor<Scalar>& value() {
    GINN_ASSERT(fx_);
    return *fx_;
  }
  const Tensor<Scalar>& value() const override {
    GINN_ASSERT(fx_);
    return *fx_;
  }
  const Tensor<Scalar>& grad() const override { return dfx_; }
  using Node<Scalar>::grad;

  bool has_grad() const override { return has_grad_; }
  virtual void has_grad(bool hg) { has_grad_ = hg; }

  size_t id() const { return id_; }
  std::mutex& access() { return *access_; }

  // Create
  WeightNode(DevPtr dev = cpu(), const Shape& shape = {0})
      : id_(next_id()),
        dfx_(dev),
        fx_(std::make_shared<Tensor<Scalar>>(dev, shape)),
        access_(std::make_shared<std::mutex>()) {
    this->forwarded = true;
  }

  WeightNode(const WeightNode& other)
      : Node<Scalar>((const Node<Scalar>&)other),
        id_(next_id()),
        dfx_(other.dfx_),
        fx_(std::make_shared<Tensor<Scalar>>(*other.fx_)),
        access_(std::make_shared<std::mutex>()) {}

  // TODO: is this safe? maybe deep copy here, rely on tie() when needed?
  void operator=(const WeightNode& other) {
    dfx_ = other.dfx_;
    fx_ = std::make_shared<Tensor<Scalar>>(*other.fx_);
  }

  void reset_forwarded() override {}

  // TODO: Is the following safe with tied copies? Probably not
  void move_to(const DevPtr& to) {
    fx_->move_to(to);
    dfx_.move_to(to);
  }

  // Tie this weight to other. value() is shared but grad() is not.
  void tie(const Ptr<WeightNode>& other) {
    if (other) {
      id_ = other->id_;
      fx_ = other->fx_;
      dfx_.move_to(fx_->dev());
      dfx_ = other->dfx_;
      access_ = other->access_;
      this->has_grad_ = other->has_grad_;
    }
  }

  auto copy(Copy mode = Copy::Tied) {
    if (mode == Copy::Deep) { return make_ptr<WeightNode>(*this); }
    auto tied_copy = make_ptr<WeightNode>(this->fx_->dev());
    tied_copy->id_ = this->id_;
    tied_copy->fx_ = this->fx_;
    tied_copy->dfx_ = this->dfx_;
    tied_copy->access_ = this->access_;
    tied_copy->has_grad_ = this->has_grad_;
    return tied_copy;
  }

  template <typename RhsScalar>
  void fill(RhsScalar val) {
    value().fill(Scalar(val));
  }

  void set_zero() { value().set_zero(); }
  void set_ones() { value().set_ones(); }
  void set_random() { value().set_random(); }

  Shape shape() const override { return value().shape(); }

  std::string name() const override {
    std::string shp = "{";
    for (size_t d : this->shape()) { shp += std::to_string(d) + ", "; }
    return "WeightNode" + std::to_string(id_) +
           (label.empty() ? "" : (" (" + label + ")")) + shp;
  }
};

template <typename Scalar>
using WeightPtr = Ptr<WeightNode<Scalar>>;

template <typename Scalar>
using ConstWeightPtr = Ptr<const WeightNode<Scalar>>;

template <typename Scalar = Real>
auto Weight(DevPtr dev = cpu(), const Shape& s = {0}) {
  return make_ptr<WeightNode<Scalar>>(dev, s);
}
template <typename Scalar = Real>
auto Weight(DevPtr dev, std::initializer_list<Size> shape) {
  return Weight<Scalar>(dev, Shape(shape));
}
template <typename Scalar>
auto Weight(const WeightNode<Scalar>& other) {
  return make_ptr<WeightNode<Scalar>>(other);
}

template <typename Scalar = Real>
auto FixedWeight(DevPtr dev = cpu(), const Shape& s = {0}) {
  auto w = Weight<Scalar>(dev, s);
  w->has_grad_ = false;
  return w;
}

template <typename Scalar = Real>
auto FixedWeight(DevPtr dev, std::initializer_list<Size> shape) {
  auto w = Weight<Scalar>(dev, Shape(shape));
  w->has_grad(false);
  return w;
}

} // end namespace ginn

#endif
