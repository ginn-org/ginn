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

#ifndef GINN_NODE_COMMON_H
#define GINN_NODE_COMMON_H

#include <ginn/node.h>
#include <ginn/node/data.h>
#include <ginn/util/traits.h>

namespace ginn {

template <typename Scalar>
class AddScalarNode : public BaseDataNode<Scalar> {
  static_assert(not std::is_same_v<Scalar, bool>);
  static_assert(ginn::is_arithmetic_v<Scalar>);

 protected:
  NodePtr<Scalar> in_;
  Scalar val_;

  void forward_() override {
    value().resize(in_->value().shape());
    value() = in_->value().t() + val_;
  }

  void backward_() override {
    if (in_->has_grad()) { in_->grad() += grad().t(); }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  template <typename RightScalar,
            typename = std::enable_if_t<ginn::is_arithmetic_v<RightScalar>>>
  AddScalarNode(NodePtr<Scalar> a, RightScalar b)
      : BaseDataNode<Scalar>({a}), in_(a), val_(Scalar(b)) {}

  std::string name() const override { return "AddScalar"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(AddScalar);

template <typename Scalar>
class AddNode : public BaseDataNode<Scalar> {
 protected:
  std::vector<NodePtr<Scalar>> ins_;

  // Helper to add N input tensors like:
  // in(0)->value().t() + ... + in(N-1)->value().t().
  // Eigen benefits from compile time unrolling of this sum.
  template <size_t... N>
  auto add_helper(size_t i, std::integer_sequence<size_t, N...>) {
    return (ins_[i + N]->value().t() + ...);
  }

  template <size_t N>
  auto add_helper(size_t i) {
    return add_helper(i, std::make_integer_sequence<size_t, N>());
  }

  void forward_() override {
    value().resize(ins_[0]->value().shape());

    auto compute = [&](const auto& rhs, bool accumulate) {
      accumulate ? (value() += rhs) : (value() = rhs);
    };

    for (size_t i = 0; i < ins_.size(); i += 4) {
      size_t remaining = std::min(ins_.size() - i, (size_t)4);
      if (remaining == 0) { remaining = 4; }
      switch (remaining) {
      case 1: compute(ins_[i]->value().t(), i > 0); break;
      case 2: compute(add_helper<2>(i), i > 0); break;
      case 3: compute(add_helper<3>(i), i > 0); break;
      case 4: compute(add_helper<4>(i), i > 0); break;
      }
    }
  }

  void backward_() override {
    for (size_t i = 0; i < ins_.size(); i++) {
      if (ins_[i]->has_grad()) { ins_[i]->grad() += grad().t(); }
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  AddNode(const std::vector<NodePtr<Scalar>>& ins)
      : BaseDataNode<Scalar>(ins), ins_(ins) {}

  template <typename... Args>
  AddNode(const NodePtr<Scalar>& in, const Args&... args)
      : AddNode(std::vector<NodePtr<Scalar>>{in, args...}) {
  }

  void set_ins(const std::vector<BaseNodePtr>& ins) override {
    BaseNode::ins_ = ins;
    ins_ = derived_cast<Node<Scalar>>(ins);
  }

  std::string name() const override { return "Add"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Add);

template <typename Scalar>
class SubtractNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> left_, right_;

  void forward_() override {
    value().resize(left_->shape());
    value() = left_->value().t() - right_->value().t();
  }

  void backward_() override {
    if (left_->has_grad()) { left_->grad() += grad().t(); }
    if (right_->has_grad()) { right_->grad() -= grad().t(); }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  SubtractNode(NodePtr<Scalar> a, NodePtr<Scalar> b)
      : BaseDataNode<Scalar>({a, b}), left_(a), right_(b) {}

  std::string name() const override { return "Subtract"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Subtract);

template <typename Scalar>
class ProdScalarNode : public BaseDataNode<Scalar> {
 protected:
  NodePtr<Scalar> in_;
  Scalar val_;

  void forward_() override {
    value().resize(in_->value().shape());
    value() = in_->value().t() * val_;
  }

  void backward_() override {
    if (in_->has_grad()) { in_->grad() += grad().t() * val_; }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  template <typename RightScalar,
            typename = std::enable_if_t<ginn::is_arithmetic_v<RightScalar>>>
  ProdScalarNode(NodePtr<Scalar> a, RightScalar b)
      : BaseDataNode<Scalar>({a}), in_(a), val_(Scalar(b)) {}

  std::string name() const override { return "ProdScalar"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(ProdScalar);

template <typename Scalar>
class CwiseProdNode : public BaseDataNode<Scalar> {
 protected:
  NodePtr<Scalar> a_, b_;

  void forward_() override {
    value().resize(a_->value().shape());
    value() = a_->value().t() * b_->value().t();
  }

  void backward_() override {
    if (a_->has_grad()) { a_->grad() += grad().t() * b_->value().t(); }
    if (b_->has_grad()) { b_->grad() += grad().t() * a_->value().t(); }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  CwiseProdNode(NodePtr<Scalar> a, NodePtr<Scalar> b)
      : BaseDataNode<Scalar>({a, b}), a_(a), b_(b) {}

  std::string name() const override { return "CwiseProd"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(CwiseProd);

template <typename Scalar>
class CwiseProdAddNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> a_, b_, c_;
  bool broadcast_;
  Scalar multiplier_bias_{0};

  void forward_() override {
    const auto s0 = a_->shape(), s1 = b_->shape(), s2 = c_->shape();
    if (s0 == s1 and s1 == s2) {
      broadcast_ = false;
    } else if (s1.size() == 1 and s2.size() == 1) {
      broadcast_ = true;
      GINN_ASSERT(s0[0] == s1[0] and s1[0] == s2[0],
                  "Unexpected shapes for CwiseProdAdd!");
    } else {
      GINN_THROW("Unexpected shapes for CwiseProdAdd!");
    }

    value().resize(s0);
    auto a_t = a_->value().t();
    auto b_t = b_->value().t();
    auto c_t = c_->value().t();

    if (not broadcast_) {
      value() = a_t * (b_t + multiplier_bias_) + c_t;
    } else {
      auto cols = a_->shape2()[1];
      value() = a_t * (b_t + multiplier_bias_).broadcast(Index<2>{1, cols}) +
                c_t.broadcast(Index<2>{1, cols});
    }
  }

  void backward_() override {
    auto a_t = a_->value().t();
    auto b_t = b_->value().t();

    if (not broadcast_) {
      if (a_->has_grad()) {
        a_->grad() += grad().t() * (b_t + multiplier_bias_);
      }
      if (b_->has_grad()) { b_->grad() += grad().t() * a_t; }
      if (c_->has_grad()) { c_->grad() += grad().t(); }
    } else {
      if (a_->has_grad()) {
        const auto cols = a_->shape2()[1];
        a_->grad() +=
            grad().t() * (b_t + multiplier_bias_).broadcast(Index<2>{1, cols});
      }
      if (b_->has_grad()) { b_->grad() += (grad().t() * a_t).sum(Index<1>{1}); }
      if (c_->has_grad()) { c_->grad() += grad().t().sum(Index<1>{1}); }
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  CwiseProdAddNode(NodePtr<Scalar> a,
                   NodePtr<Scalar> b,
                   NodePtr<Scalar> c,
                   Scalar multiplier_bias = Scalar(0))
      : BaseDataNode<Scalar>({a, b, c}),
        a_(a),
        b_(b),
        c_(c),
        multiplier_bias_(multiplier_bias) {}

  std::string name() const override { return "CwiseProdAdd"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(CwiseProdAdd);

template <typename Scalar>
class CwiseMaxNode : public BaseDataNode<Scalar> {
 private:
  std::vector<NodePtr<Scalar>> ins_;

  void forward_() override {
    value().resize(ins_[0]->value().shape());
    value() = ins_[0]->value().t();
    for (size_t i = 1; i < ins_.size(); i++) {
      value() = value().t().cwiseMax(ins_[i]->value().t());
    }
  }

  void backward_() override {
    for (size_t i = 0; i < ins_.size(); i++) {
      if (ins_[i]->has_grad()) {
        ins_[i]->grad() +=
            grad().t() *
            (ins_[i]->value().t() == value().t()).template cast<Scalar>();
      }
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  CwiseMaxNode(const std::vector<NodePtr<Scalar>> ins)
      : BaseDataNode<Scalar>(ins), ins_(ins) {}
  template <typename... Args>
  CwiseMaxNode(const NodePtr<Scalar>& in, Args&&... args)
      : BaseDataNode<Scalar>(std::vector<NodePtr<Scalar>>{in, args...}),
        ins_(std::vector<NodePtr<Scalar>>{in, args...}) {}

  std::string name() const override { return "CwiseMax"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(CwiseMax);

// TODO: explicit seed?
template <typename Scalar>
class DropoutNode : public BaseDataNode<Scalar> {
  static_assert(
      ginn::is_floating_point_v<Scalar>,
      "Dropout scalar needs to be floating point because of scaling!");

 protected:
  NodePtr<Scalar> in_;
  Real p_;
  Tensor<bool> mask_;

  void forward_() override {
    value().resize(in_->value().shape());
    if (p_ == 1.) {
      value().set_zero();
    } else {
      mask_.resize(in_->shape());
      mask_ = (mask_.t().template cast<Real>().random() >= p_);
      Scalar tmp(1. / (1. - p_));
      // TODO: should I benchmark this against using .select()?
      value() = in_->value().t() * mask_.t().template cast<Scalar>() * tmp;
    }
  }

  void backward_() override {
    if (in_->has_grad() and p_ < 1.) {
      Scalar tmp(1. / (1. - p_));
      in_->grad() += grad().t() * mask_.t().template cast<Scalar>() * tmp;
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  std::string name() const override { return "Dropout"; }

  DropoutNode(const NodePtr<Scalar>& in, Real p)
      : BaseDataNode<Scalar>(std::vector<BaseNodePtr>{in}),
        in_(in),
        p_(p),
        mask_(in->dev()) {
    GINN_ASSERT(p_ >= 0 and p_ <= 1, "Dropout probability is not in [0, 1]!");
  }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Dropout);

// Arithmetic operator overloads

template <typename Left,
          typename Right,
          typename = std::enable_if_t<ginn::is_node_ptr_v<Left> or
                                      ginn::is_node_ptr_v<Right>>>
auto operator+(const Left& a, const Right& b) {
  if constexpr (ginn::is_node_ptr_v<Left>) {
    if constexpr (ginn::is_node_ptr_v<Right>) {
      return Add(a, b);
    } else if constexpr (ginn::is_arithmetic_v<Right>) {
      return AddScalar(a, b);
    } else {
      GINN_THROW("Unexpected argument type in operator+!");
    }
  } else {
    if constexpr (ginn::is_node_ptr_v<Right>) {
      return AddScalar(b, a);
    } else {
      GINN_THROW("Unexpected argument type in operator+!");
    }
  }
}

template <typename Left,
          typename Right,
          typename = std::enable_if_t<ginn::is_node_ptr_v<Left> or
                                      ginn::is_node_ptr_v<Right>>>
auto operator-(const Left& a, const Right& b) {
  if constexpr (ginn::is_node_ptr_v<Left>) {
    if constexpr (ginn::is_node_ptr_v<Right>) {
      return Subtract(a, b);
    } else if constexpr (ginn::is_arithmetic_v<Right>) {
      return AddScalar(a, -b);
    } else {
      GINN_THROW("Unexpected argument type in operator+!");
    }
  } else {
    if constexpr (ginn::is_node_ptr_v<Right>) {
      GINN_THROW("TODO: Maybe subtract scalar node?");
    } else {
      GINN_THROW("Unexpected argument type in operator+!");
    }
  }
}

template <typename Node>
auto operator-(const Ptr<Node>& in) {
  using Scalar = typename Node::Scalar;
  return in * Scalar(-1);
}

template <typename Left,
          typename Right,
          typename = std::enable_if_t<ginn::is_node_ptr_v<Left> xor
                                      ginn::is_node_ptr_v<Right>>>
auto operator*(const Left& a, const Right& b) {
  if constexpr (ginn::is_node_ptr_v<Left>) {
    if constexpr (ginn::is_node_ptr_v<Right>) {
      // this path should not happen based on the SFINAE condition above
      GINN_THROW("Programming error!");
    } else if constexpr (ginn::is_arithmetic_v<Right>) {
      return ProdScalar(a, b);
    } else {
      GINN_THROW("Unexpected argument type in operator*!");
    }
  } else {
    if constexpr (ginn::is_node_ptr_v<Right>) {
      return ProdScalar(b, a);
    } else {
      GINN_THROW("Unexpected argument type in operator*!");
    }
  }
}

} // namespace ginn
#endif
