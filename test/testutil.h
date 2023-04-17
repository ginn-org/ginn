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

#ifndef GINN_TESTUTIL_H
#define GINN_TESTUTIL_H

#include <catch2/catch.hpp>

#include <ginn/node/common.h>
#include <ginn/node/reduce.h>
#include <ginn/node/weight.h>
#include <ginn/util/tensorio.h>
#include <ginn/util/traits.h>

namespace ginn {

// Approximate tensor matcher to be able to run:
//   CHECK(tensor1 == Close(tensor2));
//   in Catch, when checking approximate equality of float tensors.

template <typename Scalar>
class Close {
 private:
  const Tensor<Scalar>& value;
  mutable Catch::Detail::Approx approx = Catch::Detail::Approx::custom();

  bool equalityComparisonImpl(const Tensor<Scalar>& other) const {
    if (value.shape() != other.shape()) { return false; }
    for (Size i = 0; i < value.size(); ++i) {
      if (other.v()[i] != approx(value.v()[i])) { return false; }
    }
    return true;
  }

 public:
  explicit Close(const Tensor<Scalar>& value) : value(value) {}

  Close operator-() const;

  auto& margin(double margin) {
    approx.margin(margin);
    return *this;
  }
  auto& epsilon(double eps) {
    approx.epsilon(eps);
    return *this;
  }
  auto& scale(double s) {
    approx.scale(s);
    return *this;
  }

  Close operator()(const Tensor<Scalar>& t) const {
    Close<Scalar> appr(t);
    appr.approx = this->approx;
    return appr;
  }

  friend bool operator==(const Tensor<Scalar>& lhs, Close const& rhs) {
    return rhs.equalityComparisonImpl(lhs);
  }

  friend bool operator==(Close const& lhs, const Tensor<Scalar>& rhs) {
    return operator==(rhs, lhs);
  }

  friend bool operator!=(Tensor<Scalar> const& lhs, Close const& rhs) {
    return !operator==(lhs, rhs);
  }

  friend bool operator!=(Close const& lhs, Tensor<Scalar> const& rhs) {
    return !operator==(rhs, lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, Close<Scalar> const& rhs) {
    o << rhs.value;
    return o;
  }
};

// Compute numeric grad for gradient checks, using finite differences
template <typename Expr, typename Weight, typename Mask>
inline Tensor<Real> numeric_grad(Expr e, Weight w, Mask mask, Real eps = 1e-4) {
  auto s = Sum(CwiseProd(mask, e));

  auto g = Graph(s);
  Tensor<Real> rval(w->dev(), w->shape());

  for (uint i = 0; i < w->size(); i++) {
    auto value = w->value().m();
    Real tmp = value(i);
    value(i) = tmp + eps;
    g.reset_forwarded();
    g.forward();
    Real s_plus = e->value().m().cwiseProduct(mask->value().m()).sum();

    value(i) = tmp - eps;
    g.reset_forwarded();
    g.forward();
    Real s_minus = e->value().m().cwiseProduct(mask->value().m()).sum();

    rval.v()(i) = (s_plus - s_minus) / (2 * eps);
    value(i) = tmp;
  }

  return rval;
}

// Compute analytic grad for gradient checks, by calling backward()
template <typename Expr, typename Weight, typename Mask>
inline Tensor<Real> analytic_grad(Expr e, Weight w, Mask mask) {
  auto s = Sum(CwiseProd(mask, e));

  auto g = Graph(s);
  g.reset_forwarded(); // gradcheck reuses expressions
  g.forward();
  g.reset_grad();
  g.backward(1.);

  return w->grad();
}

template <typename NodeFunc, typename NodeContainer>
inline void check_grad(NodeFunc f_e,
                       const NodeContainer& ins,
                       bool randomize_inputs = false,
                       Real eps = 1e-4,
                       Real delta = 1e-4) {
  static_assert(ginn::is_node_ptr_v<typename NodeContainer::value_type>,
                "ins should contain derived Node pointer types!");

  if (randomize_inputs) {
    for (BaseNodePtr w : ins) {
      if (auto w_ = dynamic_ptr_cast<BaseDataNode<>>(w)) {
        w_->value().set_random();
      } else if (auto w_ = dynamic_ptr_cast<WeightNode<>>(w)) {
        w_->value().set_random();
      }
    }
  }

  NodePtr<Real> e = f_e();
  e = e + e; // to make sure gradient accumulates over input repetition
  auto g = Graph(e);
  g.reset_forwarded();
  g.forward(); // to init all shapes

  for (BaseNodePtr w : ins) {
    if ((dynamic_ptr_cast<BaseDataNode<>>(w) or
         dynamic_ptr_cast<WeightNode<>>(w)) and
        w->has_grad()) {
      w->reset_grad();
      auto wr = dynamic_ptr_cast<Node<Real>>(w);
      auto mask = FixedRandom(e->dev(), e->value().shape());
      auto ng = numeric_grad(e, wr, mask, delta);
      auto ag = analytic_grad(e, wr, mask);
      CHECK(ag == Close(ng).scale(eps));
    }
  }
}

template <typename NodeContainer>
inline void check_grad(NodePtr<Real> e,
                       const NodeContainer& ins,
                       bool randomize_inputs = false,
                       Real eps = 1e-4,
                       Real delta = 1e-4) {
  check_grad([e]() { return e; }, ins, randomize_inputs, eps, delta);
}

} // namespace ginn

#endif
