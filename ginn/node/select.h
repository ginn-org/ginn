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

#ifndef GINN_NODE_SELECT_H
#define GINN_NODE_SELECT_H

#include <ginn/node.h>
#include <ginn/node/data.h>

namespace ginn {

template <typename Scalar>
class SelectNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<bool> if_;
  NodePtr<Scalar> then_, else_;

  void forward_() override {
    GINN_ASSERT(then_->shape() == Shape{} or then_->shape() == if_->shape());
    GINN_ASSERT(else_->shape() == Shape{} or else_->shape() == if_->shape());

    bool then_scalar = then_->shape() == Shape{};
    bool else_scalar = else_->shape() == Shape{};

    value().resize(if_->shape());
    Index<2> cast{value().rows(), value().cols()};

    TensorMap<bool, 2> if_t = if_->value().t();
    auto then_t = then_->value().t();
    auto else_t = else_->value().t();

    // TODO: Inspect why broadcast breaks for view<1>... So arbitrary!
    if (not then_scalar and not else_scalar) {
      value() = if_t.select(then_t, else_t);
    } else if (not then_scalar) {
      // value() = (if_v != Scalar(0.)) --> eg this one... segfault on gpu.
      //              .select(then_v, else_v.broadcast(Index<1>{n}).eval());
      value() = if_t.select(then_t, else_t.broadcast(cast));
    } else if (not else_scalar) {
      value() = if_t.select(then_t.broadcast(cast), else_t);
    } else {
      value() = if_t.select(then_t.broadcast(cast), else_t.broadcast(cast));
    }
  }

  void backward_() override {
    bool then_scalar = then_->shape() == Shape{};
    bool else_scalar = else_->shape() == Shape{};

    if (then_->has_grad()) {
      auto cmp = if_->value().t().template cast<Scalar>();
      if (then_scalar) {
        then_->grad() += (cmp * grad().t()).sum();
      } else {
        then_->grad() += cmp * grad().t();
      }
    }
    if (else_->has_grad()) {
      auto cmp = (if_->value().t() == false).template cast<Scalar>();
      if (else_scalar) {
        else_->grad() += (cmp * grad().t()).sum();
      } else {
        else_->grad() += cmp * grad().t();
      }
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  SelectNode(const NodePtr<bool>& if_n,
             const NodePtr<Scalar>& then_n,
             const NodePtr<Scalar>& else_n)
      : BaseDataNode<Scalar>({if_n, then_n, else_n}),
        if_(if_n),
        then_(then_n),
        else_(else_n) {}

  SelectNode(const NodePtr<bool>& if_n,
             const NodePtr<Scalar>& then_n,
             Scalar else_val)
      : SelectNode(if_n, then_n, Constant(if_n->dev(), {}, else_val)) {}

  SelectNode(const NodePtr<bool>& if_n,
             Scalar then_val,
             const NodePtr<Scalar>& else_n)
      : SelectNode(if_n, Constant(if_n->dev(), {}, then_val), else_n) {}

  SelectNode(const NodePtr<bool>& if_n, Scalar then_val, Scalar else_val)
      : SelectNode(if_n,
                   Constant(if_n->dev(), {}, then_val),
                   Constant(if_n->dev(), {}, else_val)) {}

  std::string name() const override { return "Select"; }
};

template <typename Then, typename Else>
auto Select(const NodePtr<bool>& if_, Then&& then_, Else&& else_) {
  if constexpr (is_node_ptr_v<std::decay_t<Then>>) {
    using Scalar = typename std::decay_t<Then>::element_type::Scalar;
    return make_ref<SelectNode<Scalar>>(
        if_, std::forward<Then>(then_), std::forward<Else>(else_));
  } else {
    using Scalar = typename std::decay_t<Then>;
    return make_ref<SelectNode<Scalar>>(
        if_, std::forward<Then>(then_), std::forward<Else>(else_));
  }
}

// Mask is very close to a Select with a scalar, however mask is allowed to
// broadcast.
template <typename Scalar>
class MaskNode : public BaseDataNode<Scalar> {
 private:
  // TODO: check if any part of this should be Int or bool nodes
  NodePtr<Scalar> in_, mask_;
  Scalar mask_val_;

  void forward_() override {
    value().resize(in_->shape());

    auto mask_s = mask_->shape();
    auto in_s = in_->shape();
    GINN_ASSERT(mask_s.size() == in_s.size());
    for (size_t i = 0; i < in_s.size(); i++) {
      if (mask_s[i] != in_s[i]) {
        for (size_t j = i; j < in_s.size(); j++) {
          GINN_ASSERT(mask_s[i] == 1, "Unexpected mask shape!");
        }
        break;
      }
    }
    Size batch_size = in_->size() / mask_->size();
    auto mask_m = mask_->value().reshaped({Size(mask_->size()), 1});
    auto in_m = in_->value().reshaped({Size(mask_->size()), batch_size});
    auto value_m = value().reshaped({Size(mask_->size()), batch_size});

    Tensor<Scalar> val(dev(), {}, mask_val_);
    Index<2> cast{in_m.rows(), in_m.cols()};
    value_m = (mask_m.t().broadcast(Index<2>{1, batch_size}) != Scalar(0))
                  .select(in_m.t(), val.t().broadcast(cast));
  }

  void backward_() override {
    if (in_->has_grad()) {
      Size batch_size = in_->size() / mask_->size();
      auto mask_m = mask_->value().reshaped({Size(mask_->size()), 1});
      auto d_in_m = in_->grad().reshaped({Size(mask_->size()), batch_size});
      auto d_m = grad().reshaped({Size(mask_->size()), batch_size});

      Tensor<Scalar> zero(dev(), {}, Scalar(0));
      Index<2> cast{d_in_m.rows(), d_in_m.cols()};
      d_in_m += (mask_m.t().broadcast(Index<2>{1, batch_size}) != Scalar(0))
                    .select(d_m.t(), zero.t().broadcast(cast));
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  template <typename RhsScalar>
  MaskNode(const NodePtr<Scalar>& in,
           const NodePtr<Scalar>& mask,
           RhsScalar mask_val)
      : BaseDataNode<Scalar>({in, mask}),
        in_(in),
        mask_(mask),
        mask_val_(mask_val) {}

  std::string name() const override { return "Mask"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Mask);

} // namespace ginn

#endif
