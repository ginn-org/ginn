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

#ifndef GINN_NODE_INPLACE_H
#define GINN_NODE_INPLACE_H

#include <ginn/node/common.h>
#include <ginn/node/layernorm.h>
#include <ginn/node/layout.h>

namespace ginn {

// TODO: If the input to an InPlace node uses its value() when running
//   backward_(), the result will be incorrect because InPlace node will have
//   overwritten the value().
//   .
//   Considering we take the input nodes as the base Node type, defining a
//   traits class to check if "backward_() uses value()" won't be possible. We
//   can add a virtual function to Node with the failsafe default, but is there
//   a non intrusive way to do it instead, similar to what a traits class would
//   look like?

template <typename Scalar>
class InPlaceAddNode : public AddNode<Scalar> {
 protected:
  using AddNode<Scalar>::ins_;

  void forward_() override { AddNode<Scalar>::forward_(); }

  void backward_() override {
    for (size_t i = 1; i < ins_.size(); i++) {
      if (ins_[i]->has_grad()) { ins_[i]->grad() += grad().t(); }
    }
  }

 public:
  using AddNode<Scalar>::AddNode;

  const Tensor<Scalar>& value() const override { return ins_[0]->value(); }
  const Tensor<Scalar>& grad() const override { return ins_[0]->grad(); }

  bool has_grad() const override { return ins_[0]->has_grad(); }

  std::string name() const override { return "InPlaceAdd"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceAdd);

template <typename Scalar>
class InPlaceAddScalarNode : public AddScalarNode<Scalar> {
 protected:
  using AddScalarNode<Scalar>::in_;

  void backward_() override {}

 public:
  using AddScalarNode<Scalar>::AddScalarNode;

  const Tensor<Scalar>& value() const override { return in_->value(); }
  const Tensor<Scalar>& grad() const override { return in_->grad(); }
  bool has_grad() const override { return in_->has_grad(); }

  std::string name() const override { return "InPlaceAddScalarNode"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceAddScalar);

template <typename Scalar>
class InPlaceCwiseProdNode : public CwiseProdNode<Scalar> {
 protected:
  using CwiseProdNode<Scalar>::a_;
  using CwiseProdNode<Scalar>::b_;

  void backward_() override {
    if (b_->has_grad()) {
      b_->grad() += grad().t() * (a_->value().t() / b_->value().t());
    }
    if (a_->has_grad()) { a_->grad() = grad().t() * b_->value().t(); }
  }

 public:
  using CwiseProdNode<Scalar>::CwiseProdNode;

  const Tensor<Scalar>& value() const override { return a_->value(); }
  const Tensor<Scalar>& grad() const override { return a_->grad(); }
  bool has_grad() const override { return a_->has_grad(); }

  std::string name() const override { return "InPlaceCwiseProd"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceCwiseProd);

template <typename Scalar>
class InPlaceProdScalarNode : public ProdScalarNode<Scalar> {
 protected:
  using ProdScalarNode<Scalar>::in_;

  void backward_() override {
    if (in_->has_grad()) { in_->grad() = grad().t() * this->val_; }
  }

 public:
  using ProdScalarNode<Scalar>::ProdScalarNode;

  const Tensor<Scalar>& value() const override { return in_->value(); }
  const Tensor<Scalar>& grad() const override { return in_->grad(); }
  bool has_grad() const override { return in_->has_grad(); }

  std::string name() const override { return "InPlaceProdScalar"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceProdScalar);

template <typename Scalar>
class InPlaceDropoutNode : public DropoutNode<Scalar> {
 protected:
  using DropoutNode<Scalar>::in_;
  using DropoutNode<Scalar>::p_;
  using DropoutNode<Scalar>::mask_;

  void backward_() override {
    if (in_->has_grad() and p_ < 1.) {
      Scalar tmp(1. / (1. - p_));
      grad() = (grad().t() * mask_.t().template cast<Scalar>() * tmp);
    }
  }

 public:
  using DropoutNode<Scalar>::DropoutNode;

  const Tensor<Scalar>& value() const override { return in_->value(); }
  const Tensor<Scalar>& grad() const override { return in_->grad(); }
  using DropoutNode<Scalar>::grad;
  bool has_grad() const override { return in_->has_grad(); }

  std::string name() const override { return "InPlaceDropout"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceDropout);

template <typename Scalar>
class InPlaceMaskNode : public BaseDataNode<Scalar> {
 private:
  // TODO: Consider making mask_ a NodePtr<bool> instead
  NodePtr<Scalar> in_, mask_;
  Scalar mask_val_;

  void forward_() override {
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
    Size ms = mask_->size();
    Size batch_size = in_->size() / ms;
    auto mask_m = mask_->value().reshaped({ms, 1});
    auto in_m = in_->value().reshaped({ms, batch_size});

    Tensor<Scalar> val(this->dev(), {}, mask_val_);
    Index<2> cast{in_m.rows(), in_m.cols()};
    in_m = (mask_m.t().broadcast(Index<2>{1, batch_size}) != Scalar(0))
               .select(in_m.t(), val.t().broadcast(cast));
  }

  void backward_() override {
    if (in_->has_grad()) {
      Size ms = mask_->size();
      Size batch_size = in_->size() / ms;
      auto mask_m = mask_->value().reshaped({ms, 1});
      auto d_in_m = in_->grad().reshaped({ms, batch_size});

      Tensor<Scalar> zero(this->dev(), {}, Scalar(0));
      Index<2> cast{d_in_m.rows(), d_in_m.cols()};
      d_in_m = (mask_m.t().broadcast(Index<2>{1, batch_size}) != Scalar(0))
                   .select(d_in_m.t(), zero.t().broadcast(cast));
    }
  }

 public:
  const Tensor<Scalar>& value() const override { return in_->value(); }
  const Tensor<Scalar>& grad() const override { return in_->grad(); }
  bool has_grad() const override { return in_->has_grad(); }

  template <typename MaskScalar,
            typename = std::enable_if_t<ginn::is_arithmetic_v<MaskScalar>>>
  InPlaceMaskNode(NodePtr<Scalar> in, NodePtr<Scalar> mask, MaskScalar mask_val)
      : BaseDataNode<Scalar>({in, mask}),
        in_(in),
        mask_(mask),
        mask_val_(Scalar(mask_val)) {}

  std::string name() const override { return "InPlaceMask"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceMask);

template <typename Scalar>
class InPlaceLayerNormNode : public LayerNormNode<Scalar> {
 protected:
  using LayerNormNode<Scalar>::in_;
  using LayerNormNode<Scalar>::std_;

  void backward_() override {
    if (in_->has_grad()) {
      const auto rows = this->shape2()[0];
      const auto cols = this->shape2()[1];

      auto bc = [=](const auto& e) { return e.broadcast(Index<2>{rows, 1}); };

      Tensor<Scalar> grad_mean(this->dev(), {1, cols}),
          dot(this->dev(), {1, cols});
      grad_mean = grad().t().mean(Index<1>{0});
      dot = (grad().t() * value().t()).mean(Index<1>{0});

      in_->grad() =
          (grad().t() - bc(grad_mean.t()) - value().t() * bc(dot.t())) /
          std_.t().broadcast(Index<2>{rows, 1});
    }
  }

 public:
  using LayerNormNode<Scalar>::LayerNormNode;

  const Tensor<Scalar>& value() const override { return in_->value(); }
  const Tensor<Scalar>& grad() const override { return in_->grad(); }
  bool has_grad() const override { return in_->has_grad(); }

  std::string name() const override { return "InPlaceLayerNorm"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlaceLayerNorm);

// TODO: This seems to be breaking. Maybe an aliasing issue when using Eigen
// shuffle()?
// TODO: is this still breaking or no? check
template <typename Scalar>
class InPlacePermuteNode : public PermuteNode<Scalar> {
 protected:
  using PermuteNode<Scalar>::in_;
  using PermuteNode<Scalar>::indices_;

  void forward_() override {
    Shape s(indices_.size());
    for (size_t i = 0; i < s.size(); i++) { s[i] = in_->shape()[indices_[i]]; }
    this->value().map(in_->value(), s);

    switch (indices_.size()) {
    case 2: this->template forward_helper<2>(); break;
    case 3: this->template forward_helper<3>(); break;
    case 4: this->template forward_helper<4>(); break;
    case 5: this->template forward_helper<5>(); break;
    case 6: this->template forward_helper<6>(); break;
    case 7: this->template forward_helper<7>(); break;
    case 8: this->template forward_helper<8>(); break;
    default: GINN_THROW("Unexpected number of indices in Permute!");
    }
  }

 public:
  InPlacePermuteNode(NodePtr<Scalar> in, Shape indices)
      : PermuteNode<Scalar>(in, std::move(indices)) {
    this->overwrite_ = true;
  }

  bool has_grad() const override { return in_->has_grad(); }

  void init_grad() override {
    if (this->has_grad()) {
      BaseDataNode<Scalar>::init_grad();
      Shape s = PermuteNode<Scalar>::permute(in_->shape(), indices_);
      this->grad().map(in_->grad(), s);
    }
  }

  std::string name() const override { return "InPlacePermuteNode"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlacePermute);

template <typename Node>
auto InPlacePermute(Ptr<Node> in, std::initializer_list<Size> s) {
  return InPlacePermute(in, Shape(s));
}

template <typename Scalar>
auto InPlacePermute(NodePtr<Scalar> x, std::initializer_list<Size> indices) {
  return InPlacePermute(std::move(x), Shape(indices));
}

} // namespace ginn

#endif
