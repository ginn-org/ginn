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

#ifndef GINN_NODE_LAYERNORM_H
#define GINN_NODE_LAYERNORM_H

#include <ginn/node.h>
#include <ginn/node/data.h>
#include <ginn/prod.h>

namespace ginn {

template <typename Scalar>
class LayerNormNode : public BaseDataNode<Scalar> {
 protected:
  NodePtr<Scalar> in_;
  Scalar eps_;
  Tensor<Scalar> std_;

  void forward_() override {
    value().resize(in_->shape());

    const auto rows = this->shape2()[0];
    const auto cols = this->shape2()[1];

    Tensor<Scalar> mean(dev(), {1, cols});
    std_.resize({1, cols});

    auto bc = [=](const auto& e) { return e.broadcast(Index<2>{rows, 1}); };

    mean = in_->value().t().mean(Index<1>{0});
    std_ = ((in_->value().t() - bc(mean.t())).square().mean(Index<1>{0}) + eps_)
               .sqrt();
    value() = (in_->value().t() - bc(mean.t())) / bc(std_.t());
  }

  void backward_() override {
    if (in_->has_grad()) {
      const auto rows = this->shape2()[0];
      const auto cols = this->shape2()[1];

      auto bc = [=](const auto& e) { return e.broadcast(Index<2>{rows, 1}); };

      Tensor<Scalar> grad_mean(dev(), {1, cols}), dot(dev(), {1, cols});
      grad_mean = grad().t().mean(Index<1>{0});
      if (dev()->kind() == CPU) {
        dot = (grad().t() * value().t()).sum(Index<1>{0});
#ifdef GINN_ENABLE_GPU
      } else if (dev()->kind() == GPU) {
        auto grad_ = grad().reshaped({1, grad().rows(), grad().cols()});
        auto value_ = value().reshaped({value().rows(), 1, value().cols()});
        auto dot_ = dot.reshaped({1, 1, dot.cols()});
        internal::gpu_batched_prod(dot_, grad_, value_);
#endif
      } else {
        GINN_THROW("Unexpected device in LayerNormNode::backward_()!");
      }

      in_->grad() += (grad().t() - bc(grad_mean.t()) -
                      value().t() * bc(dot.t() / Scalar(rows))) /
                     bc(std_.t());
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  LayerNormNode(NodePtr<Scalar> in, Scalar eps = Scalar(1e-8))
      : BaseDataNode<Scalar>({in}), in_(in), eps_(eps), std_(dev()) {}

  std::string name() const override { return "LayerNorm"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(LayerNorm);

} // namespace ginn

#endif
