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

#ifndef GINN_NLNODE_H
#define GINN_NLNODE_H

#include <ginn/node/data.h>
#include <ginn/nonlin.h>

// Nodes that implement nonlinearities / activations

namespace ginn {

// Take a nonlinearity from `nonlin.h` and turn it into a node, e.g:
//   GINN_MAKE_UNARY_NODE_AND_FACTORY(Abcdef) takes AbcdefOp and makes
//   AbcdefNode class, and Abcdef() factory function.

#define GINN_MAKE_UNARY_NODE_AND_FACTORY(F)                                    \
  template <typename Scalar>                                                   \
  class F##Node : public BaseDataNode<Scalar> {                                \
   protected:                                                                  \
    NodePtr<Scalar> in_;                                                       \
    F##Op<Scalar> op_{};                                                       \
                                                                               \
    void forward_() override {                                                 \
      value().resize(in_->value().shape());                                    \
      op_.forward(value(), in_->value());                                      \
    }                                                                          \
    void backward_() override {                                                \
      if (in_->has_grad()) {                                                   \
        op_.backward(in_->grad(), grad(), in_->value(), value(), true);        \
      }                                                                        \
    }                                                                          \
                                                                               \
   public:                                                                     \
    using BaseDataNode<Scalar>::value;                                         \
    using BaseDataNode<Scalar>::grad;                                          \
    using UnaryOp = F##Op<Scalar>;                                             \
    F##Node(NodePtr<Scalar> in) : BaseDataNode<Scalar>({in}), in_(in) {}       \
    void set_ins(const std::vector<BaseNodePtr>& ins) override {               \
      GINN_ASSERT(ins.size() == 1);                                            \
      BaseNode::ins_ = ins;                                                    \
      in_ = dynamic_ref_cast<Node<Scalar>>(ins.front());                       \
    }                                                                          \
    std::string name() const override { return #F; }                           \
  };                                                                           \
                                                                               \
  GINN_MAKE_SCALAR_FORWARDING_FACTORY(F);

GINN_MAKE_UNARY_NODE_AND_FACTORY(Identity);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Tanh);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Sigmoid);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Relu);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Softmax);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Log);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Sqrt);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Gelu);
GINN_MAKE_UNARY_NODE_AND_FACTORY(Gelu2);

#define GINN_MAKE_INPLACE_UNARY_NODE_AND_FACTORY(F)                            \
  template <typename Scalar>                                                   \
  class InPlace##F##Node : public F##Node<Scalar> {                            \
   protected:                                                                  \
    using F##Node<Scalar>::in_;                                                \
    using F##Node<Scalar>::op_;                                                \
                                                                               \
    void backward_() override {                                                \
      if (in_->has_grad()) {                                                   \
        op_.backward(in_->grad(), grad(), in_->value(), value(), false);       \
      }                                                                        \
    }                                                                          \
                                                                               \
   public:                                                                     \
    const Tensor<Scalar>& value() const override { return in_->value(); }      \
    const Tensor<Scalar>& grad() const override { return in_->grad(); }        \
                                                                               \
    bool has_grad() const override { return in_->has_grad(); }                 \
                                                                               \
    using UnaryOp = F##Op<Scalar>;                                             \
    using F##Node<Scalar>::F##Node;                                            \
    std::string name() const override { return "InPlace" #F; }                 \
  };                                                                           \
                                                                               \
  GINN_MAKE_SCALAR_FORWARDING_FACTORY(InPlace##F);

GINN_MAKE_INPLACE_UNARY_NODE_AND_FACTORY(Sigmoid);

} // end namespace ginn

#endif
