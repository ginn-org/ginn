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

#ifndef GINN_NODE_AFFINE_H
#define GINN_NODE_AFFINE_H

#include <ginn/node/data.h>
#include <ginn/nonlin.h>
#include <ginn/prod.h>
#include <ginn/util/fmt.h>

namespace ginn {

// TODO: Separate LeftScalar and RightScalar types and mixed precision
//   Affine with cublasGemmEx

template <typename Scalar>
class AffineNode : public BaseDataNode<Scalar> {
 private:
  std::vector<NodePtr<Scalar>> ins_;
  Tensor<Scalar> preactiv_, dpreactiv_;
  std::unique_ptr<NonlinOp<Scalar>> nonlin_;

  void forward_() override {
    auto& a = ins_[0]->value();
    auto& b = ins_[1]->value();
    auto& bias = ins_.back()->value();

    GINN_ASSERT(a.shape().size() == 2);
    Shape new_s = b.shape();
    new_s[0] = a.rows();

    Tensor<Scalar>& affine =
        nonlin_->backward_requires_input() ? preactiv_ : value();
    affine.resize(new_s);

    if (dev()->kind() == CPU) {
      affine.m() = (a.m() * b.m()).colwise() + bias.v();
      for (size_t i = 2; i < ins_.size() - 1; i += 2) {
        auto &a = ins_[i]->value(), &b = ins_[i + 1]->value();
        GINN_ASSERT(a.shape().size() == 2);
        affine.m().noalias() += a.m() * b.m();
      }
#ifdef GINN_ENABLE_GPU
    } else if (dev()->kind() == GPU) {
      affine = bias.t().broadcast(Index<2>{1, b.cols()});
      internal::gpu_prod(affine, a, b, internal::ProdResult::Add);
      for (size_t i = 2; i < ins_.size() - 1; i += 2) {
        auto &a = ins_[i]->value(), &b = ins_[i + 1]->value();
        GINN_ASSERT(a.shape().size() == 2);
        internal::gpu_prod(affine, a, b, internal::ProdResult::Add);
      }
#endif
    } else {
      GINN_THROW("Unexpected device type in AffineNode::forward!");
    }

    if (nonlin_->backward_requires_input()) {
      // In this case preactivations should be stored separately from value().
      // preactiv_ was initialized but value() was not.
      value().resize(new_s);
    }
    if (not nonlin_->is_identity()) {
      nonlin_->forward(value(), affine);
    } // else, affine _is_ value() and value() already holds the result.
  }

  void backward_() override {
    Tensor<Scalar>& daffine = nonlin_->is_identity() ? grad() : dpreactiv_;
    if (not nonlin_->is_identity()) {
      dpreactiv_.resize(grad().shape());
      dpreactiv_.set_zero();
      nonlin_->backward(dpreactiv_, grad(), preactiv_, value(), true);
      // preactiv_ is an empty tensor if Nonlin::backward_requires_input==false,
      // which is okay since NonlinOp::backward_() is _not_ using it.
    }

    if (dev()->kind() == CPU) {
      auto& bias = ins_.back();
      if (bias->has_grad()) {
        bias->grad().m().noalias() += daffine.m().rowwise().sum();
      }
      for (size_t i = 0; i < ins_.size() - 1; i += 2) {
        auto &a = ins_[i], &b = ins_[i + 1];
        if (b->has_grad()) {
          b->grad().m().noalias() += a->value().m().transpose() * daffine.m();
        }
        if (a->has_grad()) {
          a->grad().m().noalias() += daffine.m() * b->value().m().transpose();
        }
      }
#ifdef GINN_ENABLE_GPU
    } else if (dev()->kind() == GPU) {
      using namespace internal;
      auto& bias = ins_.back();
      if (bias->has_grad()) { bias->grad() += daffine.t().sum(Index<1>{1}); }
      for (size_t i = 0; i < ins_.size() - 1; i += 2) {
        auto &a = ins_[i], &b = ins_[i + 1];
        if (b->has_grad()) {
          gpu_prod(b->grad(),
                   a->value(),
                   daffine,
                   ProdResult::Add,
                   ProdTranspose::First);
        }
        if (a->has_grad()) {
          gpu_prod(a->grad(),
                   daffine,
                   b->value(),
                   ProdResult::Add,
                   ProdTranspose::Second);
        }
      }
#endif
    } else {
      GINN_THROW("Unexpected device in AffineNode::backward!");
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  void set_ins(const std::vector<BaseNodePtr>& ins) override {
    BaseNode::ins_ = ins;
    ins_ = derived_cast<Node<Scalar>>(ins);
  }

  template <typename NonlinType>
  AffineNode(NonlinType nonlin, const std::vector<NodePtr<Scalar>>& ins)
      : BaseDataNode<Scalar>(ins),
        ins_(ins),
        preactiv_(dev()),
        dpreactiv_(dev()),
        nonlin_(std::make_unique<NonlinType>(nonlin)) {
    GINN_ASSERT(ins.size() > 0);
    GINN_ASSERT(ins.size() % 2); // force bias for now.
  }

  AffineNode(std::unique_ptr<NonlinOp<Scalar>> nonlin,
             const std::vector<NodePtr<Scalar>>& ins)
      : BaseDataNode<Scalar>(ins),
        ins_(ins),
        preactiv_(dev()),
        dpreactiv_(dev()),
        nonlin_(std::move(nonlin)) {
    GINN_ASSERT(ins.size() > 0);
    GINN_ASSERT(ins.size() % 2); // force bias for now.
  }

  AffineNode(const std::vector<NodePtr<Scalar>>& ins)
      : AffineNode<Scalar>(IdentityOp<Scalar>(), ins) {}

  template <typename... Args>
  AffineNode(const NodePtr<Scalar>& in, Args&&... args)
      : AffineNode<Scalar>(std::vector<NodePtr<Scalar>>({in, args...})) {}

  template <typename NonlinType,
            typename... Args,
            typename = std::enable_if_t<
                std::is_base_of_v<NonlinOp<Scalar>, NonlinType>>>
  AffineNode(NonlinType nonlin, const NodePtr<Scalar>& in, Args&&... args)
      : AffineNode<Scalar>(nonlin,
                           std::vector<NodePtr<Scalar>>({in, args...})) {}

  template <typename NonlinType,
            typename... Args,
            typename = std::enable_if_t<
                std::is_base_of_v<NonlinOp<Scalar>, NonlinType>>>
  AffineNode(std::unique_ptr<NonlinType> nonlin,
             const NodePtr<Scalar>& in,
             Args&&... args)
      : AffineNode<Scalar>(std::move(nonlin),
                           std::vector<NodePtr<Scalar>>({in, args...})) {}

  std::string name() const override { return "AffineNode"; }
};

// TODO: See if its possible to trim the ctors and factory functions for Affine,
//   this is too much clutter.

template <typename Scalar>
auto Affine(const std::vector<NodePtr<Scalar>>& ins) {
  return make_ptr<AffineNode<Scalar>>(ins);
}

template <typename Node,
          typename... Args,
          typename = std::enable_if_t<std::is_base_of_v<BaseNode, Node>>>
auto Affine(Ptr<Node> in, Args&&... args) {
  using Scalar = typename Node::Scalar;
  return make_ptr<AffineNode<Scalar>>(in, std::forward<Args>(args)...);
}

template <typename Node,
          typename NonlinType,
          typename... Args,
          typename = std::enable_if_t<
              std::is_base_of_v<NonlinOp<typename Node::Scalar>, NonlinType>>>
auto Affine(NonlinType nonlin, Ptr<Node> in, Args&&... args) {
  using Scalar = typename Node::Scalar;
  return make_ptr<AffineNode<Scalar>>(nonlin, in, std::forward<Args>(args)...);
}

template <typename Node,
          typename NonlinType,
          typename... Args,
          typename = std::enable_if_t<
              std::is_base_of_v<NonlinOp<typename Node::Scalar>, NonlinType>>>
auto Affine(std::unique_ptr<NonlinType> nonlin, Ptr<Node> in, Args&&... args) {
  using Scalar = typename Node::Scalar;
  return make_ptr<AffineNode<Scalar>>(
      std::move(nonlin), in, std::forward<Args>(args)...);
}

template <template <typename> typename Nonlin, typename Node, typename... Args>
auto Affine(Ptr<Node> in, Args&&... args) {
  using Scalar = typename Node::Scalar;
  return make_ptr<AffineNode<Scalar>>(
      Nonlin<Scalar>(), in, std::forward<Args>(args)...);
}

/*
// Convenience method to be able to call, e.g., Affine<Sigmoid> instead of
// Affine<SigmoidOp>.
template <auto(*NonlinFactory)(const NodePtr&)>
auto Affine(const std::vector<NodePtr>& ins) {
  using NonlinOp =
      typename decltype(NonlinFactory(NodePtr()))::element_type::UnaryOp;
  return Affine<NonlinOp>(ins);
}
 */

} // namespace ginn

#endif
