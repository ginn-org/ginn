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

#ifndef GINN_NODE_PROD_H
#define GINN_NODE_PROD_H

#include <ginn/node/data.h>
#include <ginn/prod.h>

namespace ginn {

template <typename Scalar>
class ProdNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> a_, b_;

  void forward_() override {
    value().resize(Shape{a_->value().rows(), b_->value().cols()});
    if (dev()->type() == CPU) {
      value().m() = a_->value().m() * b_->value().m();
    }
#ifdef GINN_ENABLE_GPU
    else if (dev()->type() == GPU) {
      internal::gpu_prod(value(), a_->value(), b_->value());
    }
#endif
  }

  void backward_() override {
    if (dev()->type() == CPU) {
      if (b_->has_grad()) {
        b_->grad().m().noalias() += a_->value().m().transpose() * grad().m();
      }
      if (a_->has_grad()) {
        a_->grad().m().noalias() += grad().m() * b_->value().m().transpose();
      }
#ifdef GINN_ENABLE_GPU
    } else if (dev()->type() == GPU) {
      if (b_->has_grad()) {
        internal::gpu_prod(b_->grad(),
                           a_->value(),
                           grad(),
                           internal::ProdResult::Add,
                           internal::ProdTranspose::First);
      }
      if (a_->has_grad()) {
        internal::gpu_prod(a_->grad(),
                           grad(),
                           b_->value(),
                           internal::ProdResult::Add,
                           internal::ProdTranspose::Second);
      }
#endif
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  ProdNode(const NodePtr<Scalar>& a, const NodePtr<Scalar>& b)
      : BaseDataNode<Scalar>({a, b}), a_(a), b_(b) {}

  void set_ins(const std::vector<BaseNodePtr>& ins) override {
    GINN_ASSERT(ins.size() == 2);
    BaseNode::ins_ = ins;
    a_ = dynamic_ref_cast<Node<Scalar>>(ins[0]);
    b_ = dynamic_ref_cast<Node<Scalar>>(ins[1]);
  }

  std::string name() const override { return "Prod"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Prod);

template <typename Scalar>
class BatchedProdNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> a_, b_;

  void forward_() override {
    auto sa = a_->shape(), sb = b_->shape();
    GINN_ASSERT(sa.size() >= 3 and sb.size() >= 3,
                "BatchedProd expects at least rank 3 inputs!");
    GINN_ASSERT(sa[1] == sb[0], "Mismatching inner dimensions in BatchedProd!");
    for (size_t i = 2; i < sa.size(); i++) {
      GINN_ASSERT(sa[i] == sb[i],
                  "Mismatching batch dimensions in BatchedProd!");
    }

    auto rows = sa[0], cols = sb[1];
    Shape s = sb;
    s[0] = rows;
    value().resize(s);

    if (dev()->type() == CPU) {
      auto batches = b_->value().cols() / cols;

      for (Size i = 0; i < batches; i++) { // trivial serial loop over batch
        Tensor<Scalar> ai, bi, vi;
        ai.map(a_->value(), {sa[0], sa[1]}, sa[0] * sa[1] * i);
        bi.map(b_->value(), {sb[0], sb[1]}, sb[0] * sb[1] * i);
        vi.map(value(), {sa[0], sb[1]}, sa[0] * sb[1] * i);
        vi.m() = ai.m() * bi.m();
      }
#ifdef GINN_ENABLE_GPU
    } else if (dev()->type() == GPU) {
      internal::gpu_batched_prod(value(), a_->value(), b_->value());
#endif
    } else {
      GINN_THROW("Unexpected device!");
    }
  }

  void backward_() override {
    auto sa = a_->shape(), sb = b_->shape();
    if (dev()->type() == CPU) {
      if (a_->has_grad() or b_->has_grad()) {
        auto batches = b_->value().cols() / sb[1];
        for (Size i = 0; i < batches; i++) { // trivial serial loop over batch
          Tensor<Scalar> gi;
          gi.map(grad(), {sa[0], sb[1]}, sa[0] * sb[1] * i);
          if (a_->has_grad()) {
            Tensor<Scalar> agi, bi;
            agi.map(a_->grad(), {sa[0], sa[1]}, sa[0] * sa[1] * i);
            bi.map(b_->value(), {sb[0], sb[1]}, sb[0] * sb[1] * i);
            agi.m() += gi.m() * bi.m().transpose();
          }
          if (b_->has_grad()) {
            Tensor<Scalar> bgi, ai;
            ai.map(a_->value(), {sa[0], sa[1]}, sa[0] * sa[1] * i);
            bgi.map(b_->grad(), {sb[0], sb[1]}, sb[0] * sb[1] * i);
            bgi.m() += ai.m().transpose() * gi.m();
          }
        }
      }
#ifdef GINN_ENABLE_GPU
    } else if (dev()->type() == GPU) {
      if (a_->has_grad()) {
        internal::gpu_batched_prod(a_->grad(),
                                   grad(),
                                   b_->value(),
                                   internal::ProdResult::Add,
                                   internal::ProdTranspose::Second);
      }
      if (b_->has_grad()) {
        internal::gpu_batched_prod(b_->grad(),
                                   a_->value(),
                                   grad(),
                                   internal::ProdResult::Add,
                                   internal::ProdTranspose::First);
      }
#endif
    } else {
      GINN_THROW("Unexpected device!");
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  BatchedProdNode(const NodePtr<Scalar>& a, const NodePtr<Scalar>& b)
      : BaseDataNode<Scalar>({a, b}), a_(a), b_(b) {}

  std::string name() const override { return "BatchedProd"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(BatchedProd);

template <typename Left, typename Right>
auto operator*(const Ptr<Left>& a, const Ptr<Right>& b) {
  return Prod(a, b);
}

} // namespace ginn

#endif
