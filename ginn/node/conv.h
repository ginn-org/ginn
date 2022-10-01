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

#ifndef GINN_NODE_CONV_H
#define GINN_NODE_CONV_H

#include <ginn/node/data.h>

#include <ginn/eigenop/eigen_conv.h>

namespace ginn {

// 2d Spatial Convolution
// Input shape: (channels, height, width, others (e.g. batch dim))
// Filters / Kernel shape: (filters, channels, kernel height, kernel width)
template <typename Scalar>
class Conv2dNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> in_, filter_;
  size_t row_stride_ = 1; // TODO: Should these be Size instead?
  size_t col_stride_ = 1;

  static unsigned dim(Size input_len, size_t stride) {
    return (input_len + stride - 1) / stride;
  }

  void forward_() override {
    Shape s(Tensor<Scalar>::reduce(in_->shape(), 4));
    Shape filt_s = filter_->shape();
    s[0] = filt_s.at(0);
    s[1] = dim(s[1], row_stride_);
    s[2] = dim(s[2], col_stride_);
    value().resize(s);

    auto input = in_->value().template view<4>();
    auto filters = filter_->value().template view<4>();
    value() = eigen::Conv2d(input, filters, row_stride_, col_stride_);
  }

  void backward_() override {
    auto input = in_->value().template view<4>();
    auto filters = filter_->value().template view<4>();
    if (in_->has_grad()) {
      in_->grad() += eigen::Conv2dBackwardInput(
          input, filters, grad().template view<4>(), row_stride_, col_stride_);
    }
    if (filter_->has_grad()) {
      filter_->grad() += eigen::Conv2dBackwardKernel(
          input, filters, grad().template view<4>(), row_stride_, col_stride_);
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  Conv2dNode(NodePtr<Scalar> input,
             NodePtr<Scalar> filters,
             size_t a_row_stride = 1,
             size_t a_col_stride = 1)
      : BaseDataNode<Scalar>({input, filters}),
        in_(input),
        filter_(filters),
        row_stride_(a_row_stride),
        col_stride_(a_col_stride) {}

  std::string name() const override { return "Conv2d"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Conv2d);

// 1d Spatial Convolution
// Input shape: (channels, length, others (e.g. batch dim))
// Filters / Kernel shape: (filters, channels, kernel length)
template <typename Scalar>
class Conv1dNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> in_, filter_;
  size_t stride_ = 1; // TODO: Should these be Size instead?

  static unsigned dim(Size input_len, size_t stride) {
    return (input_len + stride - 1) / stride;
  }

  void forward_() override {
    Shape is(Tensor<Scalar>::reduce(in_->value().shape(), 3));
    Shape s(Tensor<Scalar>::reduce(in_->value().shape(), 3));
    Shape ks = filter_->value().shape();

    s[0] = ks.at(0);
    s[1] = dim(s[1], stride_);
    value().resize(s);

    auto out = value().reshaped({s[0], s[1], 1, s[2]});
    auto in = in_->value().reshaped({is[0], is[1], 1, is[2]});
    auto k = filter_->value().reshaped({ks[0], ks[1], ks[2], 1});

    out =
        eigen::Conv2d(in.template view<4>(), k.template view<4>(), stride_, 1);
  }

  void backward_() override {
    Shape is(Tensor<Scalar>::reduce(in_->value().shape(), 3));
    Shape s = value().shape();
    Shape ks = filter_->value().shape();

    // Tensor<Scalar> in, k, g, in_g, k_g;
    auto in = in_->value().reshaped({is[0], is[1], 1, is[2]});
    auto k = filter_->value().reshaped({ks[0], ks[1], ks[2], 1});

    auto g = grad().reshaped({s[0], s[1], 1, s[2]});

    if (in_->has_grad()) {
      auto in_g = in_->grad().reshaped({is[0], is[1], 1, is[2]});
      in_g += eigen::Conv2dBackwardInput(in.template view<4>(),
                                         k.template view<4>(),
                                         g.template view<4>(),
                                         stride_,
                                         1);
    }
    if (filter_->has_grad()) {
      auto k_g = filter_->grad().reshaped({ks[0], ks[1], ks[2], 1});
      k_g += eigen::Conv2dBackwardKernel(in.template view<4>(),
                                         k.template view<4>(),
                                         g.template view<4>(),
                                         stride_,
                                         1);
    }
  }

 public:
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  Conv1dNode(NodePtr<Scalar> input, NodePtr<Scalar> filters, size_t stride = 1)
      : BaseDataNode<Scalar>({input, filters}),
        in_(input),
        filter_(filters),
        stride_(stride) {}

  std::string name() const override { return "Conv1d"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(Conv1d);

} // end namespace ginn

#endif
