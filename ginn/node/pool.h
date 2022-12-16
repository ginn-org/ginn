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

#ifndef GINN_POOL_H
#define GINN_POOL_H

#include <ginn/node/data.h>

#include <ginn/eigenop/eigen_pool.h>

namespace ginn {

template <typename Scalar>
class MaxPool2dNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> in_;
  size_t rows_, cols_;
  size_t row_stride_ = 1;
  size_t col_stride_ = 1;

  static unsigned dim(Size input_len, size_t stride) {
    return (input_len + stride - 1) / stride;
  }

  void forward_() override {
    Shape s(Tensor<Scalar>::reduce(in_->shape(), 4));
    s[1] = dim(s[1], row_stride_);
    s[2] = dim(s[2], col_stride_);
    value().resize(s);

    auto input = in_->value().template view<4>();
    value() = eigen::MaxPool2d(input, rows_, cols_, row_stride_, col_stride_);
  }

  void backward_() override {
    if (in_->has_grad()) {
      auto input = in_->value().template view<4>();
      auto d_input = in_->grad().template view<4>();
      auto output = value().template view<4>();
      auto d_output = grad().template view<4>();

      if (dev()->kind() == CPU) {
        eigen::MaxPool2dBackwardLoop(d_input,
                                     input,
                                     output,
                                     d_output,
                                     rows_,
                                     cols_,
                                     row_stride_,
                                     col_stride_);
#ifdef GINN_ENABLE_GPU
      } else {
        in_->grad() += eigen::MaxPool2dBackward(
            input, output, d_output, rows_, cols_, row_stride_, col_stride_);
#endif
      }
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  MaxPool2dNode(NodePtr<Scalar> input,
                size_t a_rows,
                size_t a_cols,
                size_t a_row_stride = 1,
                size_t a_col_stride = 1)
      : BaseDataNode<Scalar>({input}),
        in_(input),
        rows_(a_rows),
        cols_(a_cols),
        row_stride_(a_row_stride),
        col_stride_(a_col_stride) {}

  std::string name() const override { return "MaxPool2d"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(MaxPool2d);

template <typename Scalar>
class MaxPool1dNode : public BaseDataNode<Scalar> {
 private:
  NodePtr<Scalar> in_;
  size_t len_, stride_ = 1;

  static unsigned dim(Size input_len, size_t stride) {
    return (input_len + stride - 1) / stride;
  }

  void forward_() override {
    Shape is(Tensor<Scalar>::reduce(in_->value().shape(), 3));
    Shape s(Tensor<Scalar>::reduce(in_->value().shape(), 3));
    s[1] = dim(s[1], stride_);
    value().resize(s);

    auto out = value().reshaped({s[0], s[1], 1, s[2]});
    auto inp = in_->value().reshaped({is[0], is[1], 1, is[2]});

    out = eigen::MaxPool2d(inp.template view<4>(), len_, 1, stride_, 1);
  }

  void backward_() override {
    if (in_->has_grad()) {
      Shape is(Tensor<Scalar>::reduce(in_->value().shape(), 3));
      Shape s = value().shape();

      auto out = value().reshaped({s[0], s[1], 1, s[2]});
      auto inp = in_->value().reshaped({is[0], is[1], 1, is[2]});
      auto g = grad().reshaped({s[0], s[1], 1, s[2]});
      auto inp_g = in_->grad().reshaped({is[0], is[1], 1, is[2]});

      if (dev()->kind() == CPU) {
        auto inp_g_ = inp_g.template view<4>();
        eigen::MaxPool2dBackwardLoop(inp_g_,
                                     inp.template view<4>(),
                                     out.template view<4>(),
                                     g.template view<4>(),
                                     len_,
                                     1,
                                     stride_,
                                     1);
#ifdef GINN_ENABLE_GPU
      } else {
        inp_g += eigen::MaxPool2dBackward(inp.template view<4>(),
                                          out.template view<4>(),
                                          g.template view<4>(),
                                          len_,
                                          1,
                                          stride_,
                                          1);
#endif
      }
    }
  }

 public:
  using BaseDataNode<Scalar>::dev;
  using BaseDataNode<Scalar>::value;
  using BaseDataNode<Scalar>::grad;

  MaxPool1dNode(NodePtr<Scalar> input, size_t len, size_t stride = 1)
      : BaseDataNode<Scalar>({input}), in_(input), len_(len), stride_(stride) {}

  std::string name() const override { return "MaxPool1d"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(MaxPool1d);

} // namespace ginn

#endif
