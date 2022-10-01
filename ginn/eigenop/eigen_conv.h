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

#ifndef GINN_EIGEN_CONV_H
#define GINN_EIGEN_CONV_H

#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include "helpers.h"

namespace ginn {
namespace eigen {

// kernel: filters x channels x rows x cols
// input: channels x rows x cols x others... (e.g. batch)
template <typename Input, typename Kernel>
EIGEN_STRONG_INLINE auto Conv2d(const Input& input,
                                const Kernel& kernel,
                                DenseIndex row_stride = 1,
                                DenseIndex col_stride = 1,
                                PaddingType padding_type = PADDING_SAME,
                                DenseIndex row_in_stride = 1,
                                DenseIndex col_in_stride = 1) {
  static_assert_col_major<Input, Kernel>();

  auto in_dims = dims(input);
  auto k_dims = dims(kernel);

  DenseIndex out_rows, out_cols;
  if (padding_type == PADDING_SAME) {
    out_rows = divup(in_dims[1], row_stride);
    out_cols = divup(in_dims[2], col_stride);
  } else {
    eigen_assert(false and "Not implemented yet!");
  }

  const auto inner_dim = k_dims[1] * k_dims[2] * k_dims[3];

  Dims<2> left_dims{k_dims[0], inner_dim};

  Dims<2> right_dims{inner_dim, out_rows * out_cols};
  using rank_t = decltype(ndims<Input>());
  for (rank_t i = 3; i < ndims<Input>(); i++) { right_dims[1] *= in_dims[i]; }

  auto out_dims = dsizes(input);
  out_dims[0] = k_dims[0];
  out_dims[1] = out_rows;
  out_dims[2] = out_cols;

  array<IndexPair<DenseIndex>, 1> contract_dims{IndexPair<DenseIndex>(1, 0)};

  auto left = kernel.reshape(left_dims);
  auto right = input
                   .extract_image_patches(k_dims[2],
                                          k_dims[3],
                                          row_stride,
                                          col_stride,
                                          row_in_stride,
                                          col_in_stride,
                                          padding_type)
                   .reshape(right_dims);
  return left.contract(right, contract_dims).reshape(out_dims);
}

template <typename Input, typename Kernel, typename DOutput>
EIGEN_STRONG_INLINE auto
Conv2dBackwardKernel(const Input& input,
                     const Kernel& kernel,
                     const DOutput& d_output,
                     DenseIndex row_stride = 1,
                     DenseIndex col_stride = 1,
                     PaddingType padding_type = PADDING_SAME,
                     DenseIndex row_in_stride = 1,
                     DenseIndex col_in_stride = 1) {
  static_assert_col_major<Input, Kernel, DOutput>();

  auto in_dims = dims(input);
  auto k_dims = dims(kernel);

  DenseIndex out_rows, out_cols;
  if (padding_type == PADDING_SAME) {
    out_rows = divup(in_dims[1], row_stride);
    out_cols = divup(in_dims[2], col_stride);
  } else {
    eigen_assert(false and "Not implemented yet!");
  }

  const auto inner_dim = k_dims[1] * k_dims[2] * k_dims[3];

  Dims<2> out_2d{k_dims[0], nelems(d_output) / k_dims[0]};

  Dims<2> right_dims{inner_dim, out_rows * out_cols};
  using rank_t = decltype(ndims<Input>());
  for (rank_t i = 3; i < ndims<Input>(); i++) { right_dims[1] *= in_dims[i]; }

  array<IndexPair<DenseIndex>, 1> contract_dims{IndexPair<DenseIndex>(1, 0)};

  auto left = d_output.reshape(out_2d);
  auto right = input
                   .extract_image_patches(k_dims[2],  // input
                                          k_dims[3],  // extracted
                                          row_stride, // & transposed
                                          col_stride,
                                          row_in_stride,
                                          col_in_stride,
                                          padding_type)
                   .reshape(right_dims)
                   .shuffle(Dims<2>{1, 0});
  return left.contract(right, contract_dims).reshape(k_dims);
}

template <typename Input, typename Kernel, typename DOutput>
EIGEN_STRONG_INLINE auto
Conv2dBackwardInput(const Input& input,
                    const Kernel& kernel,
                    const DOutput& d_output,
                    DenseIndex row_stride = 1,
                    DenseIndex col_stride = 1,
                    PaddingType /*padding_type*/ = PADDING_SAME,
                    DenseIndex row_in_stride = 1,
                    DenseIndex col_in_stride = 1) {
  static_assert_col_major<Input, Kernel, DOutput>();

  auto out_dims = dims(d_output);

  auto [_out_feat, out_rows, out_cols, _out_batch] = out_dims;
  auto [_in_feat, in_rows, in_cols, _in_batch] = dims(input);
  auto [k_filters, k_channels, k_rows, k_cols] = dims(kernel);
  // effective kernel size after inflation
  auto k_rows_eff = k_rows + (row_in_stride - 1) * (k_rows - 1);
  auto k_cols_eff = k_cols + (col_in_stride - 1) * (k_cols - 1);

  auto inner_dim = k_filters * k_rows * k_cols;

  Indices<0, 0, 1, 1> spatial;

  Dims<4> transpose{1, 0, 2, 3}; // no IndexList for shuffle?

  Dims<2> k_dims_2d{k_channels, inner_dim};

  auto left = kernel.reverse(spatial).shuffle(transpose).reshape(k_dims_2d);

  Dims<2> out_dims_2d{inner_dim, in_rows * in_cols};
  for (int i = 3; i < ndims<DOutput>(); ++i) { out_dims_2d[1] *= out_dims[i]; }

  const static auto pads = [](auto ins, auto outs, auto kerns, auto stride) {
    auto excess = std::max<DenseIndex>(0, (outs - 1) * stride + kerns - ins);
    auto shortage = ins - (outs - 1) * stride;
    auto front = kerns - 1 - excess / 2;
    auto back = shortage - 1 + excess / 2;
    return std::make_pair(front, back);
  };

  auto [pad_top, pad_bottom] = pads(in_rows, out_rows, k_rows_eff, row_stride);
  auto [pad_left, pad_right] = pads(in_cols, out_cols, k_cols_eff, col_stride);

  using Scalar = typename Eigen::internal::traits<Input>::Scalar;
  auto right = d_output
                   .extract_image_patches(k_rows,
                                          k_cols,
                                          1,
                                          1,
                                          row_in_stride,
                                          col_in_stride,
                                          row_stride,
                                          col_stride,
                                          pad_top,
                                          pad_bottom,
                                          pad_left,
                                          pad_right,
                                          Scalar(0))
                   .reshape(out_dims_2d);

  array<IndexPair<DenseIndex>, 1> contract_dims{IndexPair<DenseIndex>(1, 0)};
  return left.contract(right, contract_dims).reshape(dsizes(input));
}

} // namespace eigen
} // namespace ginn

#endif
