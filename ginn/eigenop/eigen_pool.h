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

#ifndef GINN_EIGEN_POOL_H
#define GINN_EIGEN_POOL_H

#include "helpers.h"

namespace ginn {
namespace eigen {

template <typename Input>
EIGEN_STRONG_INLINE auto MaxPool2d(const Input& input,
                                   DenseIndex rows,
                                   DenseIndex cols,
                                   DenseIndex row_stride,
                                   DenseIndex col_stride,
                                   PaddingType padding_type = PADDING_SAME,
                                   DenseIndex row_in_stride = 1,
                                   DenseIndex col_in_stride = 1) {
  using namespace internal;

  static_assert(ndims<Input>() == 4, "Input needs to be an order 4 tensor!");
  static_assert_col_major<Input>();

  auto in_dims = dims(input);

  DenseIndex out_rows, out_cols;

  if (padding_type == PADDING_SAME) {
    out_rows = divup(in_dims[1], row_stride);
    out_cols = divup(in_dims[2], col_stride);
  } else {
    eigen_assert(false and "Not implemented yet!");
  }

  Dims<4> out_dims{in_dims[0], out_rows, out_cols, in_dims[3]};

  return input
      .extract_image_patches(
          rows,
          cols,
          row_stride,
          col_stride,
          row_in_stride,
          col_in_stride,
          padding_type,
          std::numeric_limits<
              typename Eigen::internal::traits<Input>::Scalar>::lowest())
      .maximum(Indices<1, 2>())
      .reshape(out_dims);
}

#ifdef GINN_ENABLE_GPU
template <typename Input, typename Output, typename DOutput>
EIGEN_STRONG_INLINE auto
MaxPool2dBackward(const Input& input,
                  const Output& output,
                  const DOutput& d_output,
                  DenseIndex rows,
                  DenseIndex cols,
                  DenseIndex row_stride,
                  DenseIndex col_stride,
                  /*PaddingType padding_type = PADDING_SAME,*/
                  DenseIndex row_in_stride = 1,
                  DenseIndex col_in_stride = 1) {
  using namespace internal;
  using Scalar = typename Eigen::internal::traits<Input>::Scalar;

  //Eigen::internal::array_prod(Eigen::array<Eigen::DenseIndex, 4UL>());

  static_assert_col_major<Input>();
  static_assert_col_major<Output>();
  static_assert_col_major<DOutput>();

  auto in_dims = dsizes(input);
  auto out_dims = dims(d_output);

  auto channels = in_dims[0];
  auto in_rows = in_dims[1];
  auto in_cols = in_dims[2];
  auto out_rows = out_dims[1];
  auto out_cols = out_dims[2];

  const static auto pads = [](auto ins, auto outs, auto patchs, auto stride) {
    auto excess = std::max<DenseIndex>(0, (outs - 1) * stride + patchs - ins);
    auto shortage = ins - (outs - 1) * stride;
    auto front = patchs - 1 - excess / 2;
    auto back = shortage - 1 + excess / 2;
    return std::make_pair(front, back);
  };

  auto rows_eff =
      rows + (row_in_stride - 1) * (rows - 1); // eff rows after inflation
  auto cols_eff = cols + (col_in_stride - 1) * (cols - 1);

  auto [pad_top, pad_bottom] = pads(in_rows, out_rows, rows_eff, row_stride);
  auto [pad_left, pad_right] = pads(in_cols, out_cols, cols_eff, col_stride);

  using Scalar = typename Eigen::internal::traits<Input>::Scalar;

  auto output_patches = output.extract_image_patches(rows,
                                                     cols,
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
                                                     Scalar(0.));
  auto is_max =
      (input.reshape(Dims<5>{channels, 1, 1, in_rows * in_cols, in_dims[3]})
           .broadcast(Dims<5>{1, rows, cols, 1, 1}) == output_patches)
          .template cast<Scalar>();

  auto d_output_patches = d_output.extract_image_patches(rows,
                                                         cols,
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
                                                         Scalar(0.));
  return (d_output_patches * is_max).sum(Index<2>({1, 2})).reshape(in_dims);
}
#endif

// This one works faster on CPU
template <typename DInput, typename Input, typename Output, typename DOutput>
EIGEN_STRONG_INLINE void MaxPool2dBackwardLoop(DInput& d_input,
                                               const Input& input,
                                               const Output& output,
                                               const DOutput& d_output,
                                               DenseIndex rows,
                                               DenseIndex cols,
                                               DenseIndex row_stride,
                                               DenseIndex col_stride,
                                               DenseIndex row_in_stride = 1,
                                               DenseIndex col_in_stride = 1) {
  using Scalar = typename Eigen::internal::traits<DInput>::Scalar;
  using PairScalar = Pair<DenseIndex, Scalar>;
  using IndexValueTensor = Eigen::Tensor<PairScalar, 4>;

  auto out_dims = dims(output);
  // TODO: use ginn::Tensors with appropriate dev()
  // Can take tmp and index pairs as args.
  IndexValueTensor tmp(out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
  IndexValueTensor index_pairs(
      out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
  index_pairs = input.index_pairs();
  tmp = index_pairs
            .extract_image_patches(
                rows,
                cols,
                row_stride,
                col_stride,
                row_in_stride,
                col_in_stride,
                PADDING_SAME,
                PairScalar(0, std::numeric_limits<Scalar>::lowest()))
            .reduce(Indices<1, 2>(),
                    Eigen::internal::ArgMaxPairReducer<PairScalar>())
            .reshape(dsizes(output));

  for (DenseIndex i = 0; i < out_dims[0]; i++) {
    for (DenseIndex r = 0; r < out_dims[1]; r++) {
      for (DenseIndex c = 0; c < out_dims[2]; c++) {
        for (DenseIndex k = 0; k < out_dims[3]; k++) {
          d_input(tmp(i, r, c, k).first) += d_output(i, r, c, k);
        }
      }
    }
  }
}

} // namespace eigen
} // namespace ginn

#endif
