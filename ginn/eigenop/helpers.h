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

#ifndef GINN_EIGENOP_HELPERS_H
#define GINN_EIGENOP_HELPERS_H

// TODO: What will the namespace be???
// TODO: Putting these under Eigen might be dangerous for Eigen users

#include <ginn/def.h>

#include <unsupported/Eigen/CXX11/Tensor>

namespace ginn {
namespace eigen {

using namespace Eigen; // Expose Eigen under ginn::eigen

template <typename Expression>
using TensorRefLike =
    TensorRef<Tensor<typename Eigen::internal::traits<Expression>::Scalar,
                     Eigen::internal::traits<Expression>::NumDimensions,
                     Eigen::internal::traits<Expression>::Layout,
                     typename Eigen::internal::traits<Expression>::Index>>;

template <size_t N>
using Dims = DSizes<DenseIndex, N>;

template <size_t... Ns>
using Indices = IndexList<type2index<Ns>...>;

template <typename Expression>
constexpr auto ndims() {
  return Eigen::internal::traits<Expression>::NumDimensions;
}

template <typename Expression>
auto dsizes(const Expression& expr) {
  return TensorRefLike<Expression>(expr).dimensions();
}

template <size_t N, typename Left, typename Right>
void assign(Left& left, const Right& right) {
  left[N - 1] = right[N - 1];
  if constexpr (N > 1) { assign<N - 1>(left, right); }
}

template <size_t N>
auto as_array(const DSizes<DenseIndex, N>& dims) {
  std::array<DenseIndex, N> a;
  assign<N>(a, dims);
  return a;
}

template <typename Expression>
auto dims(const Expression& expr) {
  // Returning as array to enable structured bindings declarations
  return as_array<ndims<Expression>()>(dsizes(expr));
}

template <typename Expression>
auto nelems(const Expression& expr) {
  TensorRefLike<Expression> eref(expr);
  return eref.size();
}

template <typename Expression, typename... Expressions>
constexpr void static_assert_col_major() {
  static_assert(Eigen::internal::traits<Expression>::Layout == ColMajor,
                "Only ColMajor order is supported!");
  if constexpr (sizeof...(Expressions) > 0) {
    static_assert_col_major<Expressions...>();
  }
}

} // namespace eigen
} // namespace ginn

#endif
