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

#ifndef GINN_UTIL_AMAX_H
#define GINN_UTIL_AMAX_H

#include <Eigen/Dense>
#include <algorithm>
#include <ginn/def.h>
#include <ginn/except.h>
#include <ginn/tensor.h>
#include <vector>

namespace ginn {

template <typename T>
size_t argmax(const std::vector<T>& v) {
  return std::max_element(v.begin(), v.end()) - v.begin();
}

template <typename Scalar>
RowVector<Int> argmax(const MatrixMap<Scalar>& x) {
  RowVector<Int> index(x.cols());
  for (int i = 0; i < x.cols(); i++) {
    auto begin = x.data() + x.rows() * i;
    index(i) = std::max_element(begin, begin + x.rows()) - begin;
  }
  return index;
}

template <typename Scalar>
size_t argmax(const VectorMap<Scalar>& x) {
  return std::max_element(x.data(), x.data() + x.size()) - x.data();
}

// Simple flattened argmax (as if tensor was a flat vector)
template <typename Scalar>
auto argmax(const Tensor<Scalar>& t) {
  using Index = typename TensorMap<Scalar, 1>::Index;
  Tensor<Index> rt(t.dev(), Shape{});
  rt = t.t().argmax();
  return rt.maybe_copy_to(cpu()).item();
}

// Argmax w.r.t. a dim / axis
template <typename Scalar>
auto argmax(const Tensor<Scalar>& t, Size dim) {
  using Index = typename TensorMap<Scalar, 1>::Index;
  auto s = t.shape();
  GINN_ASSERT(dim < (Size)s.size(), "Index dim is bigger than tensor rank!");
  GINN_ASSERT(s.size() <= 8, "Unsupported large rank!");
  s.erase(s.begin() + dim);
  Tensor<Index> rt(t.dev(), s);
  rt = t.template view<8>().argmax(dim);
  return rt;
}

} // namespace ginn

#endif
