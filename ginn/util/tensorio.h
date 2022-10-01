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

#ifndef GINN_UTIL_TENSORIO_H
#define GINN_UTIL_TENSORIO_H

#include <ginn/tensor.h>
#include <ginn/util/fmt.h>

namespace ginn {

template <typename Scalar>
unsigned short max_width(const Tensor<Scalar>& t) {
  unsigned short w = 0;
  for (Scalar x : t) {
    std::string s = fmt::format("{: .6}", x);
    w = std::max<unsigned short>(w, s.size());
  }
  return w;
}

template <typename Scalar>
void print_helper(std::ostream& os,
                  const Tensor<Scalar>& t,
                  Size start,
                  Size stride,
                  Size remaining_rank,
                  unsigned short width) {
  Size rank = t.shape().size();
  auto w = std::to_string(width);
  os << "{";
  for (Size i = 0; i < t.shape()[rank - remaining_rank]; i++) {
    Size j = start + i * stride;
    if (i > 0) { os << (remaining_rank < rank or rank == 1 ? ", " : ",\n "); }
    if (remaining_rank == 1) {
      os << fmt::format("{: " + w + ".6}", t.v()[j]);
    } else {
      print_helper(os,
                   t,
                   j,
                   stride * t.shape()[rank - remaining_rank],
                   remaining_rank - 1,
                   width);
    }
  }
  os << "}";
}

template <typename Scalar>
void print(std::ostream& os, const Tensor<Scalar>& t) {
  if (t.dev().type() != CPU) {
    auto t_ = t.copy_to(cpu());
    print(os, t_);
  }

  if (t.shape().size() == 0) {
    os << t.item();
  } else {
    auto w = max_width(t);
    print_helper(os, t, 0, 1, t.shape().size(), w);
  }
}

// TODO: does this need to be exposed outside of ginn namespace?
template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const Tensor<Scalar>& t) {
  print(os, t);
  return os;
}

} // namespace ginn

#endif
