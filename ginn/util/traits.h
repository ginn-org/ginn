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

#ifndef GINN_UTIL_TRAITS_H
#define GINN_UTIL_TRAITS_H

#include <ginn/def.h>
#include <type_traits>

namespace ginn {

// ginn::is_arithmetic == "either std::is_arithmetic, or is ginn::Half"
// TODO: should I make this tighter? E.g. exclude bool or char?
template <typename T>
struct is_arithmetic {
  static constexpr bool value =
      std::is_arithmetic_v<T> or std::is_same_v<std::remove_cv_t<T>, Half>;
};

template <typename T>
constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

// ginn::is_floating_point == "either std::is_floating_point, or is ginn::Half"
template <typename T>
struct is_floating_point {
  static constexpr bool value =
      std::is_floating_point_v<T> or std::is_same_v<std::remove_cv_t<T>, Half>;
};

template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

//
template <typename T>
class Ptr;

// ginn::is_node_ptr == "T is a Ptr<U> for some U",
// assumes Ptr constrains U to derive from Node.
template <typename T>
struct is_node_ptr {
  static constexpr bool value = false;
};

template <typename T>
struct is_node_ptr<Ptr<T>> {
  static constexpr bool value = true;
};

template <typename T>
constexpr bool is_node_ptr_v = is_node_ptr<T>::value;

template <typename T>
using if_node_ptr_t = std::enable_if_t<is_node_ptr_v<T>>;

// Get innermost type for nested std::vectors
template <typename T>
struct innermost {
  using type = T;
};

template <typename T>
struct innermost<std::vector<T>> {
  using type = typename innermost<T>::type;
};

template <typename T>
using innermost_t = typename innermost<T>::type;

} // namespace ginn

#endif
