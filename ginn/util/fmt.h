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

#ifndef GINN_UTIL_FMT_H
#define GINN_UTIL_FMT_H

#ifdef FMT_FORMAT_H_
#ifndef FMT_HEADER_ONLY
static_assert(false,
              "Ginn uses fmt in header only mode but you pre-included "
              "fmt in normal mode!");
#endif
#endif

#define FMT_HEADER_ONLY

#include <fmt/color.h>
#include <fmt/format.h>

#include <iostream>

namespace ginn {

namespace internal {

// Example: fmt::format("Op {} took {:.2f} {}", a, 3.145, "secs");
// Desired syntactic sugar: ("Op {} took {:.2f} {}"_f, a, 3.145, "secs");

// MemberTypeHelper is used to turn string literals (i.e. char[N]) to
// std::string
template <typename T>
struct MemberTypeHelper {
  using type = T;
};

template <int N>
struct MemberTypeHelper<char[N]> {
  using type = std::string;
};

template <typename T>
using member_t = typename MemberTypeHelper<T>::type;

template <typename... Args>
struct FormatHelper {
  static_assert(sizeof...(Args) == 0,
                "Programming error! Base case must have 0 template args!");
  std::string s_;

  template <typename... Args2>
  std::string realize(const Args2&... args) const {
    return fmt::format(s_, args...);
  }

  FormatHelper(std::string s) : s_(std::move(s)) {}
};

template <typename Arg, typename... Args>
struct FormatHelper<Arg, Args...> {
  FormatHelper<Args...> head_;
  member_t<Arg> arg_;

  template <typename... Args2>
  std::string realize(const Args2&... args) const {
    return head_.realize(arg_, args...);
  }

  FormatHelper(FormatHelper<Args...> head, const Arg& arg)
      : head_(head), arg_(arg) {}

  operator std::string() const { return realize(); }
};

template <typename Arg, typename... Args>
auto operator,(FormatHelper<Args...> head, const Arg& arg) {
  return FormatHelper<Arg, Args...>(head, arg);
}

template <typename... Args>
std::ostream& operator<<(std::ostream& out, const FormatHelper<Args...>& fh) {
  out << (std::string)fh;
  return out;
}

} // namespace internal

namespace literals {

inline auto operator"" _f(const char* s, size_t) {
  return internal::FormatHelper(std::string(s));
}

} // namespace literals

} // namespace ginn

#endif
