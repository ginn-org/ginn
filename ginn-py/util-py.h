#ifndef GINN_PY_UTIL_PY_H
#define GINN_PY_UTIL_PY_H

#include <string>

namespace ginn {
namespace python {

// loop over types and execute a template lambda over each type
// useful for applying some logic to every scalar type
template <typename... Scalars, typename F>
void for_each(F f) {
  (f(Scalars()), ...);
}

template <typename F, size_t... Ints>
void for_range_helper(F f, std::index_sequence<Ints...>) {
  (f(std::array<bool, Ints>()), ...);
}

template <size_t N, typename F>
void for_range(F f) {
  for_range_helper(f, std::make_index_sequence<N>());
}

template <typename Scalar>
std::string scalar_name() {
  if constexpr (std::is_same_v<Scalar, Real>) { return "Real"; }
  if constexpr (std::is_same_v<Scalar, Half>) { return "Half"; }
  if constexpr (std::is_same_v<Scalar, Int>) { return "Int"; }
  if constexpr (std::is_same_v<Scalar, bool>) { return "Bool"; }
  return "";
}

template <typename Scalar>
auto name(std::string name) {
  return scalar_name<Scalar>() + name;
}

} // namespace python
} // namespace ginn
#endif
