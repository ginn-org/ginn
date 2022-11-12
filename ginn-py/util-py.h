#ifndef GINN_PY_UTIL_PY_H
#define GINN_PY_UTIL_PY_H

#include <string>
#include <vector>

#include <ginn/node/data.h>

namespace ginn {
namespace python {

namespace py = pybind11;

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
  // is there a cheaper way to return safe (const char *)s?
  static std::vector<std::string> names;
  names.emplace_back(scalar_name<Scalar>() + name);
  return names.back().c_str();
}

// Syntactic shorthand for nodes that derive from BaseDataNode, which are many
template <typename Scalar, template <class> typename Node>
using PyNode =
    py::class_<Node<Scalar>, BaseDataNode<Scalar>, Ptr<Node<Scalar>>>;

// I use this useless indirection to help nvcc 11.1 not bork during template
// deduction -- for some reason assigning to a temporary helps. Should be
// unnecessary once I upgrade nvcc.
#define FP(F)                                                                  \
  [&]() {                                                                      \
    auto fp = F;                                                               \
    return fp;                                                                 \
  }()

} // namespace python
} // namespace ginn
#endif
