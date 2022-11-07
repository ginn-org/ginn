#ifndef GINN_PY_TENSOR_PY_H
#define GINN_PY_TENSOR_PY_H

#include <ginn/def.h>

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <ginn/tensor.h>
#include <ginn/util/tensorio.h>

#include <ginn-py/util-py.h>

namespace ginn {
namespace python {

namespace py = pybind11;

// This is a helper type enum to dispatch things based on the scalar type
// on the Python side.
enum class Scalar_ { Real, Half, Int, Bool };

template <typename Scalar>
Scalar_ scalar_() {
  if constexpr (std::is_same_v<Scalar, Real>) { return Scalar_::Real; }
  if constexpr (std::is_same_v<Scalar, Half>) { return Scalar_::Half; }
  if constexpr (std::is_same_v<Scalar, Int>) { return Scalar_::Int; }
  if constexpr (std::is_same_v<Scalar, bool>) { return Scalar_::Bool; }
  GINN_THROW("Unexpected scalar type!");
  return {};
}

template <typename Scalar>
std::string tensor_name() {
  return scalar_name<Scalar>() + "Tensor";
}

template <typename... Args>
py::object Tensor_(Args&&... args, Scalar_ scalar) {
  if (scalar == Scalar_::Real) {
    return py::cast(Tensor<Real>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Half) {
    return py::cast(Tensor<Half>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Int) {
    return py::cast(Tensor<Int>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Bool) {
    return py::cast(Tensor<bool>(std::forward<Args>(args)...));
  } else {
    GINN_THROW("Unexpected Scalar type!");
    return {};
  }
}

void bind_tensor(py::module_& m);

} // namespace python
} // namespace ginn

#endif
