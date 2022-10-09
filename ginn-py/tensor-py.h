#ifndef GINN_PY_TENSOR_PY_H
#define GINN_PY_TENSOR_PY_H

#include <pybind11/pybind11.h>

#include <ginn/tensor.h>
#include <ginn/util/tensorio.h>

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename Scalar>
std::string scalar_name() {
  if constexpr (std::is_same_v<Scalar, Real>) { return "Real"; }
  if constexpr (std::is_same_v<Scalar, Half>) { return "Half"; }
  if constexpr (std::is_same_v<Scalar, Int>) { return "Int"; }
  if constexpr (std::is_same_v<Scalar, bool>) { return "Bool"; }
  return "";
}

template <typename Scalar>
std::string tensor_name() {
  return scalar_name<Scalar>() + "Tensor";
}

template <typename Scalar>
void bind_tensor(py::module_& m) {
  py::class_<Tensor<Scalar>>(m, tensor_name<Scalar>().c_str())
      .def(py::init<const Shape&, const std::vector<Scalar>&>())
      .def(py::init<Device&, const Shape&, const std::vector<Scalar>&>())
      .def("shape", &Tensor<Scalar>::shape)
      .def("real", &Tensor<Scalar>::template cast<Real>)
      .def("half", &Tensor<Scalar>::template cast<Half>)
      .def("integral", &Tensor<Scalar>::template cast<Int>)
      .def("boolean", &Tensor<Scalar>::template cast<bool>)
      .def("__repr__", [&](const Tensor<Scalar>& t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });
}

} // namespace python
} // namespace ginn

#endif
