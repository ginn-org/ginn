#ifndef GINN_PY_TENSOR_PY_H
#define GINN_PY_TENSOR_PY_H

#include <pybind11/operators.h>
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
auto declare_tensor_of(py::module_& m) {
  return py::class_<Tensor<Scalar>>(m, tensor_name<Scalar>().c_str());
}

template <typename Scalar, typename PyClass>
void bind_tensor_of(PyClass& m) {
  using namespace pybind11::literals;
  using T = Tensor<Scalar>;

  m.def(py::init<>())
      .def(py::init<DevPtr>(), "device"_a)
      .def(py::init<const Shape&, const std::vector<Scalar>&>(),
           "shape"_a,
           "val"_a)
      .def(py::init<DevPtr, const Shape&>(), "device"_a, "shape"_a)
      .def(py::init<DevPtr, const Shape&, const std::vector<Scalar>&>(),
           "device"_a,
           "shape"_a,
           "val"_a)
      .def("dev", &T::dev)
      .def("shape", &T::shape)
      .def("real", &T::template cast<Real>)
      .def("half", &T::template cast<Half>)
      .def("integral", &T::template cast<Int>)
      .def("boolean", &T::template cast<bool>)
      .def("set", &T::template set<1>)
      .def("set", &T::template set<2>)
      .def("set", &T::template set<3>)
      .def("set", &T::template set<4>)
      .def(py::self == py::self)
      .def("__repr__", [&](const T& t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });
}

inline void bind_tensor(py::module_& m) {
  // making pybind know all tensor types first, so method docs contain the
  // appropriate python types throughout.
  auto mr = declare_tensor_of<Real>(m);
  auto mi = declare_tensor_of<Int>(m);
  auto mh = declare_tensor_of<Half>(m);
  auto mb = declare_tensor_of<bool>(m);

  bind_tensor_of<Real>(mr);
  bind_tensor_of<Int>(mi);
  bind_tensor_of<Half>(mh);
  bind_tensor_of<bool>(mb);
}

} // namespace python
} // namespace ginn

#endif
