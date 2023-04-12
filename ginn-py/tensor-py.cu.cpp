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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn-py/tensor-py.h>

#include <ginn-py/util-py.h>

namespace ginn {
namespace python {

template <typename Scalar>
auto declare_tensor_of(py::module_& m) {
  return py::class_<Tensor<Scalar>>(m, name<Scalar>("Tensor"));
}

template <typename Scalar, typename PyClass>
void bind_tensor_of(PyClass& m) {
  using namespace pybind11::literals;
  using T = Tensor<Scalar>;

  m.def(py::init<>());
  m.def(py::init<DevPtr>(), "device"_a);
  m.def(py::init<Shape>(), "shape"_a);
  m.def(py::init<Shape, const std::vector<Scalar>&>(), "shape"_a, "val"_a);
  m.def(py::init<DevPtr, Shape>(), "device"_a, "shape"_a);
  m.def(py::init<DevPtr, Shape, std::vector<Scalar>>(),
        "device"_a,
        "shape"_a,
        "val"_a);
  m.def_property_readonly("dev", &T::dev);
  m.def_property("shape", &T::shape, &T::resize);
  m.def_property_readonly("size", py::overload_cast<>(&T::size, py::const_));
  if constexpr (std::is_same_v<Scalar, Half>) {
    m.def("list", [](const T& t) { return t.template cast<Real>().vector(); });
  } else {
    m.def("list", &T::vector);
  }
  m.def("v", py::overload_cast<>(&T::v));
  m.def("m", py::overload_cast<>(&T::m));
  m.def("item", &T::item);
  m.def("real", &T::template cast<Real>);
  m.def("half", &T::template cast<Half>);
  m.def("int", &T::template cast<Int>);
  m.def("bool", &T::template cast<bool>);
  m.def("cast", [](const T& t, Scalar_ scalar) -> py::object {
    switch (scalar) {
    case Scalar_::Real: return py::cast(t.template cast<Real>());
    case Scalar_::Half: return py::cast(t.template cast<Half>());
    case Scalar_::Int: return py::cast(t.template cast<Int>());
    case Scalar_::Bool: return py::cast(t.template cast<bool>());
    default: return {};
    }
  });
  m.def("set_zero", &T::set_zero);
  m.def("set_ones", &T::set_ones);
  m.def("set_random", &T::set_random);

  if constexpr (std::is_same_v<Scalar, bool>) {
    for_range<5>([&](auto arr) { m.def("set", &T::template set<arr.size()>); });
  } else {
    for_range<5>(
        [&](auto arr) { m.def("set", &T::template set<arr.size(), Real>); });
    for_range<5>(
        [&](auto arr) { m.def("set", &T::template set<arr.size(), Int>); });
  }

  m.def("copy_to", &T::copy_to, "device"_a);
  m.def("move_to", &T::move_to, "device"_a);
  m.def("maybe_copy_to", &T::maybe_copy_to, "device"_a);
  m.def_property_readonly("scalar", [](const T&) { return scalar_<Scalar>(); });
  m.def("__repr__", [&](const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
  });
  m.def(
      "__eq__",
      [](const T& a, const T& b) { return a == b; },
      py::is_operator());
}

GINN_PY_MAKE_SCALAR_DISPATCHER(Tensor)

void bind_tensor(py::module_& m) {
  using namespace py::literals;

  py::enum_<Scalar_>(m, "Scalar")
      .value("Real", Scalar_::Real)
      .value("Half", Scalar_::Half)
      .value("Int", Scalar_::Int)
      .value("Bool", Scalar_::Bool)
      .export_values();

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

  // Why is &Tensor_<> an unknown type? -- might be a nvcc11.1 thing.
  m.def("Tensor",
        static_cast<py::object (*)(Scalar_)>(&Tensor_<>),
        "scalar"_a = Scalar_::Real);
  m.def("Tensor",
        &Tensor_<const DevPtr&>,
        "device"_a,
        "scalar"_a = Scalar_::Real);
  m.def("Tensor",
        &Tensor_<const DevPtr&, const Shape&>,
        "device"_a,
        "shape"_a,
        "scalar"_a = Scalar_::Real);
  m.def(
      "Tensor", &Tensor_<const Shape&>, "shape"_a, "scalar"_a = Scalar_::Real);
}

} // namespace python
} // namespace ginn
