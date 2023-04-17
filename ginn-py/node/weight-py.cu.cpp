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

#include <ginn/node/weight.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include <ginn-py/node/weight-py.h>

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename Scalar>
auto declare_weight_of(py::module_& m) {
  return py::class_<WeightNode<Scalar>, Node<Scalar>, Ptr<WeightNode<Scalar>>>(
      m, name<Scalar>("WeightNode"));
}

template <typename Scalar, typename PyClass>
void bind_weight_of(PyClass& m) {
  using namespace py::literals;
  using T = WeightNode<Scalar>;

  m.def_property_readonly("id", &T::id)
      .def("move_to", &T::move_to, "device"_a)
      .def("tie", &T::tie, "weight"_a)
      .def("copy", &T::copy, "mode"_a)
      .def("fill", &T::template fill<Real>, "val"_a)
      .def("fill", &T::template fill<Int>, "val"_a)
      .def("set_zero", &T::set_zero)
      .def("set_ones", &T::set_ones)
      .def("set_random", &T::set_random);
}

GINN_PY_MAKE_FLOATING_SCALAR_DISPATCHER(Weight)

void bind_weight_node(py::module_& m) {
  using namespace py::literals;

  py::enum_<Copy>(m, "Copy")
      .value("Tied", Copy::Tied)
      .value("Deep", Copy::Deep);

  auto rw = declare_weight_of<Real>(m);
  auto hw = declare_weight_of<Half>(m);

  bind_weight_of<Real>(rw);
  bind_weight_of<Half>(hw);

  m.def("Weight",
        &Weight_<const DevPtr&, const Shape&>,
        "device"_a = cpu(),
        "shape"_a = Shape{0},
        "scalar"_a = Scalar_::Real);
}

} // namespace python
} // namespace ginn
