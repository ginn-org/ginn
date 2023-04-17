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

#include <ginn/init/init.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include <ginn-py/init-py.h>

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename Scalar>
void bind_init_of(py::module_& m) {
  using namespace py::literals;
  using T = Initer<Scalar>;

  py::class_<T>(m, name<Scalar>("Initer"))
      .def("init", py::overload_cast<Tensor<Scalar>&>(&T::init))
      .def("init", py::overload_cast<const WeightPtr<Scalar>&>(&T::init))
      .def("init",
           py::overload_cast<const std::vector<WeightPtr<Scalar>>&>(&T::init));

  py::class_<init::Xavier<Scalar>, T>(m, name<Scalar>("Xavier"))
      .def(py::init<>());
  py::class_<init::Uniform<Scalar>, T>(m, name<Scalar>("Uniform"))
      .def(py::init<Real>(), "range"_a)
      .def(py::init<Int>(), "range"_a);
}

using namespace ginn::init;
GINN_PY_MAKE_FLOATING_SCALAR_DISPATCHER(Xavier)
GINN_PY_MAKE_FLOATING_SCALAR_DISPATCHER(Uniform)

void bind_init(py::module_& m) {
  using namespace py::literals;

  bind_init_of<Real>(m);
  bind_init_of<Half>(m);

  m.def("Xavier", FP(&Xavier_<>), "scalar"_a = Scalar_::Real);
  m.def("Uniform",
        FP(&Uniform_<Real>),
        "range"_a = Real(1),
        "scalar"_a = Scalar_::Real);
  m.def("Uniform",
        FP(&Uniform_<Int>),
        "range"_a = Int(1),
        "scalar"_a = Scalar_::Real);
}

} // namespace python
} // namespace ginn
