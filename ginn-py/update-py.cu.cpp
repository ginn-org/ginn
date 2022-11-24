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

#include <ginn/update/update.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "update-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

using namespace ginn::update;

template <typename Scalar>
void bind_update_of(py::module_& m) {
  using namespace py::literals;
  using T = Updater<Scalar>;

  py::class_<T, std::unique_ptr<T>>(m, name<Scalar>("Updater"))
      .def("update",
           py::overload_cast<const std::vector<WeightPtr<Scalar>>&>(&T::update))
      .def("update", py::overload_cast<Graph&>(&T::update));

  py::class_<Sgd<Scalar>, T>(m, name<Scalar>("Sgd"))
      .def(py::init<Real, Real>(), "lr"_a = 0.1, "clip"_a = 5.);
  py::class_<Adam<Scalar>, T>(m, name<Scalar>("Adam"))
      .def(py::init<Real, Real, Real, Real, Real>(),
           "lr"_a = 0.1,
           "clip"_a = 5.,
           "eps"_a = 1e-8,
           "beta_1"_a = 0.9,
           "beta_2"_a = 0.999);
}

// Given a template class / function such as Initer<Scalar>(), define
//   a non-template version Initer_(..., Scalar_ s) that dispatches based on the
//   scalar.
#define GINN_PY_MAKE_DISPATCHER_UNIQUE(F)                                      \
  template <typename... Args>                                                  \
  py::object F##_(Args&&... args, Scalar_ scalar) {                            \
    if (scalar == Scalar_::Real) {                                             \
      return py::cast(std::make_unique<F<Real>>(std::forward<Args>(args)...)); \
    } else if (scalar == Scalar_::Half) {                                      \
      return py::cast(std::make_unique<F<Half>>(std::forward<Args>(args)...)); \
    } else {                                                                   \
      GINN_THROW("Unexpected Scalar type!");                                   \
      return {};                                                               \
    }                                                                          \
  }

GINN_PY_MAKE_DISPATCHER_UNIQUE(Sgd)
GINN_PY_MAKE_DISPATCHER_UNIQUE(Adam)

void bind_update(py::module_& m) {
  using namespace py::literals;

  bind_update_of<Real>(m);
  bind_update_of<Half>(m);

  m.def("Sgd",
        &Sgd_<Real, Real>,
        "lr"_a = 0.1,
        "clip"_a = 5.,
        "scalar"_a = Scalar_::Real);
  m.def("Adam",
        &Adam_<Real, Real, Real, Real, Real>,
        "lr"_a = 1e-3,
        "clip"_a = 5.,
        "eps"_a = 1e-8,
        "beta_1"_a = 0.9,
        "beta_2"_a = 0.999,
        "scalar"_a = Scalar_::Real);
}

} // namespace python
} // namespace ginn
