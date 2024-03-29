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

#include <ginn/node/select.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include <ginn-py/node/select-py.h>

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_select_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, SelectNode>(m, name<Scalar>("SelectNode"));
    m.def("Select", &Select<const Np&, const Np&>);
    m.def("Select", &Select<const Np&, const Scalar&>);
    m.def("Select", &Select<const Scalar&, const Np&>);
    m.def("Select", &Select<const Scalar&, const Scalar&>);

    PyNode<Scalar, MaskNode>(m, name<Scalar>("MaskNode"));
    m.def("Mask", &Mask<const Np&, const Np&, const Real&>);
    m.def("Mask", &Mask<const Np&, const Np&, const Int&>);
  });
}

} // namespace python
} // namespace ginn
