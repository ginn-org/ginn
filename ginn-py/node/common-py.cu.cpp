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

#include <ginn/node/common.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include <ginn-py/node/common-py.h>

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_common_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, AddNode>(m, name<Scalar>("AddNode"));
    m.def("Add", &Add<const Np&, const Np&>);
    m.def("Add", &Add<const std::vector<Np>&>);

    PyNode<Scalar, AddScalarNode>(m, name<Scalar>("AddScalarNode"));
    m.def("AddScalar", &AddScalar<const Np&, const Real&>);
    m.def("AddScalar", &AddScalar<const Np&, const Int&>);

    PyNode<Scalar, SubtractScalarNode>(m, name<Scalar>("SubtractScalarNode"));
    m.def("SubtractScalar", &SubtractScalar<Real, Np>);
    m.def("SubtractScalar", &SubtractScalar<Int, Np>);

    PyNode<Scalar, ProdScalarNode>(m, name<Scalar>("ProdScalarNode"));
    m.def("ProdScalar", &ProdScalar<const Np&, const Real&>);
    m.def("ProdScalar", &ProdScalar<const Np&, const Int&>);

    PyNode<Scalar, CwiseProdNode>(m, name<Scalar>("CwiseProdNode"));
    m.def("CwiseProd", &CwiseProd<const Np&, const Np&>);

    PyNode<Scalar, CwiseProdAddNode>(m, name<Scalar>("CwiseProdAddNode"));
    m.def("CwiseProdAdd",
          &CwiseProdAdd<const Np&, const Np&, const Np&, const Real&>,
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a = 0.);
    m.def("CwiseProdAdd",
          &CwiseProdAdd<const Np&, const Np&, const Np&, const Int&>,
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a);

    PyNode<Scalar, CwiseMaxNode>(m, name<Scalar>("CwiseMaxNode"));
    m.def("CwiseMax", &CwiseMax<const std::vector<Np>&>);
  });
}

} // namespace python
} // namespace ginn
