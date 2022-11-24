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

#include <ginn/node/pick.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include "pick-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_pick_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, PickNode>(m, name<Scalar>("PickNode"));
    m.def("Pick", FP((&Pick<const Np&, const DataPtr<Int>&>)));
    m.def("Pick", FP((&Pick<const Np&, const std::vector<Int>&>)));
    m.def("Pick", FP((&Pick<const Np&, const Int&>)));
  });

  for_each<Real, Half>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    py::class_<PickSoftmaxNode<Scalar>,
               PickNode<Scalar>,
               Ptr<PickSoftmaxNode<Scalar>>>(m,
                                             name<Scalar>("PickSoftmaxNode"));
    m.def("PickSoftmax", FP((&PickSoftmax<const Np&, const DataPtr<Int>&>)));
    m.def("PickSoftmax",
          FP((&PickSoftmax<const Np&, const std::vector<Int>&>)));
    m.def("PickSoftmax", FP((&PickSoftmax<const Np&, const Int&>)));

    py::class_<PickNegLogSoftmaxNode<Scalar>,
               PickSoftmaxNode<Scalar>,
               Ptr<PickNegLogSoftmaxNode<Scalar>>>(
        m, name<Scalar>("PickNegLogSoftmaxNode"));
    m.def("PickNegLogSoftmax",
          FP((&PickNegLogSoftmax<const Np&, const DataPtr<Int>&>)));
    m.def("PickNegLogSoftmax",
          FP((&PickNegLogSoftmax<const Np&, const std::vector<Int>&>)));
    m.def("PickNegLogSoftmax", FP((&PickNegLogSoftmax<const Np&, const Int&>)));

    PyNode<Scalar, PickNegLogSigmoidNode>(
        m, name<Scalar>("PickNegLogSigmoidNode"));
    m.def("PickNegLogSigmoid",
          FP((&PickNegLogSigmoid<const Np&, const DataPtr<Int>&>)));
    m.def("PickNegLogSigmoid",
          FP((&PickNegLogSigmoid<const Np&, const std::vector<Int>&>)));
    m.def("PickNegLogSigmoid", FP((&PickNegLogSigmoid<const Np&, const Int&>)));
  });
}

} // namespace python
} // namespace ginn
