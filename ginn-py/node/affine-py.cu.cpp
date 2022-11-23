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

#include <ginn/node/affine.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include "affine-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_affine_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, AffineNode>(m, name<Scalar>("AffineNode"));
    m.def("Affine",
          static_cast<Ptr<AffineNode<Scalar>> (*)(const std::vector<Np>&)>(
              &Affine<Scalar>));
  });
}

} // namespace python
} // namespace ginn