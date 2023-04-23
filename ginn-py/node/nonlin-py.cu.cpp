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

#include <ginn/node/nlnode.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include <ginn-py/node/nonlin-py.h>

#define GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(F)                                 \
  py::class_<F##Node<Scalar>, BaseDataNode<Scalar>, Ptr<F##Node<Scalar>>>(     \
      m, name<Scalar>(#F "Node"));                                             \
  m.def(#F,                                                                    \
        py::overload_cast<const NodePtr<Scalar>&>(         \
            &F<const NodePtr<Scalar>&>))

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_nonlin_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half>([&](auto scalar) {
    using Scalar = decltype(scalar);

    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Identity);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Tanh);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Sigmoid);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Relu);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Softmax);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Log);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Sqrt);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Gelu);
    GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(Gelu2);
  });
}

} // namespace python
} // namespace ginn
