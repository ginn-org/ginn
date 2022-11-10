#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/nlnode.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "nonlin-py.h"

#define GINN_PY_MAKE_UNARY_NODE_AND_FACTORY(F)                                 \
  py::class_<F##Node<Scalar>, BaseDataNode<Scalar>, Ptr<F##Node<Scalar>>>(     \
      m, name<Scalar>(#F "Node").c_str());                                     \
  m.def(#F,                                                                    \
        static_cast<Ptr<F##Node<Scalar>> (*)(const NodePtr<Scalar>&)>(         \
            &F<const NodePtr<Scalar>&>));

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