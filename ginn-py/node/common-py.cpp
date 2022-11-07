#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/common.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "common-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_common_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);

    py::class_<AddNode<Scalar>, BaseDataNode<Scalar>, Ptr<AddNode<Scalar>>>(
        m, name<Scalar>("AddNode").c_str());
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def("Add",
          static_cast<Ptr<AddNode<Scalar>> (*)(NodePtr<Scalar>&,
                                               NodePtr<Scalar>&)>(
              &Add<NodePtr<Scalar>&, NodePtr<Scalar>&>));
    m.def("Add",
          static_cast<Ptr<AddNode<Scalar>> (*)(
              const std::vector<NodePtr<Scalar>>&)>(
              &Add<const std::vector<NodePtr<Scalar>>&>));

    py::class_<AddScalarNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<AddScalarNode<Scalar>>>(
        m, name<Scalar>("AddScalarNode").c_str());
    m.def("AddScalar",
          static_cast<Ptr<AddScalarNode<Scalar>> (*)(NodePtr<Scalar>&, Real&)>(
              &AddScalar<NodePtr<Scalar>&, Real&>));
    m.def("AddScalar",
          static_cast<Ptr<AddScalarNode<Scalar>> (*)(NodePtr<Scalar>&, Int&)>(
              &AddScalar<NodePtr<Scalar>&, Int&>));
  });
}

} // namespace python
} // namespace ginn