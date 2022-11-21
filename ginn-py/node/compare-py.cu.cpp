#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/compare.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include "compare-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_compare_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    py::class_<LessThanNode<Scalar>,
               BaseDataNode<bool>,
               Ptr<LessThanNode<Scalar>>>(m, name<Scalar>("LessThanNode"));
    m.def("LessThan", FP((&LessThan<const Np&, const Np&>)));
  });
}

} // namespace python
} // namespace ginn
