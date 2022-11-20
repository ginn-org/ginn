#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/prod.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include "prod-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_prod_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, ProdNode>(m, name<Scalar>("ProdNode"));
    m.def("Prod", FP((&Prod<const Np&, const Np&>)));

    PyNode<Scalar, BatchedProdNode>(m, name<Scalar>("BatchedProdNode"));
    m.def("BatchedProd", FP((&BatchedProd<const Np&, const Np&>)));
  });
}

} // namespace python
} // namespace ginn
