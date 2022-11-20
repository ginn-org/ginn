#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/select.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include "select-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_select_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, SelectNode>(m, name<Scalar>("SelectNode"));
    m.def("Select", FP((&Select<const Np&, const Np&>)));
    m.def("Select", FP((&Select<const Np&, const Scalar&>)));
    m.def("Select", FP((&Select<const Scalar&, const Np&>)));
    m.def("Select", FP((&Select<const Scalar&, const Scalar&>)));

    PyNode<Scalar, MaskNode>(m, name<Scalar>("MaskNode"));
    m.def("Mask", FP((&Mask<const Np&, const Np&, const Real&>)));
    m.def("Mask", FP((&Mask<const Np&, const Np&, const Int&>)));
  });
}

} // namespace python
} // namespace ginn