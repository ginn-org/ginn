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
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, AddNode>(m, name<Scalar>("AddNode"));
    m.def("Add", FP((&Add<const Np&, const Np&>)));
    m.def("Add", FP((&Add<const std::vector<Np>&>)));

    PyNode<Scalar, AddScalarNode>(m, name<Scalar>("AddScalarNode"));
    m.def("AddScalar", FP((&AddScalar<const Np&, const Real&>)));
    m.def("AddScalar", FP((&AddScalar<const Np&, const Int&>)));

    PyNode<Scalar, SubtractScalarNode>(m, name<Scalar>("SubtractScalarNode"));
    m.def("SubtractScalar", FP((&SubtractScalar<Real, Np>)));
    m.def("SubtractScalar", FP((&SubtractScalar<Int, Np>)));

    PyNode<Scalar, ProdScalarNode>(m, name<Scalar>("ProdScalarNode"));
    m.def("ProdScalar", FP((&ProdScalar<const Np&, const Real&>)));
    m.def("ProdScalar", FP((&ProdScalar<const Np&, const Int&>)));

    PyNode<Scalar, CwiseProdNode>(m, name<Scalar>("CwiseProdNode"));
    m.def("CwiseProd", FP((&CwiseProd<const Np&, const Np&>)));

    PyNode<Scalar, CwiseProdAddNode>(m, name<Scalar>("CwiseProdAddNode"));
    m.def("CwiseProdAdd",
          FP((&CwiseProdAdd<const Np&, const Np&, const Np&, const Real&>)),
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a = 0.);
    m.def("CwiseProdAdd",
          FP((&CwiseProdAdd<const Np&, const Np&, const Np&, const Int&>)),
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a);

    PyNode<Scalar, CwiseMaxNode>(m, name<Scalar>("CwiseMaxNode"));
    m.def("CwiseMax", FP((&CwiseMax<const std::vector<Np>&>)));
  });
}

} // namespace python
} // namespace ginn