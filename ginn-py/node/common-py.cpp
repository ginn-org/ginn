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

template <typename PyClass, typename Rval, typename... Args>
void factory(PyClass& m, const char* name, Rval (*fp)(Args...)) {
  m.def(name, fp);
}

void bind_common_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, AddNode>(m, name<Scalar>("AddNode"));

    // nvcc 11.1 forces me to use an explicit static cast here.
    // auto fp = &Add<Np&, Np&>;
    // m.def("Add", (fp));
    m.def("Add", FP((&Add<Np&, Np&>)));
    m.def("Add", FP((&Add<const std::vector<Np>&>)));

    PyNode<Scalar, AddScalarNode>(m, name<Scalar>("AddScalarNode"));
    m.def("AddScalar", FP((&AddScalar<const Np&, Real&>)));
    m.def("AddScalar", FP((&AddScalar<const Np&, Int&>)));

    PyNode<Scalar, SubtractScalarNode>(m, name<Scalar>("SubtractScalarNode"));
    m.def("SubtractScalar", FP((&SubtractScalar<Real, Np>)));
    m.def("SubtractScalar", FP((&SubtractScalar<Int, Np>)));

    PyNode<Scalar, ProdScalarNode>(m, name<Scalar>("ProdScalarNode"));
    m.def("ProdScalar", FP((&ProdScalar<const Np&, Real&>)));
    m.def("ProdScalar", FP((&ProdScalar<const Np&, Int&>)));

    PyNode<Scalar, CwiseProdNode>(m, name<Scalar>("CwiseProdNode"));
    m.def("CwiseProd", FP((&CwiseProd<const Np&, const Np&>)));

    PyNode<Scalar, CwiseProdAddNode>(m, name<Scalar>("CwiseProdAddNode"));
    m.def("CwiseProdAdd",
          FP((&CwiseProdAdd<const Np&, const Np&, const Np&, Real&>)),
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a = 0.);
    m.def("CwiseProdAdd",
          FP((&CwiseProdAdd<const Np&, const Np&, const Np&, Int&>)),
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