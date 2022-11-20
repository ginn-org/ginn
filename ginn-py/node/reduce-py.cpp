#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/reduce.h>

#include <ginn-py/node-py.h>
#include <ginn-py/util-py.h>

#include "reduce-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_reduce_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);
    using Np = NodePtr<Scalar>;

    PyNode<Scalar, SumNode>(m, name<Scalar>("SumNode"));
    m.def("Sum", FP(&Sum<const Np&>));

    PyNode<Scalar, AxisSumNode>(m, name<Scalar>("AxisSumNode"));
    m.def("AxisSum", FP((&AxisSum<const Np&, const Shape&>)));

    PyNode<Scalar, MeanNode>(m, name<Scalar>("MeanNode"));
    m.def("Mean", FP(&Mean<const Np&>));

    PyNode<Scalar, VarianceNode>(m, name<Scalar>("VarianceNode"));
    m.def("Variance", FP(&Variance<const Np&>));
  });
}

} // namespace python
} // namespace ginn
