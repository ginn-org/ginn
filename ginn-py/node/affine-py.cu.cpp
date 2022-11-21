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
