#ifndef GINN_PY_WEIGHT_PY_H
#define GINN_PY_WEIGHT_PY_H

#include <pybind11/pybind11.h>

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_weight_node(py::module_& m);

} // namespace python
} // namespace ginn

#endif
