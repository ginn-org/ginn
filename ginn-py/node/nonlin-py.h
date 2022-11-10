#ifndef GINN_PY_NONLIN_PY_H
#define GINN_PY_NONLIN_PY_H

#include <pybind11/pybind11.h>

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_nonlin_nodes(py::module_& m);

} // namespace python
} // namespace ginn
#endif
