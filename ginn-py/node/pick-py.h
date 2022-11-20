#ifndef GINN_PY_PICK_PY_H
#define GINN_PY_PICK_PY_H

#include <pybind11/pybind11.h>

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_pick_nodes(py::module_& m);

} // namespace python
} // namespace ginn

#endif
