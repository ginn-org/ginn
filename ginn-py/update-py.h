#ifndef GINN_PY_UPDATE_PY_H
#define GINN_PY_UPDATE_PY_H

#include <pybind11/pybind11.h>

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_update(py::module_& m);

} // namespace python
} // namespace ginn

#endif
