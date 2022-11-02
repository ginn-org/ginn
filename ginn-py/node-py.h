#ifndef GINN_PY_NODE_PY_H
#define GINN_PY_NODE_PY_H

#include <pybind11/pybind11.h>

#include <ginn/node.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, ginn::Ptr<T>);

namespace ginn {
namespace python {

void bind_node(py::module_& m);

} // namespace python
} // namespace ginn

#endif