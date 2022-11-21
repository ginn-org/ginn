#include <pybind11/pybind11.h>

#include <ginn-py/dev-py.h>
#include <ginn-py/init-py.h>
#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/update-py.h>

namespace py = pybind11;

PYBIND11_MODULE(ginn, m) {
  using namespace ginn;
  using namespace ginn::python;

  m.doc() = "pybind11 example plugin"; // optional module docstring

  bind_dev(m);
  bind_tensor(m);
  bind_node(m);
  bind_init(m);
  bind_update(m);
}
