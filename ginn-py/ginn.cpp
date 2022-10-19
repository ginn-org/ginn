#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/util/util.h>

#include <ginn-py/dev-py.h>
#include <ginn-py/tensor-py.h>

namespace py = pybind11;

void dummy(py::type x) { std::cout << std::string(py::str(x)); }

PYBIND11_MODULE(ginn, m) {
  using namespace ginn;
  using namespace ginn::python;

  m.doc() = "pybind11 example plugin"; // optional module docstring

  py::module_ m2 = m.def_submodule("util", "A submodule of 'example'");
  m2.def("iota", &iota<int>, "A function");
  m2.def("dummy", &dummy, "A function");

  bind_dev(m);
  bind_tensor(m);
}
