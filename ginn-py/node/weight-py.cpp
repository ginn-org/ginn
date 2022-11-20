#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/weight.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "weight-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename Scalar>
auto declare_weight_of(py::module_& m) {
  return py::class_<WeightNode<Scalar>, Node<Scalar>, Ptr<WeightNode<Scalar>>>(
      m, name<Scalar>("WeightNode"));
}

template <typename Scalar, typename PyClass>
void bind_weight_of(PyClass& m) {
  using namespace py::literals;
  using T = WeightNode<Scalar>;

  m.def_property_readonly("id", &T::id)
      .def("move_to", &T::move_to, "device"_a)
      .def("tie", &T::tie, "weight"_a)
      .def("copy", &T::copy, "mode"_a)
      .def("fill", &T::template fill<Real>, "val"_a)
      .def("fill", &T::template fill<Int>, "val"_a)
      .def("set_zero", &T::set_zero)
      .def("set_ones", &T::set_ones)
      .def("set_random", &T::set_random);
}

template <typename... Args>
py::object Weight_(Args&&... args, Scalar_ scalar) {
  if (scalar == Scalar_::Real) {
    return py::cast(Weight<Real>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Half) {
    return py::cast(Weight<Half>(std::forward<Args>(args)...));
  } else {
    GINN_THROW("Unexpected Scalar type!");
    return {};
  }
}

void bind_weight_node(py::module_& m) {
  using namespace py::literals;

  py::enum_<Copy>(m, "Copy")
      .value("Tied", Copy::Tied)
      .value("Deep", Copy::Deep);

  auto rw = declare_weight_of<Real>(m);
  auto hw = declare_weight_of<Half>(m);

  bind_weight_of<Real>(rw);
  bind_weight_of<Half>(hw);

  m.def("Weight",
        &Weight_<const DevPtr&, const Shape&>,
        "device"_a = cpu(),
        "shape"_a = Shape{0},
        "scalar"_a = Scalar_::Real);
}

} // namespace python
} // namespace ginn
