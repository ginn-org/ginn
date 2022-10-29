#ifndef GINN_PY_TENSOR_PY_H
#define GINN_PY_TENSOR_PY_H

#include <ginn/def.h>

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <ginn/tensor.h>
#include <ginn/util/tensorio.h>

namespace ginn {
namespace python {

namespace py = pybind11;

// This is a helper type enum to dispatch things based on the scalar type
// on the Python side.
enum class Scalar_ { Real, Half, Int, Bool };

template <typename Scalar>
Scalar_ scalar_() {
  if constexpr (std::is_same_v<Scalar, Real>) { return Scalar_::Real; }
  if constexpr (std::is_same_v<Scalar, Half>) { return Scalar_::Half; }
  if constexpr (std::is_same_v<Scalar, Int>) { return Scalar_::Int; }
  if constexpr (std::is_same_v<Scalar, bool>) { return Scalar_::Bool; }
  GINN_THROW("Unexpected scalar type!");
  return {};
}

template <typename Scalar>
std::string scalar_name() {
  if constexpr (std::is_same_v<Scalar, Real>) { return "Real"; }
  if constexpr (std::is_same_v<Scalar, Half>) { return "Half"; }
  if constexpr (std::is_same_v<Scalar, Int>) { return "Int"; }
  if constexpr (std::is_same_v<Scalar, bool>) { return "Bool"; }
  return "";
}

template <typename Scalar>
std::string tensor_name() {
  return scalar_name<Scalar>() + "Tensor";
}

template <typename Scalar>
auto declare_tensor_of(py::module_& m) {
  return py::class_<Tensor<Scalar>>(m, tensor_name<Scalar>().c_str());
}

template <typename Scalar, typename PyClass>
void bind_tensor_of(PyClass& m) {
  using namespace pybind11::literals;
  using T = Tensor<Scalar>;

  m.def(py::init<>());
  m.def(py::init<DevPtr>(), "device"_a);
  m.def(py::init<Shape>(), "shape"_a);
  m.def(py::init<Shape, const std::vector<Scalar>&>(), "shape"_a, "val"_a);
  m.def(py::init<DevPtr, Shape>(), "device"_a, "shape"_a);
  m.def(py::init<DevPtr, Shape, std::vector<Scalar>>(),
        "device"_a,
        "shape"_a,
        "val"_a);
  m.def("dev", &T::dev);
  m.def("shape", &T::shape);
  m.def("size", py::overload_cast<>(&T::size, py::const_));
  m.def("list", &T::vector);
  m.def("v", static_cast<VectorMap<Scalar> (T::*)()>(&T::v));
  m.def("m", static_cast<MatrixMap<Scalar> (T::*)()>(&T::m));
  m.def("real", &T::template cast<Real>);
  m.def("half", &T::template cast<Half>);
  m.def("int", &T::template cast<Int>);
  m.def("bool", &T::template cast<bool>);
  m.def("set_zero", &T::set_zero);
  m.def("set_ones", &T::set_ones);
  m.def("set_random", &T::set_random);
  m.def("set", &T::template set<0>);
  m.def("set", &T::template set<1>);
  m.def("set", &T::template set<2>);
  m.def("set", &T::template set<3>);
  m.def("set", &T::template set<4>);
  m.def("resize", &T::resize);
  m.def("copy_to", &T::copy_to, "device"_a);
  m.def("move_to", &T::move_to, "device"_a);
  m.def("maybe_copy_to", &T::maybe_copy_to, "device"_a);
  m.def_property_readonly("scalar", [](const T&) { return scalar_<Scalar>(); });
  m.def("__repr__", [&](const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
  });
  m.def(
      "__eq__",
      [](const T& a, const T& b) { return a == b; },
      py::is_operator());
}

template <typename... Args>
py::object Tensor_(Args&&... args, Scalar_ scalar) {
  if (scalar == Scalar_::Real) {
    return py::cast(Tensor<Real>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Half) {
    return py::cast(Tensor<Half>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Int) {
    return py::cast(Tensor<Int>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Bool) {
    return py::cast(Tensor<bool>(std::forward<Args>(args)...));
  } else {
    GINN_THROW("Unexpected Scalar type!");
    return {};
  }
}

inline void bind_tensor(py::module_& m) {
  using namespace py::literals;

  py::enum_<Scalar_>(m, "Scalar")
      .value("Real", Scalar_::Real)
      .value("Half", Scalar_::Half)
      .value("Int", Scalar_::Int)
      .value("Bool", Scalar_::Bool);

  // making pybind know all tensor types first, so method docs contain the
  // appropriate python types throughout.
  auto mr = declare_tensor_of<Real>(m);
  auto mi = declare_tensor_of<Int>(m);
  auto mh = declare_tensor_of<Half>(m);
  auto mb = declare_tensor_of<bool>(m);

  bind_tensor_of<Real>(mr);
  bind_tensor_of<Int>(mi);
  bind_tensor_of<Half>(mh);
  bind_tensor_of<bool>(mb);

  // NOTE: Not sure how to instantiate factory functions with value type args
  // when perfect forwarding was used (template Arg&&). Is using references
  // safe?

  // Why is &Tensor_<> an unknown type? -- might be a nvcc11.1 thing.
  m.def("Tensor",
        static_cast<py::object (*)(Scalar_)>(&Tensor_<>),
        "scalar"_a = Scalar_::Real);
  m.def("Tensor", &Tensor_<DevPtr&>, "device"_a, "scalar"_a = Scalar_::Real);
  m.def("Tensor",
        &Tensor_<DevPtr&, Shape&>,
        "device"_a,
        "shape"_a,
        "scalar"_a = Scalar_::Real);
  m.def("Tensor", &Tensor_<Shape&>, "shape"_a, "scalar"_a = Scalar_::Real);
}

} // namespace python
} // namespace ginn

#endif
