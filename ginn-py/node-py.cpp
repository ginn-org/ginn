#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node.h>
#include <ginn/node/common.h>
#include <ginn/node/data.h>

#include <ginn-py/node/common-py.h>
#include <ginn-py/node/layout-py.h>
#include <ginn-py/node/nonlin-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "node-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename Scalar>
auto declare_node_of(py::module_& m) {
  using namespace pybind11::literals;

  auto node = py::class_<Node<Scalar>, BaseNode, Ptr<Node<Scalar>>>(
      m, name<Scalar>("Node"));
  py::class_<BaseDataNode<Scalar>, Node<Scalar>, Ptr<BaseDataNode<Scalar>>>(
      m, name<Scalar>("BaseDataNode"))
      .def_property(
          "has_grad",
          py::overload_cast<>(&BaseDataNode<Scalar>::has_grad, py::const_),
          py::overload_cast<bool>(&BaseDataNode<Scalar>::has_grad));
  auto data =
      py::class_<DataNode<Scalar>, BaseDataNode<Scalar>, Ptr<DataNode<Scalar>>>(
          m, name<Scalar>("DataNode"));

  return std::make_tuple(node, data);
}

template <typename Scalar, typename PyClass>
void bind_node_of(PyClass& m) {
  using namespace pybind11::literals;
  using T = Node<Scalar>;

  m.def("dev", &T::dev)
      .def("size", py::overload_cast<>(&T::size, py::const_))
      .def_property_readonly("shape",
                             py::overload_cast<>(&T::shape, py::const_))
      // TODO: Setters for value & grad
      .def_property_readonly("value", py::overload_cast<>(&T::value))
      .def_property_readonly("grad", py::overload_cast<>(&T::grad))
      .def_property_readonly("name", &T::name);
  m.def_property_readonly("scalar", [](const T&) { return scalar_<Scalar>(); });
}

template <typename Scalar, typename PyClass>
void bind_data_of(PyClass& m) {
  using namespace pybind11::literals;
  using T = DataNode<Scalar>;

  m.def("move_to", &T::move_to, "device"_a)
      .def("fill", &T::fill, "val"_a)
      .def("set_zero", &T::set_zero)
      .def("set_ones", &T::set_ones)
      .def("set_random", &T::set_random);
  m.def("cast", [](const T& t, Scalar_ scalar) -> py::object {
    switch (scalar) {
    case Scalar_::Real: return py::cast(t.template cast<Real>());
    case Scalar_::Half: return py::cast(t.template cast<Half>());
    case Scalar_::Int: return py::cast(t.template cast<Int>());
    case Scalar_::Bool: return py::cast(t.template cast<bool>());
    }
  });
}

template <typename... Args>
py::object Data_(Args&&... args, Scalar_ scalar) {
  if (scalar == Scalar_::Real) {
    return py::cast(Data<Real>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Half) {
    return py::cast(Data<Half>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Int) {
    return py::cast(Data<Int>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Bool) {
    return py::cast(Data<bool>(std::forward<Args>(args)...));
  } else {
    GINN_THROW("Unexpected Scalar type!");
    return {};
  }
}

template <typename... Args>
py::object Random_(Args&&... args, Scalar_ scalar) {
  if (scalar == Scalar_::Real) {
    return py::cast(Random<Real>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Half) {
    return py::cast(Random<Half>(std::forward<Args>(args)...));
    // TODO: Int & bool here
  } else {
    GINN_THROW("Unexpected Scalar type!");
    return {};
  }
}

template <typename Scalar, typename NodeClass>
void op_overload_helper(Scalar, NodeClass& nc) {
  using N = const NodePtr<Scalar>&;
  nc.def(
      "__add__", [&](N a, N b) { return a + b; }, py::is_operator());
  for (auto fun : {"__add__", "__radd__"}) {
    for_each<Real, Int>([&](auto s) {
      nc.def(
          fun, [&](N a, decltype(s) b) { return a + b; }, py::is_operator());
    });
  }
  for (auto fun : {"__mul__", "__rmul__"}) {
    for_each<Real, Int>([&](auto s) {
      nc.def(
          fun, [&](N a, decltype(s) b) { return a * b; }, py::is_operator());
    });
  }
  for_each<Real, Int>([&](auto s) {
    nc.def(
        "__sub__",
        [&](N a, decltype(s) b) { return a - b; },
        py::is_operator());
    nc.def(
        "__rsub__",
        [&](N a, decltype(s) b) { return b - a; },
        py::is_operator());
  });
}

void bind_node(py::module_& m) {
  using namespace pybind11::literals;

  py::class_<BaseNode, Ptr<BaseNode>>(m, "BaseNode");
  py::class_<Graph>(m, "Graph")
      .def(py::init<Ptr<BaseNode>>())
      .def("forward", &Graph::forward)
      .def("reset_grad", &Graph::reset_grad)
      .def("backward", &Graph::backward, "loss_coeff"_a);

  // making pybind know all node types first, so method docs contain the
  // appropriate python types throughout.
  auto [rnode, rdata] = declare_node_of<Real>(m);
  auto [inode, idata] = declare_node_of<Int>(m);
  auto [hnode, hdata] = declare_node_of<Half>(m);
  auto [bnode, bdata] = declare_node_of<bool>(m);

  bind_node_of<Real>(rnode);
  bind_node_of<Int>(inode);
  bind_node_of<Half>(hnode);
  bind_node_of<bool>(bnode);

  bind_data_of<Real>(rdata);
  bind_data_of<Int>(idata);
  bind_data_of<Half>(hdata);
  bind_data_of<bool>(bdata);

  // NOTE: Not sure how to instantiate factory functions with value type args
  // when perfect forwarding was used (template Arg&&). Is using references
  // safe?

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def(name<Scalar>("Data"),
          static_cast<DataPtr<Scalar> (*)(DevPtr&, Shape&)>(
              &Data<Scalar, DevPtr&, Shape&>),
          "dev"_a,
          "shape"_a);
  });

  m.def("Data", &Data_<DevPtr&>, "device"_a, "scalar"_a = Scalar_::Real);
  m.def("Data",
        &Data_<DevPtr&, Shape&>,
        "device"_a,
        "shape"_a,
        "scalar"_a = Scalar_::Real);
  m.def("Data", &Data_<Shape&>, "shape"_a, "scalar"_a = Scalar_::Real);

  m.def("Random",
        &Random_<DevPtr&, Shape&>,
        "device"_a,
        "shape"_a,
        "scalar"_a = Scalar_::Real);
  m.def("Random", &Random_<Shape&>, "shape"_a, "scalar"_a = Scalar_::Real);

  for_range<5>([&](auto arr) {
    constexpr size_t N = arr.size();
    m.def("Values",
          static_cast<DataPtr<Real> (*)(NestedInitList<N, Real>)>(
              &Values<N, Real>),
          "values"_a);
    m.def("Values",
          static_cast<DataPtr<Real> (*)(DevPtr, NestedInitList<N, Real>)>(
              &Values<N, Real>),
          "dev"_a,
          "values"_a);
  });

  bind_common_nodes(m);
  bind_layout_nodes(m);
  bind_nonlin_nodes(m);

  // add operator overloads to base node because we cannot do free functions
  op_overload_helper(Real(), rnode);
  op_overload_helper(Half(), hnode);
  op_overload_helper(Int(), inode);
}

} // namespace python
} // namespace ginn