#ifndef GINN_PY_NODE_PY_H
#define GINN_PY_NODE_PY_H

#include <string>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <ginn/node.h>
#include <ginn/node/common.h>
#include <ginn/node/data.h>

#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, ginn::Ptr<T>);

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename Scalar>
auto name(std::string name) {
  return scalar_name<Scalar>() + name;
}

template <typename Scalar>
auto declare_node_of(py::module_& m) {
  using namespace pybind11::literals;

  auto node = py::class_<Node<Scalar>, BaseNode, Ptr<Node<Scalar>>>(
      m, name<Scalar>("Node").c_str());
  py::class_<BaseDataNode<Scalar>, Node<Scalar>, Ptr<BaseDataNode<Scalar>>>(
      m, name<Scalar>("BaseDataNode").c_str())
      .def_property(
          "has_grad",
          py::overload_cast<>(&BaseDataNode<Scalar>::has_grad, py::const_),
          py::overload_cast<bool>(&BaseDataNode<Scalar>::has_grad));
  auto data =
      py::class_<DataNode<Scalar>, BaseDataNode<Scalar>, Ptr<DataNode<Scalar>>>(
          m, name<Scalar>("DataNode").c_str());

  return std::make_tuple(node, data);
}

template <typename Scalar, typename PyClass>
void bind_node_of(PyClass& m) {
  using namespace pybind11::literals;
  using T = Node<Scalar>;

  m.def("dev", &T::dev)
      .def("size", py::overload_cast<>(&T::size, py::const_))
      .def("shape", py::overload_cast<>(&T::shape, py::const_));
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
}

inline void bind_node(py::module_& m) {
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

  // Not sure how to instantiate factory functions with value type args when
  // perfect forwarding was used (template Arg&&). Is using references safe?
  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    m.def(name<Scalar>("Data").c_str(),
          py::overload_cast<DevPtr&, Shape&>(&Data<Scalar, DevPtr&, Shape&>));
  });

  for_each<Real, Half>([&](auto scalar) {
    using Scalar = decltype(scalar);
    py::class_<AddNode<Scalar>, BaseDataNode<Scalar>, Ptr<AddNode<Scalar>>>(
        m, name<Scalar>("AddNode").c_str());
    m.def("Add", &Add<NodePtr<Scalar>&, NodePtr<Scalar>&>);
  });
}

} // namespace python
} // namespace ginn

#endif
