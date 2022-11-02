#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node.h>
#include <ginn/node/common.h>
#include <ginn/node/data.h>
#include <ginn/node/layout.h>

#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "node-py.h"

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
      .def("shape", py::overload_cast<>(&T::shape, py::const_))
      .def("value", py::overload_cast<>(&T::value))
      .def("grad", py::overload_cast<>(&T::grad))
      .def("name", &T::name);
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
    m.def(name<Scalar>("Data").c_str(),
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

  m.def("Values",
        static_cast<DataPtr<Real> (*)(NestedInitList<0, Real>)>(
            &Values<0, Real>));
  m.def("Values",
        static_cast<DataPtr<Real> (*)(NestedInitList<1, Real>)>(
            &Values<1, Real>));
  m.def("Values",
        static_cast<DataPtr<Real> (*)(NestedInitList<2, Real>)>(
            &Values<2, Real>));
  m.def("Values",
        static_cast<DataPtr<Real> (*)(NestedInitList<3, Real>)>(
            &Values<3, Real>));
  m.def("Values",
        static_cast<DataPtr<Real> (*)(NestedInitList<4, Real>)>(
            &Values<4, Real>));

  for_each<Real, Half>([&](auto scalar) {
    using Scalar = decltype(scalar);

    py::class_<AddNode<Scalar>, BaseDataNode<Scalar>, Ptr<AddNode<Scalar>>>(
        m, name<Scalar>("AddNode").c_str());
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def("Add",
          static_cast<Ptr<AddNode<Scalar>> (*)(NodePtr<Scalar>&,
                                               NodePtr<Scalar>&)>(
              &Add<NodePtr<Scalar>&, NodePtr<Scalar>&>));
  });

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);

    py::class_<StackNode<Scalar>, BaseDataNode<Scalar>, Ptr<StackNode<Scalar>>>(
        m, name<Scalar>("StackNode").c_str());
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def(
        "Stack",
        static_cast<Ptr<StackNode<Scalar>> (*)(
            const std::vector<std::vector<NodePtr<Scalar>>>&)>(&Stack<Scalar>));

    py::class_<CatNode<Scalar>, BaseDataNode<Scalar>, Ptr<CatNode<Scalar>>>(
        m, name<Scalar>("CatNode").c_str());
    m.def("Cat",
          static_cast<Ptr<CatNode<Scalar>> (*)(
              const std::vector<NodePtr<Scalar>>&)>(
              &Cat<const std::vector<NodePtr<Scalar>>&>));

    py::class_<RowwiseCatNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<RowwiseCatNode<Scalar>>>(
        m, name<Scalar>("RowwiseCatNode").c_str());
    m.def("RowwiseCat",
          static_cast<Ptr<RowwiseCatNode<Scalar>> (*)(
              const std::vector<NodePtr<Scalar>>&)>(
              &RowwiseCat<const std::vector<NodePtr<Scalar>>&>));

    py::class_<ReshapeNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<ReshapeNode<Scalar>>>(m,
                                         name<Scalar>("ReshapeNode").c_str());
    m.def("Reshape",
          static_cast<Ptr<ReshapeNode<Scalar>> (*)(
              const NodePtr<Scalar>&, const typename ReshapeNode<Scalar>::LazyShape&)>(
              &Reshape<const NodePtr<Scalar>&,
                       const typename ReshapeNode<Scalar>::LazyShape&>));
    m.def("Reshape",
          static_cast<Ptr<ReshapeNode<Scalar>> (*)(NodePtr<Scalar>&, Shape&)>(
              &Reshape<NodePtr<Scalar>&, Shape&>),
          "in"_a,
          "shape"_a);

    py::class_<RankViewNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<RankViewNode<Scalar>>>(m,
                                          name<Scalar>("RankViewNode").c_str());
    m.def("RankView",
          static_cast<Ptr<RankViewNode<Scalar>> (*)(NodePtr<Scalar>&, Size&)>(
              &RankView<NodePtr<Scalar>&, Size&>),
          "in"_a,
          "rank"_a);
  });

  py::class_<DimNode, BaseNode, DimPtr>(m, "DimNode")
      .def("value", &DimNode::value);
  m.def("Dim", static_cast<DimPtr (*)(Size&)>(&Dim<Size&>), "dims"_a);
  m.def("Dim",
        static_cast<DimPtr (*)(BaseNodePtr&, Size&)>(&Dim<BaseNodePtr&, Size&>),
        "in"_a,
        "dim_idx"_a);
}

} // namespace python
} // namespace ginn