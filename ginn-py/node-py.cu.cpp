// Copyright 2022 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node.h>
#include <ginn/node/common.h>
#include <ginn/node/compare.h> // need for operator<(), etc
#include <ginn/node/data.h>
#include <ginn/node/prod.h> // need for operator*()

#include <ginn-py/node/affine-py.h>
#include <ginn-py/node/common-py.h>
#include <ginn-py/node/compare-py.h>
#include <ginn-py/node/layout-py.h>
#include <ginn-py/node/nonlin-py.h>
#include <ginn-py/node/pick-py.h>
#include <ginn-py/node/prod-py.h>
#include <ginn-py/node/reduce-py.h>
#include <ginn-py/node/select-py.h>
#include <ginn-py/node/weight-py.h>

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
                  m, name<Scalar>("Node"))
                  .def("forward", &BaseNode::forward)
                  .def("reset_forwarded", &BaseNode::reset_forwarded)
                  .def("reset_grad", &BaseNode::reset_grad)
                  .def_property(
                      "forwarded",
                      [](const BaseNode& n) { return n.forwarded; },
                      [](BaseNode& n, bool val) { n.forwarded = val; });
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

  m.def_property_readonly("dev", &T::dev)
      .def_property_readonly("size", py::overload_cast<>(&T::size, py::const_))
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
  m.def("real", &T::template cast<Real>);
  m.def("half", &T::template cast<Half>);
  m.def("int", &T::template cast<Int>);
  m.def("bool", &T::template cast<bool>);
}

GINN_PY_MAKE_SCALAR_DISPATCHER(Data)
GINN_PY_MAKE_FLOATING_SCALAR_DISPATCHER(Random) // TODO: Int & bool

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

  nc.def(
      "__neg__", [&](N a) { return 0. - a; }, py::is_operator());
  nc.def(
      "__lt__", [&](N a, N b) { return a < b; }, py::is_operator());
  nc.def(
      "__mul__", [&](N a, N b) { return a * b; }, py::is_operator());
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

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def(name<Scalar>("Data"),
          FP((&Data<Scalar, const DevPtr&, const Shape&>)),
          "dev"_a,
          "shape"_a);
  });

  m.def("Data", &Data_<const DevPtr&>, "device"_a, "scalar"_a = Scalar_::Real);
  m.def("Data",
        &Data_<const DevPtr&, const Shape&>,
        "device"_a,
        "shape"_a,
        "scalar"_a = Scalar_::Real);
  m.def("Data", &Data_<const Shape&>, "shape"_a, "scalar"_a = Scalar_::Real);
  m.def("Data", [&](const Matrix<Real>& m) {
    auto x = Data(cpu(), {m.rows(), m.cols()});
    x->value().m() = m;
    return x;
  });

  m.def("Random",
        &Random_<const DevPtr&, const Shape&>,
        "device"_a,
        "shape"_a,
        "scalar"_a = Scalar_::Real);
  m.def(
      "Random", &Random_<const Shape&>, "shape"_a, "scalar"_a = Scalar_::Real);

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

  bind_affine_nodes(m);
  bind_common_nodes(m);
  bind_compare_nodes(m);
  bind_layout_nodes(m);
  bind_nonlin_nodes(m);
  bind_pick_nodes(m);
  bind_prod_nodes(m);
  bind_reduce_nodes(m);
  bind_select_nodes(m);

  bind_weight_node(m);

  // add operator overloads to base node because we cannot do free functions
  op_overload_helper(Real(), rnode);
  op_overload_helper(Half(), hnode);
  op_overload_helper(Int(), inode);
}

} // namespace python
} // namespace ginn
