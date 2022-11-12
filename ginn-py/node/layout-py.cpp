#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/layout.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "layout-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

template <typename... Args>
py::object UpperTri_(Args&&... args, Scalar_ scalar) {
  if (scalar == Scalar_::Real) {
    return py::cast(UpperTri<Real>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Half) {
    return py::cast(UpperTri<Half>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Int) {
    return py::cast(UpperTri<Int>(std::forward<Args>(args)...));
  } else if (scalar == Scalar_::Bool) {
    return py::cast(UpperTri<bool>(std::forward<Args>(args)...));
  } else {
    GINN_THROW("Unexpected Scalar type!");
    return {};
  }
}

void bind_layout_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);

    PyNode<Scalar, StackNode>(m, name<Scalar>("StackNode"));
    m.def("Stack", FP((&Stack<Scalar>)));
  });

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);

    PyNode<Scalar, CatNode>(m, name<Scalar>("CatNode"));
    m.def("Cat", FP((&Cat<const std::vector<NodePtr<Scalar>>&>)));

    PyNode<Scalar, RowwiseCatNode>(m, name<Scalar>("RowwiseCatNode"));
    m.def("RowwiseCat", FP((&RowwiseCat<const std::vector<NodePtr<Scalar>>&>)));

    PyNode<Scalar, ReshapeNode>(m, name<Scalar>("ReshapeNode"));
    m.def("Reshape",
          FP((&Reshape<const NodePtr<Scalar>&,
                       const typename ReshapeNode<Scalar>::LazyShape&>)));
    m.def(
        "Reshape", FP((&Reshape<NodePtr<Scalar>&, Shape&>)), "in"_a, "shape"_a);

    PyNode<Scalar, RankViewNode>(m, name<Scalar>("RankViewNode"));
    m.def(
        "RankView", FP((&RankView<NodePtr<Scalar>&, Size&>)), "in"_a, "rank"_a);

    PyNode<Scalar, SliceNode>(m, name<Scalar>("SliceNode"));
    m.def("Slice",
          FP((&Slice<const NodePtr<Scalar>&, Shape&, Shape&>)),
          "in"_a,
          "offsets"_a,
          "sizes"_a);

    PyNode<Scalar, ChipNode>(m, name<Scalar>("ChipNode"));
    m.def("Chip",
          FP((&Chip<const NodePtr<Scalar>&, Size&, Size&>)),
          "in"_a,
          "offset"_a,
          "dim"_a);

    PyNode<Scalar, PermuteNode>(m, name<Scalar>("PermuteNode"));
    m.def("Permute",
          FP((&Permute<const NodePtr<Scalar>&, Shape&>)),
          "in"_a,
          "indices"_a);
    m.def("Transpose",
          FP((&Transpose<const NodePtr<Scalar>&>)),
          "in"_a,
          "i"_a,
          "j"_a);

    PyNode<Scalar, RowBroadcastNode>(m, name<Scalar>("RowBroadcastNode"));
    m.def("RowBroadcast",
          FP((&RowBroadcast<const NodePtr<Scalar>&, DimPtr&>)),
          "in"_a,
          "size"_a);
    m.def("RowBroadcast",
          FP((&RowBroadcast<const NodePtr<Scalar>&, Size&>)),
          "in"_a,
          "size"_a);

    PyNode<Scalar, ColBroadcastNode>(m, name<Scalar>("ColBroadcastNode"));
    m.def("ColBroadcast",
          FP((&ColBroadcast<const NodePtr<Scalar>&, DimPtr&>)),
          "in"_a,
          "size"_a);
    m.def("ColBroadcast",
          FP((&ColBroadcast<const NodePtr<Scalar>&, Size&>)),
          "in"_a,
          "size"_a);

    PyNode<Scalar, UpperTriNode>(m, name<Scalar>("UpperTriNode"));
    m.def("UpperTri",

          &UpperTri_<DevPtr&, DimPtr&>,
          "dev"_a,
          "size"_a,
          "scalar"_a = Scalar_::Real);
    m.def("UpperTri",

          &UpperTri_<DevPtr&, Size&>,
          "dev"_a,
          "size"_a,
          "scalar"_a = Scalar_::Real);
  });

  py::class_<DimNode, BaseNode, DimPtr>(m, "DimNode")
      .def_property_readonly("value", &DimNode::value);
  m.def("Dim", FP(&Dim<Size&>), "dims"_a);
  m.def("Dim", FP((&Dim<BaseNodePtr&, Size&>)), "in"_a, "dim_idx"_a);
}

} // namespace python
} // namespace ginn