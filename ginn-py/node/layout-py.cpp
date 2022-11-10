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

template <typename Scalar, template <class> typename Node>
using PyNode =
    py::class_<Node<Scalar>, BaseDataNode<Scalar>, Ptr<Node<Scalar>>>;

void bind_layout_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);

    // py::class_<StackNode<Scalar>, BaseDataNode<Scalar>,
    // Ptr<StackNode<Scalar>>>(
    //     m, name<Scalar>("StackNode").c_str());
    PyNode<Scalar, StackNode>(m, name<Scalar>("StackNode").c_str());
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def(
        "Stack",
        static_cast<Ptr<StackNode<Scalar>> (*)(
            const std::vector<std::vector<NodePtr<Scalar>>>&)>(&Stack<Scalar>));
  });

  for_each<Real, Half, Int, bool>([&](auto scalar) {
    using Scalar = decltype(scalar);

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
              const NodePtr<Scalar>&,
              const typename ReshapeNode<Scalar>::LazyShape&)>(
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

    py::class_<SliceNode<Scalar>, BaseDataNode<Scalar>, Ptr<SliceNode<Scalar>>>(
        m, name<Scalar>("SliceNode").c_str());
    m.def("Slice",
          static_cast<Ptr<SliceNode<Scalar>> (*)(
              const NodePtr<Scalar>&, Shape&, Shape&)>(
              &Slice<const NodePtr<Scalar>&, Shape&, Shape&>),
          "in"_a,
          "offsets"_a,
          "sizes"_a);

    py::class_<ChipNode<Scalar>, BaseDataNode<Scalar>, Ptr<ChipNode<Scalar>>>(
        m, name<Scalar>("ChipNode").c_str());
    m.def("Chip",
          static_cast<Ptr<ChipNode<Scalar>> (*)(
              const NodePtr<Scalar>&, Size&, Size&)>(
              &Chip<const NodePtr<Scalar>&, Size&, Size&>),
          "in"_a,
          "offset"_a,
          "dim"_a);

    py::class_<PermuteNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<PermuteNode<Scalar>>>(m,
                                         name<Scalar>("PermuteNode").c_str());
    m.def("Permute",
          static_cast<Ptr<PermuteNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                   Shape&)>(
              &Permute<const NodePtr<Scalar>&, Shape&>),
          "in"_a,
          "indices"_a);
    m.def("Transpose",
          static_cast<Ptr<PermuteNode<Scalar>> (*)(
              const NodePtr<Scalar>&, Size, Size)>(
              &Transpose<const NodePtr<Scalar>&>),
          "in"_a,
          "i"_a,
          "j"_a);

    py::class_<RowBroadcastNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<RowBroadcastNode<Scalar>>>(
        m, name<Scalar>("RowBroadcastNode").c_str());
    m.def("RowBroadcast",
          static_cast<Ptr<RowBroadcastNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                        DimPtr&)>(
              &RowBroadcast<const NodePtr<Scalar>&, DimPtr&>),
          "in"_a,
          "size"_a);
    m.def("RowBroadcast",
          static_cast<Ptr<RowBroadcastNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                        Size&)>(
              &RowBroadcast<const NodePtr<Scalar>&, Size&>),
          "in"_a,
          "size"_a);

    py::class_<ColBroadcastNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<ColBroadcastNode<Scalar>>>(
        m, name<Scalar>("ColBroadcastNode").c_str());
    m.def("ColBroadcast",
          static_cast<Ptr<ColBroadcastNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                        DimPtr&)>(
              &ColBroadcast<const NodePtr<Scalar>&, DimPtr&>),
          "in"_a,
          "size"_a);
    m.def("ColBroadcast",
          static_cast<Ptr<ColBroadcastNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                        Size&)>(
              &ColBroadcast<const NodePtr<Scalar>&, Size&>),
          "in"_a,
          "size"_a);

    py::class_<UpperTriNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<UpperTriNode<Scalar>>>(m,
                                          name<Scalar>("UpperTriNode").c_str());
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
  m.def("Dim", static_cast<DimPtr (*)(Size&)>(&Dim<Size&>), "dims"_a);
  m.def("Dim",
        static_cast<DimPtr (*)(BaseNodePtr&, Size&)>(&Dim<BaseNodePtr&, Size&>),
        "in"_a,
        "dim_idx"_a);
}

} // namespace python
} // namespace ginn