#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ginn/node/common.h>

#include <ginn-py/node-py.h>
#include <ginn-py/tensor-py.h>
#include <ginn-py/util-py.h>

#include "common-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_common_nodes(py::module_& m) {
  using namespace py::literals;

  for_each<Real, Half, Int>([&](auto scalar) {
    using Scalar = decltype(scalar);

    py::class_<AddNode<Scalar>, BaseDataNode<Scalar>, Ptr<AddNode<Scalar>>>(
        m, name<Scalar>("AddNode").c_str());
    // nvcc 11.1 forces me to use an explicit static cast here.
    m.def("Add",
          static_cast<Ptr<AddNode<Scalar>> (*)(NodePtr<Scalar>&,
                                               NodePtr<Scalar>&)>(
              &Add<NodePtr<Scalar>&, NodePtr<Scalar>&>));
    m.def("Add",
          static_cast<Ptr<AddNode<Scalar>> (*)(
              const std::vector<NodePtr<Scalar>>&)>(
              &Add<const std::vector<NodePtr<Scalar>>&>));

    py::class_<AddScalarNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<AddScalarNode<Scalar>>>(
        m, name<Scalar>("AddScalarNode").c_str());
    m.def("AddScalar",
          static_cast<Ptr<AddScalarNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                     Real&)>(
              &AddScalar<const NodePtr<Scalar>&, Real&>));
    m.def("AddScalar",
          static_cast<Ptr<AddScalarNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                     Int&)>(
              &AddScalar<const NodePtr<Scalar>&, Int&>));

    py::class_<SubtractScalarNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<SubtractScalarNode<Scalar>>>(
        m, name<Scalar>("SubtractScalarNode").c_str());
    m.def(
        "SubtractScalar",
        static_cast<Ptr<SubtractScalarNode<Scalar>> (*)(Real, NodePtr<Scalar>)>(
            &SubtractScalar<Real, NodePtr<Scalar>>));
    m.def(
        "SubtractScalar",
        static_cast<Ptr<SubtractScalarNode<Scalar>> (*)(Int, NodePtr<Scalar>)>(
            &SubtractScalar<Int, NodePtr<Scalar>>));

    py::class_<ProdScalarNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<ProdScalarNode<Scalar>>>(
        m, name<Scalar>("ProdScalarNode").c_str());
    m.def("ProdScalar",
          static_cast<Ptr<ProdScalarNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                      Real&)>(
              &ProdScalar<const NodePtr<Scalar>&, Real&>));
    m.def("ProdScalar",
          static_cast<Ptr<ProdScalarNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                      Int&)>(
              &ProdScalar<const NodePtr<Scalar>&, Int&>));

    py::class_<CwiseProdNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<CwiseProdNode<Scalar>>>(
        m, name<Scalar>("CwiseProdNode").c_str());
    m.def("CwiseProd",
          static_cast<Ptr<CwiseProdNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                     const NodePtr<Scalar>&)>(
              &CwiseProd<const NodePtr<Scalar>&, const NodePtr<Scalar>&>));

    py::class_<CwiseProdAddNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<CwiseProdAddNode<Scalar>>>(
        m, name<Scalar>("CwiseProdAddNode").c_str());
    m.def("CwiseProdAdd",
          static_cast<Ptr<CwiseProdAddNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                        const NodePtr<Scalar>&,
                                                        const NodePtr<Scalar>&,
                                                        Real&)>(
              &CwiseProdAdd<const NodePtr<Scalar>&,
                            const NodePtr<Scalar>&,
                            const NodePtr<Scalar>&,
                            Real&>),
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a = 0.);
    m.def("CwiseProdAdd",
          static_cast<Ptr<CwiseProdAddNode<Scalar>> (*)(const NodePtr<Scalar>&,
                                                        const NodePtr<Scalar>&,
                                                        const NodePtr<Scalar>&,
                                                        Int&)>(
              &CwiseProdAdd<const NodePtr<Scalar>&,
                            const NodePtr<Scalar>&,
                            const NodePtr<Scalar>&,
                            Int&>),
          "in"_a,
          "multiplicand"_a,
          "addend"_a,
          "multiplicand_bias"_a);

    py::class_<CwiseMaxNode<Scalar>,
               BaseDataNode<Scalar>,
               Ptr<CwiseMaxNode<Scalar>>>(m,
                                          name<Scalar>("CwiseMaxNode").c_str());
    m.def("CwiseMax",
          static_cast<Ptr<CwiseMaxNode<Scalar>> (*)(
              const std::vector<NodePtr<Scalar>>&)>(
              &CwiseMax<const std::vector<NodePtr<Scalar>>&>));
  });
}

} // namespace python
} // namespace ginn