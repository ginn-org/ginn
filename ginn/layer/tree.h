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

#ifndef GINN_LAYER_TREE_H
#define GINN_LAYER_TREE_H

#include <ginn/layer/layer.h>
#include <ginn/model/treelstm.h>
#include <ginn/util/tree.h>

namespace ginn {

template <typename Scalar = Real>
using NodeTree = tree::Tree<NodePtr<Scalar>>;

template <typename Scalar = Real>
class TreeLstmLayerNode : public LayerNode<NodeTree<Scalar>(NodeTree<Scalar>)> {
 public:
  using Model = TreeLstm<Scalar>;
  using State =
      typename Model::State; // h & c for treelstm as a pair of NodePtrs
  using Reduce = typename Model::Reduce;

  // TODO: maybe remove namespace ginn::tree and put Tree under ginn?
  template <typename T>
  using Tree = tree::Tree<T>;

 private:
  Real drop_p_;
  Model model_;

 public:
  TreeLstmLayerNode() = default;
  TreeLstmLayerNode(Device& dev,
                    Size labels,
                    Size dim,
                    Size xdim,
                    Real drop_p = 0.,
                    Reduce reduce = Reduce::add) {
    init(dev, labels, dim, xdim, drop_p, reduce);
  }

  void init(Device& dev,
            Size labels,
            Size dim,
            Size xdim,
            Real drop_p = 0.,
            Reduce reduce = Reduce::add) {
    drop_p_ = drop_p;
    model_.init(dev, labels, dim, xdim, reduce);
  }

  std::vector<BaseNodePtr> weights_() override {
    return base_cast(model_.weights());
  }

  NodeTree<Scalar> run(const NodeTree<Scalar>& x) override {
    auto hc = clone_empty<State>(x); // create empty node trees
    auto h = clone_empty<NodePtr<Real>>(x);

    for (size_t i = 0; i < x.size(); i++) {
      std::vector<typename Model::Child> children;
      for (size_t j = 0; j < hc.children_at(i).size(); j++) {
        auto& n = hc.children_at(i)[j];
        auto& [h, c] = n->data;
        children.emplace_back(j, h, c);
      }
      hc[i] = model_.step(x[i], children);
      h[i] = hc[i].first;
    }
    return h;
  }

  LayerPtr<NodeTree<Scalar>(NodeTree<Scalar>)> copy(Copy mode) override {
    auto rval = std::make_shared<TreeLstmLayerNode>();
    rval->drop_p_ = drop_p_;
    rval->model_ = model_.copy(mode);
    return rval;
  }
};

GINN_MAKE_TEMPLATE_LAYER_FACTORY(TreeLstmLayer);

} // namespace ginn

#endif
