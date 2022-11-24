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

#ifndef GINN_AUTOBATCH_H
#define GINN_AUTOBATCH_H

#include <iostream>
#include <stack>
#include <unordered_map>

#include <ginn/node.h>
#include <ginn/node/affine.h>
#include <ginn/node/common.h>
#include <ginn/node/layout.h>
#include <ginn/node/prod.h>

#include <ginn/util/timer.h>

// TODO: Now that Affine contains nonlin ops too, make sure each Affine
// occurence is replaced by a composition of Affine+Nonlin as separate nodes
// before Affines are transformed into sums of products.

// TODO: As I switch to Scalar as a template parameter, for now I'm supporting
// autobatching only for Real nodes. Revisit to handle mixed / other scalar
// Affine and Prod types.

// NOTE: Don't forget to check affine doesn't have nonlin (nonlin == id)

namespace ginn {

class ExtendedGraph {
 private:
  template <typename Key, typename Value>
  using Map = std::unordered_map<Key, Value>;

  BaseNodePtr sink_;
  std::vector<BaseNodePtr> list_;
  std::vector<ProdNode<Real>*> prods_;
  std::vector<std::pair<AffineNode<Real>*, size_t>> affines_;
  Map<BaseNode*, std::vector<std::pair<BaseNodePtr, size_t>>> outs_;
  Map<const BaseNode*, size_t> heights_;

  Map<BaseNode*, Map<size_t, std::vector<ProdNode<Real>*>>> groups_;
  // groups_[lhs][height] = {Prod nodes that depend on lhs at given height}

  void compute_outs() {
    outs_.clear();
    for (size_t list_idx = 0; list_idx < list_.size(); list_idx++) {
      auto& node = list_[list_idx];
      auto& ins = node->ins();
      for (size_t i = 0; i < ins.size(); i++) {
        auto in = ins[i].get();
        if (dynamic_cast<AffineNode<Real>*>(in) or
            dynamic_cast<ProdNode<Real>*>(in)) {
          outs_[in].emplace_back(node, i);
        }
      }

      if (auto affine = dynamic_cast<AffineNode<Real>*>(node.get())) {
        affines_.emplace_back(affine, list_idx);
      } else if (auto prod = dynamic_cast<ProdNode<Real>*>(node.get())) {
        prods_.emplace_back(prod);
      }
    }
  }

  // Replace Affines with sum of products. E.g. Affine(W, x, V, h, b)
  // turns into Add({Prod(W, x), Prod(V, h), b})
  void replace_affines() {
    std::vector<NodePtr<Real>> add_ins;

    for (auto& [affine, list_idx] : affines_) {
      const auto& ins = affine->ins();
      // auto bias = Reshape(ins.back(), {ins.back()->value().size(), 1});
      auto& bias = ins.back();
      add_ins.reserve(ins.size());
      for (size_t i = 0; (i + 1) < ins.size(); i += 2) {
        auto a = dynamic_ptr_cast<Node<Real>>(ins[i]);
        auto b = dynamic_ptr_cast<Node<Real>>(ins[i + 1]);
        GINN_ASSERT(a);
        GINN_ASSERT(b);
        auto prod = a * b;
        add_ins.push_back(prod);
        prods_.push_back(prod.get());
      }
      auto dim = Dim(ins.at(1)->cols());
      auto biasx = ColBroadcast(dynamic_ptr_cast<Node<Real>>(bias), dim);
      add_ins.push_back(biasx);
      auto add = Add(add_ins);
      for (size_t i = 0; i < add_ins.size(); i++) {
        outs_[add_ins[i].get()].emplace_back(add, i);
      }
      for (auto& [out_node, idx] : outs_[(BaseNode*)affine]) {
        auto ins = out_node->ins();
        ins[idx] = add;
        out_node->set_ins(ins);
      }
      outs_.erase(affine);
      if (affine == sink_.get()) { sink_ = add; }
      add_ins.clear();
    }
  }

  void height_(const BaseNode* sink) {
    std::stack<const BaseNode*> s;
    s.push(sink);

    while (not s.empty()) {
      auto n = s.top();
      const auto& ins = n->ins();
      size_t max = 0;
      bool ins_done = true;
      for (auto& in : ins) {
        if (heights_.find(&*in) != heights_.end()) {
          max = std::max(max, heights_[&*in]);
        } else {
          s.push(&*in);
          ins_done = false;
        }
      }
      if (ins_done) {
        heights_[n] = max + 1;
        s.pop();
      }
    }
  }

  void group_by_height() {
    heights_.clear();
    groups_.clear();
    height_(sink_.get());
    for (auto prod : prods_) {
      auto& left = prod->ins()[0];
      groups_[left.get()][heights_.at(prod)].push_back(prod);
    }
  }

  void merge_prods() {
    std::vector<NodePtr<Real>> cat_ins;
    std::vector<BaseNodePtr> uncats;
    for (auto& p : groups_) {
      for (auto& q : p.second) {
        auto& prods = q.second;
        if (prods.size() == 1) { continue; } // only 1 Prod with that height

        for (auto prod : prods) {
          cat_ins.push_back(dynamic_ptr_cast<Node<Real>>(
              prod->in(1))); // collect all rhs nodes
        }
        auto cat = RowwiseCat(cat_ins);
        auto batched_prod = dynamic_ptr_cast<Node<Real>>(prods.front()->in(0)) *
                            cat; // lhs * batched rhs

        uncats.reserve(cat_ins.size());
        for (size_t i = 0; i < cat_ins.size(); i++) {
          uncats.push_back(RowwiseUncat(batched_prod, i, cat));
        }

        for (size_t i = 0; i < prods.size(); i++) {
          for (auto& [out, index] : outs_.at(prods[i])) {
            auto ins = out->ins();
            ins[index] = uncats[i];
            out->set_ins(ins);
            // out->in(index) = uncats[i];
          }
        }

        cat_ins.clear();
        uncats.clear();
      }
    }
  }

 public:
  ExtendedGraph(const Graph& g) {
    list_ = g.nodes();
    sink_ = list_.back();
    GINN_TIME(compute_outs());
  }

  auto& autobatch() {
    GINN_TIME(replace_affines());
    GINN_TIME(group_by_height());
    GINN_TIME(merge_prods());
    return *this;
  }

  Graph graph() { return Graph(sink_); }
};

Graph Autobatch(const Graph& g) { return ExtendedGraph(g).autobatch().graph(); }

} // namespace ginn

#endif
