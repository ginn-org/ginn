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

#ifndef GINN_MODEL_TREELSTM_H
#define GINN_MODEL_TREELSTM_H

#include <ginn/node/affine.h>
#include <ginn/node/nlnode.h>
#include <ginn/node/weight.h>

namespace ginn {

// N-ary tree LSTM.
// See
//   Tai, Kai Sheng, Richard Socher, and Christopher D. Manning. "Improved
//   semantic representations from tree-structured long short-term memory
//   networks." arXiv preprint arXiv:1503.00075 (2015).
//   Section 3.2
// for details.
//
// TODO: Add peephole connections a la
//   Zhu, Xiaodan, Parinaz Sobihani, and Hongyu Guo. "Long short-term memory
//   over recursive structures." International Conference on Machine Learning.
//   2015.
template <typename Scalar = Real>
class TreeLstm {
 public:
  using State =
      std::pair<NodePtr<Scalar>, NodePtr<Scalar>>; // {h, c} for a node
  using Child = std::tuple<unsigned, NodePtr<Scalar>, NodePtr<Scalar>>;
  enum class Reduce { add, max };

  Size labels; // # of edge labels
  Reduce reduce = Reduce::add;

  WeightPtr<Scalar> Wi, Wf, Wu, Wo, // 1 of each
      bi, bf, bu, bo;
  std::vector<WeightPtr<Scalar>> Ui, Uu, Uo;      // N of each
  std::vector<std::vector<WeightPtr<Scalar>>> Uf; // NxN of each

 public:
  TreeLstm() = default;
  TreeLstm(Device& dev,
           Size labels,
           Size dim,
           Size xdim,
           Reduce reduce = Reduce::add) {
    init(dev, labels, dim, xdim, reduce);
  }

  Device& dev() const {
    GINN_ASSERT(Wi, "Uninitialized TreeLstm does not have a device!");
    return Wi->dev();
  }

  TreeLstm copy(Copy mode) {
    TreeLstm rval;
    rval.init(dev(), labels, 0, 0, reduce);
    auto rval_ws = rval.weights();
    auto this_ws = weights();
    for (size_t i = 0; i < rval_ws.size(); i++) {
      if (mode == Copy::Tied) {
        rval_ws[i]->tie(this_ws[i]);
      } else {
        *rval_ws[i] = *this_ws[i];
      }
    }
    return rval;
  }

  TreeLstm(const TreeLstm& l) : labels(l.labels) {
    GINN_THROW("Not implemented yet!");
  }

  TreeLstm& operator=(const TreeLstm&) = default;

  auto weights() {
    std::vector<WeightPtr<Scalar>> ws{Wi, Wf, Wu, Wo, bi, bf, bu, bo};
    for (auto W : {&Ui, &Uu, &Uo}) {
      ws.insert(ws.end(), W->begin(), W->end());
    }
    for (auto& W : Uf) { ws.insert(ws.end(), W.begin(), W.end()); }
    return ws;
  }

  void init(Device& dev, Size a_labels, Size dim, Size xdim, Reduce a_reduce) {
    labels = a_labels;
    reduce = a_reduce;
    Ui.resize(labels);
    Uu.resize(labels);
    Uo.resize(labels);
    Uf.resize(labels);
    for (auto& Uf_ : Uf) { Uf_.resize(labels); }

    for (auto& w : {&Wi, &Wf, &Wu, &Wo}) { *w = Weight(dev, {dim, xdim}); }
    for (auto& W : {&Ui, &Uu, &Uo}) {
      for (auto& w : *W) { w = Weight(dev, {dim, dim}); }
    }
    for (auto& W : Uf) {
      for (auto& w : W) { w = Weight(dev, {dim, dim}); }
    }
    for (auto& w : {&bi, &bf, &bu, &bo}) { *w = Weight(dev, {dim}); }
  }

  State step(const NodePtr<Scalar>& x, const std::vector<Child>& children) {
    // TODO: make x optional, by nullptr or 0-size checking
    auto label_of = [](auto& child) { return std::get<0>(child); };
    auto h_of = [](auto& child) { return std::get<1>(child); };
    auto c_of = [](auto& child) { return std::get<2>(child); };

    std::vector<NodePtr<Scalar>> ins = {Wi, x};
    for (auto& ch : children) { ins += {Ui[label_of(ch)], h_of(ch)}; }
    ins.push_back(bi);
    auto i = Sigmoid(Affine((ins)));

    std::vector<NodePtr<Scalar>> f(children.size());
    for (size_t i = 0; i < f.size(); i++) {
      std::vector<NodePtr<Scalar>> f_i = {Wf, x};
      unsigned i_label = label_of(children[i]);
      for (auto& ch : children) {
        f_i += {Uf[i_label][label_of(ch)], h_of(ch)};
      }
      f_i.push_back(bf);
      f[i] = Sigmoid(Affine(f_i));
    }

    ins = {Wo, x};
    for (auto& ch : children) { ins += {Uo[label_of(ch)], h_of(ch)}; }
    ins.push_back(bo);
    auto o = Sigmoid(Affine((ins)));

    ins = {Wu, x};
    for (auto& ch : children) { ins += {Uu[label_of(ch)], h_of(ch)}; }
    ins.push_back(bu);
    auto u = Sigmoid(Affine((ins)));

    ins = {};
    for (unsigned j = 0; j < f.size(); j++) {
      ins.push_back(CwiseProd(f[j], c_of(children[j])));
    }

    NodePtr<Scalar> c = CwiseProd(i, u);
    if (not ins.empty()) {
      if (reduce == Reduce::add) {
        c = c + Add((ins));
      } else if (reduce == Reduce::max) {
        c = c + CwiseMax((ins));
      } else {
        GINN_THROW("Unexpected reduction type!");
      }
    }

    auto h = CwiseProd(o, Tanh(c));

    return {h, c};
  }
};

} // end namespace ginn

#endif
