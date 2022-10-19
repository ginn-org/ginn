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

#ifndef GINN_UPDATE_H
#define GINN_UPDATE_H

#include <ginn/node/weight.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace ginn {

template <typename Scalar>
class Updater {
 public:
  bool guard = true;
  virtual void update(WeightNode<Scalar>*) = 0;

  void update(const std::unordered_set<WeightNode<Scalar>*>& ws) {
    for (auto w : ws) { update(w); }
  }

  void update(const std::vector<WeightNode<Scalar>*>& ws) {
    for (auto w : ws) { update(w); }
  }

  void update(const std::vector<Ptr<WeightNode<Scalar>>>& ws) {
    for (auto w : ws) { update(w.get()); }
  }

  void update(Graph& g) {
    std::unordered_set<WeightNode<Scalar>*> ws;
    for (auto n : g.nodes()) {
      auto n_ =
          dynamic_cast<WeightNode<Scalar>*>(n.get()); // is this expensive?
      if (n_ and n_->has_grad()) { ws.insert(n_); }
    }
    update(ws);
  }
};

namespace update {

template <typename Scalar>
class Sgd : public Updater<Scalar> {
 public:
  using Updater<Scalar>::update;
  Real lr, clip;

  Sgd(Real a_lr = 1e-1, Real a_clip = 5.) : lr(a_lr), clip(a_clip) {}

  void update(WeightNode<Scalar>* w) override {
    if (this->guard) {
      std::lock_guard<std::mutex> l(w->access());
      update(w->value(), w->grad());
    } else {
      update(w->value(), w->grad());
    }
  }

  void update(Tensor<Scalar>& w, Tensor<Scalar>& d) {
    d = d.t().cwiseMin(clip).cwiseMax(-clip);
    w += -lr * d.t();
  }
};

template <typename Scalar>
class Adam : public Updater<Scalar> {
 public:
  struct History {
    Tensor<Scalar> m, v;
    Real beta_1_t, beta_2_t;
  };

  using Updater<Scalar>::update;
  std::unordered_map<size_t, History> weight_histories;
  std::mutex history_access;
  // std::unordered_map<std::pair<Device, Tensor*>, Tensor> m__, v__; TODO
  Real lr_, clip_, eps_, beta_1_, beta_2_;

  Adam(Real lr = 1e-3,
       Real clip = 5.,
       Real eps = 1e-8,
       Real beta_1 = 0.9,
       Real beta_2 = 0.999)
      : lr_(lr), clip_(clip), eps_(eps), beta_1_(beta_1), beta_2_(beta_2) {}

  template <typename WeightPtr>
  void init(WeightPtr w) {
    GINN_ASSERT(w->value().size() > 0);
    Tensor<Scalar> m(w->dev(), w->value().shape()),
        v(w->dev(), w->value().shape());
    m.set_zero();
    v.set_zero();
    weight_histories.emplace(
        std::make_pair(w->id(), History{std::move(m), std::move(v), 1., 1.}));
  }

  template <typename WeightPtr>
  void init(const std::vector<WeightPtr>& weights) {
    for (auto& w : weights) { init(w); }
  }

  void update(Tensor<Scalar>& w, Tensor<Scalar>& d, size_t id) {
    History* h;
    {
      std::lock_guard<std::mutex> lh(history_access);
      h = &(weight_histories[id]);
    }
    auto& m = h->m;
    auto& v = h->v;

    Scalar clip(clip_), beta_1(beta_1_), beta_2(beta_2_);
    Scalar eps(eps_); // TODO: this is likely to underflow for Half?

    d = d.t().cwiseMin(clip).cwiseMax(-clip);
    m = beta_1 * m.t() + (1. - beta_1) * d.t();
    v = beta_2 * v.t() + (1. - beta_2) * d.t().square();
    h->beta_1_t *= beta_1_;
    h->beta_2_t *= beta_2_;
    w += (Scalar(-lr_ * (1. / (1. - h->beta_1_t))) * m.t()) /
         (eps + (Scalar(1. / (1. - h->beta_2_t)) * v.t()).sqrt());
  }
  void update(WeightNode<Scalar>* w) override {
    if (this->guard) {
      std::lock_guard<std::mutex> l(w->access());
      if (w->dev()->type() == CPU and
          !w->grad().m().array().isFinite().all()) { // TODO: GPU?
        if (w->grad().m().array().isInf().any()) {
          std::cerr << "Gradient has an inf!" << std::endl;
        }
        if (Eigen::isnan(w->grad().m().array()).any()) {
          std::cerr << "Gradient has a nan!" << std::endl;
        }
        throw std::domain_error("Gradient is not finite!");
      }
      {
        std::lock_guard<std::mutex> lh(history_access);
        if (weight_histories.find(w->id()) == weight_histories.end()) {
          init(w);
        }
      }
      update(w->value(), w->grad(), w->id());
    } else {
      {
        std::lock_guard<std::mutex> lh(history_access);
        if (weight_histories.find(w->id()) == weight_histories.end()) {
          init(w);
        }
      }
      update(w->value(), w->grad(), w->id());
    }
  }
};

} // end namespace update
} // end namespace ginn
#endif
