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

#ifndef GINN_MODEL_LSTM_H
#define GINN_MODEL_LSTM_H

#include <ginn/node/affine.h>
#include <ginn/node/common.h>
#include <ginn/node/nlnode.h>
#include <ginn/node/weight.h>
#include <ginn/util/util.h>

namespace ginn {

template <typename Scalar = Real>
class Lstm {
 public:
  using State = std::pair<NodePtr<Scalar>, NodePtr<Scalar>>;

  // clang-format off
  WeightPtr<Scalar> Wix, Wfx, Wcx, Wox,
                    Wih, Wfh, Wch, Woh,
                    Wic, Wfc,      Woc,
                    bi,  bf,  bc,  bo;
  // clang-format on

  Lstm() = default;
  Lstm(Device& dev, Size dim, Size xdim) { init(dev, dim, xdim); }

  Device& dev() const {
    GINN_ASSERT(Wix, "Uninitialized Lstm does not have a device!");
    return Wix->dev();
  }

  std::vector<WeightPtr<Scalar>> weights() {
    // clang-format off
    return {Wix, Wfx, Wcx, Wox,
            Wih, Wfh, Wch, Woh,
            Wic, Wfc,      Woc,
            bi,  bf,  bc,  bo};
    // clang-format on 
  };
  // clang-format on

  void init(Device& dev, Size dim, Size xdim) {
    for (auto& w : {&Wix, &Wfx, &Wcx, &Wox}) {
      *w = Weight<Scalar>(dev, {dim, xdim});
    }
    for (auto& w : {&Wih, &Wfh, &Wch, &Woh, &Wic, &Wfc, &Woc}) {
      *w = Weight<Scalar>(dev, {dim, dim});
    }
    for (auto& w : {&bi, &bf, &bc, &bo}) { *w = Weight<Scalar>(dev, {dim}); }
  }

  void tie(Lstm& other) {
    init(other.dev(), 0, 0);
    auto my_weights = weights();
    auto other_weights = other.weights();
    for (size_t i = 0; i < my_weights.size(); i++) {
      my_weights[i]->tie(other_weights[i]);
    }
  }

  Lstm copy(Copy mode) {
    Lstm rval;
    rval.init(dev(), 0, 0);
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

  template <typename DataPtrPair>
  State step(const NodePtr<Scalar>& x, const DataPtrPair& past) {
    auto [h_past, c_past] = past;

    auto i = Affine<SigmoidOp>(Wix, x, Wih, h_past, Wic, c_past, bi);
    auto f = Affine<SigmoidOp>(Wfx, x, Wfh, h_past, Wfc, c_past, bf);
    auto g = Affine<TanhOp>(Wcx, x, Wch, h_past, bc);
    auto c = CwiseProd(f, c_past) + CwiseProd(i, g);
    auto o = Affine<SigmoidOp>(Wox, x, Woh, h_past, Woc, c, bo);
    auto h_ = Tanh(c);
    auto h = CwiseProd(o, h_);

    return {h, c};
  }
};

} // end namespace ginn

#endif
