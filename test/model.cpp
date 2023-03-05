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

#define CATCH_CONFIG_MAIN // so that Catch is responsible for main()

#include <catch2/catch.hpp>

#include <ginn/model/lstm.h>

using namespace ginn;

TEST_CASE("Lstm") {
  // Compare to PyTorch output
  Lstm lstm(cpu(), 3, 3);

  for (auto& w : {lstm.Wix,
                  lstm.Wfx,
                  lstm.Wcx,
                  lstm.Wox,
                  lstm.Wih,
                  lstm.Wfh,
                  lstm.Wch,
                  lstm.Woh}) {
    w->value().set<2>({{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}, {0.6, 0.7, -1.1}});
  }
  // At the time of writing, PyTorch's Lstm doesn't have peephole connections,
  // so set these weights to 0
  for (auto& w : {lstm.Wic, lstm.Wfc, lstm.Woc}) { w->value().set_zero(); }
  for (auto& b : {lstm.bi, lstm.bf, lstm.bc, lstm.bo}) {
    b->value().v() << 0.0, 0.3, 0.6;
  }

  Lstm<>::State hidden{Data(cpu(), {3, 1}), Data(cpu(), {3, 1})};
  hidden.first->value().v() << 0.3, 0.5, 0.7;
  hidden.second->value().v() << -0.3, -0.5, -0.7;

  std::vector<DataPtr<>> inputs;
  for (size_t i = 0; i < 5; i++) {
    inputs.push_back(Data(cpu(), {3, 1}));
    inputs.back()->value().v() << 0.25, -0.5, 0.75;
  }

  for (auto& x : inputs) { hidden = lstm.step(x, hidden); }

  Graph(hidden.first).forward();

  auto hv = hidden.first->value().v();
  auto cv = hidden.second->value().v();
  CHECK(hv(0) == Approx(0.067673));
  CHECK(hv(1) == Approx(0.504593));
  CHECK(hv(2) == Approx(-0.030797));
  CHECK(cv(0) == Approx(0.127502));
  CHECK(cv(1) == Approx(0.972858));
  CHECK(cv(2) == Approx(-0.062305));
}
// (tensor([[[ 0.067673,  0.504593, -0.030797]]], grad_fn=<StackBackward>),
// tensor([[[ 0.127502,  0.972858, -0.062305]]], grad_fn=<StackBackward>))
