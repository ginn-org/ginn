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

#define CATCH_CONFIG_MAIN     // so that Catch is responsible for main()
#define GINN_DOUBLE_PRECISION // so that Reals are doubles

#include <catch2/catch.hpp>

#include <algorithm>
#include <ginn/autobatch.h>
#include <ginn/node/affine.h>
#include <ginn/node/common.h>
#include <ginn/node/reduce.h>
#include <ginn/util/util.h>
#include <iostream>

using namespace ginn;

void check(const Tensor<Real>& a, const Tensor<Real>& b, Real eps = 1e-4) {
  REQUIRE(a.shape() == b.shape());
  for (Size i = 0; i < a.size(); i++) {
    CHECK(a.v()(i) == Approx(b.v()(i)).scale(eps));
  }
}

void check(NodePtr<Real> e,
           const std::vector<NodePtr<Real>>& ins,
           Real eps = 1e-4) {
  auto g = Graph(e);
  g.reset_forwarded();
  g.forward();

  auto sink = dynamic_ptr_cast<Node<Real>>(g.sink());

  Tensor<Real> sink_before = sink->value();

  g.reset_grad();
  g.backward(1.);

  std::vector<Tensor<Real>> d_ins_before;
  for (auto in : ins) { d_ins_before.push_back(in->grad()); }

  g = Autobatch(g);
  g.reset_forwarded();
  g.forward();

  Tensor<Real> sink_after = sink->value();

  g.reset_grad();
  g.backward(1.);

  std::vector<Tensor<Real>> d_ins_after;
  for (auto in : ins) { d_ins_after.push_back(in->grad()); }

  check(sink_before, sink_after);
  for (size_t i = 0; i < d_ins_before.size(); i++) {
    check(d_ins_before[i], d_ins_after[i], eps);
  }
}

TEST_CASE("Autobatch", "[autobatch]") {
  auto x = Random(cpu(), {2, 2});
  auto y = Random(cpu(), {2, 2});
  auto b = Random(cpu(), {2});

  auto h = Affine(x, y, x, y, b);
  auto t = Affine(h, y, b);
  auto g = Graph(t);
  g.forward();
  Tensor<Real> before = t->value();

  g.reset_forwarded();

  g = Autobatch(g);
  auto sink = dynamic_ptr_cast<Node<Real>>(g.sink());
  g.forward();
  Tensor<Real> after = sink->value();

  check(before, after);
}

TEST_CASE("Autobatch grad", "[autobatch]") {
  auto x = Random(cpu(), {2, 2});
  auto y = Random(cpu(), {2, 2});
  auto b = Random(cpu(), {2});

  auto h = Affine(x, y, x, y, b);
  auto t = Affine(h, y, b);
  auto s = Sum(t);

  auto g = Graph(s);
  g.forward();
  Tensor<Real> s_before = s->value();
  g.reset_grad();
  g.backward(1.);

  Tensor<Real> dx_before = x->grad();
  Tensor<Real> dy_before = y->grad();
  Tensor<Real> db_before = b->grad();

  g = Autobatch(g);
  auto sink = dynamic_ptr_cast<Node<Real>>(g.sink());

  g.reset_forwarded();
  g.forward();
  Tensor<Real> s_after = sink->value();

  g.reset_grad();
  g.backward(1.);

  Tensor<Real> dx_after = x->grad();
  Tensor<Real> dy_after = y->grad();
  Tensor<Real> db_after = b->grad();

  check(s_before, s_after);
  check(dx_before, dx_after);
  check(dy_before, dy_after);
  check(db_before, db_after);
}

TEST_CASE("Autobatch check helper", "[autobatch]") {
  auto x = Random(cpu(), {2, 2});
  auto y = Random(cpu(), {2, 2});
  auto b = Random(cpu(), {2});

  auto h = Affine(x, y, x, y, b);
  auto t = Affine(h, y, b);
  auto s = Sum(t);

  check(s, {x, y, b});
}

TEST_CASE("Recurrent", "[autobatch]") {
  std::vector<NodePtr<Real>> x;
  for (size_t i = 0; i < 3; i++) {
    for (size_t t = 0; t < 4; t++) { x.push_back(Random(cpu(), {2})); }
  }

  auto W = Random(cpu(), {3, 2});
  auto V = Random(cpu(), {3, 3});
  auto b = Random(cpu(), {3});

  std::vector<NodePtr<Real>> hs;
  for (size_t i = 0; i < 3; i++) {
    NodePtr<Real> h = Zero(cpu(), {3});
    for (size_t t = 0; t < 4; t++) { h = Affine(W, x.at(4 * i + t), V, h, b); }
    hs.push_back(h);
  }
  auto s = Sum(Add(hs));

  check(s, x + std::vector<NodePtr<Real>>{W, V, b});
}

TEST_CASE("Recurrent mixed", "[autobatch]") {
  std::vector<NodePtr<Real>> x;
  for (size_t i = 0; i < 3; i++) {
    for (size_t t = 0; t < 4; t++) { x.push_back(Random(cpu(), {2})); }
  }

  auto W = Random(cpu(), {3, 2});
  auto V = Random(cpu(), {3, 3});
  auto b = Random(cpu(), {3});

  std::vector<NodePtr<Real>> hs;
  for (size_t i = 0; i < 3; i++) {
    NodePtr<Real> h = Zero(cpu(), {3});
    for (size_t t = 0; t < 4; t++) {
      if ((i + t) % 2) {
        h = Affine(W, x.at(4 * i + t), V, h, b);
      } else {
        h = W * x.at(4 * i + t) + V * h + b;
      }
    }
    hs.push_back(h);
  }
  auto s = Sum(Add(hs));

  check(s, x + std::vector<NodePtr<Real>>{W, V, b});
}
