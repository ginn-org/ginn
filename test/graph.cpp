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

#include <algorithm>
#include <ginn/node/common.h>
#include <ginn/node/nlnode.h>
#include <iostream>

using namespace ginn;

TEST_CASE("Forwardedness", "[graph]") {
  // Data nodes are constructed as "forwarded", therefore we add a dummy
  // Identity
  auto a = Identity(Data(cpu(), {2, 3}));
  auto b = Identity(Data(cpu(), {2, 3}));
  auto c = a + b;

  auto g = Graph(c);
  CHECK(g.nodes().size() == 5);

  SECTION("Full forward") {
    g.forward();
    for (auto e : g.nodes()) { CHECK(e->forwarded); }
  }

  SECTION("Partial forward") {
    Graph(a).forward();
    CHECK(a->forwarded);
    CHECK(not b->forwarded);
    CHECK(not c->forwarded);
  }

  SECTION("Partial forward composed") {
    auto d = c + a;
    g.forward();
    for (auto e : g.nodes()) { CHECK(e->forwarded); }
    CHECK(not d->forwarded);
  }
}

TEST_CASE("Gradfulness", "[graph]") {
  auto a = Random(cpu(), {2, 3});
  auto b = FixedRandom(cpu(), {2, 3});
  auto c = a + b;
  auto g = Graph(c);

  CHECK(a->has_grad());
  CHECK(not b->has_grad());
  CHECK(c->has_grad());

  SECTION("No reset grad") {
    CHECK(a->grad().size() == 0);
    CHECK(b->grad().size() == 0);
    CHECK(c->grad().size() == 0);
  }

  SECTION("Full init grad") {
    g.forward();
    g.init_grad();

    CHECK(a->grad().shape() == Shape{2, 3});
    CHECK(b->grad().size() == 0);
    CHECK(c->grad().shape() == Shape{2, 3});
  }

  SECTION("Full reset grad") {
    g.forward();
    g.reset_grad();

    CHECK(a->grad().shape() == Shape{2, 3});
    CHECK((a->grad().m().array() == 0.).all());
    CHECK(b->grad().size() == 0);
    CHECK(c->grad().shape() == Shape{2, 3});
    CHECK((c->grad().m().array() == 0.).all());
  }

  SECTION("Conservative init grad") {        // Check that init_grad
    g.forward();                             // does not override the existing
                                             // gradients if they have the
    a->grad() = Tensor<Real>(cpu(), {2, 3}); // appropriate dimensions.
    a->grad().m() << 1., 2., 3., 4., 5., 6.;

    g.init_grad();
    Matrix<Real> expected(2, 3);
    expected << 1., 2., 3., 4., 5., 6.;
    CHECK(a->grad().m() == expected);
  }

  SECTION("Destructive init grad") {         // Check that init_grad
    g.forward();                             // DOES override the existing
                                             // gradients if they don't have the
    a->grad() = Tensor<Real>(cpu(), {1, 3}); // appropriate dimensions.
    a->grad().m() << 1., 2., 3.;

    g.init_grad();
    CHECK(a->grad().shape() == Shape{2, 3});
  }
}
