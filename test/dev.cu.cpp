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

#include "testutil.h"
#include <catch2/catch.hpp>

#include <algorithm>
#include <iostream>

#include <ginn/node/common.h>
#include <ginn/node/inplace.h>

using namespace ginn;

TEMPLATE_TEST_CASE("Cpu tensors", "", Real, Int, Half) {
  using Scalar = TestType;
  auto dev = PreallocCpu(100);

  Tensor<Scalar> t1(cpu(), {2, 3});

  CHECK(dev->used() == 0);

  Tensor<Scalar> t2(dev, {1, 2});
  CHECK(dev->used() == (t2.size() * sizeof(Scalar)));

  Tensor<Scalar> t3(dev);
  CHECK(dev->used() == (t2.size() * sizeof(Scalar)));

  t3 = t2; // should copy
  CHECK(dev->used() == (2 * t2.size() * sizeof(Scalar)));

  auto t4 = t1.maybe_copy_to(dev); // should avoid copy
  CHECK(dev->used() == (2 * t2.size() * sizeof(Scalar)));

  t1.move_to(dev);
  CHECK(dev->used() == ((2 * t2.size() + t1.size()) * sizeof(Scalar)));

  t2.move_to(dev); // should be a no-op
  CHECK(dev->used() == ((2 * t2.size() + t1.size()) * sizeof(Scalar)));
}

TEMPLATE_TEST_CASE("Cpu nodes", "", Real, Int, Half) {
  using Scalar = TestType;
  auto dev = PreallocCpu(200);

  auto x = Values<2>({{1, 2, 3}, {4, 5, 6}})->cast<Scalar>();
  auto y = Values<2>({{1, 2, 3}, {4, 5, 6}})->cast<Scalar>();
  x->has_grad(false);

  CHECK(dev->used() == 0);

  x->move_to(dev);
  CHECK(dev->used() == (x->size() * sizeof(Scalar)));

  y->move_to(dev); // should move both value() and grad(), but grad() is empty
  CHECK(dev->used() == (2 * x->size() * sizeof(Scalar)));

  auto z = x + y;
  Graph g(z);
  CHECK(dev->used() == (2 * x->size() * sizeof(Scalar)));

  g.forward();
  CHECK(dev->used() == (3 * x->size() * sizeof(Scalar)));

  g.reset_grad(); // only z and y have grads
  CHECK(dev->used() == (5 * x->size() * sizeof(Scalar)));
}

TEST_CASE("InPlaceAdd", "[inplace]") {
  auto dev1 = PreallocCpu(1000);
  auto dev2 = PreallocCpu(1000);

  auto W = Random(dev1, {2, 3});
  auto V = Random(dev1, {2, 3});

  auto a = Random(dev2, {2, 3});
  auto b = Random(dev2, {2, 3});

  a->value() = W->value();
  b->value() = V->value();

  auto U = W + V;
  auto c = InPlaceAdd(a, b);

  Graph gU(U);
  gU.forward();
  CHECK(dev1->used() == sizeof(Real) * (3 * W->size()));

  Graph gc(c);
  gc.forward();
  CHECK(dev2->used() == sizeof(Real) * (2 * W->size()));

  CHECK(U->value() == Close(c->value()));

  gU.reset_grad();
  gU.backward(1.);
  CHECK(dev1->used() == sizeof(Real) * (6 * W->size()));

  gc.reset_grad();
  gc.backward(1.);
  CHECK(dev2->used() == sizeof(Real) * (4 * W->size()));

  CHECK(W->grad() == Close(a->grad()));
  CHECK(V->grad() == Close(b->grad()));
}

#ifdef GINN_ENABLE_GPU

TEMPLATE_TEST_CASE("Gpu tensors", "", Real, Int, Half) {
  using Scalar = TestType;
  auto dev = PreallocGpu(100);

  Tensor<Scalar> t1(gpu(), {2, 3});
  CHECK(dev->used() == 0);

  Tensor<Scalar> t2(dev, {1, 2});
  CHECK(dev->used() == (t2.size() * sizeof(Scalar)));

  Tensor<Scalar> t3(dev);
  CHECK(dev->used() == (t2.size() * sizeof(Scalar)));

  t3 = t2; // should copy
  CHECK(dev->used() == (2 * t2.size() * sizeof(Scalar)));

  auto t4 = t1.maybe_copy_to(dev); // should avoid copy
  CHECK(dev->used() == (2 * t2.size() * sizeof(Scalar)));

  t1.move_to(dev);
  CHECK(dev->used() == ((2 * t2.size() + t1.size()) * sizeof(Scalar)));

  t2.move_to(dev); // should be a no-op
  CHECK(dev->used() == ((2 * t2.size() + t1.size()) * sizeof(Scalar)));

  Tensor<Scalar> t5(cpu(), {1, 3});
  auto t6 = t5.maybe_copy_to(dev); // should copy
  CHECK(dev->used() ==
        ((2 * t2.size() + t1.size() + t5.size()) * sizeof(Scalar)));
}

TEMPLATE_TEST_CASE("Gpu nodes", "", Real, Int, Half) {
  using Scalar = TestType;
  auto dev = PreallocGpu(200);

  auto x = Values<2>({{1, 2, 3}, {4, 5, 6}})->cast<Scalar>();
  auto y = Values<2>({{1, 2, 3}, {4, 5, 6}})->cast<Scalar>();
  x->has_grad(false);

  CHECK(dev->used() == 0);

  x->move_to(dev);
  CHECK(dev->used() == (x->size() * sizeof(Scalar)));

  y->move_to(dev); // should move both value() and grad(), but grad() is empty
  CHECK(dev->used() == (2 * x->size() * sizeof(Scalar)));

  auto z = x + y;
  Graph g(z);
  CHECK(dev->used() == (2 * x->size() * sizeof(Scalar)));

  g.forward();
  CHECK(dev->used() == (3 * x->size() * sizeof(Scalar)));

  g.reset_grad(); // only z and y have grads
  CHECK(dev->used() == (5 * x->size() * sizeof(Scalar)));
}

#endif
