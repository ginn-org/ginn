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
#ifndef GINN_ENABLE_GPU
#define GINN_DOUBLE_PRECISION // grad checks require double precision
#endif

#include <catch.hpp>

#include <ginn/node/common.h>
#include <ginn/node/inplace.h>

using namespace ginn;
using namespace ginn;

auto& Dev = cpu();

TEMPLATE_TEST_CASE("Dropout", "[common][inplace]", Real, Half) {
  using Scalar = TestType;

  auto drop = [](auto in, Real p, bool inplace) {
    return inplace ? (NodePtr<Scalar>)InPlaceDropout(1 * in, p)
                   : (NodePtr<Scalar>)Dropout(in, p);
  };

  SECTION("Cpu sample") {
    auto x = Ones(Dev, {100, 100, 100, 100})->cast<Scalar>();
    Real p = GENERATE(0.25, 0.5, 0.75);
    bool inplace = GENERATE(false, true);

    auto y = drop(x, p, inplace);
    Graph(y).forward();

    double mean = y->value().template cast<double>().m().sum() / y->size();
    CHECK(mean == Approx(1.).epsilon(1e-3));
  }
  SECTION("Cpu extremes") {
    auto x = Ones(Dev, {100, 100, 100, 100})->cast<Scalar>();
    Real p = GENERATE(0., 1.);
    bool inplace = GENERATE(false, true);

    auto y = drop(x, p, inplace);
    Graph(y).forward();

    if (p == 0) {
      CHECK((y->value().m().array() == Scalar(1)).all());
    } else if (p == 1) {
      CHECK((y->value().m().array() == Scalar(0)).all());
    }

    CHECK_THROWS(Dropout(x, -1.));
    CHECK_THROWS(Dropout(x, 1.1));
  }
#ifdef GINN_ENABLE_GPU
  SECTION("Gpu sample") {
    auto x = Ones(gpu(), {100, 100, 100, 100})->cast<Scalar>();
    Real p = GENERATE(0.25, 0.5, 0.75);
    bool inplace = GENERATE(false, true);

    auto y = drop(x, p, inplace);
    Graph(y).forward();

    Tensor<double> sum(gpu(), Shape{});
    sum = y->value().t().template cast<double>().sum();
    double mean = sum.copy_to(cpu()).item() / y->size();
    CHECK(mean ==
          Approx(1.).epsilon(1e-2)); // Gpu is float hence bigger epsilon
  }
  SECTION("Gpu extremes") {
    auto x = Ones(gpu(), {100, 100, 100, 100})->cast<Scalar>();
    Real p = GENERATE(0., 1.);
    bool inplace = GENERATE(false, true);

    auto y = drop(x, p, inplace);
    Graph(y).forward();

    Tensor<bool> valid(gpu(), Shape{});

    if (p == 0) {
      valid = (y->value().t() == Scalar(1)).all();
      CHECK(valid.item());
    } else if (p == 1) {
      valid = (y->value().t() == Scalar(0)).all();
      CHECK(valid.item());
    }
  }
#endif
}
