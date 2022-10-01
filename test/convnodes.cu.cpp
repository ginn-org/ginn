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

#include <algorithm>
#include <iostream>

#include <ginn/node/conv.h>
#include <ginn/node/pool.h>

#include "check_node.h"

using namespace ginn;

auto& Dev = cpu();

// clang-format off

// With Half, this does not build (with cuda)
// With __half, it builds but crashes at runtime
TEMPLATE_TEST_CASE("Conv2d", "[conv]", Real) {
  using Scalar = TestType;
  auto x = Values<4>({{{{1}, {4}, {7}},
                       {{2}, {5}, {8}},
                       {{3}, {6}, {9}}}})->cast<Scalar>();
  REQUIRE(x->shape() == Shape{1, 3, 3, 1});

  auto f = Values<4>({{{{1, 3},
                        {2, 4}}}})->cast<Scalar>();
  REQUIRE(f->shape() == Shape{1, 1, 2, 2});

  auto y = Values<4>({{{{37}, {67}, {23}},
                       {{47}, {77}, {26}},
                       {{21}, {33}, { 9}}}})->cast<Scalar>();
  REQUIRE(y->shape() == Shape{1, 3, 3, 1});

  auto y2 = Values<4>({{{{37}, {67}, {23}},
                        {{21}, {33}, { 9}}}})->cast<Scalar>();
  REQUIRE(y2->shape() == Shape{1, 2, 3, 1});

  auto y3 = Values<4>({{{{37}},
                        {{21}}}})->cast<Scalar>();
  REQUIRE(y3->shape() == Shape{1, 2, 1, 1});

  SECTION("Basic")            { check(Conv2d(x, f),       y ); }
  SECTION("Row stride")       { check(Conv2d(x, f, 2, 1), y2); }
  SECTION("Row & col stride") { check(Conv2d(x, f, 2, 3), y3); }
  SECTION("Grad or cuda") {
    CHECK_(Conv2d(x, f),       {x, f}, true);
    CHECK_(Conv2d(x, f, 2, 1), {x, f}, true);
    CHECK_(Conv2d(x, f, 2, 3), {x, f}, true);
  }
}

TEST_CASE("Conv2d 2", "[conv]") {
  auto x = Values<4>({{{{1}, {5}, {9}, {3}},
                       {{2}, {6}, {0}, {4}},
                       {{3}, {7}, {1}, {5}},
                       {{4}, {8}, {2}, {6}}}});
  REQUIRE(x->shape() == Shape{1, 4, 4, 1});

  auto f = Values<4>({{{{1, 4, 7},
                        {2, 5, 8},
                        {3, 6, 9}}}});
  REQUIRE(f->shape() == Shape{1, 1, 3, 3});

  auto y = Values<4>({{{{111}, {141}, {133}, {57}},
                       {{178}, {178}, {178}, {74}},
                       {{217}, {153}, {183}, {85}},
                       {{145}, {102}, {120}, {55}}}});
  REQUIRE(y->shape() == Shape{1, 4, 4, 1});

  SECTION("Basic") {
    check(Conv2d(x, f), y);
    CHECK_(Conv2d(x, f), {x, f}, true);
  }
}

TEST_CASE("Conv2d 3", "[conv]") {
  auto x = Values<4>({{{ {2}, {3} }}});
  auto f = Values<4>({{{{4, 5, 6}}}});
  auto y = Values<4>({{{{23}}}});

  REQUIRE(x->shape() == Shape{1, 1, 2, 1});
  REQUIRE(f->shape() == Shape{1, 1, 1, 3});
  REQUIRE(y->shape() == Shape{1, 1, 1, 1});

  SECTION("Strided") {
    check(Conv2d(x, f, 1, 2), y);
    CHECK_(Conv2d(x, f), {x, f}, true);
  }
}

TEST_CASE("Conv1d", "[forward]") {
  auto x = Values<3>({{{1}, {4}, {7}},
                      {{2}, {5}, {8}},
                      {{3}, {6}, {9}}});
  REQUIRE(x->shape() == Shape{3, 3, 1});

  auto f = Values<3>({{{1, 4},
                       {2, 5},
                       {3, 6}}});
  REQUIRE(f->shape() == Shape{1, 3, 2});

  auto y = Values<3>({{{91}, {154}, {50}}});
  REQUIRE(y->shape() == Shape{1, 3, 1});

  auto y2 = Values<3>({{{91}, {50}}});
  REQUIRE(y2->shape() == Shape{1, 2, 1});

  SECTION("Basic")  { check(Conv1d(x, f),    y ); }
  SECTION("Stride") { check(Conv1d(x, f, 2), y2); }
  SECTION("Grad or cuda") {
    CHECK_(Conv1d(x, f),       {x, f}, true);
    CHECK_(Conv1d(x, f, 2),    {x, f}, true);
  }
}

TEST_CASE("Conv1d more filters", "[forward]") {
  auto x = Values<3>({{{1}, {4}, {7}},
                      {{2}, {5}, {8}},
                      {{3}, {6}, {9}}});
  REQUIRE(x->shape() == Shape{3, 3, 1});

  auto f = Values<3>({{{1, 4},
                       {2, 5},
                       {3, 6}},
                      {{1.5, 4.5},
                       {2.5, 5.5},
                       {3.5, 6.5}}});
  REQUIRE(f->shape() == Shape{2, 3, 2});

  auto y = Values<3>({{{ 91  }, {154  }, {50}},
                      {{101.5}, {173.5}, {62}}});
  REQUIRE(y->shape() == Shape{2, 3, 1});

  auto y2 = Values<3>({{{ 91  }, {50}},
                       {{101.5}, {62}}});
  REQUIRE(y2->shape() == Shape{2, 2, 1});

  SECTION("Basic")  { check(Conv1d(x, f),    y ); }
  SECTION("Stride") { check(Conv1d(x, f, 2), y2); }
  SECTION("Grad or cuda") {
    CHECK_(Conv1d(x, f),       {x, f}, true);
    CHECK_(Conv1d(x, f, 2),    {x, f}, true);
  }
}

TEMPLATE_TEST_CASE("MaxPool2d", "[pool]", Real, Half) {
  using Scalar = TestType;
  auto x = Values<4>({{{{1}, {4}, {7}},
                       {{2}, {5}, {8}},
                       {{3}, {6}, {9}}}})->cast<Scalar>();
  REQUIRE(x->shape() == Shape{1, 3, 3, 1});

  auto y = Values<4>({{{{5}, {8}, {8}},
                       {{6}, {9}, {9}},
                       {{6}, {9}, {9}}}})->cast<Scalar>();
  REQUIRE(y->shape() == Shape{1, 3, 3, 1});

  auto y2 = Values<4>({{{{5}, {8}, {8}},
                        {{6}, {9}, {9}}}})->cast<Scalar>();
  REQUIRE(y2->shape() == Shape{1, 2, 3, 1});

  auto y3 = Values<4>({{{{5}},
                        {{6}}}})->cast<Scalar>();
  REQUIRE(y3->shape() == Shape{1, 2, 1, 1});

  SECTION("Basic")            { check(MaxPool2d(x, 2, 2),       y ); }
  SECTION("Row stride")       { check(MaxPool2d(x, 2, 2, 2),    y2); }
  SECTION("Row & col stride") { check(MaxPool2d(x, 2, 2, 2, 3), y3); }
  SECTION("Grad or cuda") {
    CHECK_(MaxPool2d(x, 2, 2),       {x}, true); 
    CHECK_(MaxPool2d(x, 2, 2, 2),    {x}, true); 
    CHECK_(MaxPool2d(x, 2, 2, 2, 3), {x}, true); 
  }
}

TEMPLATE_TEST_CASE("MaxPool1d", "[pool]", Real, Half) {
  using Scalar = TestType;
  auto x = Values<3>({{{9}, {6}, {3}},
                      {{2}, {5}, {8}},
                      {{4}, {1}, {7}}})->cast<Scalar>();
  REQUIRE(x->shape() == Shape{3, 3, 1});

  SECTION("Basic") {
    auto y = Values<3>({{{9}, {6}, {3}},
                        {{5}, {8}, {8}},
                        {{4}, {7}, {7}}})->cast<Scalar>();
    REQUIRE(y->shape() == Shape{3, 3, 1});
    check(MaxPool1d(x, 2), y);
  }

  SECTION("Stride") {
    auto y = Values<3>({{{9}, {3}},
                        {{5}, {8}},
                        {{4}, {7}}})->cast<Scalar>();
    REQUIRE(y->shape() == Shape{3, 2, 1});
    check(MaxPool1d(x, 2, 2), y);
  }

  SECTION("Grad or cuda") {
    CHECK_(MaxPool1d(x, 2),    {x}, true);
    CHECK_(MaxPool1d(x, 2, 2), {x}, true);
  }
}

// clang-format on
