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

#include <catch2/catch.hpp>

#include <algorithm>
#include <iostream>

#include <ginn/node/affine.h>
#include <ginn/node/common.h>
#include <ginn/node/compare.h>
#include <ginn/node/inplace.h>
#include <ginn/node/layernorm.h>
#include <ginn/node/layout.h>
#include <ginn/node/nlnode.h>
#include <ginn/node/pick.h>
#include <ginn/node/prod.h>
#include <ginn/node/reduce.h>
#include <ginn/node/select.h>
#include <ginn/node/weight.h>

#include "check_node.h"

using namespace ginn;

auto Dev = cpu();

// clang-format off

TEMPLATE_TEST_CASE("Dim", "[layout]", Real, Int, Half, void) {
  BaseNodePtr x;
  if constexpr(std::is_same_v<TestType, void>) {
    x = Random(Dev, {3, 2, 1});
  } else {
    x = Random<TestType>(Dev, {3, 2, 1});
  }
  auto d0 = Dim(x, 0);
  auto d1 = Dim(x, 1);
  auto d2 = Dim(x, 2);
  SECTION("Basic") {
    Graph(d0).forward();
    Graph(d1).forward();
    Graph(d2).forward();
    CHECK(d0->value() == 3);
    CHECK(d1->value() == 2);
    CHECK(d2->value() == 1);
  }
}

TEMPLATE_TEST_CASE("Stack", "[forward] [layout]", Real, Int, Half) {
  using Scalar = TestType;

  SECTION("of rank-1") {
    auto a = Values<1>({ 1,  2,  3,  4})->cast<Scalar>();
    auto b = Values<1>({ 5,  6,  7,  8})->cast<Scalar>();
    auto c = Values<1>({ 9, 10, 11, 12})->cast<Scalar>();
    auto d = Values<1>({13, 14, 15, 16})->cast<Scalar>();
    auto e = Values<1>({17, 18, 19, 20})->cast<Scalar>();
    auto f = Values<1>({21, 22, 23, 24})->cast<Scalar>();

    auto expected = Values<3>({{{1, 5}, {9, 13}, {17, 21}},
                               {{2, 6}, {10, 14}, {18, 22}},
                               {{3, 7}, {11, 15}, {19, 23}},
                               {{4, 8}, {12, 16}, {20, 24}}})->cast<Scalar>();

    check(Stack<Scalar>({{a, b}, {c, d}, {e, f}}), expected);
    CHECK_(Stack<Scalar>({{a, b}}),                 {a, b}            , true);
    CHECK_(Stack<Scalar>({{a, b}, {c, d}}),         {a, b, c, d}      );
    CHECK_(Stack<Scalar>({{a, b}, {c, d}, {e, f}}), {a, b, c, d, e, f});
    CHECK_(Stack<Scalar>({{a, b}, {a, b}}),         {a, b}            );
    CHECK_(Stack<Scalar>({{a, a}, {a, a}, {a, a}}), {a}               );
  }

  SECTION("of rank-2") {
    auto a = Values<2>({{ 1,  2}, { 3,  4}})->cast<Scalar>();
    auto b = Values<2>({{ 5,  6}, { 7,  8}})->cast<Scalar>();
    auto c = Values<2>({{ 9, 10}, {11, 12}})->cast<Scalar>();
    auto d = Values<2>({{13, 14}, {15, 16}})->cast<Scalar>();
    auto e = Values<2>({{17, 18}, {19, 20}})->cast<Scalar>();
    auto f = Values<2>({{21, 22}, {23, 24}})->cast<Scalar>();

    auto expected = Values<4>({{{{1, 5}, {9, 13}, {17, 21}},
                                {{2, 6}, {10, 14}, {18, 22}}},
                               {{{3, 7}, {11, 15}, {19, 23}},
                                {{4, 8}, {12, 16}, {20, 24}}}})->cast<Scalar>();

    check(Stack<Scalar>({{a, b}, {c, d}, {e, f}}), expected);
    CHECK_(Stack<Scalar>({{a, b}}),                 {a, b}            );
    CHECK_(Stack<Scalar>({{a, b}, {c, d}}),         {a, b, c, d}      );
    CHECK_(Stack<Scalar>({{a, b}, {c, d}, {e, f}}), {a, b, c, d, e, f});
    CHECK_(Stack<Scalar>({{a, b}, {a, b}}),         {a, b}            );
    CHECK_(Stack<Scalar>({{a, a}, {a, a}, {a, a}}), {a}               );
  }

  SECTION("Errors") {
    auto a = Values<1>({ 1,  2,  3,  4});
    auto b = Values<1>({ 5,  6,  7,  8});
    auto c = Values<1>({ 9, 10, 11, 12});
    auto d = Values<1>({13, 14, 15, 16});
    auto e = Values<1>({17, 18, 19});
    auto f = Values<1>({21, 22, 23, 24});

    CHECK_THROWS(Stack<Real>({{a, b}, {c, d}, {e, f}})->forward());
    CHECK_THROWS(Stack<Real>({{a, b}, {c, d}, {f}})->forward());
    CHECK_THROWS(Stack<Real>({})->forward());
    CHECK_THROWS(Stack<Real>({{}})->forward());
  }
}

TEMPLATE_TEST_CASE("Cat", "[forward] [layout]", Real, Int, Half) {
  auto Vals = [](NestedInitList<2, Int> vals) {
    return Values<2, Int>(vals)->template cast<TestType>();
  };
  auto a = Vals({{1, 2}});
  auto b = Vals({{3, 4},
                 {5, 6}});
  auto c = Vals({{7, 8},
                 {9, 0}});

  auto cat = Cat(a, b, c);

  auto res = Vals({{1, 2},
                   {3, 4},
                   {5, 6},
                   {7, 8},
                   {9, 0}});

  check(cat, res);
  CHECK_(Cat(a, b, c), {a, b, c}, true);
  CHECK_(Cat(a, b, a), {a, b},    true);
  CHECK_(Cat(a, b),    {a, b},    true);
  CHECK_(Cat(a, a, a), {a},       true);
}

TEMPLATE_TEST_CASE("RowwiseCat", "[layout]", Real, Int, Half) {
  using Scalar = TestType;

  auto Vals = [](NestedInitList<2, Int> vals) {
    return Values<2, Int>(vals)->template cast<Scalar>();
  };
  auto a = Vals({{1},
                 {2}});
  auto b = Vals({{3, 4},
                 {5, 6}});
  auto c = Vals({{7, 8, 9},
                 {0, 1, 2}});

  SECTION("Forward") {
    auto cat = RowwiseCat(a, b, c);

    auto res = Vals({{1, 3, 4, 7, 8, 9},
                     {2, 5, 6, 0, 1, 2}});

    check(cat, res);
    CHECK(cat->offsets() == std::vector<Size>{0, 1, 3});
    CHECK(cat->extents() == std::vector<Size>{1, 2, 3});

    auto m = cat + Scalar(1);
    auto uncat0 = RowwiseUncat(m, 0, cat);
    auto uncat1 = RowwiseUncat(m, 1, cat);
    auto uncat2 = RowwiseUncat(m, 2, cat);

    auto a_ = Vals({{2},
                    {3}});
    auto b_ = Vals({{4, 5},
                    {6, 7}});
    auto c_ = Vals({{8, 9, 10},
                    {1, 2,  3}});

    check(uncat0, a_);
    check(uncat1, b_);
    check(uncat2, c_);
  }

  SECTION("Grad or cuda") {
    CHECK_(RowwiseCat(a, b, c),    {a, b, c}, true);
    for (size_t i = 0; i < 3; i++) {
      check_expr([&]() {
        auto cat = RowwiseCat(a, b, c);
        return RowwiseUncat(cat + 1, i, cat);
      }, {a, b, c}, true);
    }
  }
}

TEMPLATE_TEST_CASE("Reshape", "[layout]", Real, Half, Int) {
  using Scalar = TestType;
  auto W = Values<2>({{1, 2, 3, 4, 5, 6}})->cast<Scalar>();

  SECTION("Forward") {
    auto col = Values<2>({{1}, {2}, {3}, {4}, {5}, {6}})->cast<Scalar>();
    auto mat = Values<2>({{1, 4},
                          {2, 5},
                          {3, 6}})->cast<Scalar>();


    check(Reshape(W, Shape{6, 1}), col);
    check(Reshape(W, Shape{3, 2}), mat);
  }

  SECTION("Grad or cuda") {
    CHECK_(Reshape(W, Shape{6, 1}),    {W}, true);
    CHECK_(Reshape(W, Shape{3, 2}),    {W}, true);
    CHECK_(Reshape(W, Shape{2, 3}),    {W}, true);
    CHECK_(Reshape(W, Shape{6}),       {W}, true);
    CHECK_(Reshape(W, Shape{3, 1, 2}), {W}, true);
  }
}

TEMPLATE_TEST_CASE("RankView", "[layout]", Real, Half, Int) {
  using Scalar = TestType;
  auto W = Values<2>({{1, 4},
                      {2, 5},
                      {3, 6}})->cast<Scalar>();

  SECTION("Forward") {
    auto col = Values<1>({1, 2, 3, 4, 5, 6})->cast<Scalar>();
    auto mat = Values<2>({{1, 4},
                          {2, 5},
                          {3, 6}})->cast<Scalar>();
    auto ten = Values<3>({{{1}, {4}},
                          {{2}, {5}},
                          {{3}, {6}}})->cast<Scalar>();

    check(RankView(W, 1), col);
    check(RankView(W, 2), mat);
    check(RankView(W, 3), ten);
  }

  SECTION("Grad or cuda") {
    CHECK_(RankView(W, 1), {W}, true);
    CHECK_(RankView(W, 2), {W}, true);
    CHECK_(RankView(W, 3), {W}, true);
  }
}

TEMPLATE_TEST_CASE("Slice", "[layout]", Real, Half, Int) {
  using Scalar = TestType;
  auto x = Values<2>({{1, 2},
                      {3, 4},
                      {5, 6},
                      {7, 8}})->cast<Scalar>();
  REQUIRE(x->shape() == Shape{4, 2});

  SECTION("Row subset") {
    auto out = Values<2>({{3, 4},
                          {5, 6},
                          {7, 8}})->cast<Scalar>();
    REQUIRE(out->shape() == Shape{3, 2});
    check(Slice(x, Shape{1, 0}, Shape{3, 2}), out);
  }

  SECTION("Col subset") {
    auto out = Values<2>({{2},
                          {4},
                          {6},
                          {8}})->cast<Scalar>();
    REQUIRE(out->shape() == Shape{4, 1});
    check(Slice(x, Shape{0, 1}, Shape{4, 1}), out);
  }

  SECTION("Row & col subset") {
    auto out = Values<2>({{5},
                          {7}})->cast<Scalar>();
    REQUIRE(out->shape() == Shape{2, 1});
    check(Slice(x, Shape{2, 0}, Shape{2, 1}), out);
  }
}

TEMPLATE_TEST_CASE("Chip", "[layout]", Real, Int, Half) {
  using Scalar = TestType;
  auto x = Values<2>({{1, 2},
                      {3, 4},
                      {5, 6},
                      {7, 8}});
  auto y = Values<1>({5, 6});
  auto z = Values<1>({2, 4, 6, 8});

  NodePtr<Scalar> x_, y_, z_;
  if constexpr(std::is_same_v<Scalar, Real>) {
    x_ = x, y_ = y, z_ = z;
  } else {
    x_ = x->cast<Scalar>();
    y_ = y->cast<Scalar>();
    z_ = z->cast<Scalar>();
  }

  SECTION("Forward") {
    check(Chip(x_, 2, 0), y_);
    check(Chip(x_, 1, 1), z_);
  }

  SECTION("Grad or cuda") {
    SECTION("Basic") {
      CHECK_(Chip(x_, 0, 0), {x_}, true);
      CHECK_(Chip(x_, 1, 1), {x_}, true);
      CHECK_(Chip(x_, 2, 0), {x_}, true);
    }

    SECTION("High rank") {
      auto x = Random(Dev, {4, 2, 3})->cast<Scalar>();
      CHECK_(Chip(x, 3, 0), {x}, true);
      CHECK_(Chip(x, 0, 1), {x}, true);
      CHECK_(Chip(x, 1, 2), {x}, true);
      CHECK_(Chip(x, 2, 2), {x}, true);
    }
  }
}

TEMPLATE_TEST_CASE("Permute", "[layout]", Real, Half, Int) {
  using Scalar = TestType;

  auto a = Values<3>({{{ 1,  2,  3,  4}, { 5,  6,  7,  8}},
                      {{ 9, 10, 11, 12}, {13, 14, 15, 16}},
                      {{17, 18, 19, 20}, {21, 22, 23, 24}}})->cast<Scalar>();
  CHECK(a->shape() == Shape{3, 2, 4});

  auto b = Values<3>({{{ 1,  5}, { 2,  6}, { 3,  7}, { 4,  8}},
                      {{ 9, 13}, {10, 14}, {11, 15}, {12, 16}},
                      {{17, 21}, {18, 22}, {19, 23}, {20, 24}}})->cast<Scalar>();
  auto c = Values<3>({{{ 1,  9, 17}, { 5, 13, 21}},
                      {{ 2, 10, 18}, { 6, 14, 22}},
                      {{ 3, 11, 19}, { 7, 15, 23}},
                      {{ 4, 12, 20}, { 8, 16, 24}}})->cast<Scalar>();
  auto d = Values<3>({{{ 1,  2,  3,  4}, { 9, 10, 11, 12}, {17, 18, 19, 20}},
                      {{ 5,  6,  7,  8}, {13, 14, 15, 16}, {21, 22, 23, 24}}})
                    ->cast<Scalar>();
  auto e = Values<3>({{{ 1,  9, 17}, { 2, 10, 18}, { 3, 11, 19}, { 4, 12, 20}},
                      {{ 5, 13, 21}, { 6, 14, 22}, { 7, 15, 23}, { 8, 16, 24}}})
                    ->cast<Scalar>();
  auto f = Values<3>({{{ 1,  5}, { 9, 13}, {17, 21}},
                      {{ 2,  6}, {10, 14}, {18, 22}},
                      {{ 3,  7}, {11, 15}, {19, 23}},
                      {{ 4,  8}, {12, 16}, {20, 24}}})->cast<Scalar>();

  bool inplace = GENERATE(false, true);

  auto permute = [&] (NodePtr<Scalar> x, Shape s) {
    return inplace ? InPlacePermute(x * 1, s) : Permute(x, s);
  };

  check(permute(a, Shape{0, 2, 1}), b);
  check(Transpose(a, 1, 2),         b);
  check(permute(a, Shape{2, 1, 0}), c);
  check(Transpose(a, 2, 0),         c);
  check(permute(a, Shape{1, 0, 2}), d);
  check(Transpose(a, 0, 1),         d);
  check(permute(a, Shape{1, 2, 0}), e);
  check(permute(a, Shape{2, 0, 1}), f);
  check(permute(a, Shape{0, 1, 2}), a);

  CHECK_(permute(a, Shape{0, 2, 1}), {a}, true);
  CHECK_(permute(a, Shape{2, 1, 0}), {a}, true);
  CHECK_(permute(a, Shape{1, 0, 2}), {a}, true);
  CHECK_(permute(a, Shape{1, 2, 0}), {a}, true);
  CHECK_(permute(a, Shape{2, 0, 1}), {a}, true);
  CHECK_(permute(a, Shape{0, 1, 2}), {a}, true);
  CHECK_(Transpose(a, 0, 1), {a}, true);
  CHECK_(Transpose(a, 1, 0), {a}, true);
  CHECK_(Transpose(a, 0, 2), {a}, true);
  CHECK_(Transpose(a, 2, 0), {a}, true);
  CHECK_(Transpose(a, 1, 2), {a}, true);
  CHECK_(Transpose(a, 2, 1), {a}, true);
}

TEMPLATE_TEST_CASE("Broadcast", "[layout]", Real, Half, Int) {
  using Scalar = TestType;
  auto a = Values<2>({{1},
                      {2},
                      {3}})->cast<Scalar>();
  auto b = Values<2>({{0.1, 1.2, 2.3}})->cast<Scalar>();

  REQUIRE(a->shape() == Shape{3, 1});
  REQUIRE(b->shape() == Shape{1, 3});

  SECTION("RowBroadcast") {
    auto b3 = Values<2>({{0.1, 1.2, 2.3},
                         {0.1, 1.2, 2.3},
                         {0.1, 1.2, 2.3}})->cast<Scalar>();

    check(RowBroadcast(b, 3        ), b3);
    check(RowBroadcast(b, 1        ), b );
    check(RowBroadcast(b, Dim(a, 0)), b3);
    check(RowBroadcast(b, Dim(a, 1)), b );
    CHECK_(RowBroadcast(b, 3        ), {b}, true);
    CHECK_(RowBroadcast(b, 1        ), {b}, true);
    CHECK_(RowBroadcast(b, Dim(a, 0)), {b}, true);
    CHECK_(RowBroadcast(b, Dim(a, 1)), {b}, true);
  }

  SECTION("ColBroadcast") {
    auto a3 = Values<2>({{1, 1, 1},
                         {2, 2, 2},
                         {3, 3, 3}})->cast<Scalar>();

    check(ColBroadcast(a, 3        ), a3);
    check(ColBroadcast(a, 1        ), a );
    check(ColBroadcast(a, Dim(b, 1)), a3);
    check(ColBroadcast(a, Dim(b, 0)), a );
    CHECK_(ColBroadcast(a, 3        ), {a}, true);
    CHECK_(ColBroadcast(a, 1        ), {a}, true);
    CHECK_(ColBroadcast(a, Dim(b, 0)), {a}, true);
    CHECK_(ColBroadcast(a, Dim(b, 1)), {a}, true);
  }
}

TEMPLATE_TEST_CASE("UpperTri", "[layout]", Real, Int) {
  using Scalar = TestType;
  auto tri2 = Values<2>({{1, 1},
                         {0, 1}})->cast<Scalar>();
  auto tri5 = Values<2>({{1, 1, 1, 1, 1},
                         {0, 1, 1, 1, 1},
                         {0, 0, 1, 1, 1},
                         {0, 0, 0, 1, 1},
                         {0, 0, 0, 0, 1}})->cast<Scalar>();

  check(UpperTri<Scalar>(Dev, 2), tri2);
  check(UpperTri<Scalar>(Dev, 5), tri5);

#ifdef GINN_ENABLE_GPU
  // UpperTri itself is a terminal node that doesn't have the move_to() method,
  // hence the special treatment here. TODO
  check(DeviceTransfer(UpperTri<Scalar>(gpu(), 5), cpu()), tri5);
#endif
}

TEMPLATE_TEST_CASE("Add subtract", "[arithmetic]", Real, Half, Int) {
  using Scalar = TestType;

  auto a = Values<2>({{1, 4},
                      {2, 5},
                      {3, 6}})->cast<Scalar>();
  auto b = Values<2>({{-1, 4},
                      {-2, 5},
                      {-3, 6}})->cast<Scalar>();
  auto c = Values<2>({{1, -4},
                      {2, -5},
                      {3, -6}})->cast<Scalar>();

  SECTION("Add") {
    auto e = Values<2>({{0,  8},
                        {0, 10},
                        {0, 12}})->cast<Scalar>();

    check(Add(a, b), e);
    check(a + b    , e);
    CHECK_(Add(a, b), {a, b}, true);
    CHECK_(a + b,     {a, b}, true);
  }
  SECTION("Add longer") {
    auto e = Values<2>({{1, 4},
                        {2, 5},
                        {3, 6}})->cast<Scalar>();

    check(Add(a, b, c), e);
    check(a + b + c   , e);
    CHECK_(Add(a, b, c), {a, b, c}, true);
    CHECK_(a + b + c   , {a, b, c}, true);
  }
  SECTION("Add w/ repeat") {
    auto e = Values<2>({{1, 12},
                        {2, 15},
                        {3, 18}})->cast<Scalar>();

    check(Add(a, b, a), e);
    check(a + b + a   , e);
    CHECK_(Add(a, b, a), {a, b}, true);
    CHECK_(a + b + a   , {a, b}, true);
  }

  SECTION("Add scalar") {
    auto e = Values<2>({{2, 5},
                        {3, 6},
                        {4, 7}})->cast<Scalar>();

    Scalar s{1};
    check(AddScalar(a, s), e);
    check(a + s          , e);
    check(s + a          , e);
    CHECK_(AddScalar(a, s), {a}, true);
    CHECK_(a + s          , {a}, true);
    CHECK_(s + a          , {a}, true);
  }

  SECTION("Subtract") {
    auto e = Values<2>({{2, 0},
                        {4, 0},
                        {6, 0}})->cast<Scalar>();

    check(Subtract(a, b), e);
    check(a - b         , e);
    CHECK_(Subtract(a, b), {a, b}, true);
    CHECK_(a - b         , {a, b}, true);
  }

  SECTION("Subtract scalar") {
    auto e = Values<2>({{ 0, -3},
                        {-1, -4},
                        {-2, -5}})->cast<Scalar>();
    check(SubtractScalar(1, a), e);
    check(1 - a,                e);
    check(1.0 - a,              e);
    CHECK_(SubtractScalar(1, a), {a});
    CHECK_(1 - a,                {a});
    CHECK_(1.0 - a,              {a});

    auto a2 = Values<2>({{1, 4},
                         {2, 5},
                         {3, 6}})->cast<Scalar>();
    auto e2 = Values<2>({{0, 3},
                         {1, 4},
                         {2, 5}})->cast<Scalar>();
    check(AddScalar(a2, -1), e2);
    check(a2 - 1,            e2);
    check(a2 - 1.,           e2);
    CHECK_(AddScalar(a2, -1), {a2});
    CHECK_(a2 - 1,            {a2});
    CHECK_(a2 - 1.,           {a2});
  }

  SECTION("Unary -") {
    auto e = Values<2>({{-1, -4},
                        {-2, -5},
                        {-3, -6}})->cast<Scalar>();

    check(-a, e);
    CHECK_(-a, {a}, true);
  }

  SECTION("Prod scalar") {
    auto e = Values<2>({{2,  8},
                        {4, 10},
                        {6, 12}})->cast<Scalar>();

    Scalar s{2};
    check(ProdScalar(a, s), e);
    check(a * s           , e);
    check(s * a           , e);
    CHECK_(ProdScalar(a, s), {a}, true);
    CHECK_(a * s           , {a}, true);
    CHECK_(s * a           , {a}, true);
  }

  SECTION("CwiseProd") {
    auto e = Values<2>({{-1, 16 },
                        {-4, 25},
                        {-9, 36}})->cast<Scalar>();

    check(CwiseProd(a, b), e);
    CHECK_(CwiseProd(a, b), {a, b}, true);
  }

  SECTION("CwiseProdAdd") {
    Real eps = std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6;

    SECTION("Regular") {
      auto e = Values<2>({{ 0, 12 },
                          {-2, 20},
                          {-6, 30}})->cast<Scalar>();

      check(CwiseProdAdd(a, b, c), e);
      CHECK_(CwiseProdAdd(a, b, c), {a, b, c}, true, eps);
    }

    SECTION("Regular w/ bias") {
      // a * (b+1) + c
      auto e = Values<2>({{ 1, 16 },
                          { 0, 25},
                          {-3, 36}})->cast<Scalar>();

      check(CwiseProdAdd(a, b, c, Scalar(1)), e);
      CHECK_(CwiseProdAdd(a, b, c, Scalar(1)), {a, b, c}, true, eps);
    }

    SECTION("Broadcast") {
      auto b = Values<1>({-1,
                          -2,
                          -3})->cast<Scalar>();
      auto c = Values<1>({4,
                          5,
                          6})->cast<Scalar>();

      SECTION("w/o bias") {
        auto e = Values<2>({{ 3,   0},
                            { 1,  -5},
                            {-3, -12}})->cast<Scalar>();

        check(CwiseProdAdd(a, b, c), e);
        CHECK_(CwiseProdAdd(a, b, c), {a, b, c}, true);
      }

      SECTION("w/ bias") {
        auto e = Values<2>({{4,  4},
                            {3,  0},
                            {0, -6}})->cast<Scalar>();

        check(CwiseProdAdd(a, b, c, Scalar(1)), e);
        CHECK_(CwiseProdAdd(a, b, c, Scalar(1)), {a, b, c}, true);
      }
    }

    SECTION("Wrong shapes") {
      SECTION("b only column") {
        auto b = Values<1>({-1,
                            -2,
                            -3})->cast<Scalar>();
        CHECK_THROWS(Graph(CwiseProdAdd(a, b, c)).forward());
      }
      SECTION("c only column") {
        auto c = Values<1>({-1,
                            -2,
                            -3})->cast<Scalar>();
        CHECK_THROWS(Graph(CwiseProdAdd(a, b, c)).forward());
      }
      SECTION("b transposed") {
        auto b = Values<2>({{-1, -2, -3},
                            { 4,  5,  6}})->cast<Scalar>();
        CHECK_THROWS(Graph(CwiseProdAdd(a, b, c)).forward());
      }
      SECTION("Incorrect rows") {
        auto b = Values<1>({-1,
                            -2})->cast<Scalar>();
        auto c = Values<1>({4,
                            5})->cast<Scalar>();
        CHECK_THROWS(Graph(CwiseProdAdd(a, b, c)).forward());
      }
      SECTION("Rank 0") {
        auto b = Values<0>(-1)->cast<Scalar>();
        auto c = Values<0>(4)->cast<Scalar>();
        CHECK_THROWS(Graph(CwiseProdAdd(a, b, c)).forward());
      }
    }
  }

  SECTION("CwiseMax") {
    check(CwiseMax(b, c), a);
    // Warning: CwiseMax does not have derivative at points where argmax is
    // more than one. Gradcheck might fail if two maxes are within gradcheck eps.
    CHECK_(CwiseMax(a, b, c), {a, b, c}, true, 1e-6);
  }
}

TEMPLATE_TEST_CASE("Weight", "[weight]", Real, Half) {
  using Scalar = TestType;
  auto W = Weight<Scalar>(Dev, {2, 3});
  W->set_random();
  auto t = W->value();

  CHECK(W->forwarded);
  W->reset_forwarded();
  CHECK(W->forwarded);

  W->forward();
  CHECK(W->value() == t);

  W->reset_grad();

  auto W2 = W->copy(Copy::Tied);
  CHECK(&W2->value() == &W->value());
  CHECK(&W2->grad()  != &W->grad() );

  auto W3 = W->copy(Copy::Deep);
  CHECK(W3->value()  == W->value() );
  CHECK(&W3->value() != &W->value());
  CHECK(&W3->grad()  != &W->grad() );

  auto W4 = Weight(*W); // use copy ctor, should deep copy
  CHECK(W4->value()  == W->value() );
  CHECK(&W4->value() != &W->value());
  CHECK(&W4->grad()  != &W->grad() );
}

TEMPLATE_TEST_CASE("Nonlin", "[nlnode]", Real, Half) {
  using Scalar = TestType;
  auto W = Values<2>({{-1, -2, -3},
                      { 4,  5,  6}})->cast<Scalar>();
  auto tanhW = Values<2>({{-0.76159415, -0.96402758, -0.99505475},
                          { 0.99932929,  0.99990920,  0.99998771}})->cast<Scalar>();
  auto reluW = Values<2>({{0, 0, 0},
                          {4, 5, 6}})->cast<Scalar>();
  auto sigmW = Values<2>({{0.26894142, 0.11920292, 0.04742587},
                          {0.98201379, 0.99330714, 0.99752737}})->cast<Scalar>();
  auto smaxW = Values<2>({{0.00669285, 9.11051194e-04, 1.23394576e-04},
                          {0.99330715, 9.99088949e-01, 9.99876605e-01}})->cast<Scalar>();
  auto absW  = Values<2>({{1, 2, 3},
                          {4, 5, 6}})->cast<Scalar>();
  auto logaW = Values<2>({{0,          0.69314718, 1.09861229},
                          {1.38629436, 1.60943791, 1.79175947}})->cast<Scalar>();

  check(Identity(W), W    );
  check(Tanh(W),     tanhW);
  check(Relu(W),     reluW);
  check(Sigmoid(W),  sigmW, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6); // TODO: how can i make this more accurate?
  check(Softmax(W),  smaxW, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6); // TODO: how can i make this more accurate?
  check(Sqrt(CwiseProd(W, W)), absW);
  CHECK_THROWS(check(Sqrt(W), W));
  check(Log(absW),   logaW);
  // TODO: Gelu forward
  // TODO: Gelu2 forward

  CHECK_(Identity(W), {W}, true);
  CHECK_(Tanh(W),     {W}, true);
  CHECK_(Relu(W),     {W}, true);
  CHECK_(Sigmoid(W),  {W}, true);
  CHECK_(Softmax(W),  {W}, true);
  CHECK_(Sqrt(W + Scalar(3)),  {W}, true);
  CHECK_(Log(W + Scalar(1.5)), {W}, true);
  CHECK_(Gelu(W),     {W}, true, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-4);
  CHECK_(Gelu2(W),    {W}, true, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-4);
}

TEMPLATE_TEST_CASE("Nonlin Extreme", "[nlnode]", Real, Half) {
  using Scalar = TestType;
  auto x = Values<2>({{10000.}, {-10000.}})->cast<Scalar>();
  auto x2 = Values<2>({{5.}, {-std::numeric_limits<Real>::infinity()}})->cast<Scalar>();

  REQUIRE(x->shape() == Shape{2, 1});

  auto tanhx    = Values<2>({{1.}, {-1.}})->cast<Scalar>();
  auto sigmoidx = Values<2>({{1.}, {0.}})->cast<Scalar>();
  auto smaxx    = Values<2>({{1.}, {0.}})->cast<Scalar>();
  auto smaxx2   = Values<2>({{1., 1.}})->cast<Scalar>();

  REQUIRE(smaxx2->shape() == Shape{1, 2});

  check(Tanh(x),     tanhx   );
  check(Sigmoid(x),  sigmoidx);
  check(Softmax(Reshape(x, Shape{1, 2})), smaxx2  );
  check(Softmax(x),  smaxx);
  check(Softmax(x2), smaxx);

  CHECK_(Tanh(x),     {x});
  CHECK_(Sigmoid(x),  {x});
  CHECK_(Softmax(Reshape(x, Shape{1, 2})), {x});
  CHECK_(Softmax(x),  {x});
  CHECK_(Softmax(x2), {x2});
}

TEMPLATE_TEST_CASE("Pickles", "[pick][nlnode]", Real, Half) {
  using Scalar = TestType;
  auto W = Values<2>({{-0.5,  0.55, -0.45},
                      { 1. ,  2.  , -1.  },
                      { 0. ,  0.  ,  0.  },
                      { 0.3, -0.33,  1.3 }})->cast<Scalar>();
  //auto sm = Values<2>({{0.106884, 0.159876,  0.112361},
  //                     {0.47902,  0.68157,   0.0648268},
  //                     {0.176222, 0.0922404, 0.176218},
  //                     {0.237874, 0.0663138, 0.646594}});
  auto p     = Values<2>({{0.3, 0.55, -1.}})->cast<Scalar>();
  auto psm   = Values<2>({{0.23787436, 0.15987601, 0.06482681}})->cast<Scalar>();
  auto pnlsm = Values<2>({{1.43601264, 1.8333567,  2.73603603}})->cast<Scalar>();

  auto iv = std::vector<Int>{3, 0, 1};
  auto it = Values<1>({3, 0, 1})->cast<Int>();
  it->has_grad(false);

  SECTION("Pick") {
    check(Pick(W, iv), p);
    check(Pick(W, it), p);
    CHECK_(Pick(W, iv), {W},     true);
    CHECK_(Pick(W, it), {W, it}, true);
  }
  SECTION("PickSoftmax") {
    check(PickSoftmax(W, iv),   psm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(PickSoftmax(W, it),   psm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(Pick(Softmax(W), iv), psm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(Pick(Softmax(W), it), psm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    CHECK_(PickSoftmax(W, iv),   {W},     true);
    CHECK_(PickSoftmax(W, it),   {W, it}, true);
    CHECK_(Pick(Softmax(W), iv), {W},     true);
    CHECK_(Pick(Softmax(W), it), {W, it}, true);
  }
  SECTION("PickNegLogSoftmax") {
    check(PickNegLogSoftmax(W, iv),   pnlsm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(PickNegLogSoftmax(W, it),   pnlsm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(-Log(PickSoftmax(W, iv)),   pnlsm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(-Log(PickSoftmax(W, it)),   pnlsm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(Pick(-Log(Softmax(W)), iv), pnlsm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    check(Pick(-Log(Softmax(W)), it), pnlsm, std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6);
    CHECK_(PickNegLogSoftmax(W, iv),   {W},     true);
    CHECK_(PickNegLogSoftmax(W, it),   {W, it}, true);
    CHECK_(-Log(PickSoftmax(W, iv)),   {W},     true);
    CHECK_(-Log(PickSoftmax(W, it)),   {W, it}, true);
    CHECK_(Pick(-Log(Softmax(W)), iv), {W},     true);
    CHECK_(Pick(-Log(Softmax(W)), it), {W, it}, true);
  }
}

TEMPLATE_TEST_CASE("PickNegLogSigmoid", "[pick][nlnode]", Real, Half) {
  using Scalar = TestType;
  auto W = Values<2>({{-0.5,  0.55, -0.45},
                      { 1. ,  2.  , -1.  },
                      { 0. ,  0.  ,  0.  },
                      { 0.3, -0.33,  1.3 }})->cast<Scalar>();
  auto p = Values<2>({{0, 1, 0},
                      {1, 0, 1},
                      {0, 1, 0},
                      {1, 0, 1}})->cast<Int>();
  p->has_grad(false);
  auto pnls = Values<2>({{0.47407698, 0.45549248, 0.49324895},
                         {0.31326169, 2.12692801, 1.31326169},
                         {0.69314718, 0.69314718, 0.69314718},
                         {0.55435524, 0.54169836, 0.24100845}})->cast<Scalar>();

  check(PickNegLogSigmoid(W, p), pnls, std::is_same_v<Scalar, Half> ? 2e-3 : 1e-6);
  CHECK_(PickNegLogSigmoid(W, p), {W, p}, true);
}

TEMPLATE_TEST_CASE("Select", "[select]", Real, Half, Int) {
  using Scalar = TestType;
  auto if_ = Values<2>({{1., 0.},
                        {0., 1.},
                        {1., 0.}})->cast<bool>();
  if_->has_grad(false); // TODO: maybe integral nodes should have this default?
  auto a = Values<2>({{1., 2.},
                      {3., 4.},
                      {5., 6.}})->cast<Scalar>();
  auto b = Values<2>({{.1, .2},
                      {.3, .4},
                      {.5, .6}})->cast<Scalar>();

  auto y1 = Values<2>({{1., .2},
                       {.3, 4.},
                       {5., .6}})->cast<Scalar>();
  auto y2 = Values<2>({{1., 7.},
                       {7., 4.},
                       {5., 7.}})->cast<Scalar>();
  auto y3 = Values<2>({{7., .2},
                       {.3, 7.},
                       {7., .6}})->cast<Scalar>();
  auto y4 = Values<2>({{7., -2},
                       {-2, 7.},
                       {7., -2}})->cast<Scalar>();


  check(Select(if_, a, b),                  y1);
  check(Select(if_, a, Scalar(7)),          y2);
  check(Select(if_, Scalar(7), b),          y3);
  check(Select(if_, Scalar(7), Scalar(-2)), y4);
  CHECK_(Select(if_, a, b),                  {if_, a, b});
  CHECK_(Select(if_, a, Scalar(7)),          {if_, a});
  CHECK_(Select(if_, Scalar(7), b),          {if_, b});
  CHECK_(Select(if_, Scalar(7), Scalar(-2)), {if_});
}

TEMPLATE_TEST_CASE("Mask", "[select]", Real, Half, Int) {
  using Scalar = TestType;
  auto mask = Values<2>({{1., 0.},
                         {0., 1.},
                         {1., 0.}})->cast<Scalar>();
  mask->has_grad(false);
  auto mask2 = Values<2>({{1.},
                          {0.},
                          {1.}})->cast<Scalar>();
  mask2->has_grad(false);
  auto a = Values<2>({{1., 2.},
                      {3., 4.},
                      {5., 6.}})->cast<Scalar>();

  SECTION("Forward") {
    auto y1 = Values<2>({{ 1., -1.},
                         {-1.,  4.},
                         { 5., -1.}})->cast<Scalar>();
    auto y2 = Values<2>({{1., 2.},
                         {7., 7.},
                         {5., 6.}})->cast<Scalar>();

    check(Mask(a, mask,  Scalar(-1)), y1);
    check(Mask(a, mask2, Scalar(7)),  y2);
  }

  SECTION("Grad or cuda") {
    CHECK_(Mask(a, mask,  Scalar(-1)), {a, mask });
    CHECK_(Mask(a, mask2, Scalar(7) ), {a, mask2});

    auto W  = Random(Dev, {2, 3, 2})->cast<Scalar>();
    auto m1 = Random(Dev, {2, 3, 2})->cast<Scalar>();
    auto m2 = Random(Dev, {2, 3, 1})->cast<Scalar>();
    auto m3 = Random(Dev, {2, 1, 1})->cast<Scalar>();

    auto set_mask = [](auto& m) {
      m->set_random();
      m->value() = (m->value().t() > Scalar(0)).template cast<Scalar>();
      m->has_grad(false);
    };

    set_mask(m1);
    set_mask(m2);
    set_mask(m3);

    CHECK_(Mask(W, m1, Scalar( 3.)), {W, m1, m2, m3});
    CHECK_(Mask(W, m2, Scalar(-3.)), {W, m1, m2, m3});
    CHECK_(Mask(W, m3, Scalar( 0.)), {W, m1, m2, m3});
  }
}

TEMPLATE_TEST_CASE("LayerNorm", "[layernorm][inplace]", Real, Half) {
  // TODO: forward test
  using Scalar = TestType;
  auto x = Random(Dev, {3, 2})->cast<Scalar>();
  auto y = Random(Dev, {3, 2, 4})->cast<Scalar>();
  x->value() = x->value().t() * Scalar(3) + Scalar(2);
  y->value() = y->value().t() * Scalar(2.5) - Scalar(2);

  #ifdef GINN_ENABLE_GPU
  auto eps2 = std::is_same_v<Scalar, Half> ? 2e-2 : 1e-4;
  #else
  auto eps2 = 1e-4;
  #endif

  CHECK_(LayerNorm(x), {x}, false, eps2);
  CHECK_(LayerNorm(y), {y}, false, eps2);
  CHECK_(InPlaceLayerNorm(1 * x), {x}, false, eps2);
  CHECK_(InPlaceLayerNorm(1 * y), {y}, false, eps2);
  //SECTION("Layer") {
  //  auto ln1 = LayerNormLayer();
  //  auto ln2 = LayerNormLayer(Dev, 3);
  //  ln2->gamma->value().set_random();
  //  ln2->gamma->value() = ln2->gamma->value().t() + Real(1);
  //  ln2->beta->value().set_zero();
  //  CHECK_(ln1->run(x), {x});
  //  CHECK_(ln2->run(x), {x, ln2->gamma, ln2->beta});
  //}
}

TEMPLATE_TEST_CASE("Sums", "[reduce]", Real, Half) {
  using Scalar = TestType;
  auto W = Values<2>({{1, 2, 3},
                      {4, 5, 6}})->cast<Scalar>();
  SECTION("Sum") {
    auto Wsum = Values<0>(21)->cast<Scalar>();
    check(Sum(W), Wsum);
  }

  SECTION("AxisSum") {
    auto V    = Values<3>({{{1, 2, 3},
                            {4, 5, 6}},
                           {{.1, .2, .3},
                            {.4, .5, .6}}})->cast<Scalar>();
    auto V0   = Values<2>({{1.1, 2.2, 3.3},
                           {4.4, 5.5, 6.6}})->cast<Scalar>();
    auto V01  = Values<1>({5.5, 7.7, 9.9})->cast<Scalar>();
    auto V012 = Values<0>(23.1)->cast<Scalar>();
    auto V1   = Values<2>({{5 , 7 , 9 },
                           {.5, .7, .9}})->cast<Scalar>();
    auto V12  = Values<1>({21,
                           2.1})->cast<Scalar>();
    auto V2   = Values<2>({{ 6,
                            15},
                           {.6,
                            1.5}})->cast<Scalar>();
    auto V02  = Values<1>({ 6.6,
                           16.5})->cast<Scalar>();

    Real eps = std::is_same_v<Scalar, Half> ? 1e-3 : 1e-6;
    check(AxisSum(V, {0}),       V0,   eps);
    check(AxisSum(V, {0, 1}),    V01,  eps);
    check(AxisSum(V, {0, 1, 2}), V012, eps);
    check(AxisSum(V, {1}),       V1,   eps);
    check(AxisSum(V, {1, 2}),    V12,  eps);
    check(AxisSum(V, {2}),       V2,   eps);
    check(AxisSum(V, {0, 2}),    V02,  eps);
    CHECK_THROWS(AxisSum(V, {0, 0}));
    CHECK_THROWS(AxisSum(V, {2, 0}));
    CHECK_(AxisSum(V, {0}),       {V}, true);
    CHECK_(AxisSum(V, {0, 1}),    {V}, true);
    CHECK_(AxisSum(V, {0, 1, 2}), {V}, true);
    CHECK_(AxisSum(V, {1}),       {V}, true);
    CHECK_(AxisSum(V, {1, 2}),    {V}, true);
    CHECK_(AxisSum(V, {2}),       {V}, true);
    CHECK_(AxisSum(V, {0, 2}),    {V}, true);
  }

  SECTION("Mean") {
    auto m = Values<0>(3.5)->cast<Scalar>();

    check(Mean(W), m);
    CHECK_(Mean(W), {W}, true);
  }

  SECTION("Variance") {
    auto m = Values<0>(35./12.)->cast<Scalar>();

    check(Variance(W), m);
    CHECK_(Variance(W), {W}, true);
  }
}

TEMPLATE_TEST_CASE("Comparison", "[compare]", Real, Half, Int) {
  using Scalar = TestType;
  auto a = Values<2>({{1.}, {2.}, {3.}, {4.}, {5.}})->cast<Scalar>();
  auto b = Values<2>({{5.}, {4.}, {3.}, {2.}, {1.}})->cast<Scalar>();

  REQUIRE(a->shape() == Shape{5, 1});
  REQUIRE(b->shape() == Shape{5, 1});

  auto y = Values<2>({{1.}, {1.}, {0.}, {0.}, {0.}})->cast<bool>();

  SECTION("LessThan") {
    check(LessThan(a, b), y);
    check(a < b,          y);
    CHECK_(LessThan(a, b), {a, b}, true);
    CHECK_(a < b,          {a, b}, true);
  }
}

TEMPLATE_TEST_CASE("Prod", "[prod]", Real, Half) {
  using Scalar = TestType;
  auto W = Values<2>({{1, 2, 3},
                      {4, 5, 6}})->cast<Scalar>();
  auto V = Values<2>({{ 0.6,  0.5},
                      { 0.4, -0.1},
                      {-0.2, -0.3}})->cast<Scalar>();

  auto WV = Values<2>({{0.8, -0.6},
                       {3.2, -0.3}})->cast<Scalar>();

  Real eps = std::is_same_v<Scalar, Half> ? 1e-2 : 1e-6; // TODO: 😢
  Real eps2 = std::is_same_v<Scalar, Half> ? 1e-2 : 1e-4;

  check(Prod(W, V), WV, eps);
  check(W * V,      WV, eps);
  CHECK_(Prod(W, V), {W, V}, true, eps2);
  CHECK_(W * V,      {W, V}, true, eps2);
}

TEMPLATE_TEST_CASE("BatchedProd", "[prod]", Real, Half) {
  using Scalar = TestType;

  SECTION("Basic") {
    auto a = Random({2, 3, 4})->cast<Scalar>();
    auto b = Random({3, 5, 4})->cast<Scalar>();
    a->value() = a->value().t() + Scalar(1.);
    b->value() = b->value().t() - Scalar(1.);
    auto c = BatchedProd(a, b);

    auto c0 = Chip(c, 0, 2);
    auto c1 = Chip(c, 1, 2);
    auto c2 = Chip(c, 2, 2);
    auto c3 = Chip(c, 3, 2);

    auto a0 = Chip(a, 0, 2), b0 = Chip(b, 0, 2);
    auto a1 = Chip(a, 1, 2), b1 = Chip(b, 1, 2);
    auto a2 = Chip(a, 2, 2), b2 = Chip(b, 2, 2);
    auto a3 = Chip(a, 3, 2), b3 = Chip(b, 3, 2);

    auto c0_ = a0 * b0;
    auto c1_ = a1 * b1;
    auto c2_ = a2 * b2;
    auto c3_ = a3 * b3;

    Real eps = std::is_same_v<Scalar, Half> ? 1e-2 : 1e-6; // TODO: 😢
    Real eps2 = std::is_same_v<Scalar, Half> ? 1e-2 : 1e-4;

    check(c0, c0_, eps);
    check(c1, c1_, eps);
    check(c2, c2_, eps);
    check(c3, c3_, eps);

    CHECK_(BatchedProd(a, b), {a, b}, false, eps2);
  }

  SECTION("High rank") {
    auto a = Random({2, 3, 2, 2})->cast<Scalar>();
    auto b = Random({3, 5, 2, 2})->cast<Scalar>();
    a->value() = a->value().t() + Scalar(1.);
    b->value() = b->value().t() - Scalar(1.);
    auto c = BatchedProd(a, b);

    auto ChipTwice = [](auto x, Size i, Size j) {
      return Chip(Chip(x, j, 3), i, 2);
    };

    auto c00 = ChipTwice(c, 0, 0);
    auto c01 = ChipTwice(c, 0, 1);
    auto c10 = ChipTwice(c, 1, 0);
    auto c11 = ChipTwice(c, 1, 1);

    auto a00 = ChipTwice(a, 0, 0), b00 = ChipTwice(b, 0, 0);
    auto a01 = ChipTwice(a, 0, 1), b01 = ChipTwice(b, 0, 1);
    auto a10 = ChipTwice(a, 1, 0), b10 = ChipTwice(b, 1, 0);
    auto a11 = ChipTwice(a, 1, 1), b11 = ChipTwice(b, 1, 1);

    auto c00_ = a00 * b00;
    auto c01_ = a01 * b01;
    auto c10_ = a10 * b10;
    auto c11_ = a11 * b11;

    Real eps = std::is_same_v<Scalar, Half> ? 1e-2 : 1e-6; // TODO: 😢
    Real eps2 = std::is_same_v<Scalar, Half> ? 1e-2 : 1e-4;

    check(c00, c00_, eps);
    check(c01, c01_, eps);
    check(c10, c10_, eps);
    check(c11, c11_, eps);

    CHECK_(BatchedProd(a, b), {a, b}, false, eps2);
  }
}

TEMPLATE_TEST_CASE("Affine", "[affine]", Real, Half) {
  using Scalar = TestType;
  Real eps = std::is_same_v<Scalar, Half> ? 2e-3 : 1e-6; // TODO: 😢
  Real eps2 = std::is_same_v<Scalar, Half> ? 2e-3 : 1e-4;

  auto W = Values<2>({{1, 2, 3},
                      {4, 5, 6}})->cast<Scalar>();
  auto V = Values<2>({{ 0.6},
                      { 0.4},
                      {-0.2}})->cast<Scalar>();
  auto b = Values<2>({{0.01},
                      {0.02}})->cast<Scalar>();

  SECTION("Affine")        {
    auto WVb = Values<2>({{0.81},
                          {3.22}})->cast<Scalar>();
    check(Affine(W, V, b), WVb, eps);
    check(W * V + b,       WVb, eps);
    CHECK_(Affine(W, V, b), {W, V, b}, true, eps2);
  }
  SECTION("AffineSigmoid") {
    auto sigmWVb = Values<2>({{0.692109},
                              {0.96158 }})->cast<Scalar>();
    check(Affine(SigmoidOp<Scalar>(), W, V, b), sigmWVb, eps);
    check(Affine<SigmoidOp>(W, V, b),           sigmWVb, eps);
    CHECK_(Affine(SigmoidOp<Scalar>(), W, V, b), {W, V, b}, true, eps2);
    CHECK_(Affine<SigmoidOp>(W, V, b),           {W, V, b}, true, eps2);
  }
  SECTION("AffineSoftmax") {
    auto smaxWVb = Values<2>({{0.0824133},
                              {0.917587 }})->cast<Scalar>();
    check(Affine<SoftmaxOp>(W, V, b), smaxWVb, eps);
    CHECK_(Affine<SoftmaxOp>(W, V, b), {W, V, b}, true, eps2);
  }

  SECTION("Other nonlins") {
    CHECK_(Affine<TanhOp>(W, V, b), {W, V, b}, true, eps2);
    CHECK_(Affine<ReluOp>(W * 10, V, b), {W, V, b}, true, eps2);
    CHECK_(Affine<Gelu2Op>(W, V, b), {W, V, b}, true, eps2);
    CHECK_(Affine<GeluOp>(W, V, b), {W, V, b}, true, eps2);
  }
}

TEMPLATE_TEST_CASE("Affine w/ broadcast", "[affine]", Real, Half) {
  using Scalar = TestType;
  Real eps = std::is_same_v<Scalar, Half> ? 2e-3 : 1e-6; // TODO: 😢
  Real eps2 = std::is_same_v<Scalar, Half> ? 2e-3 : 1e-4;
  auto W = Values<2>({{1, 2, 3},
                      {4, 5, 6}})->cast<Scalar>();
  auto V = Values<2>({{ 6,  5},
                      { 4, -1},
                      {-2, -3}})->cast<Scalar>();
  auto b = Values<2>({{0.1},
                      {0.2}})->cast<Scalar>();

  auto WVb = Values<2>({{ 8.1, -5.9},
                        {32.2, -2.8}})->cast<Scalar>();

  SECTION("Affine") {
    check(Affine(W, V, b), WVb, eps);
    CHECK_(Affine(W, V, b), {W, V, b}, true, eps2);
  }
}

TEMPLATE_TEST_CASE("Affine w/ high rank", "[affine]", Real, Half) {
  using Scalar = TestType;
  Real eps = std::is_same_v<Scalar, Half> ? 2e-3 : 1e-6; // TODO: 😢
  Real eps2 = std::is_same_v<Scalar, Half> ? 2e-3 : 1e-4;
  auto W = Values<2>({{1, 2, 3}, // 2 x 3
                      {4, 5, 6}})->cast<Scalar>();
  auto V = Values<3>({{{ 6,  5},{ 6,  5}}, // 3 x 2 x 2
                      {{ 4, -1},{ 4, -1}},
                      {{-2, -3},{-2, -3}}})->cast<Scalar>();
  auto b = Values<1>({0.1, // 2
                      0.2})->cast<Scalar>();

  auto WVb = Values<3>({{{ 8.1, -5.9},{ 8.1, -5.9}}, // 2 x 2 x 2
                        {{32.2, -2.8},{32.2, -2.8}}})->cast<Scalar>();

  SECTION("Affine") {
    check(Affine(W, V, b), WVb, eps);
    CHECK_(Affine(W, V, b), {W, V, b}, true, eps2);
  }
}

TEMPLATE_TEST_CASE("InPlaceAdd", "[inplace][arithmetic]", Real, Half) {
  using Scalar = TestType;
  auto W = Random(Dev, {2, 3})->cast<Scalar>();
  auto V = Random(Dev, {2, 3})->cast<Scalar>();
  auto U = Random(Dev, {2, 3})->cast<Scalar>();

  auto a1 = Add(W, V, U);
  auto a2 = Add(W, V, V);
  Graph(a1).forward();
  Graph(a2).forward();

  // Need one level of indirection because of how gradcheck perturbs values and
  //   how in-place nodes reuse value tensors. Hence the multiply by 1. This
  //   only applies to testing.
  check(InPlaceAdd(W * 1, V, U), a1);
  check(InPlaceAdd(W * 1, V, V), a2);
  CHECK_(InPlaceAdd(W * 1, V, U), {W, V, U});
  CHECK_(InPlaceAdd(W * 1, V, V), {W, V});
}

TEMPLATE_TEST_CASE("InPlaceAddScalar", "[inplace][arithmetic]", Real, Half) {
  using Scalar = TestType;
  auto W = Random(Dev, {2, 3})->cast<Scalar>();

  auto a1 = W + 3.14;
  auto a2 = W - 5;
  Graph(a1).forward();
  Graph(a2).forward();

  check(InPlaceAddScalar(1 * W, 3.14), a1);
  check(InPlaceAddScalar(1 * W, -5), a2);
  CHECK_(InPlaceAddScalar(1 * W, 3.14), {W});
  CHECK_(InPlaceAddScalar(1 * W, -5), {W});
}

TEMPLATE_TEST_CASE("InPlaceCwiseProd", "[inplace][arithmetic]", Real, Half) {
  using Scalar = TestType;
  auto W = Random(Dev, {2, 3, 2})->cast<Scalar>();
  auto V = Random(Dev, {2, 3, 2})->cast<Scalar>();

  auto p = CwiseProd(W, V);
  Graph(p).forward();

  check(InPlaceCwiseProd(1 * W, V), p);
  CHECK_(InPlaceCwiseProd(1 * W, V), {W, V});
}

TEMPLATE_TEST_CASE("InPlaceProdScalar", "[inplace][arithmetic]", Real, Half) {
  using Scalar = TestType;
  auto W = Random(Dev, {2, 3})->cast<Scalar>();

  auto p1 = W * 3.14;
  auto p2 = W * -5;
  Graph(p1).forward();
  Graph(p2).forward();

  check(InPlaceProdScalar(1 * W, 3.14), p1);
  check(InPlaceProdScalar(1 * W, -5), p2);
  CHECK_(InPlaceProdScalar(1 * W, 3.14), {W});
  CHECK_(InPlaceProdScalar(1 * W, -5), {W});
}

TEMPLATE_TEST_CASE("InPlaceMask", "[inplace]", Real, Half) {
  using Scalar = TestType;
  auto W = Random(Dev, {2, 3, 2})->cast<Scalar>();
  auto m1 = Random(Dev, {2, 3, 2})->cast<Scalar>();
  auto m2 = Random(Dev, {2, 3, 1})->cast<Scalar>();
  auto m3 = Random(Dev, {2, 1, 1})->cast<Scalar>();

  auto set_mask = [](auto& m) {
    m->set_random();
    m->value() = (m->value().t() > Scalar(0)).template cast<Scalar>();
    m->has_grad(false);
  };

  set_mask(m1);
  set_mask(m2);
  set_mask(m3);

  CHECK_(InPlaceMask(W * 1, m1, 3.), {W, m1, m2, m3});
  CHECK_(InPlaceMask(W * 1, m2, -3.), {W, m1, m2, m3});
  CHECK_(InPlaceMask(W * 1, m3, 0.), {W, m1, m2, m3});
}

TEMPLATE_TEST_CASE("InPlaceSigmoid", "[inplace][nlnode]", Real, Half) {
  using Scalar = TestType;
  auto W = Random(Dev, {2, 3})->cast<Scalar>();

  auto s = Sigmoid(W);
  Graph(s).forward();

  check(InPlaceSigmoid(1 * W), s);
  CHECK_(InPlaceSigmoid(1 * W), {W});
}

// clang-format on
