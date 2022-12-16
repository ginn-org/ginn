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

#include <catch.hpp>

#include <algorithm>
#include <iostream>
#include <tuple>

#include <ginn/eigenop/helpers.h>
#include <ginn/tensor.h>
#ifdef GINN_ENABLE_GPU
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#endif

using namespace ginn;
using namespace ginn::literals;

// Different ways of making a tensor since each one of them is tested with the
// same body
#define TENSOR(t, ...) TensorType t(__VA_ARGS__)
#define COPIED_TENSOR(t, ...)                                                  \
  TensorType t_(__VA_ARGS__);                                                  \
  TensorType t(t_)
#define MOVED_TENSOR(t, ...)                                                   \
  TensorType t_(__VA_ARGS__);                                                  \
  TensorType t(std::move(t_))
#define GPU_ASSIGNED_TENSOR(t, ...)                                            \
  TensorType t_(__VA_ARGS__);                                                  \
  TensorType t(gpu());                                                         \
  t = t_
#define CPU_ASSIGNED_TENSOR(t, ...)                                            \
  TensorType t_(__VA_ARGS__);                                                  \
  TensorType t(cpu());                                                         \
  t = t_

// Take one of the tensor makers above and test it with this body
#define TEST_CTOR(TENSOR_CTOR, name, init_dev, DEV_TYPE)                       \
  TEMPLATE_TEST_CASE(                                                          \
      name, "[tensor]", Tensor<Real>, Tensor<Int>, Tensor<Half>) {             \
    using TensorType = TestType;                                               \
    using Scalar = typename TensorType::Scalar;                                \
                                                                               \
    SECTION("Device") {                                                        \
      TENSOR_CTOR(t, init_dev);                                                \
      CHECK(t.dev()->kind() == DEV_TYPE);                                      \
      CHECK(t.size() == 0);                                                    \
    }                                                                          \
                                                                               \
    SECTION("Shape") {                                                         \
      TENSOR_CTOR(t, init_dev, {2, 1, 3});                                     \
      CHECK(t.dev()->kind() == DEV_TYPE);                                      \
      CHECK(t.size() == 6);                                                    \
      CHECK(t.shape().size() == 3);                                            \
    }                                                                          \
                                                                               \
    SECTION("Value") {                                                         \
      static_assert(std::is_same_v<Scalar, Real> or                            \
                        std::is_same_v<Scalar, Half> or                        \
                        std::is_same_v<Scalar, Int>,                           \
                    "Unexpected Scalar type!");                                \
      Scalar val;                                                              \
      if constexpr (std::is_same_v<Scalar, Real>) {                            \
        val = 0.6;                                                             \
      } else if constexpr (std::is_same_v<Scalar, Int>) {                      \
        val = 4;                                                               \
      } else if constexpr (std::is_same_v<Scalar, Half>) {                     \
        val = Half{0.6};                                                       \
      }                                                                        \
      TENSOR_CTOR(t, init_dev, {2, 1, 3}, val);                                \
      CHECK(t.dev()->kind() == DEV_TYPE);                                      \
      CHECK(t.size() == 6);                                                    \
      CHECK(t.shape().size() == 3);                                            \
      t.move_to(cpu());                                                        \
      for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == val); }          \
    }                                                                          \
  }

TEST_CTOR(TENSOR, "Ctors", cpu(), CPU)
TEST_CTOR(COPIED_TENSOR, "Copy", cpu(), CPU)
TEST_CTOR(MOVED_TENSOR, "Move", cpu(), CPU)
TEST_CTOR(CPU_ASSIGNED_TENSOR, "Cpu Assign", cpu(), CPU)
#ifdef GINN_ENABLE_GPU
TEST_CTOR(TENSOR, "Gpu Ctors", gpu(), GPU)
TEST_CTOR(COPIED_TENSOR, "Gpu Copy", gpu(), GPU)
TEST_CTOR(MOVED_TENSOR, "Gpu Move", gpu(), GPU)
TEST_CTOR(GPU_ASSIGNED_TENSOR, "Gpu Gpu Assign", gpu(), GPU)
TEST_CTOR(CPU_ASSIGNED_TENSOR, "Gpu Cpu Assign", gpu(), CPU)
#endif

#ifdef GINN_ENABLE_GPU
using Typelist = std::tuple<std::tuple<Tensor<Real>, GpuDevice>,
                            std::tuple<Tensor<Int>, GpuDevice>,
                            std::tuple<Tensor<Half>, GpuDevice>,
                            std::tuple<Tensor<Real>, CpuDevice>,
                            std::tuple<Tensor<Int>, CpuDevice>,
                            std::tuple<Tensor<Half>, CpuDevice>>;
#else
using Typelist = std::tuple<std::tuple<Tensor<Real>, CpuDevice>,
                            std::tuple<Tensor<Int>, CpuDevice>,
                            std::tuple<Tensor<Half>, CpuDevice>>;
#endif

TEMPLATE_LIST_TEST_CASE("Resize", "[tensor]", Typelist) {
  using TensorType = typename std::tuple_element<0, TestType>::type;
  using DeviceType = typename std::tuple_element<1, TestType>::type;
  using Scalar = typename TensorType::Scalar;

  auto dev = std::make_shared<DeviceType>();
  TensorType t(dev);
  if constexpr (std::is_same_v<Scalar, Half>) {
    t = TensorType(dev, {2, 1, 3}, {1_h, 2_h, 3_h, 4_h, 5_h, 6_h});
  } else {
    t = TensorType(dev, {2, 1, 3}, {1, 2, 3, 4, 5, 6});
  }

  SECTION("Same size vector") {
    t.resize({6});
    CHECK(t.size() == 6);
    CHECK(t.shape() == Shape{6});
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == (i + 1)); }
  }

  SECTION("Same size matrix") {
    t.resize({2, 3});
    CHECK(t.size() == 6);
    CHECK(t.shape() == Shape{2, 3});
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == (i + 1)); }
  }

  SECTION("Different size vector") {
    t.resize({4});
    CHECK(t.size() == 4);
    CHECK(t.shape() == Shape{4});
  }

  SECTION("Different size matrix") {
    t.resize({2, 2});
    CHECK(t.size() == 4);
    CHECK(t.shape() == Shape{2, 2});
  }
}

TEMPLATE_LIST_TEST_CASE("View", "[tensor]", Typelist) {
  using TensorType = typename std::tuple_element<0, TestType>::type;
  using DeviceType = typename std::tuple_element<1, TestType>::type;
  using Scalar = typename TensorType::Scalar;

  // TODO: this is hideous
  auto equals = [](const auto& a, const auto& b, auto dev_kind) -> bool {
    if (dev_kind == CPU) { return Eigen::Tensor<bool, 0>((a == b).all())(0); }
#ifdef GINN_ENABLE_GPU
    if (eigen::ndims<std::decay_t<decltype(a)>>() !=
        eigen::ndims<std::decay_t<decltype(b)>>()) {
      return false;
    }
    return thrust::equal(thrust::device_ptr<Scalar>(a.data()),
                         thrust::device_ptr<Scalar>(a.data()) + a.size(),
                         thrust::device_ptr<Scalar>(b.data()));
#endif
    return false;
  };

  auto dev = std::make_shared<DeviceType>();
  TensorType t(dev);
  if constexpr (std::is_same_v<Scalar, Half>) {
    t = TensorType(dev, {2, 1, 3}, {1_h, 2_h, 3_h, 4_h, 5_h, 6_h});
  } else {
    t = TensorType(dev, {2, 1, 3}, {1, 2, 3, 4, 5, 6});
  }

  if (dev->kind() == CPU) {
    CHECK(t.v() == VectorMap<Scalar>(t.data(), 6));
    CHECK(t.m() == MatrixMap<Scalar>(t.data(), 2, 3));
  } else {
    CHECK_THROWS(t.v());
    CHECK_THROWS(t.m());
  }

  CHECK(equals(
      t.template view<1>(), TensorMap<Scalar, 1>(t.data(), 6), dev->kind()));

  CHECK(equals(t.t(), TensorMap<Scalar, 2>(t.data(), 2, 3), dev->kind()));

  CHECK(equals(t.template view<3>(),
               TensorMap<Scalar, 3>(t.data(), 2, 1, 3),
               dev->kind()));

  CHECK(equals(t.template view<4>(),
               TensorMap<Scalar, 4>(t.data(), 2, 1, 3, 1),
               dev->kind()));

  CHECK(equals(t.template view<5>(),
               TensorMap<Scalar, 5>(t.data(), 2, 1, 3, 1, 1),
               dev->kind()));
}

TEMPLATE_LIST_TEST_CASE("Set values", "[tensor]", Typelist) {
  using TensorType = typename std::tuple_element<0, TestType>::type;
  using DeviceType = typename std::tuple_element<1, TestType>::type;
  using Scalar = typename TensorType::Scalar;

  auto dev = std::make_shared<DeviceType>();
  TensorType t(dev);
  if constexpr (std::is_same_v<Scalar, Half>) {
    t = TensorType(dev, {2, 1, 3}, {1_h, 2_h, 3_h, 4_h, 5_h, 6_h});
  } else {
    t = TensorType(dev, {2, 1, 3}, {1, 2, 3, 4, 5, 6});
  }

  SECTION("Zero") {
    t.set_zero();
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == 0); }
  }

  SECTION("Ones") {
    t.set_ones();
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == 1); }
  }

  SECTION("Constant") {
    static_assert(std::is_same_v<Scalar, Real> or std::is_same_v<Scalar, Int> or
                      std::is_same_v<Scalar, Half>,
                  "Unexpected Scalar type!");
    Scalar val;
    if constexpr (std::is_same_v<Scalar, Real>) {
      val = 0.12345;
    } else if constexpr (std::is_same_v<Scalar, Int>) {
      val = 12345;
    } else if constexpr (std::is_same_v<Scalar, Half>) {
      val = 0.12345_h;
    }
    t.fill(val);
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == val); }
  }

  SECTION("Random") {
    if constexpr (std::is_same_v<Real, Scalar>) {
      t.set_random();
      t.move_to(cpu());
      for (Size i = 0; i < t.size(); i++) {
        CHECK(t.v()[i] <= 1.);
        CHECK(t.v()[i] >= -1.);
        if (i > 0) { CHECK(t.v()[i] != t.v()[i - 1]); }
      }
    }
  }

  SECTION("Values") {
    t.set(5, 4, 3, 2, 1, 0);
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == 5 - i); }
  }

  SECTION("Longer") {
    t.set(5, 4, 3, 2, 1, 0, 1, 2, 3);
    t.move_to(cpu());
    for (Size i = 0; i < t.size(); i++) { CHECK(t.v()[i] == 5 - i); }
  }

  SECTION("Shorter") {
    t.set(5, 4, 3);
    t.move_to(cpu());
    CHECK(t.v()[0] == 5);
    CHECK(t.v()[1] == 4);
    CHECK(t.v()[2] == 3);
    CHECK(t.v()[3] == 4);
    CHECK(t.v()[4] == 5);
    CHECK(t.v()[5] == 6);
  }
}

TEMPLATE_LIST_TEST_CASE("Serialize", "[tensor]", Typelist) {
  using TensorType = typename std::tuple_element<0, TestType>::type;
  using DeviceType = typename std::tuple_element<1, TestType>::type;
  using Scalar = typename TensorType::Scalar;

  auto dev = std::make_shared<DeviceType>();
  TensorType t(dev), t2(dev);
  if constexpr (std::is_same_v<Scalar, Half>) {
    t = TensorType(dev, {2, 1, 3}, {1_h, 2_h, 3_h, 4_h, 5_h, 6_h});
  } else {
    t = TensorType(dev, {2, 1, 3}, {1, 2, 3, 4, 5, 6});
  }

  std::ostringstream out;
  t.save(out);

  std::istringstream in(out.str());
  t2.load(in);

  CHECK(t == t2);
}

TEMPLATE_LIST_TEST_CASE("Errors", "[tensor]", Typelist) {
  using TensorType = typename std::tuple_element<0, TestType>::type;
  using DeviceType = typename std::tuple_element<1, TestType>::type;
  using Scalar = typename TensorType::Scalar;

  auto dev = std::make_shared<DeviceType>();
  TensorType t1(dev, {1, 2}), t2(dev, {1, 3}), t3(dev);

  SECTION("Misshaped map") {
    t3.map(t1, {2, 1});
    CHECK_THROWS_AS(t3 = t2, ginn::RuntimeError);
    CHECK_THROWS_AS(t3.map(t2, {2, 1}), ginn::RuntimeError);
  }

  SECTION("Misshaped value ctor") {
    CHECK_THROWS_MATCHES(t3 = TensorType({1, 3}, {Scalar(1), Scalar(2)}),
                         ginn::RuntimeError,
                         Catch::Message("Size of Shape (3) does not match "
                                        "size of values (2)!"));
  }
}
