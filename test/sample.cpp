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
#include <cstdlib>
#include <vector>

#include <ginn/except.h>
#include <ginn/util/sample.h>

using namespace ginn;

std::vector<double> sample_dist(AliasSampler sampler, size_t n = 10000000) {
  std::vector<double> dist(sampler.num_classes(), 0.0);

  for (size_t i = 0; i < n; ++i) { ++dist[sampler()]; }

  for (size_t i = 0; i < dist.size(); ++i) { dist[i] /= n; }

  return dist;
}

bool dists_are_close(const std::vector<double>& d1,
                     const std::vector<double>& d2) {
  GINN_ASSERT(d1.size() == d2.size());

  for (size_t i = 0; i < d1.size(); ++i) {
    if (std::abs(d1[i] - d2[i]) >= (d1[i] * 0.01)) { return false; }
  }

  return true;
}

TEST_CASE("AliasSampler",
          "check alias sampling matches ground truth distribution") {
  std::vector<double> probs1(2, 0.5);
  AliasSampler sampler1(probs1);

  SECTION("Balanced binary distribution") {
    CHECK(dists_are_close(probs1, sample_dist(sampler1)));
  }

  std::vector<double> probs2(10, 0.1);
  AliasSampler sampler2(probs2);

  SECTION("Balanced 10-class distribution") {
    CHECK(dists_are_close(probs2, sample_dist(sampler2)));
  }

  std::vector<double> probs3(50, 0.02);
  AliasSampler sampler3(probs3);

  SECTION("Balanced 50-class distribution") {
    CHECK(dists_are_close(probs3, sample_dist(sampler3)));
  }

  std::vector<double> probs4{0.1, 0.9};
  AliasSampler sampler4(probs4);

  SECTION("Unbalanced binary distribution") {
    CHECK(dists_are_close(probs4, sample_dist(sampler4)));
  }

  std::vector<double> probs5{
      0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.2, 0.2, 0.2, 0.2};
  AliasSampler sampler5(probs5);

  SECTION("Unbalanced 10-class distribution") {
    CHECK(dists_are_close(probs5, sample_dist(sampler5)));
  }
}
