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

#ifndef GINN_UTIL_SAMPLE_H
#define GINN_UTIL_SAMPLE_H

#include <algorithm>
#include <random>
#include <vector>

#include <ginn/except.h>
#include <ginn/tensor.h>

namespace ginn {

/// Algorithm to sample from a fixed categorical distribution in constant time.
/// Implements Vose's Alias Method as described in:
/// https://www.keithschwarz.com/darts-dice-coins/

class AliasSampler {
 public:
  using Index = size_t; // Class index, 0-indexed

 private:
  std::vector<Index> alias; // alias class for each bucket
  std::vector<double> prob; // threshold for selecting the alias class
  std::uniform_int_distribution<Index> macro_dist;
  std::uniform_real_distribution<double> micro_dist;
  std::default_random_engine rng;
  size_t n;

  void init_alias_table(const std::vector<double>& probs) {
    // Ensure this is a valid probability distribution
    GINN_ASSERT(std::all_of(
        probs.begin(), probs.end(), [](double p) { return p >= 0.0; }));
    double probSum = std::accumulate(probs.begin(), probs.end(), 0.0);
    GINN_ASSERT((0.9999 <= probSum) and (probSum <= 1.0001));

    // Step 2
    std::vector<Index> small;
    std::vector<Index> large;

    // Step 3
    std::vector<double> scaledProbs = probs;
    for (size_t i = 0; i < scaledProbs.size(); ++i) { scaledProbs[i] *= n; }

    // Step 4
    for (size_t i = 0; i < scaledProbs.size(); ++i) {
      double p_i = scaledProbs[i];

      if (p_i < 1.0) {
        small.push_back(i);
      } else {
        large.push_back(i);
      }
    }

    // Step 5
    Index l;
    Index g;

    while (not(small.empty() or large.empty())) {
      l = small.back();
      g = large.back();
      small.pop_back();
      large.pop_back();

      prob[l] = scaledProbs[l];
      alias[l] = g;
      scaledProbs[g] = (scaledProbs[g] + scaledProbs[l]) - 1;
      if (scaledProbs[g] < 1.0) {
        small.push_back(g);
      } else {
        large.push_back(g);
      }
    }

    // Step 6
    while (not large.empty()) {
      g = large.front();
      large.erase(large.begin());
      prob[g] = 1.0;
    }

    // Step 7
    while (not small.empty()) {
      l = small.front();
      small.erase(small.begin());
      prob[l] = 1.0;
    }
  }

 public:
  AliasSampler(const std::vector<double>& probs)
      : alias(probs.size(), 0),
        prob(probs.size(), 0.0),
        macro_dist(1, probs.size()),
        micro_dist(0.0, 1.0),
        rng(),
        n(probs.size()) {
    init_alias_table(probs);
  }

  void set_seed(unsigned seed) { rng.seed(seed); }

  Index sample() {
    Index bucket = macro_dist(rng) - 1;
    double r = micro_dist(rng);
    if (r <= prob[bucket]) {
      return bucket;
    } else {
      return alias[bucket];
    }
  }

  size_t num_classes() { return n; }

  Index operator()() { return sample(); }
};

} // namespace ginn

#endif
