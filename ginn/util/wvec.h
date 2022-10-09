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

#ifndef GINN_UTIL_WVEC_H
#define GINN_UTIL_WVEC_H

#include <numeric>
#include <string>
#include <unordered_set>

#include <ginn/node/weight.h>
#include <ginn/util/lookup.h>
#include <ginn/util/util.h>

namespace ginn {
namespace util {

// TODO: add expected dimensionality for sanity checking
template <typename Scalar>
void load_wvecs(LookupTable<std::string, WeightPtr<Scalar>>& table,
                Device& dev,
                const std::string& fname,
                const std::unordered_set<std::string>& vocab,
                bool fixed = false,
                size_t max = std::numeric_limits<size_t>::max()) {
  size_t count = 0;
  for (const auto& line : lines(fname)) {
    auto v = split(line);
    std::string w = v[0];
    if (vocab.find(w) != vocab.end()) {
      if (fixed) {
        table.insert(w, FixedWeight<Scalar>(cpu(), {Size(v.size()) - 1}));
      } else {
        table.insert(w, Weight<Scalar>(cpu(), {Size(v.size()) - 1}));
      }
      for (size_t i = 0; i < v.size() - 1; i++)
        table[w]->value().v()(i) = Scalar(std::stod(v[i + 1]));
      table[w]->move_to(dev);
    }
    count++;

    if (max == count) { break; }
  }
}

} // namespace util
} // namespace ginn

#endif