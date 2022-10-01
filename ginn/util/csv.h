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

#ifndef GINN_UTIL_CSV_H
#define GINN_UTIL_CSV_H

#include <Eigen/Dense>
#include <fstream>
#include <ginn/except.h>
#include <ginn/util/util.h>

namespace ginn {

// Naive CSV / TSV reader. Does not attempt to handle quote escapes.

template <typename EigenMatrixT>
EigenMatrixT read_csv(const std::string& fname, char delim) {
  EigenMatrixT X;
  size_t i = 0;
  std::string line;
  std::ifstream in(fname);
  GINN_ASSERT(in, "File " + fname + " could not be opened!");

  for (i = 0;; i++) {
    if (not std::getline(in, line)) { break; }
    if (line[0] == '#') { continue; }
    auto v = split(line, delim);
    if (i == 0) { X = EigenMatrixT(1, v.size()); }
    GINN_ASSERT(v.size() == (size_t)X.cols(), "Inconsistent number of values!");

    if (i >= (size_t)X.rows()) {
      X.conservativeResize(2 * i, Eigen::NoChange_t());
    }
    for (size_t j = 0; j < v.size(); j++) { X(i, j) = std::stod(v[j]); }
  }

  X.conservativeResize(i, Eigen::NoChange_t());
  return X;
}

} // namespace ginn

#endif
