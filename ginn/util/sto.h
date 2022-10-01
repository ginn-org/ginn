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

#ifndef GINN_UTIL_STO_H
#define GINN_UTIL_STO_H

#include <sstream>
#include <string>

namespace ginn {

// Convert string to things
namespace sto {

template <typename T>
T sto(const std::string& s) {
  T thing;
  char _;
  std::istringstream ss(s);
  if (not(ss >> thing) or (ss >> _)) {
    throw std::runtime_error("Unexpected string for type: " + s);
  }
  return thing;
}

template <>
inline bool sto<bool>(const std::string& s) {
  if (s == "true" or s == "True" or s == "1") { return true; }
  if (s == "false" or s == "False" or s == "0") { return false; }
  throw std::runtime_error("Unexpected boolean string: " + s);
  return false;
}

template <>
inline std::string sto<std::string>(const std::string& s) {
  return s;
}

} // namespace sto

} // namespace ginn

#endif
