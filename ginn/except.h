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

#ifndef GINN_EXCEPT_H
#define GINN_EXCEPT_H

#include <stdexcept>
#include <string>

namespace ginn {

class EigenError : public std::runtime_error {
 public:
  EigenError(const std::string& what) : std::runtime_error(what) {}
};

// is it worth to have different error types for different kind of errors?
class RuntimeError : public std::runtime_error {
 public:
  RuntimeError(const std::string& what) : std::runtime_error(what) {}
};

class CudaError : public std::runtime_error {
 public:
  CudaError(const std::string& what) : std::runtime_error(what) {}
};

} // namespace ginn

#define GINN_OVERLOAD(_1, _2, MACRO, ...) MACRO

#define GINN_ASSERT(...)                                                       \
  GINN_OVERLOAD(__VA_ARGS__, GINN_ASSERT2, GINN_ASSERT1)(__VA_ARGS__)

#define GINN_ASSERT2(statement, message)                                       \
  do {                                                                         \
    if (not(statement)) { throw ginn::RuntimeError(message); }                 \
  } while (0)

#define GINN_ASSERT1(statement)                                                \
  GINN_ASSERT2(statement, "Assertion " #statement " failed!")

#define GINN_THROW(message) GINN_ASSERT(false, message)

#ifndef GINN_ENABLE_GPU

// TODO: Changing Eigen assertions to throws breaks gpu kernels. See what can
//   be done here.

#undef eigen_assert
#define eigen_assert(statement)                                                \
  do {                                                                         \
    if (not(statement)) {                                                      \
      throw ginn::EigenError("Assertion " #statement " failed.");              \
    }                                                                          \
  } while (0)

#endif

#endif
