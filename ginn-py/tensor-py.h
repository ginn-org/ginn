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

#ifndef GINN_PY_TENSOR_PY_H
#define GINN_PY_TENSOR_PY_H

#include <ginn/def.h>

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <ginn/tensor.h>
#include <ginn/util/tensorio.h>

#include <ginn-py/util-py.h>

namespace ginn {
namespace python {

namespace py = pybind11;

// This is a helper type enum to dispatch things based on the scalar type
// on the Python side.
enum class Scalar_ { Real, Half, Int, Bool };

template <typename Scalar>
Scalar_ scalar_() {
  if constexpr (std::is_same_v<Scalar, Real>) { return Scalar_::Real; }
  else if constexpr (std::is_same_v<Scalar, Half>) { return Scalar_::Half; }
  else if constexpr (std::is_same_v<Scalar, Int>) { return Scalar_::Int; }
  else if constexpr (std::is_same_v<Scalar, bool>) { return Scalar_::Bool; }
  else { 
    GINN_THROW("Unexpected scalar type!");
    return {};
  }
}

void bind_tensor(py::module_& m);

} // namespace python
} // namespace ginn

#endif
