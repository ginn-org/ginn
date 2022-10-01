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

#ifndef GINN_CORE_H
#define GINN_CORE_H

#ifdef GINN_ENABLE_GPU // cuda
#define EIGEN_USE_GPU
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#endif

#include "def.h" // fundamentals
#ifdef GINN_ENABLE_GPU
#include "cudef.h"
#endif
#include "prod.h"
#include "tensor.h"

#endif
