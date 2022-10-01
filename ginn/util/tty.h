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

#ifndef GINN_UTIL_TTY_H
#define GINN_UTIL_TTY_H

// TODO: This is not really portable, have availability checks
// and fallbacks

#include <iostream>
#include <unistd.h>

namespace ginn {
namespace internal {

bool istty() { return ::isatty(STDOUT_FILENO); }

} // namespace internal
} // namespace ginn

#endif // GINN_UTIL_TTY_H
