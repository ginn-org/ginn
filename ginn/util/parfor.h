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

#ifndef GINN_UTIL_PARFOR_H
#define GINN_UTIL_PARFOR_H

#include <atomic>
#include <thread>
#include <vector>

namespace ginn {

template <typename F>
inline void
parallel_for(size_t begin, size_t end, F f, unsigned n_threads = 8) {
  std::vector<std::thread> threads(n_threads);
  std::atomic<size_t> i{begin};
  for (unsigned ti = 0; ti < n_threads; ti++) {
    auto& t = threads[ti];
    t = std::thread([&, ti]() {
      while (true) {
        size_t i_ = i++;
        if (i_ >= end) { break; }
        f(i_, ti);
      }
    });
  }

  for (auto& t : threads) { t.join(); }
}

template <typename F>
inline void
parallel_for_except(size_t begin, size_t end, F f, unsigned n_threads = 8) {
  std::vector<std::thread> threads(n_threads);
  std::vector<std::exception_ptr> excepts(n_threads, nullptr);
  std::atomic<size_t> i{begin};
  for (unsigned ti = 0; ti < n_threads; ti++) {
    auto& t = threads[ti];
    t = std::thread([&, ti]() {
      while (true) {
        size_t i_ = i++;
        if (i_ >= end) { break; }

        try {
          f(i_, ti);
        } catch (...) { excepts[ti] = std::current_exception(); }
      }
    });
  }

  for (auto& t : threads) { t.join(); }
  for (auto& e : excepts) {
    if (e) { std::rethrow_exception(e); }
  }
}

} // namespace ginn

#endif
