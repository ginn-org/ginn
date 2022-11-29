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

#ifndef GINN_TIMER_H
#define GINN_TIMER_H

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tblr.h>

#include <ginn/except.h>

namespace ginn {
namespace timer {

using Duration = std::chrono::microseconds;
using Rep = decltype(Duration().count());
using Clock = std::chrono::system_clock;
using TimePoint = std::chrono::time_point<Clock>;

namespace internal {

template <typename Key, typename Value>
using Map = std::unordered_map<Key, Value>;

Map<std::string, TimePoint> tic_starts;
Map<std::string, Duration> totals;
Map<std::string, unsigned long long> count;

TimePoint total_start;
Duration total;

std::string simplify(Rep micros) {
  auto us = micros;
  auto ms = us / 1000;
  us = us % 1000;
  auto secs = ms / 1000;
  ms = ms % 1000;
  auto mins = secs / 60;
  secs = secs % 60;
  auto hrs = mins / 60;
  mins = mins % 60;
  std::stringstream ss;
  ss << std::setprecision(2);
  const static int num_elems = 2;

  auto delim = [](auto& ss) {
    if (!ss.str().empty()) { ss << " "; }
  };

  int done = num_elems;
  if (hrs > 0 and done-- > 0) {
    delim(ss);
    ss << hrs << "h";
  }
  if (mins > 0 and done-- > 0) {
    delim(ss);
    ss << mins << "m";
  }
  if (secs > 0 and done-- > 0) {
    delim(ss);
    ss << secs << "s";
  }
  if (ms > 0 and done-- > 0) {
    delim(ss);
    ss << ms << "ms";
  }
  if (done-- > 0) {
    delim(ss);
    ss << us << "μs";
  }
  return ss.str();
}

std::string make_bar(float percent, int width = 20) {
  const std::array<std::string, 8> parts{
      "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};

  int on = percent * width / 100.;
  int partial = percent * parts.size() * width / 100. - on * parts.size();
  if (on >= width) {
    on = width;
    partial = 0;
  } else if (on < 0) {
    on = 0;
    partial = 0;
  }
  GINN_ASSERT(size_t(partial) != parts.size());

  std::string s;
  for (int i = 0; i < on; i++) { s += parts.back(); }
  if (partial > 0) { s += parts.at(partial - 1); }

  return s;
}

} // namespace internal

void tic(const std::string& name = "") {
  using namespace internal;
  auto now = Clock::now();
  if (tic_starts.empty()) { total_start = now; }
  tic_starts[name] = now;
}

auto toc(const std::string& name = "") {
  using namespace internal;
  auto now = Clock::now();
  auto duration = now - tic_starts[name];
  totals[name] += std::chrono::duration_cast<Duration>(duration);
  tic_starts.erase(name);
  if (tic_starts.empty()) {
    auto duration = now - total_start;
    total += std::chrono::duration_cast<Duration>(duration);
  }
  count[name]++;
  return std::chrono::duration_cast<Duration>(duration).count();
}

auto get(const std::string& name = "") {
  return internal::totals.at(name).count();
}

// human readable toc()
enum Readability { HumanReadable };

enum class TimerSort { Duration, Name };

std::string toc(std::string name, Readability) {
  return internal::simplify(toc(name));
}
std::string toc(Readability) { return internal::simplify(toc()); }

template <typename Duration = std::chrono::microseconds>
void print(TimerSort sort_by = TimerSort::Duration,
           std::ostream& out = std::cout,
           bool simple = true) {
  using namespace internal;
  using namespace tblr;
  Table t;
  t.aligns({Left, Right, Right, Right, Left})
      .layout(indented_list())
      .precision(2)
      .fixed();

  std::vector<std::string> keys;
  for (auto& p : totals) { keys.push_back(p.first); }

  std::sort(keys.begin(), keys.end(), [sort_by](auto& a, auto& b) {
    // For some reason, nvcc complains if I don't have the namespace here.
    if (sort_by == TimerSort::Duration) {
      return internal::totals[a] > internal::totals[b];
    } else {
      return a < b;
    }
  });

  Duration max_dur =
      std::max_element(internal::totals.begin(),
                       internal::totals.end(),
                       [](auto& a, auto& b) { return a.second < b.second; })
          ->second;

  t << "Name" << (simple ? "time" : "μs") << "#"
    << "%" << ("relative" + std::string(12, '_')) << endr;
  for (auto& key : keys) {
    long long dur = std::chrono::duration_cast<Duration>(totals[key]).count();
    t << (Cell() << "[" << key << "]");
    if (simple) {
      t << simplify(dur);
    } else {
      t << dur;
    }
    auto percent = totals[key] * 100. / total;
    auto percent_by_max = totals[key] * 100. / max_dur;

    t << count[key] << percent << internal::make_bar(percent_by_max) << endr;
  }

  out << "Timing:\n" << t;
}

template <typename Duration = std::chrono::microseconds>
void print(std::ostream& out, bool simple = true) {
  print(TimerSort::Duration, out, simple);
}

void reset() {
  using namespace internal;
  tic_starts.clear();
  totals.clear();
  total = Duration();
  count.clear();
}

const std::string date_time(const std::string format = "%Y-%m-%d_%H.%M.%S") {
  std::time_t tt = Clock::to_time_t(Clock::now());
  char buff[50];
  strftime(buff, 50, format.c_str(), localtime(&tt));
  return buff;
}

template <typename Func>
void time(const std::string& name, Func f) {
  tic(name);
  f();
  toc(name);
}

} // end namespace timer
} // end namespace ginn

#define GINN_TIME(e)                                                           \
  ginn::timer::tic(#e);                                                        \
  e;                                                                           \
  ginn::timer::toc(#e)

#endif
