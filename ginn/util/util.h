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

#ifndef GINN_UTIL_H
#define GINN_UTIL_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <ginn/def.h>
#include <ginn/except.h>

namespace ginn {

namespace internal {

class FileLines {
 private:
  std::optional<std::ifstream> in_;
  std::ifstream* inp_;

 public:
  class FileLinesIter {
   private:
    std::ifstream& in_;
    bool eof_ = false;
    std::string line;

   public:
    const std::string& operator*() const { return line; };
    bool operator!=(const FileLinesIter& right) const {
      return eof_ != right.eof_;
    }
    FileLinesIter& operator++() {
      eof_ = not std::getline(in_, line);
      return *this;
    }
    FileLinesIter(std::ifstream& in, bool eof = false) : in_(in), eof_(eof) {}
  };

  FileLinesIter begin() { return ++FileLinesIter(*inp_); }
  FileLinesIter end() { return FileLinesIter(*inp_, true); };

  FileLines(const std::string& fname) : in_(fname), inp_(&*in_) {
    GINN_ASSERT(*in_, "File " + fname + " could not be opened!");
  }
  FileLines(std::ifstream& in) : inp_(&in) {}
};

} // namespace internal

template <
    typename String,
    typename = std::enable_if_t<std::is_convertible_v<String, std::string>>>
auto lines(const String& fname) {
  return internal::FileLines(std::string(fname));
}

auto lines(std::ifstream& in) { return internal::FileLines(in); }

template <
    typename String,
    typename = std::enable_if_t<std::is_convertible_v<String, std::string>>>
std::vector<std::string> readlines(const String& fname) {
  std::vector<std::string> lines_;
  for (auto& line : lines(fname)) { lines_.push_back(line); }
  return lines_;
}

std::vector<std::string> readlines(std::ifstream& in) {
  std::vector<std::string> lines_;
  for (auto& line : lines(in)) { lines_.push_back(line); }
  return lines_;
}

template <typename String>
std::string read(const String& fname) {
  std::ifstream in(fname);
  GINN_ASSERT(in, "File " + fname + " could not be opened!");
  return {(std::istreambuf_iterator<char>(in)),
          std::istreambuf_iterator<char>()};
}

template <typename T = std::string>
T getline(std::istream& in);

template <>
std::string getline(std::istream& in) {
  GINN_ASSERT(in, "Input stream could not be opened!");
  std::string line;
  std::getline(in, line);
  return line;
}

template <>
size_t getline<size_t>(std::istream& in) {
  return std::stoull(getline(in));
}

template <>
unsigned getline<unsigned>(std::istream& in) {
  return std::stoul(getline(in));
}

template <>
long getline<long>(std::istream& in) {
  return std::stol(getline(in));
}

template <>
float getline<float>(std::istream& in) {
  return std::stof(getline(in));
}

template <>
double getline<double>(std::istream& in) {
  return std::stod(getline(in));
}

template <>
bool getline<bool>(std::istream& in) {
  GINN_ASSERT(in, "Input stream could not be opened!");
  std::string s = getline(in);
  if (s == "1" or s == "true" or s == "True") { return true; }
  if (s == "0" or s == "false" or s == "False") { return false; }
  GINN_THROW("Cannot parse line (" + s + ") as a boolean!");
  return false;
}

// Insert elements from a container (recursively) to a set.
// Container can be list of elems, or list of list of elems, etc.
template <typename T>
void insert(std::unordered_set<T>& vocab, const T& x) {
  vocab.insert(x);
}

template <typename T, typename Container>
void insert(std::unordered_set<T>& vocab, const Container& list) {
  for (const auto& x : list) { insert(vocab, x); }
}

template <typename T, typename K>
std::vector<T> values(const std::unordered_map<K, T>& m) {
  std::vector<T> v;
  for (const auto& p : m) { v.push_back(p.second); }
  return v;
}

template <typename Container, typename T>
bool has(const Container& c, const T& item) {
  return c.find(item) != c.end();
}

template <typename Str>
bool has(const std::string& whole, const Str& part) {
  return whole.find(part) != std::string::npos;
}

std::vector<std::string> split(const std::string& s) { // splits from whitespace
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (ss >> item) elems.push_back(item);
  return elems;
}

std::vector<std::string>
split(const std::string& s, char delim, bool allow_empty = false) {
  std::vector<std::string> elems;
  std::string buffer;
  for (char c : s) {
    if (c == delim) {
      if (allow_empty or !buffer.empty()) { elems.push_back(buffer); }
      buffer.clear();
    } else {
      buffer += c;
    }
  }
  if (allow_empty or !buffer.empty()) { elems.push_back(buffer); }

  return elems;
}

bool startswith(const std::string& s, const std::string& prefix) {
  if (s.size() < prefix.size()) { return false; }
  return s.substr(0, prefix.size()) == prefix;
}

// TODO: hide these in a namespace too?

template <typename T>
std::vector<T> cat(const std::vector<T>& left, const std::vector<T>& right) {
  std::vector<T> res(left);
  for (const auto& x : right) { res.push_back(x); }
  return res;
}

template <typename Container>
void print(const Container& c) {
  for (const auto& x : c) { std::cout << x << " "; }
  std::cout << std::endl;
}

template <typename Container>
void print_map(const Container& c) {
  for (const auto& x : c) {
    std::cout << "{" << x.first << ": " << x.second << "} ";
  }
  std::cout << std::endl;
}

// TODO: is the following too intrusive? should i limit to things where T
// is a ginn type?
template <typename T>
std::vector<T> operator+(const std::vector<T>& left,
                         const std::vector<T>& right) {
  return cat(left, right);
}

template <typename T>
std::vector<T> operator+=(std::vector<T>& left, const std::vector<T>& right) {
  for (const auto& x : right) { left.push_back(x); }
  return left;
}

template <int Rank, typename T>
struct NestedInitListImpl {
  using type =
      std::initializer_list<typename NestedInitListImpl<Rank - 1, T>::type>;
};

template <typename T>
struct NestedInitListImpl<0, T> {
  using type = T;
};

template <int Rank, typename T>
using NestedInitList = typename NestedInitListImpl<Rank, T>::type;

// Helper for assigning a nested initializer list rhs to an indexable type lhs
// such that lhs(i, j, ..., k) = rhs[i][j]...[k]
template <int Rank, typename Scalar, typename T, typename... Args>
void assign(T&& lhs, NestedInitList<Rank, Scalar> rhs, Args... args) {
  typename T::Index i = 0;
  if constexpr (Rank == 0) {
    lhs(i) = rhs;
  } else if constexpr (Rank == 1) {
    for (auto it = rhs.begin(); it != rhs.end(); it++) {
      lhs(args..., i++) = *it;
    }
  } else {
    for (auto it = rhs.begin(); it != rhs.end(); it++) {
      assign<Rank - 1, Scalar>(std::forward<T>(lhs), *it, args..., i++);
    }
  }
}

// Helper for determining the shape of a nested initializer list
template <typename Int, int Rank, typename Scalar>
std::vector<Int> shape_of(NestedInitList<Rank, Scalar> list) {
  if constexpr (Rank == 0) {
    static_assert(std::is_same_v<decltype(list), Scalar>);
    return {};
  } else if constexpr (Rank == 1) {
    return {(Int)list.size()};
  } else {
    for (auto it = list.begin(); it != list.end(); it++) {
      GINN_ASSERT(it->size() == list.begin()->size(),
                  "Inner lists have size mismatch!");
    }
    return std::vector<Int>{(Int)list.size()} +
           shape_of<Int, Rank - 1, Scalar>(*list.begin());
  }
}

template <typename T, typename Rng>
std::vector<T> randperm(size_t size, Rng& g) {
  std::vector<T> perm(size);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), g);
  return perm;
}

template <typename Rng>
std::vector<size_t> randperm(size_t size, Rng& g) {
  return randperm<size_t, Rng>(size, g);
}

template <typename T = size_t>
std::vector<T> iota(size_t size) {
  std::vector<T> v(size);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

} // end namespace ginn

#endif
