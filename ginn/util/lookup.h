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

#ifndef GINN_UTIL_LOOKUP_H
#define GINN_UTIL_LOOKUP_H

#include <ginn/node/weight.h>
#include <ginn/util/indexmap.h>
#include <ginn/util/traits.h>
#include <memory>

namespace ginn {
namespace internal {

// TODO: Should I add other known containers? Recurse? Also see how Layers work
//   with containers and possibly unify these with those utilities.
template <typename T>
auto deep_copy(const std::vector<std::shared_ptr<T>>& v) {
  std::vector<std::shared_ptr<T>> copy(v.size());
  for (size_t i = 0; i < copy.size(); i++) {
    copy[i] = std::make_shared<T>(*v[i]);
  }
  return copy;
}

// TODO: doesn't 4th case cover this?
template <typename T>
auto deep_copy(const std::vector<T>& v) {
  return std::vector<T>(v);
}

template <typename T>
auto deep_copy(const std::shared_ptr<T>& x) {
  return std::make_shared<T>(*x);
}

template <typename T>
auto deep_copy(const T& x) {
  return T(x);
}

} // namespace internal

template <typename Key, typename NodePtrType>
class LookupTable {
  static_assert(ginn::is_node_ptr_v<NodePtrType>,
                "Unexpected value type in LookupTable!");

 private:
  IndexMap<Key> map_;
  std::vector<NodePtrType> storage_;
  NodePtrType unk_;
  bool has_unk_ = true;

 public:
  LookupTable(bool has_unk = true) : has_unk_(has_unk) {}

  LookupTable(const std::unordered_set<Key>& keys, bool has_unk = true)
      : LookupTable(has_unk) {
    map_ = IndexMap<Key>(keys);
    storage_.resize(map_.size());
    if (has_unk_) { unk_ = NodePtrType(); }
  }

  template <typename Function>
  LookupTable(const std::unordered_set<Key>& keys,
              Function value_initializer,
              bool has_unk = true)
      : LookupTable(keys, has_unk) {
    for (auto& val : storage_) { val = value_initializer(); }
    if (has_unk_) { unk_ = value_initializer(); }
  }

  LookupTable(const LookupTable& other)
      : map_(other.map_), has_unk_(other.has_unk_) {
    unk_ = internal::deep_copy(other.unk_);
    storage_ = internal::deep_copy(other.storage_);
  }

  LookupTable& operator=(const LookupTable& other) {
    if (this == &other) { return *this; }
    map_ = other.map_;
    has_unk_ = other.has_unk_;
    unk_ = internal::deep_copy(other.unk_);
    storage_ = internal::deep_copy(other.storage_);

    return *this;
  }

  LookupTable& operator=(LookupTable&& other) = default;

  NodePtrType& unk() {
    GINN_ASSERT(has_unk_);
    return unk_;
  }

  LookupTable<Key, NodePtrType> copy(Copy mode = Copy::Tied) {
    LookupTable<Key, NodePtrType> rval;
    rval.map_ = map_;
    rval.has_unk_ = has_unk_;
    if (has_unk_) { rval.unk_ = unk_->copy(mode); }
    for (auto w : storage_) { rval.storage_.push_back(w->copy(mode)); }
    return rval;
  }

  void tie(LookupTable& other) {
    map_ = other.map_;
    has_unk_ = other.has_unk_;
    if (has_unk_) {
      unk_ = Weight();
      unk_->tie(other.unk_);
    }
    storage_.clear();
    for (auto other_w : other.storage_) {
      auto w = Weight();
      w->tie(other_w);
      storage_.push_back(w);
    }
  }

  std::vector<NodePtrType> weights() {
    if (has_unk_) { return (storage_ + std::vector<NodePtrType>{unk_}); }
    return storage_;
  }

  NodePtrType& operator[](const Key& key) { // never creates an entry
    if (has_unk_ && !map_.has(key)) return unk_;
    return storage_.at(map_[key]);
  }

  template <typename Keys>
  std::vector<NodePtrType> operator[](const Keys& keys) {
    std::vector<NodePtrType> v;
    for (auto key : keys) { v.push_back((*this)[key]); }
    return v;
  }

  void insert(const Key& key) {
    if (not map_.has(key)) {
      map_.insert(key);
      storage_.push_back(NodePtrType());
    }
  }
  void insert(const Key& key, const NodePtrType& value) {
    if (not map_.has(key)) {
      map_.insert(key);
      storage_.push_back(value);
    } else {
      storage_.at(map_.lookup(key)) = value;
    }
  }
  void clear() {
    storage_.clear();
    map_.clear();
  }
  auto size() const { return storage_.size(); }
  const auto& keys() const { return map_.keys(); }
};

} // namespace ginn

#endif
