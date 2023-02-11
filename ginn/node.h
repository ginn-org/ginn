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

#ifndef GINN_NODE_H
#define GINN_NODE_H

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <ginn/tensor.h>
#include <ginn/util/traits.h>

namespace ginn {

class BaseNode;

template <typename NodeType>
class Ptr {
  static_assert(std::is_base_of_v<BaseNode, NodeType>,
                "Ptr only applies to classes deriving from BaseNode!");

 private:
  std::shared_ptr<NodeType> ptr_;

 public:
  using element_type = typename std::shared_ptr<NodeType>::element_type;

  Ptr() = default;
  Ptr(NodeType* ptr) : ptr_(ptr) {}
  Ptr(std::shared_ptr<NodeType> ptr) : ptr_(std::move(ptr)) {}
  Ptr(std::nullptr_t) : ptr_(nullptr) {}

  template <typename OtherNodeType,
            typename = std::enable_if_t<
                std::is_convertible_v<std::shared_ptr<OtherNodeType>,
                                      std::shared_ptr<NodeType>>>>
  Ptr(const Ptr<OtherNodeType>& other) : ptr_(other.ptr_) {}

  Ptr(const Ptr<NodeType>&) = default;
  Ptr& operator=(const Ptr&) = default;

  template <typename OtherNodeType>
  Ptr& operator=(const Ptr<OtherNodeType>& other) {
    ptr_ = other.ptr_;
    return *this;
  }

  auto operator->() { return ptr_.get(); }
  auto operator->() const { return ptr_.get(); }
  auto& operator*() { return *ptr_; }
  const auto& operator*() const { return *ptr_; }
  auto get() const { return ptr_.get(); }
  const auto& as_shared_ptr() const { return ptr_; }

  bool operator==(const Ptr<NodeType>& other) const {
    return ptr_ == other.ptr_;
  }

  explicit operator bool() const { return (bool)ptr_; }

  template <typename T>
  friend class Ptr;
};

template <typename NodeType, typename... Args>
auto make_ptr(Args&&... args) {
  return Ptr(std::make_shared<NodeType>(std::forward<Args>(args)...));
}

template <typename T, typename U>
auto dynamic_ptr_cast(const Ptr<U>& sp) {
  return Ptr(std::dynamic_pointer_cast<T>(sp.as_shared_ptr()));
}

using BaseNodePtr = Ptr<BaseNode>;
using ConstBaseNodePtr = Ptr<const BaseNode>;

// To pass lists of exprs of different types as arguments
// Explicitly casts to the base Node class
template <typename Iterable>
auto base_cast(Iterable&& v) {
  return std::vector<BaseNodePtr>(v.begin(), v.end());
}

template <typename DerivedNode, typename Container>
auto derived_cast(const Container& v) {
  std::vector<Ptr<DerivedNode>> rv;
  for (const auto& x : v) {
    if (auto x_ = dynamic_ptr_cast<DerivedNode>(x)) {
      rv.push_back(x_);
    } else {
      GINN_THROW("Failed to cast to a derived Node!");
    }
  }
  return rv;
}

class BaseNode {
 protected:
  std::vector<BaseNodePtr> ins_;

  virtual void forward_() {}
  virtual void backward_() {}

 public:
  BaseNode() = default;

  bool forwarded = false;

  const auto& ins() { return ins_; }

  std::vector<ConstBaseNodePtr> ins() const {
    return {ins_.begin(), ins_.end()};
  }

  const BaseNodePtr& in(size_t idx = 0) { return ins_.at(idx); }
  ConstBaseNodePtr in(size_t idx = 0) const { return ins_.at(idx); }

  virtual void set_ins(const std::vector<BaseNodePtr>&) {
    GINN_THROW("set_ins() for " + name() + " is not implemented!");
  }

  // Construct
  BaseNode(std::vector<BaseNodePtr> ins) : ins_(std::move(ins)) {}

  template <typename DerivedNodePtr>
  BaseNode(std::vector<DerivedNodePtr> ins) : ins_(base_cast(std::move(ins))) {}

  // template <typename Container>
  // BaseNode(const Container& ins) {
  //   for (auto& x : ins) { ins_.push_back(x); }
  // }

  virtual ~BaseNode() = default;

  virtual DevPtr dev() const = 0;
  virtual Shape shape() const = 0;
  Shape shape2() const { return Tensor<>::reduce(shape(), 2); }
  Size rows() const { return shape2()[0]; }
  Size cols() const { return shape2()[1]; }
  Size size() const { return Tensor<>::size(shape()); }

  void forward() {
    if (not forwarded) {
      forward_();
      forwarded = true;
    }
  }
  void backward() {
    if (has_grad()) { backward_(); }
  }

  virtual bool has_grad() const { return false; }
  virtual void init_grad() {}
  virtual void reset_grad() {}
  virtual void reset_forwarded() { forwarded = false; }

  virtual std::string name() const = 0;
};

template <typename ScalarType = Real>
class Node : public BaseNode {
 public:
  using Scalar = ScalarType;

 public:
  using BaseNode::BaseNode;

  virtual const Tensor<Scalar>& value() const = 0;
  virtual const Tensor<Scalar>& grad() const = 0;
  Tensor<Scalar>& value() {
    return const_cast<Tensor<Scalar>&>(const_cast<const Node&>(*this).value());
  }
  Tensor<Scalar>& grad() {
    return const_cast<Tensor<Scalar>&>(const_cast<const Node&>(*this).grad());
  }

  DevPtr dev() const override { return value().dev(); }
  Shape shape() const override { return value().shape(); }

  void init_grad() override {
    GINN_ASSERT(forwarded, "Cannot init grad on non-forwarded nodes!");
    if (has_grad() and value().size() != grad().size()) {
      grad().resize(value().shape());
    }
  }
  void reset_grad() override {
    init_grad();
    if (has_grad()) { grad().set_zero(); }
  }

  Scalar item() const { return value().item(); }
};

template <typename Scalar = Real>
using NodePtr = Ptr<Node<Scalar>>;

template <typename Scalar = Real>
using ConstNodePtr = Ptr<const Node<Scalar>>;

class Graph {
 private:
  std::vector<BaseNodePtr> list_;
  std::unordered_set<BaseNode*> visited_;

  // Define (in topological order) the graph (as list of nodes) reaching to
  // node e
  void traverse(BaseNodePtr e) {
    if (visited_.find(e.get()) == visited_.end()) {
      for (auto& in : e->ins()) { traverse(in); }
      visited_.insert(e.get());
      list_.push_back(e);
    }
  }

 public:
  Graph(BaseNodePtr e) {
    traverse(e);
    visited_.clear();
  }

  auto& forward() {
    for (auto e : list_) { e->forward(); }
    return *this;
  }

  auto& backward(double loss_coeff = 1) {
    if (auto sink = dynamic_ptr_cast<Node<Real>>(list_.back())) {
      GINN_ASSERT(sink->has_grad());
      // TODO: should i constrain this to have shape {1} ?
      sink->grad().fill(loss_coeff); // assume scalar loss.
    } else if (auto sink = dynamic_ptr_cast<Node<Half>>(list_.back())) {
      GINN_ASSERT(sink->has_grad());
      // TODO: should i constrain this to have shape {1} ?
      sink->grad().fill(Half(loss_coeff)); // assume scalar loss.
    } else {
      GINN_THROW("Unexpected scalar type in sink node!");
    }

    for (auto e = list_.rbegin(); e != list_.rend(); e++) {
      if ((*e)->has_grad()) { (*e)->backward(); }
    }
    return *this;
  }

  auto& init_grad() {
    for (auto e : list_) { e->init_grad(); }
    return *this;
  }

  auto& reset_grad() {
    for (auto e : list_) { e->reset_grad(); }
    return *this;
  }

  auto& reset_forwarded() {
    for (auto e : list_) { e->reset_forwarded(); }
    return *this;
  }

  const auto& nodes() const { return list_; }
  BaseNodePtr sink() const { return list_.back(); }
};

// Given SomeNode, define Some(),
//  that returns a ref (shared_ptr) and
//  FixedSome(), which does the same but sets has_grad to false.

#define GINN_MAKE_FACTORY(f)                                                   \
  template <typename... Args>                                                  \
  auto f(Args&&... args) {                                                     \
    return make_ptr<f##Node>(std::forward<Args>(args)...);                     \
  }                                                                            \
  template <typename... Args>                                                  \
  auto Fixed##f(Args&&... args) {                                              \
    auto n = make_ptr<f##Node>(std::forward<Args>(args)...);                   \
    n->has_grad(false);                                                        \
    return n;                                                                  \
  }                                                                            \
  static_assert(true, "Factory maker requires a semicolon")

// TODO: a version of the following where Scalar can be ommitted and defaults to
//   Real.
#define GINN_MAKE_TEMPLATE_FACTORY(f)                                          \
  template <typename Scalar, typename... Args>                                 \
  auto f(Args&&... args) {                                                     \
    return make_ptr<f##Node<Scalar>>(std::forward<Args>(args)...);             \
  }                                                                            \
  template <typename Scalar, typename... Args>                                 \
  auto Fixed##f(Args&&... args) {                                              \
    auto n = make_ptr<f##Node<Scalar>>(std::forward<Args>(args)...);           \
    n->has_grad(false);                                                        \
    return n;                                                                  \
  }                                                                            \
  static_assert(true, "Factory maker requires a semicolon")

// Helper for forwarding initializer_list<T> to std::vector<T>
#define GINN_FORWARD_INIT_LIST(f, T)                                           \
  template <typename... Args>                                                  \
  auto f(Args&&... args, std::initializer_list<T> arg) {                       \
    return f(std::forward<Args>(args)..., std::vector<T>(arg));                \
  }                                                                            \
  static_assert(true, "Factory maker requires a semicolon")

// Helper for forwarding the Scalar type of first arg
#define GINN_MAKE_SCALAR_FORWARDING_FACTORY(f)                                 \
  /*If first arg is a Ptr<Node<Scalar>> for a derived node type, forward its   \
   * Scalar */                                                                 \
  template <typename Arg, typename... Args>                                    \
  auto f(Arg&& arg, Args&&... args) {                                          \
    using NodePtr = innermost_t<Arg>;                                          \
    using Scalar = typename std::decay_t<NodePtr>::element_type::Scalar;       \
    return make_ptr<f##Node<Scalar>>(std::forward<Arg>(arg),                   \
                                     std::forward<Args>(args)...);             \
  }                                                                            \
  static_assert(true, "Factory maker requires a semicolon")

} // end namespace ginn

#endif
