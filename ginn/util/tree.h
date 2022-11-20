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

#ifndef GINN_UTIL_TREE_H
#define GINN_UTIL_TREE_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <string_view>
#include <vector>

#include <ginn/except.h>
#include <ginn/util/sto.h>

namespace ginn {
namespace tree {

template <typename T>
struct TreeNode {
  using TreeNodePtr = std::shared_ptr<TreeNode<T>>;
  using Children = std::vector<TreeNodePtr>;

  T data;
  std::vector<TreeNodePtr> children;
};

template <typename T>
using TreeNodePtr = typename TreeNode<T>::TreeNodePtr;

// perform a post-order traversal of a tree
// (equivalent to topological sorting bottom-up)
template <typename T>
std::vector<T> sort(const T& root) {
  if (root == nullptr) { return {}; }
  std::vector<T> v;
  std::stack<T> s;
  s.push(root);
  while (not s.empty()) {
    auto n = s.top();
    v.push_back(n);
    s.pop();
    for (auto& child : n->children) { s.push(child); }
  }
  std::reverse(v.begin(), v.end());
  return v;
}

template <typename Iterator>
class DereferencingIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = typename std::decay<
      decltype(*std::declval<typename Iterator::value_type>())>::type;
  using pointer = value_type*;
  using reference = value_type&;

 private:
  Iterator it_;

 public:
  DereferencingIterator(pointer p) : it_(*p) {}
  DereferencingIterator(Iterator it) : it_(it) {}

  reference operator*() const { return **it_; }
  pointer operator->() { return *it_; }
  auto& operator++() {
    it_++;
    return *this;
  }
  auto operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  bool operator==(const DereferencingIterator& other) {
    return it_ == other.it_;
  }
  bool operator!=(const DereferencingIterator& other) {
    return it_ != other.it_;
  }
};

template <typename T>
class Tree {
  // TODO: Either ensure constness of tree structure, or make sure
  // `sorted` retains sortedness upon tree change
 public: // TODO: look for ways to make these private
  TreeNodePtr<T> root;

 private:
  std::vector<T*> sorted_;
  using Children = typename TreeNode<T>::Children;
  std::vector<Children*> sorted_children_;

 public:
  using value_type = T;

  Tree() = default;
  Tree(TreeNodePtr<T> a_root) : root(a_root) {
    auto sorted_nodes = sort(root);
    for (auto& x : sorted_nodes) {
      sorted_.push_back(&(x->data));
      sorted_children_.push_back(&(x->children));
    }
  }

  size_t size() const { return sorted_.size(); }
  T& operator[](size_t t) { return *sorted_[t]; }
  const T& operator[](size_t t) const { return *sorted_[t]; }

  const auto& children_at(size_t t) const { return *sorted_children_[t]; }

  auto begin() { return DereferencingIterator(sorted_.begin()); }
  auto end() { return DereferencingIterator(sorted_.end()); }
  auto begin() const { return DereferencingIterator(sorted_.begin()); }
  auto end() const { return DereferencingIterator(sorted_.end()); }
};

// copy a tree structure with empty data (uses T's default ctor)
template <typename T, typename T2>
TreeNodePtr<T> make_like(const T2& root) {
  if (root == nullptr) { return nullptr; }
  std::stack<T2> s;
  std::stack<TreeNodePtr<T>> new_s;
  TreeNodePtr<T> new_root = std::make_shared<TreeNode<T>>();

  s.push(root);
  new_s.push(new_root);

  while (!s.empty()) {
    auto n = s.top();
    auto new_n = new_s.top();
    s.pop();
    new_s.pop();
    for (size_t i = 0; i < n->children.size(); i++) { // make new children
      new_n->children.push_back(std::make_shared<TreeNode<T>>());
    }
    for (auto& child : n->children) { s.push(child); }
    for (auto& child : new_n->children) { new_s.push(child); }
  }
  return new_root;
}

template <typename T, typename T2>
Tree<T> make_like(const Tree<T2>& other) {
  return Tree<T>(make_like<T>(other.root));
}

template <typename T, typename Printer>
void print(std::ostream& o,
           TreeNodePtr<T> n,
           const Printer& p,
           bool last_child = true,
           std::string indent = "") {
  o << indent;
  if (last_child) {
    o << (indent.empty() ? "╌╌" : "└─");
    indent += "  ";
  } else {
    o << "├─";
    indent += "│ ";
  }
  p(o, n->data);
  o << std::endl;

  for (auto& child : n->children) {
    print<T>(o, child, p, &child == &n->children.back(), indent);
  }
}

template <typename T, typename Printer>
void print(std::ostream& o, const Tree<T>& t, const Printer& p) {
  print<T>(o, t.root, p);
}

template <typename T>
void print(std::ostream& out, const Tree<T>& t) {
  print<T>(out, t.root, [](std::ostream& o, const T& data) { o << data; });
}

template <typename T>
void print(const Tree<T>& tree) {
  print<T>(std::cout, tree);
}

template <typename T, typename Reader>
TreeNodePtr<T> parse_helper(std::string_view s, Reader r) {
  std::stack<TreeNodePtr<T>> stack;

  while (not s.empty()) {
    if (s.front() == '(') {
      auto n = std::make_shared<TreeNode<T>>();
      if (not stack.empty()) {
        auto& pc = stack.top()->children;
        pc.push_back(n);
      }
      stack.push(n);
      size_t i = s.find_first_of("()", 1);
      GINN_ASSERT(i != std::string::npos);
      n->data = r(s.substr(1, i - 1));
      s.remove_prefix(i);
    } else if (s.front() == ')') {
      GINN_ASSERT(not stack.empty(),
                  "Close-paren does not have matching open-paren!");
      auto n = stack.top();
      stack.pop();
      s.remove_prefix(1);
      if (stack.empty()) {
        GINN_ASSERT(s.empty(), "Parsed a tree but string continues!");
        return n;
      }
    } else {
      GINN_ASSERT(::isspace(s.front()), "Unexpected token in tree string!");
      s.remove_prefix(1);
    }
  }

  GINN_ASSERT(stack.empty(), "Unclosed parenthesis in tree string!");

  return nullptr;
}

template <typename T, typename Reader>
Tree<T> parse(const std::string& line, Reader r) {
  auto root = parse_helper<T, Reader>(line, r);
  return Tree<T>(root);
}

template <typename T>
Tree<T> parse(const std::string& line) {
  auto r = [](std::string_view s) { return sto::sto<T>(std::string(s)); };
  return parse<T>(line, r);
}

} // namespace tree

template <typename OutContainer, typename Container>
OutContainer clone_empty(const Container&);

template <typename Out, typename In>
tree::Tree<Out> clone_empty(const tree::Tree<In>& x) {
  static_assert(std::is_default_constructible_v<Out>);
  return tree::make_like<Out, In>(x);
}

} // namespace ginn

#endif
