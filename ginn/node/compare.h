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

#ifndef GINN_NODE_COMPARE_H
#define GINN_NODE_COMPARE_H

#include <ginn/node.h>
#include <ginn/node/data.h>

namespace ginn {

// TODO: Seems like when I name the template parameter Scalar, it conflicts with
// Node::Scalar of base? Why does this not shadow?

template <typename InScalar>
class LessThanNode : public BaseDataNode<bool> {
 private:
  NodePtr<InScalar> l_, r_;

  void forward_() override {
    value().resize(l_->value().shape());
    value() = (l_->value().t() < r_->value().t());
  }

 public:
  using BaseDataNode<bool>::value;

  bool has_grad() const override { return false; }

  LessThanNode<InScalar>(NodePtr<InScalar> l, NodePtr<InScalar> r)
      : BaseDataNode<bool>(std::vector<BaseNodePtr>{l, r}), l_(l), r_(r) {}

  std::string name() const override { return "LessThan"; }
};

GINN_MAKE_SCALAR_FORWARDING_FACTORY(LessThan);

template <typename Left, typename Right>
auto operator<(const Ptr<Left>& a, const Ptr<Right>& b) {
  return LessThan(a, b);
}

} // namespace ginn

#endif
