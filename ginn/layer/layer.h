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

#ifndef GINN_LAYER_LAYER_H
#define GINN_LAYER_LAYER_H

#include <ginn/node.h>
#include <ginn/node/weight.h>
#include <memory>
#include <tuple>

namespace ginn {

template <typename Func>
struct FunctionTraits;

// Get result and argument types of a function type
template <typename Out, typename In>
struct FunctionTraits<Out(In)> {
  using result_t = Out;
  using arg_t = In;
};

template <typename Inner, typename Outer>
struct Composition;

template <typename Out, typename MidOut, typename MidIn, typename In>
struct Composition<MidIn(In), Out(MidOut)> {
  static_assert(
      std::is_convertible_v<MidIn, MidOut>,
      "For functions to be composable, output of inner function should"
      " be convertible to input of outer function!");
  using type = Out(In);
};

// Get the type of a function composed of two other functions
template <typename Inner, typename Outer>
using composition_t = typename Composition<Inner, Outer>::type;

// Take a container and make a clone of it in the type of OutContainer, such
// that the container structure is the same, but contents are default
// constructed.
template <typename OutContainer, typename Container>
OutContainer clone_empty(const Container&);

template <typename Out, typename In>
std::vector<Out> clone_empty(const std::vector<In>& x) {
  static_assert(std::is_default_constructible_v<Out>);
  return std::vector<Out>(x.size());
}

template <typename Out>
std::vector<Out> clone_empty(const std::string& x) {
  static_assert(std::is_default_constructible_v<Out>);
  return std::vector<Out>(x.size());
}

enum class Mode { Training, Inference };

class LayerNodeBase {
 private:
  Mode mode_ = Mode::Training;

 public:
  virtual std::vector<BaseNodePtr> weights_() { return {}; }

  template <typename Scalar>
  std::vector<WeightPtr<Scalar>> weights() {
    std::vector<WeightPtr<Scalar>> rval;
    for (auto w : weights_()) {
      auto ws = dynamic_ref_cast<WeightNode<Scalar>>(w);
      if (ws) { rval.push_back(ws); }
    }
    return rval;
  }

  virtual void set_mode(Mode mode) { mode_ = mode; }
  virtual Mode mode() const { return mode_; }

  virtual ~LayerNodeBase() = default;
};

using LayerBasePtr = std::shared_ptr<LayerNodeBase>;
using ConstLayerBasePtr = std::shared_ptr<const LayerNodeBase>;

template <typename Func>
class LayerNode;

template <typename Func>
using LayerPtr = std::shared_ptr<LayerNode<Func>>;

template <typename Out, typename In>
class LayerNode<Out(In)> : public LayerNodeBase {
 public:
  using function_t = Out(In);
  using arg_t = In;
  using result_t = Out;

  virtual Out run(const In&) = 0;

  virtual LayerPtr<Out(In)> copy(Copy mode) = 0;
};

#define GINN_MAKE_LAYER_FACTORY(f)                                             \
  template <typename... Args>                                                  \
  auto f(Args&&... args) {                                                     \
    return std::make_shared<f##Node>(std::forward<Args>(args)...);             \
  }

#define GINN_MAKE_TEMPLATE_LAYER_FACTORY(f)                                    \
  template <typename Func, typename... Args>                                   \
  auto f(Args&&... args) {                                                     \
    return std::make_shared<f##Node<Func>>(std::forward<Args>(args)...);       \
  }

// Abstract class for layers that are made up of other (children) layers,
// such as composition (stacking) of two layers.
template <typename Func>
class CombinedLayerNode : public LayerNode<Func> {
 public:
  virtual std::vector<ConstLayerBasePtr> children() const = 0;

  std::vector<LayerBasePtr> children() {
    std::vector<LayerBasePtr> rval;
    for (auto c : const_cast<const CombinedLayerNode*>(this)->children()) {
      rval.push_back(std::const_pointer_cast<LayerNodeBase>(c));
    }
    return rval;
  }

  std::vector<BaseNodePtr> weights_() override {
    std::vector<BaseNodePtr> w;
    for (auto c : children()) { w += c->weights_(); }
    return w;
  }

  void set_mode(Mode mode) override {
    for (auto c : children()) { c->set_mode(mode); }
  }

  Mode mode() const override {
    Mode m = children().front()->mode();
    for (auto c : children()) { GINN_ASSERT(c->mode() == m); }
    return m;
  }
};

template <template <typename...> class Container, typename InnerFunc>
class ContainerwiseLayerNode;

template <template <typename...> class Container,
          typename InnerOut,
          typename InnerIn>
class ContainerwiseLayerNode<Container, InnerOut(InnerIn)>
    : public CombinedLayerNode<Container<InnerOut>(Container<InnerIn>)> {
 public:
  using InputType = Container<InnerIn>;
  using OutputType = Container<InnerOut>;
  using InputValueType = InnerIn;
  using OutputValueType = InnerOut;

 private:
  LayerPtr<OutputValueType(InputValueType)> inner_;

 public:
  ContainerwiseLayerNode<Container, InnerOut(InnerIn)>(
      LayerPtr<InnerOut(InnerIn)> inner)
      : inner_(inner) {}

  std::vector<ConstLayerBasePtr> children() const override { return {inner_}; }

  OutputType run(const InputType& x) override {
    auto y = clone_empty<OutputValueType>(x);
    std::transform(x.begin(),
                   x.end(),
                   y.begin(),
                   [this](const InputValueType& x) { return inner_->run(x); });
    return y;
  }

  LayerPtr<OutputType(InputType)> copy(Copy mode) override {
    auto cp_inner = inner_ ? inner_->copy(mode) : nullptr;
    return std::make_shared<std::decay_t<decltype(*this)>>(cp_inner);
  }
};

template <template <typename...> class Container, typename Inner>
auto Containerwise(Inner inner) {
  using InnerFunc = typename Inner::element_type::function_t;
  return std::make_shared<ContainerwiseLayerNode<Container, InnerFunc>>(inner);
}

template <typename Func>
class JoinLayerNode;

template <typename Out, typename In, typename Out2, typename In2>
class JoinLayerNode<std::tuple<Out, Out2>(std::tuple<In, In2>)>
    : public CombinedLayerNode<std::tuple<Out, Out2>(std::tuple<In, In2>)> {
 public:
  using InputType = std::tuple<In, In2>;
  using OutputType = std::tuple<Out, Out2>;

 private:
  LayerPtr<Out(In)> left_;
  LayerPtr<Out2(In2)> right_;

 public:
  JoinLayerNode(LayerPtr<Out(In)> left, LayerPtr<Out2(In2)> right)
      : left_(left), right_(right) {
    GINN_ASSERT(left_);
    GINN_ASSERT(right_);
  }

  std::vector<ConstLayerBasePtr> children() const override {
    return {left_, right_};
  }

  OutputType run(const InputType& xs) override {
    auto& [l, r] = xs;
    return {left_->run(l), right_->run(r)};
  }

  LayerPtr<OutputType(InputType)> copy(Copy mode) override {
    return std::make_shared<JoinLayerNode<OutputType(InputType)>>(
        left_->copy(mode), right_->copy(mode));
  }
};

template <typename Left, typename Right>
auto Join(const std::shared_ptr<Left>& left,
          const std::shared_ptr<Right>& right) {
  using In = typename Left::arg_t;
  using Out = typename Left::result_t;
  using In2 = typename Right::arg_t;
  using Out2 = typename Right::result_t;
  return std::make_shared<
      JoinLayerNode<std::tuple<Out, Out2>(std::tuple<In, In2>)>>(left, right);
}

template <typename Left, typename Right>
auto operator,(const std::shared_ptr<Left>& left,
               const std::shared_ptr<Right>& right) {
  return Join(left, right);
}

template <typename Func>
class FunctionalLayerNode : public LayerNode<Func> {
 public:
  using InputType = typename FunctionTraits<Func>::arg_t;
  using OutputType = typename FunctionTraits<Func>::result_t;

 private:
  std::function<OutputType(const InputType&)> f_;

 public:
  FunctionalLayerNode(std::function<OutputType(const InputType&)> f) : f_(f) {}

  OutputType run(const InputType& x) override { return f_(x); }

  LayerPtr<Func> copy(Copy /*mode*/) override {
    return std::make_shared<FunctionalLayerNode<Func>>(f_);
  }
};

GINN_MAKE_TEMPLATE_LAYER_FACTORY(FunctionalLayer);

template <typename Func>
class StackedLayerNode;

template <typename Out, typename In>
class StackedLayerNode<Out(In)> : public CombinedLayerNode<Out(In)> {
 private:
  LayerBasePtr inner_, outer_;
  std::function<Out(const In&)> f_;
  std::function<std::shared_ptr<StackedLayerNode<Out(In)>>(Copy)> copier_;

 public:
  template <typename MidIn, typename MidOut>
  StackedLayerNode(LayerPtr<MidIn(In)> inner, LayerPtr<Out(MidOut)> outer)
      : inner_(inner), outer_(outer) {
    f_ = [inner, outer](const In& x) { return outer->run(inner->run(x)); };
    copier_ = [inner, outer](Copy mode) {
      auto inner_cp = inner->copy(mode);
      auto outer_cp = outer->copy(mode);
      return std::make_shared<StackedLayerNode<Out(In)>>(inner_cp, outer_cp);
    };
  }

  std::vector<ConstLayerBasePtr> children() const override {
    return {inner_, outer_};
  }

  Out run(const In& x) override { return f_(x); }

  LayerPtr<Out(In)> copy(Copy mode = Copy::Tied) override {
    return copier_(mode);
  }
};

template <typename Inner, typename Outer>
auto Stacked(Inner inner, Outer outer) {
  // assuming Inner & Outer are std::shared_ptr<Derived of LayerNode<Func>>
  using InnerF = typename Inner::element_type::function_t;
  using OuterF = typename Outer::element_type::function_t;
  using Comp = composition_t<InnerF, OuterF>;
  LayerPtr<InnerF> inner_ = inner; // get pointers to base type
  LayerPtr<OuterF> outer_ = outer;
  return std::make_shared<StackedLayerNode<Comp>>(inner_, outer_);
}

template <typename Inner, typename Outer>
auto operator|(const std::shared_ptr<Inner>& inner,
               const std::shared_ptr<Outer>& outer) {
  static_assert(std::is_base_of_v<LayerNodeBase, Inner>);
  static_assert(std::is_base_of_v<LayerNodeBase, Outer>);
  return Stacked(inner, outer);
}

} // namespace ginn

#endif
