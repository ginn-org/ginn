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

#ifndef NONLIN_H
#define NONLIN_H

#include <cmath>
#include <ginn/tensor.h>

namespace ginn {

// Turn a boolean reduction operation to bool
template <typename Expr>
bool make_bool(const DevPtr& dev, const Expr& e) {
  Tensor<bool> result(dev, Shape{});
  result = e;
  if (dev->type() == CPU) { return result.v()[0]; }
  return result.move_to(cpu()).v()[0];
}

template <typename Lhs, typename Rhs>
void add_or_assign(Lhs& lhs, bool accumulate, Rhs rhs) {
  if (accumulate) {
    lhs += rhs;
  } else {
    lhs = rhs;
  }
}

template <typename Scalar>
class NonlinOp {
 public:
  virtual bool backward_requires_input() const {
    // failsafe default
    return true;
  }
  virtual bool is_identity() const { return false; }
  virtual void forward(Tensor<Scalar>& /*y*/,
                       const Tensor<Scalar>& /*x*/) const = 0;
  virtual void backward(Tensor<Scalar>& /*dx*/,
                        const Tensor<Scalar>& /*dy*/,
                        const Tensor<Scalar>& /*x*/,
                        const Tensor<Scalar>& /*y*/,
                        bool /*accumulate*/) const = 0;
  virtual ~NonlinOp() = default;

  virtual std::unique_ptr<NonlinOp<Scalar>> copy() const = 0;
};

#define MAKE_NONLIN_COPY(F)                                                    \
  std::unique_ptr<NonlinOp<Scalar>> copy() const override {                    \
    return std::make_unique<F<Scalar>>(*this);                                 \
  }

template <typename Scalar>
class IdentityOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return false; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    y = x.t();
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& /*x*/,
                const Tensor<Scalar>& /*y*/,
                bool accumulate) const override {
    add_or_assign(dx, accumulate, dy.t());
  }

  MAKE_NONLIN_COPY(IdentityOp)
};

template <typename Scalar>
class TanhOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return false; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    y = x.t().tanh();
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& /*x*/,
                const Tensor<Scalar>& y,
                bool accumulate) const override {
    if constexpr (std::is_same_v<Scalar, Half>) { // more accurate for Half
      add_or_assign(dx, accumulate, dy.t() - y.t().square() * dy.t());
    } else {
      add_or_assign(dx, accumulate, (Scalar(1) - y.t().square()) * dy.t());
    }
  }

  MAKE_NONLIN_COPY(TanhOp)
};

template <typename Scalar>
class ReluOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return false; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    y = x.t().cwiseMax(Scalar(0));
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& /*x*/,
                const Tensor<Scalar>& y,
                bool accumulate) const override {
    add_or_assign(
        dx, accumulate, dy.t() * (y.t() > Scalar(0)).template cast<Scalar>());
  }

  MAKE_NONLIN_COPY(ReluOp)
};

template <typename Scalar>
class SigmoidOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return false; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    if constexpr (std::is_same_v<Scalar, Half>) {
      y = (x.t() > Scalar(0))
              .select(Scalar(1) / (Scalar(1) + (-x.t()).exp()),
                      x.t().exp() / (Scalar(1) + x.t().exp()));
    } else {
      const static Scalar h(0.5);
      y = (x.t() * h).tanh() * h + h;
    }
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& /*x*/,
                const Tensor<Scalar>& y,
                bool accumulate) const override {
    add_or_assign(dx, accumulate, y.t() * (Scalar(1) - y.t()) * dy.t());
  }

  MAKE_NONLIN_COPY(SigmoidOp)
};

template <typename Scalar>
class SoftmaxOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return true; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    Tensor<Scalar> m(x.dev(), {1, x.cols()}), t(x.dev(), x.shape2());
    m = x.t().maximum(Index<1>{0});
    t = (x.t() - m.t().broadcast(Index<2>{x.rows(), 1})).exp();
    m = t.t().sum(Index<1>{0});
    y = t.t() / m.t().broadcast(Index<2>{x.rows(), 1});
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& x,
                const Tensor<Scalar>& y,
                bool accumulate) const override {
    Tensor<Scalar> m(x.dev(), {1, x.cols()});
    m = (y.t() * dy.t()).sum(Index<1>{0});
    add_or_assign(dx,
                  accumulate,
                  y.t() * (dy.t() - m.t().broadcast(Index<2>{x.rows(), 1})));
  }

  MAKE_NONLIN_COPY(SoftmaxOp)
};

template <typename Scalar>
class LogOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return true; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    // Disabling log(0) == -inf because grad would be inf (currently 0-div err)
    GINN_ASSERT(make_bool(x.dev(), (x.t() > Scalar(0)).all()),
                "Log requires positive input!");
    y = x.t().log();
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& x,
                const Tensor<Scalar>&
                /*y*/,
                bool accumulate) const override {
    add_or_assign(dx, accumulate, dy.t() / x.t());
  }

  MAKE_NONLIN_COPY(LogOp)
};

template <typename Scalar>
class SqrtOp : public NonlinOp<Scalar> {
 public:
  bool backward_requires_input() const override { return false; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    GINN_ASSERT(make_bool(x.dev(), (x.t() >= Scalar(0)).all()),
                "Sqrt requires nonnegative input!");
    y = x.t().sqrt();
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& /*x*/,
                const Tensor<Scalar>& y,
                bool accumulate) const override {
    // TODO: unstability issues for x, y close to 0?
    add_or_assign(dx, accumulate, Scalar(0.5) * dy.t() / y.t());
  }

  MAKE_NONLIN_COPY(SqrtOp)
};

// TODO: Try UnaryExpr as well and benchmark. UnaryExpr can use erf in cmath
template <typename Scalar>
class Gelu2Op : public NonlinOp<Scalar> {
 private:
  using s = Scalar; // shorthand for math

 public:
  bool backward_requires_input() const override { return true; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    const static Scalar a(sqrt(2. / acos(-1.)));
    const static Scalar b(0.044715);
    y = s(0.5) * x.t() * (s(1) + (a * (x.t() + b * x.t().pow(s(3)))).tanh());
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& x,
                const Tensor<Scalar>& y,
                bool accumulate) const override {
    const static Scalar a(0.5 * sqrt(2. / acos(-1.)));
    const static Scalar b(0.044715 * 3 * a);
    add_or_assign(
        dx,
        accumulate,
        dy.t() * (y.t() / x.t() + (a + b * x.t() * x.t()) *
                                      (s(4) * y.t() * (s(1) - y.t() / x.t()))));
  }

  MAKE_NONLIN_COPY(Gelu2Op)
};

// When I make Erf template and pass instances to unaryExpr(), gpu builds give
//   a cuda fail. Therefore I have two separate classes for Real and Half.
struct Erf {
  Real operator()(Real x) const { return ::erf(x); }
};

struct HalfErf {
  Half operator()(Half x) const { return Half(::erf(Real(x))); }
};

template <typename Scalar>
class GeluOp : public NonlinOp<Scalar> {
 private:
  using s = Scalar; // shorthand for math

  static auto erf() {
    if constexpr (std::is_same_v<Scalar, Half>) {
      return HalfErf();
    } else {
      return Erf();
    }
  }

 public:
  bool backward_requires_input() const override { return true; }

  void forward(Tensor<Scalar>& y, const Tensor<Scalar>& x) const override {
    y = s(0.5) * x.t() * (s(1) + (x.t() / s(::sqrt(2.))).unaryExpr(erf()));
  }
  void backward(Tensor<Scalar>& dx,
                const Tensor<Scalar>& dy,
                const Tensor<Scalar>& x,
                const Tensor<Scalar>&
                /*y*/,
                bool accumulate) const override {
    const static Scalar c_1_sqrt2pi(1. / ::sqrt(2. * ::acos(-1.)));
    add_or_assign(
        dx,
        accumulate,
        dy.t() *
            //      (y.t() / x.t() + x.t() * (-x.t() * x.t() / 2_r).exp() *
            //      c_1_sqrt2pi);
            (s(0.5) * (s(1) + (x.t() / s(::sqrt(2.))).unaryExpr(erf())) +
             x.t() * (-x.t() * x.t() / s(2)).exp() * c_1_sqrt2pi));
  }

  MAKE_NONLIN_COPY(GeluOp)
};

} // end namespace ginn

#endif
