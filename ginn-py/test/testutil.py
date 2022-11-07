import ginn

# Compute numeric grad for gradient checks, using finite differences
def numeric_grad(
    node: ginn.RealNode, weight: ginn.RealNode, mask: ginn.RealNode, eps: float = 1e-4
):
    pass


"""
// Compute numeric grad for gradient checks, using finite differences
template <typename Expr, typename Weight, typename Mask>
inline Tensor<Real> numeric_grad(Expr e, Weight w, Mask mask, Real eps = 1e-4) {
    auto s = Sum(CwiseProd(mask, e));

auto g = Graph(s);
Tensor<Real> rval(w->dev(), w->shape());

for (uint i = 0; i < w->size(); i++) {
auto value = w->value().m();
Real tmp = value(i);
value(i) = tmp + eps;
g.reset_forwarded();
g.forward();
Real s_plus = e->value().m().cwiseProduct(mask->value().m()).sum();

value(i) = tmp - eps;
g.reset_forwarded();
g.forward();
Real s_minus = e->value().m().cwiseProduct(mask->value().m()).sum();

rval.v()(i) = (s_plus - s_minus) / (2 * eps);
value(i) = tmp;
}

return rval;
}
"""
