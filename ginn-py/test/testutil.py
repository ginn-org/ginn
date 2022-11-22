# Copyright 2022 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
