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

from typing import List

import ginn
import numpy as np
import pytest


# Compute numeric grad for gradient checks, using finite differences
def numeric_grad(
    node: ginn.RealNode, weight: ginn.RealNode, mask: ginn.RealNode, eps: float
) -> ginn.RealTensor:
    s = ginn.Sum(ginn.CwiseProd(mask, node))
    g = ginn.Graph(s)
    rval = ginn.RealTensor(weight.dev, weight.shape)

    for i in range(weight.size):
        value = weight.value.v()
        tmp = value[i]

        value[i] = tmp + eps
        g.reset_forwarded()
        g.forward()
        s_plus = np.sum(node.value.m() * mask.value.m())

        value[i] = tmp - eps
        g.reset_forwarded()
        g.forward()
        s_minus = np.sum(node.value.m() * mask.value.m())

        rval.v()[i] = (s_plus - s_minus) / (2.0 * eps)
        value[i] = tmp

    return rval


# Compute analytic grad for gradient checks, by calling backward()
def analytic_grad(
    node: ginn.RealNode, weight: ginn.RealNode, mask: ginn.RealNode
) -> ginn.RealTensor:
    s = ginn.Sum(ginn.CwiseProd(mask, node))
    g = ginn.Graph(s)
    g.reset_forwarded()  # gradcheck reuses expressions
    g.forward()
    g.reset_grad()
    g.backward(1.0)

    return weight.grad.copy_to(ginn.cpu())


def check_grad(node: ginn.RealNode, ins: List[ginn.RealNode], eps: float = 1e-2):
    node = node + node  # to make sure grad accumulates over input repetition
    g = ginn.Graph(node)
    g.reset_forwarded()
    g.forward()  # to init all shapes

    for w in ins:
        if w.has_grad:
            w.reset_grad()

            mask = ginn.Random(device=node.dev, shape=node.shape) * 0.5 + 1.0
            mask.has_grad = False

            ng = numeric_grad(node, w, mask, eps)
            ag = analytic_grad(node, w, mask)
            assert ag.list() == pytest.approx(ng.list(), abs=1e-5, rel=eps)
