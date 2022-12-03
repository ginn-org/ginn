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
import pytest
from test_util import check_grad

scalars = pytest.mark.parametrize(
    "scalar",
    [
        ginn.Real,
        ginn.Half,
        ginn.Int,
        ginn.Bool,
    ],
)

scalars3 = pytest.mark.parametrize("scalar", [ginn.Real, ginn.Half, ginn.Int])

scalars2 = pytest.mark.parametrize("scalar", [ginn.Real, ginn.Half])

gpu = ginn.gpu() if ginn.gpus() > 0 else None

devices = pytest.mark.parametrize(
    "dev",
    [
        ginn.cpu(),
        pytest.param(gpu, marks=pytest.mark.skipif(gpu is None, reason="requires gpu")),
    ],
)


def check(a: ginn.BaseNode, b: ginn.BaseNode, eps=1e-6):
    ginn.Graph(a).forward()
    ginn.Graph(b).forward()
    a_ = a.value.maybe_copy_to(ginn.cpu())
    b_ = b.value.maybe_copy_to(ginn.cpu())
    assert a.dev == b.dev
    assert a_.shape == b_.shape
    assert a_.list() == pytest.approx(b_.list(), rel=eps)


@scalars2
@devices
def test_dim(scalar, dev):
    x = ginn.Random(dev, [3, 2, 1], scalar=scalar)

    d0 = ginn.Dim(x, 0)
    d1 = ginn.Dim(x, 1)
    d2 = ginn.Dim(x, 2)

    ginn.Graph(d0).forward()
    ginn.Graph(d1).forward()
    ginn.Graph(d2).forward()

    assert d0.value == 3
    assert d1.value == 2
    assert d2.value == 1


class TestStack:
    @scalars3
    @devices
    def test_rank_1(self, scalar, dev):
        a = ginn.Values(values=[1, 2, 3, 4], dev=dev).cast(scalar)
        b = ginn.Values(values=[5, 6, 7, 8], dev=dev).cast(scalar)
        c = ginn.Values(values=[9, 10, 11, 12], dev=dev).cast(scalar)
        d = ginn.Values(values=[13, 14, 15, 16], dev=dev).cast(scalar)
        e = ginn.Values(values=[17, 18, 19, 20], dev=dev).cast(scalar)
        f = ginn.Values(values=[21, 22, 23, 24], dev=dev).cast(scalar)
        expected = ginn.Values(
            values=[
                [[1, 5], [9, 13], [17, 21]],
                [[2, 6], [10, 14], [18, 22]],
                [[3, 7], [11, 15], [19, 23]],
                [[4, 8], [12, 16], [20, 24]],
            ],
            dev=dev,
        ).cast(scalar)
        s = ginn.Stack([[a, b], [c, d], [e, f]])
        check(s, expected)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(s, [a, b, c, d, e, f])

    @scalars3
    @devices
    def test_rank_2(self, scalar, dev):
        a = ginn.Values(dev=dev, values=[[1, 2], [3, 4]]).cast(scalar)
        b = ginn.Values(dev=dev, values=[[5, 6], [7, 8]]).cast(scalar)
        c = ginn.Values(dev=dev, values=[[9, 10], [11, 12]]).cast(scalar)
        d = ginn.Values(dev=dev, values=[[13, 14], [15, 16]]).cast(scalar)
        e = ginn.Values(dev=dev, values=[[17, 18], [19, 20]]).cast(scalar)
        f = ginn.Values(dev=dev, values=[[21, 22], [23, 24]]).cast(scalar)
        expected = ginn.Values(
            dev=dev,
            values=[
                [[[1, 5], [9, 13], [17, 21]], [[2, 6], [10, 14], [18, 22]]],
                [[[3, 7], [11, 15], [19, 23]], [[4, 8], [12, 16], [20, 24]]],
            ],
        ).cast(scalar)
        s = ginn.Stack([[a, b], [c, d], [e, f]])
        check(s, expected)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(s, [a, b, c, d, e, f])

    @scalars3
    @devices
    def test_errors(self, scalar, dev):
        a = ginn.Values(dev, [1, 2, 3, 4]).cast(scalar)
        b = ginn.Values(dev, [5, 6, 7, 8]).cast(scalar)
        c = ginn.Values(dev, [9, 10, 11, 12]).cast(scalar)
        d = ginn.Values(dev, [13, 14, 15, 16]).cast(scalar)
        e = ginn.Values(dev, [17, 18, 19]).cast(scalar)
        f = ginn.Values(dev, [21, 22, 23, 24]).cast(scalar)

        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([[a, b], [c, d], [e, f]])).forward()
        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([[a, b], [c, d], [e]])).forward()
        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([])).forward()
        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([[]])).forward()


@scalars
@devices
def test_cat(scalar, dev):
    a = ginn.Values(dev, [[1, 2]]).cast(scalar)
    b = ginn.Values(dev, [[3, 4], [5, 6]]).cast(scalar)
    c = ginn.Values(dev, [[7, 8], [9, 0]]).cast(scalar)

    res = ginn.Values(dev, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]]).cast(scalar)

    check(ginn.Cat([a, b, c]), res)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Cat([a, b, c]), [a, b, c])


@scalars
@devices
def test_rowwise_cat(scalar, dev):
    a = ginn.Values(dev, [[1], [2]]).cast(scalar)
    b = ginn.Values(dev, [[3, 4], [5, 6]]).cast(scalar)
    c = ginn.Values(dev, [[7, 8, 9], [0, 1, 2]]).cast(scalar)

    res = ginn.Values(dev, [[1, 3, 4, 7, 8, 9], [2, 5, 6, 0, 1, 2]]).cast(scalar)

    check(ginn.RowwiseCat([a, b, c]), res)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.RowwiseCat([a, b, c]), [a, b, c])


@scalars
@devices
def test_reshape(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3, 4, 5, 6]]).cast(scalar)

    col = ginn.Values(dev, [[1], [2], [3], [4], [5], [6]]).cast(scalar)
    mat = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    check(ginn.Reshape(W, shape=[6, 1]), col)
    check(ginn.Reshape(W, shape=[3, 2]), mat)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Reshape(W, shape=[6, 1]), [W])
        check_grad(ginn.Reshape(W, shape=[3, 2]), [W])


@scalars
@devices
def test_rank_view(scalar, dev):
    W = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    col = ginn.Values(dev, [1, 2, 3, 4, 5, 6]).cast(scalar)
    mat = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    ten = ginn.Values(dev, [[[1], [4]], [[2], [5]], [[3], [6]]]).cast(scalar)

    check(ginn.RankView(W, 1), col)
    check(ginn.RankView(W, 2), mat)
    check(ginn.RankView(W, 3), ten)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.RankView(W, 1), [W])
        check_grad(ginn.RankView(W, 2), [W])
        check_grad(ginn.RankView(W, 3), [W])


@scalars
@devices
def test_slice(scalar, dev):
    x = ginn.Values(dev, [[1, 2], [3, 4], [5, 6], [7, 8]]).cast(scalar)
    assert x.shape == [4, 2]

    out = ginn.Values(dev, [[3, 4], [5, 6], [7, 8]]).cast(scalar)
    assert out.shape == [3, 2]
    check(ginn.Slice(x, [1, 0], [3, 2]), out)

    out = ginn.Values(dev, [[2], [4], [6], [8]]).cast(scalar)
    assert out.shape == [4, 1]
    check(ginn.Slice(x, offsets=[0, 1], sizes=[4, 1]), out)

    out = ginn.Values(dev, [[5], [7]]).cast(scalar)
    assert out.shape == [2, 1]
    check(ginn.Slice(x, [2, 0], [2, 1]), out)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Slice(x, [1, 0], [3, 2]), [x])
        check_grad(ginn.Slice(x, offsets=[0, 1], sizes=[4, 1]), [x])
        check_grad(ginn.Slice(x, [2, 0], [2, 1]), [x])


@scalars
@devices
def test_chip(scalar, dev):
    x = ginn.Values(dev, [[1, 2], [3, 4], [5, 6], [7, 8]]).cast(scalar)

    y = ginn.Values(dev, [5, 6]).cast(scalar)
    z = ginn.Values(dev, [2, 4, 6, 8]).cast(scalar)
    check(ginn.Chip(x, 2, 0), y)
    check(ginn.Chip(x, 1, 1), z)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Chip(x, 2, 0), [x])
        check_grad(ginn.Chip(x, 1, 1), [x])


@scalars
@devices
def test_permute(scalar, dev):
    a = ginn.Values(
        dev,
        [
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 22, 23, 24]],
        ],
    ).cast(scalar)
    assert a.shape == [3, 2, 4]

    b = ginn.Values(
        dev,
        [
            [[1, 5], [2, 6], [3, 7], [4, 8]],
            [[9, 13], [10, 14], [11, 15], [12, 16]],
            [[17, 21], [18, 22], [19, 23], [20, 24]],
        ],
    ).cast(scalar)
    c = ginn.Values(
        dev,
        [
            [[1, 9, 17], [5, 13, 21]],
            [[2, 10, 18], [6, 14, 22]],
            [[3, 11, 19], [7, 15, 23]],
            [[4, 12, 20], [8, 16, 24]],
        ],
    ).cast(scalar)
    d = ginn.Values(
        dev,
        [
            [[1, 2, 3, 4], [9, 10, 11, 12], [17, 18, 19, 20]],
            [[5, 6, 7, 8], [13, 14, 15, 16], [21, 22, 23, 24]],
        ],
    ).cast(scalar)
    e = ginn.Values(
        dev,
        [
            [[1, 9, 17], [2, 10, 18], [3, 11, 19], [4, 12, 20]],
            [[5, 13, 21], [6, 14, 22], [7, 15, 23], [8, 16, 24]],
        ],
    ).cast(scalar)
    f = ginn.Values(
        dev,
        [
            [[1, 5], [9, 13], [17, 21]],
            [[2, 6], [10, 14], [18, 22]],
            [[3, 7], [11, 15], [19, 23]],
            [[4, 8], [12, 16], [20, 24]],
        ],
    ).cast(scalar)

    check(ginn.Permute(a, [0, 2, 1]), b)
    check(ginn.Transpose(a, 1, 2), b)
    check(ginn.Permute(a, [2, 1, 0]), c)
    check(ginn.Transpose(a, 2, 0), c)
    check(ginn.Permute(a, [1, 0, 2]), d)
    check(ginn.Transpose(a, 0, 1), d)
    check(ginn.Permute(a, [1, 2, 0]), e)
    check(ginn.Permute(a, [2, 0, 1]), f)
    check(ginn.Permute(a, [0, 1, 2]), a)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Permute(a, [0, 2, 1]), [a])
        check_grad(ginn.Transpose(a, 1, 2), [a])
        check_grad(ginn.Permute(a, [2, 1, 0]), [a])
        check_grad(ginn.Transpose(a, 2, 0), [a])
        check_grad(ginn.Permute(a, [1, 0, 2]), [a])
        check_grad(ginn.Transpose(a, 0, 1), [a])
        check_grad(ginn.Permute(a, [1, 2, 0]), [a])
        check_grad(ginn.Permute(a, [2, 0, 1]), [a])
        check_grad(ginn.Permute(a, [0, 1, 2]), [a])


@scalars
@devices
def test_row_broadcast(scalar, dev):
    a = ginn.Values(dev, [[1], [2], [3]]).cast(scalar)
    b = ginn.Values(dev, [[0.1, 1.2, 2.3]]).cast(scalar)

    assert a.shape == [3, 1]
    assert b.shape == [1, 3]

    b3 = ginn.Values(dev, [[0.1, 1.2, 2.3], [0.1, 1.2, 2.3], [0.1, 1.2, 2.3]]).cast(
        scalar
    )

    check(ginn.RowBroadcast(b, 3), b3)
    check(ginn.RowBroadcast(b, 1), b)
    check(ginn.RowBroadcast(b, ginn.Dim(a, 0)), b3)
    check(ginn.RowBroadcast(b, ginn.Dim(a, 1)), b)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.RowBroadcast(b, 3), [b])
        check_grad(ginn.RowBroadcast(b, 1), [b])
        check_grad(ginn.RowBroadcast(b, ginn.Dim(a, 0)), [b])
        check_grad(ginn.RowBroadcast(b, ginn.Dim(a, 1)), [b])


@scalars
@devices
def test_col_broadcast(scalar, dev):
    a = ginn.Values(dev, [[1], [2], [3]]).cast(scalar)
    b = ginn.Values(dev, [[0.1, 1.2, 2.3]]).cast(scalar)

    assert a.shape == [3, 1]
    assert b.shape == [1, 3]

    a3 = ginn.Values(dev, [[1, 1, 1], [2, 2, 2], [3, 3, 3]]).cast(scalar)

    check(ginn.ColBroadcast(a, 3), a3)
    check(ginn.ColBroadcast(a, 1), a)
    check(ginn.ColBroadcast(a, ginn.Dim(b, 1)), a3)
    check(ginn.ColBroadcast(a, ginn.Dim(b, 0)), a)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.ColBroadcast(a, 3), [a])
        check_grad(ginn.ColBroadcast(a, 1), [a])
        check_grad(ginn.ColBroadcast(a, ginn.Dim(b, 1)), [a])
        check_grad(ginn.ColBroadcast(a, ginn.Dim(b, 0)), [a])


@scalars
@devices
def test_upper_tri(scalar, dev):
    tri2 = ginn.Values(dev, [[1, 1], [0, 1]]).cast(scalar)
    tri5 = ginn.Values(
        dev,
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ],
    ).cast(scalar)

    check(ginn.UpperTri(dev, 2, scalar=scalar), tri2)
    check(ginn.UpperTri(dev, 5, scalar=scalar), tri5)


@scalars3
@devices
def test_add(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
    c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

    e = ginn.Values(dev, [[0, 8], [0, 10], [0, 12]]).cast(scalar)
    check(ginn.Add(a, b), e)
    check(a + b, e)

    e = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    check(ginn.Add([a, b, c]), e)
    check(a + b + c, e)

    e = ginn.Values(dev, [[1, 12], [2, 15], [3, 18]]).cast(scalar)
    check(ginn.Add([a, b, a]), e)
    check(a + b + a, e)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Add(a, b), [a, b])
        check_grad(a + b, [a, b])
        check_grad(ginn.Add([a, b, c]), [a, b, c])
        check_grad(a + b + c, [a, b, c])
        check_grad(ginn.Add([a, b, a]), [a, b])
        check_grad(a + b + a, [a, b])


@scalars3
@devices
def test_add_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    e = ginn.Values(dev, [[2, 5], [3, 6], [4, 7]]).cast(scalar)

    for s in [1, 1.0]:
        check(ginn.AddScalar(a, s), e)
        check(a + s, e)
        check(s + a, e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.AddScalar(a, s), [a])
            check_grad(a + s, [a])
            check_grad(s + a, [a])


@scalars3
@devices
def test_subtract_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    e = ginn.Values(dev, [[0, -3], [-1, -4], [-2, -5]]).cast(scalar)

    for s in [1, 1.0]:
        check(ginn.SubtractScalar(s, a), e)
        check(s - a, e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.SubtractScalar(s, a), [a])
            check_grad(s - a, [a])

    e2 = ginn.Values(dev, [[0, 3], [1, 4], [2, 5]]).cast(scalar)
    for s in [1, 1.0]:
        check(ginn.AddScalar(a, -s), e2)
        check(a - s, e2)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.AddScalar(a, -s), [a])
            check_grad(a - s, [a])


@scalars3
@devices
def test_prod_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    e = ginn.Values(dev, [[2, 8], [4, 10], [6, 12]]).cast(scalar)

    for s in [2, 2.0]:
        check(ginn.ProdScalar(a, s), e)
        check(a * s, e)
        check(s * a, e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.ProdScalar(a, s), [a])
            check_grad(a * s, [a])
            check_grad(s * a, [a])


@scalars3
@devices
def test_cwise_prod(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
    e = ginn.Values(dev, [[-1, 16], [-4, 25], [-9, 36]]).cast(scalar)

    check(ginn.CwiseProd(a, b), e)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.CwiseProd(a, b), [a, b])


class TestCwiseProdAdd:
    @scalars3
    @devices
    def test_regular(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
        c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

        e = ginn.Values(dev, [[0, 12], [-2, 20], [-6, 30]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c), e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.CwiseProdAdd(a, b, c), [a, b, c])

    @scalars3
    @devices
    def test_regular_w_bias(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
        c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

        e = ginn.Values(dev, [[1, 16], [0, 25], [-3, 36]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c, 1), e)
        check(ginn.CwiseProdAdd(a, b, c, 1.0), e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.CwiseProdAdd(a, b, c, 1), [a, b, c])
            check_grad(ginn.CwiseProdAdd(a, b, c, 1.0), [a, b, c])

    @scalars3
    @devices
    def test_broadcast(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [-1, -2, -3]).cast(scalar)
        c = ginn.Values(dev, [4, 5, 6]).cast(scalar)

        e = ginn.Values(dev, [[3, 0], [1, -5], [-3, -12]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c), e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.CwiseProdAdd(a, b, c), [a, b, c])

    @scalars3
    @devices
    def test_broadcast_w_bias(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [-1, -2, -3]).cast(scalar)
        c = ginn.Values(dev, [4, 5, 6]).cast(scalar)

        e = ginn.Values(dev, [[4, 4], [3, 0], [0, -6]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c, 1), e)
        check(ginn.CwiseProdAdd(a, b, c, 1.0), e)

        if scalar == ginn.Real and dev == ginn.cpu():
            check_grad(ginn.CwiseProdAdd(a, b, c, 1), [a, b, c])
            check_grad(ginn.CwiseProdAdd(a, b, c, 1.0), [a, b, c])


@scalars3
@devices
def test_cwise_max(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
    c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

    check(ginn.CwiseMax([a, b, c]), a)

    if scalar == ginn.Real and dev == ginn.cpu():
        # gradcheck will fail for equal values
        a = ginn.Values(dev, [[1.5, 4], [2, 5], [3.7, 6]]).cast(scalar)
        b = ginn.Values(dev, [[-1, 4.2], [-2, 5.2], [-3, 6.6]]).cast(scalar)
        c = ginn.Values(dev, [[1, -4], [2.3, -5], [3, -6]]).cast(scalar)
        check_grad(ginn.CwiseMax([a, b, c]), [a, b, c])


@scalars2
@devices
def test_nonlin(scalar, dev):
    W = ginn.Values(dev, [[-1, -2, -3], [4, 5, 6]]).cast(scalar)

    tanhW = ginn.Values(
        dev,
        [[-0.76159415, -0.96402758, -0.99505475], [0.99932929, 0.99990920, 0.99998771]],
    ).cast(scalar)
    reluW = ginn.Values(dev, [[0, 0, 0], [4, 5, 6]]).cast(scalar)
    sigmW = ginn.Values(
        dev,
        [[0.26894142, 0.11920292, 0.04742587], [0.98201379, 0.99330714, 0.99752737]],
    ).cast(scalar)
    smaxW = ginn.Values(
        dev,
        [
            [0.00669285, 9.11051194e-04, 1.23394576e-04],
            [0.99330715, 9.99088949e-01, 9.99876605e-01],
        ],
    ).cast(scalar)
    absW = ginn.Values(dev, [[1, 2, 3], [4, 5, 6]]).cast(scalar)
    logaW = ginn.Values(
        dev, [[0, 0.69314718, 1.09861229], [1.38629436, 1.60943791, 1.79175947]]
    ).cast(scalar)

    check(ginn.Identity(W), W)
    check(ginn.Tanh(W), tanhW)
    check(ginn.Relu(W), reluW)
    check(ginn.Sigmoid(W), sigmW, eps=1e-3 if W.scalar == ginn.Half else 1e-6)
    check(ginn.Softmax(W), smaxW, eps=1e-3 if W.scalar == ginn.Half else 1e-6)
    check(ginn.Sqrt(ginn.CwiseProd(W, W)), absW)
    with pytest.raises(RuntimeError):
        check(ginn.Sqrt(W), W)
    check(ginn.Log(absW), logaW)
    with pytest.raises(RuntimeError):
        check(ginn.Log(W), W)
    # TODO: Gelu forward
    # TODO: Gelu2 forward

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Identity(W), [W])
        check_grad(ginn.Tanh(W * 0.1), [W])
        check_grad(ginn.Relu(W), [W])
        check_grad(ginn.Sigmoid(W), [W])
        check_grad(ginn.Softmax(W * 0.1), [W])
        check_grad(ginn.Sqrt(ginn.CwiseProd(W, W)), [W])


@scalars2
@devices
def test_nonlin_extreme(scalar, dev):
    x = ginn.Values(dev, [[10000.0], [-10000.0]]).cast(scalar)
    x2 = ginn.Values(dev, [[5.0], [-float("inf")]]).cast(scalar)

    tanhx = ginn.Values(dev, [[1], [-1]]).cast(scalar)
    sigmoidx = ginn.Values(dev, [[1], [0]]).cast(scalar)
    smaxx = ginn.Values(dev, [[1], [0]]).cast(scalar)
    smaxx2 = ginn.Values(dev, [[1], [1]]).cast(scalar)

    check(ginn.Tanh(x), tanhx)
    # curiously, the following broke when I build with -Ofast, sigmoid is less exact -- what changed?
    check(ginn.Sigmoid(x) + 1.0, sigmoidx + 1.0)
    check(ginn.Softmax(ginn.Reshape(x, [1, 2])), ginn.Reshape(smaxx2, [1, 2]))
    check(ginn.Softmax(x), smaxx)
    check(ginn.Softmax(x2), smaxx)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Sigmoid(x), [x])
        check_grad(ginn.Softmax(x), [x])
        check_grad(ginn.Softmax(x2), [x2])


@scalars2
@devices
def test_pick_and_friends(scalar, dev):
    W = ginn.Values(
        dev, [[-0.5, 0.55, -0.45], [1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [0.3, -0.33, 1.3]]
    ).cast(scalar)
    p = ginn.Values(dev, [[0.3, 0.55, -1.0]]).cast(scalar)
    psm = ginn.Values(dev, [[0.23787436, 0.15987601, 0.06482681]]).cast(scalar)
    pnlsm = ginn.Values(dev, [[1.43601264, 1.8333567, 2.73603603]]).cast(scalar)

    il = [3, 0, 1]
    it = ginn.Values(dev, [3, 0, 1]).cast(ginn.Int)
    it.has_grad = False

    eps = 1e-3 if W.scalar == ginn.Half else 1e-6

    check(ginn.Pick(W, il), p)
    check(ginn.Pick(W, it), p)
    check(ginn.PickSoftmax(W, il), psm, eps=eps)
    check(ginn.PickSoftmax(W, it), psm, eps=eps)
    check(ginn.Pick(ginn.Softmax(W), il), psm, eps=eps)
    check(ginn.Pick(ginn.Softmax(W), it), psm, eps=eps)
    check(ginn.PickNegLogSoftmax(W, il), pnlsm, eps=eps)
    check(ginn.PickNegLogSoftmax(W, it), pnlsm, eps=eps)
    check(ginn.Pick(-ginn.Log(ginn.Softmax(W)), il), pnlsm, eps=eps)
    check(ginn.Pick(-ginn.Log(ginn.Softmax(W)), it), pnlsm, eps=eps)
    check(-ginn.Log(ginn.Pick(ginn.Softmax(W), il)), pnlsm, eps=eps)
    check(-ginn.Log(ginn.Pick(ginn.Softmax(W), it)), pnlsm, eps=eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Pick(W, il), [W])
        check_grad(ginn.Pick(W, it), [W])
        check_grad(ginn.PickSoftmax(W, il), [W])
        check_grad(ginn.PickSoftmax(W, it), [W])
        check_grad(ginn.Pick(ginn.Softmax(W), il), [W])
        check_grad(ginn.Pick(ginn.Softmax(W), it), [W])
        check_grad(ginn.PickNegLogSoftmax(W, il), [W])
        check_grad(ginn.PickNegLogSoftmax(W, it), [W])


# TODO: add "Half" to the following test once I make ginn::Half known to python
@pytest.mark.parametrize("scalar", [ginn.Real, ginn.Int, ginn.Bool])
@devices
def test_select(scalar, dev):
    if_ = ginn.Values(dev, [[1, 0], [0, 1], [1, 0]]).bool()
    if_.has_grad = False  # TODO: maybe this should be default for bool?

    a = ginn.Values(dev, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).cast(scalar)
    b = ginn.Values(dev, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).cast(scalar)

    y1 = ginn.Values(dev, [[1.0, 0.2], [0.3, 4.0], [5.0, 0.6]]).cast(scalar)
    y2 = ginn.Values(dev, [[1.0, 7.0], [7.0, 4.0], [5.0, 7.0]]).cast(scalar)
    y3 = ginn.Values(dev, [[7.0, 0.2], [0.3, 7.0], [7.0, 0.6]]).cast(scalar)
    y4 = ginn.Values(dev, [[7.0, -2], [-2, 7.0], [7.0, -2]]).cast(scalar)

    def val(x):
        if scalar == ginn.Real:
            return float(x)
        elif scalar == ginn.Int:
            return int(x)
        elif scalar == ginn.Bool:
            return bool(x)

    check(ginn.Select(if_, a, b), y1)
    check(ginn.Select(if_, a, val(7)), y2)
    check(ginn.Select(if_, val(7), b), y3)
    check(ginn.Select(if_, val(7), val(-2)), y4)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Select(if_, a, b), [a, b])
        check_grad(ginn.Select(if_, a, val(7)), [a])
        check_grad(ginn.Select(if_, val(7), b), [b])


@scalars
@devices
def test_mask(scalar, dev):
    mask = ginn.Values(dev, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).cast(scalar)
    mask.has_grad = False

    mask2 = ginn.Values(dev, [[1.0], [0.0], [1.0]]).cast(scalar)
    mask2.has_grad = False

    a = ginn.Values(dev, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).cast(scalar)

    y1 = ginn.Values(dev, [[1.0, -1.0], [-1.0, 4.0], [5.0, -1.0]]).cast(scalar)
    y2 = ginn.Values(dev, [[1.0, 2.0], [7.0, 7.0], [5.0, 6.0]]).cast(scalar)

    check(ginn.Mask(a, mask, -1), y1)
    check(ginn.Mask(a, mask, -1.0), y1)
    check(ginn.Mask(a, mask2, 7), y2)
    check(ginn.Mask(a, mask2, 7.0), y2)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Mask(a, mask, -1), [a])
        check_grad(ginn.Mask(a, mask, -1.0), [a])
        check_grad(ginn.Mask(a, mask2, 7), [a])
        check_grad(ginn.Mask(a, mask2, 7.0), [a])


@scalars2
@devices
def test_axis_sum(scalar, dev):
    V = ginn.Values(
        dev, [[[1, 2, 3], [4, 5, 6]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]
    ).cast(scalar)
    V0 = ginn.Values(dev, [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]).cast(scalar)
    V01 = ginn.Values(dev, [5.5, 7.7, 9.9]).cast(scalar)
    V012 = ginn.Values(dev, 23.1).cast(scalar)
    V1 = ginn.Values(dev, [[5, 7, 9], [0.5, 0.7, 0.9]]).cast(scalar)
    V12 = ginn.Values(dev, [21, 2.1]).cast(scalar)
    V2 = ginn.Values(dev, [[6, 15], [0.6, 1.5]]).cast(scalar)
    V02 = ginn.Values(dev, [6.6, 16.5]).cast(scalar)

    eps = 1e-3 if scalar == ginn.Half else 1e-6
    check(ginn.AxisSum(V, [0]), V0, eps)
    check(ginn.AxisSum(V, [0, 1]), V01, eps)
    check(ginn.AxisSum(V, [0, 1, 2]), V012, eps)
    check(ginn.AxisSum(V, [1]), V1, eps)
    check(ginn.AxisSum(V, [1, 2]), V12, eps)
    check(ginn.AxisSum(V, [2]), V2, eps)
    check(ginn.AxisSum(V, [0, 2]), V02, eps)
    with pytest.raises(RuntimeError):
        ginn.AxisSum(V, [0, 0])
    with pytest.raises(RuntimeError):
        ginn.AxisSum(V, [2, 0])

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.AxisSum(V, [0]), [V])
        check_grad(ginn.AxisSum(V, [0, 1]), [V])
        check_grad(ginn.AxisSum(V, [0, 1, 2]), [V])
        check_grad(ginn.AxisSum(V, [1]), [V])
        check_grad(ginn.AxisSum(V, [1, 2]), [V])
        check_grad(ginn.AxisSum(V, [2]), [V])
        check_grad(ginn.AxisSum(V, [0, 2]), [V])


@scalars2
@devices
def test_reduce(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3], [4, 5, 6]]).cast(scalar)

    s = ginn.Values(dev, 21).cast(scalar)
    μ = ginn.Values(dev, 3.5).cast(scalar)
    σ2 = ginn.Values(dev, 35.0 / 12.0).cast(scalar)

    check(ginn.Sum(W), s)
    check(ginn.Mean(W), μ)
    check(ginn.Variance(W), σ2)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Sum(W), [W])
        check_grad(ginn.Mean(W), [W])
        check_grad(ginn.Variance(W), [W])


@scalars3
@devices
def test_reduce(scalar, dev):
    a = ginn.Values(dev, [[1], [2], [3], [4], [5]]).cast(scalar)
    b = ginn.Values(dev, [[5], [4], [3], [2], [1]]).cast(scalar)

    y = ginn.Values(dev, [[1], [1], [0], [0], [0]]).bool()
    check(ginn.LessThan(a, b), y)
    check(a < b, y)


@scalars2
@devices
def test_prod(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3], [4, 5, 6]]).cast(scalar)
    V = ginn.Values(dev, [[0.6, 0.5], [0.4, -0.1], [-0.2, -0.3]]).cast(scalar)

    WV = ginn.Values(dev, [[0.8, -0.6], [3.2, -0.3]]).cast(scalar)

    eps = 1e-2 if scalar == ginn.Half else 1e-6
    check(ginn.Prod(W, V), WV, eps=eps)
    check(W * V, WV, eps=eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Prod(W, V), [W, V])
        check_grad(W * V, [W, V])


@scalars2
@devices
def test_batched_prod(scalar, dev):
    a = ginn.Random(dev, [2, 3, 4]).cast(scalar)
    b = ginn.Random(dev, [3, 5, 4]).cast(scalar)
    c = ginn.BatchedProd(a, b)

    c0 = ginn.Chip(c, 0, 2)
    c1 = ginn.Chip(c, 1, 2)
    c2 = ginn.Chip(c, 2, 2)
    c3 = ginn.Chip(c, 3, 2)

    a0, b0 = ginn.Chip(a, 0, 2), ginn.Chip(b, 0, 2)
    a1, b1 = ginn.Chip(a, 1, 2), ginn.Chip(b, 1, 2)
    a2, b2 = ginn.Chip(a, 2, 2), ginn.Chip(b, 2, 2)
    a3, b3 = ginn.Chip(a, 3, 2), ginn.Chip(b, 3, 2)

    c0_ = a0 * b0
    c1_ = a1 * b1
    c2_ = a2 * b2
    c3_ = a3 * b3

    eps = 1e-2 if scalar == ginn.Half else 1e-6
    check(c0, c0_, eps)
    check(c1, c1_, eps)
    check(c2, c2_, eps)
    check(c3, c3_, eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(c, [a, b], eps=2e-2)

    a = ginn.Random(dev, [2, 3, 2, 2]).cast(scalar)
    b = ginn.Random(dev, [3, 5, 2, 2]).cast(scalar)
    c = ginn.BatchedProd(a, b)

    def ChipTwice(x, i, j):
        return ginn.Chip(ginn.Chip(x, j, 3), i, 2)

    c00 = ChipTwice(c, 0, 0)
    c01 = ChipTwice(c, 0, 1)
    c10 = ChipTwice(c, 1, 0)
    c11 = ChipTwice(c, 1, 1)

    a00, b00 = ChipTwice(a, 0, 0), ChipTwice(b, 0, 0)
    a01, b01 = ChipTwice(a, 0, 1), ChipTwice(b, 0, 1)
    a10, b10 = ChipTwice(a, 1, 0), ChipTwice(b, 1, 0)
    a11, b11 = ChipTwice(a, 1, 1), ChipTwice(b, 1, 1)

    c00_ = a00 * b00
    c01_ = a01 * b01
    c10_ = a10 * b10
    c11_ = a11 * b11

    eps = 1e-2 if scalar == ginn.Half else 1e-6
    check(c00, c00_, eps)
    check(c01, c01_, eps)
    check(c10, c10_, eps)
    check(c11, c11_, eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(c, [a, b], eps=2e-2)


@scalars2
@devices
def test_affine(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3], [4, 5, 6]]).cast(scalar)
    V = ginn.Values(dev, [[0.6], [0.4], [-0.2]]).cast(scalar)
    b = ginn.Values(dev, [[0.01], [0.02]]).cast(scalar)

    WVb = ginn.Values(dev, [[0.81], [3.22]]).cast(scalar)

    eps = 2e-3 if scalar == ginn.Half else 1e-6
    check(ginn.Affine([W, V, b]), WVb, eps=eps)
    check(W * V + b, WVb, eps=eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Affine([W, V, b]), [W, V, b])
        check_grad(W * V + b, [W, V, b])

    # TODO: add Affine with nonlins after Nonlin refactor


@scalars2
@devices
def test_affine_w_broadcast(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3], [4, 5, 6]]).cast(scalar)
    V = ginn.Values(dev, [[6, 5], [4, -1], [-2, -3]]).cast(scalar)
    b = ginn.Values(dev, [[0.1], [0.2]]).cast(scalar)

    WVb = ginn.Values(dev, [[8.1, -5.9], [32.2, -2.8]]).cast(scalar)

    eps = 2e-3 if scalar == ginn.Half else 1e-6
    check(ginn.Affine([W, V, b]), WVb, eps=eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Affine([W, V, b]), [W, V, b])


@scalars2
@devices
def test_affine_w_high_rank(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3], [4, 5, 6]]).cast(scalar)  # 2 x 3
    # 3 x 2 x 2
    V = ginn.Values(
        dev, [[[6, 5], [6, 5]], [[4, -1], [4, -1]], [[-2, -3], [-2, -3]]]
    ).cast(scalar)
    b = ginn.Values(dev, [[0.1], [0.2]]).cast(scalar)
    # 2 x 2 x 2
    WVb = ginn.Values(
        dev, [[[8.1, -5.9], [8.1, -5.9]], [[32.2, -2.8], [32.2, -2.8]]]
    ).cast(scalar)

    eps = 2e-3 if scalar == ginn.Half else 1e-6
    check(ginn.Affine([W, V, b]), WVb, eps=eps)

    if scalar == ginn.Real and dev == ginn.cpu():
        check_grad(ginn.Affine([W, V, b]), [W, V, b])


@scalars2
@devices
def test_weight(scalar, dev):
    W = ginn.Weight(scalar=scalar, device=dev, shape=[2, 3])
    W.set_random()
    t = W.value.copy_to(dev)

    assert W.forwarded
    W.reset_forwarded()
    assert W.forwarded

    W.forward()
    assert W.value == t

    W.reset_grad()

    W2 = W.copy(ginn.Copy.Tied)
    assert W2.value is W.value
    assert W2.grad is not W.grad

    W3 = W.copy(ginn.Copy.Deep)
    assert W3.value == W.value
    assert W3.value is not W.value
    assert W3.grad is not W.grad
