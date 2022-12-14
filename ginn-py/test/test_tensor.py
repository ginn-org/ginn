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

TensorTypes = [ginn.RealTensor, ginn.HalfTensor, ginn.IntTensor, ginn.BoolTensor]
ScalarTypes = [ginn.Real, ginn.Half, ginn.Int, ginn.Bool]


def test_ctors():
    for i, Tensor in enumerate(TensorTypes):
        a = Tensor()
        b = Tensor([2])
        # c = Tensor([2], [1, 2]) #TODO: support this init for Half
        d = Tensor(ginn.cpu())
        e = Tensor(ginn.cpu(), [2])
        # f = Tensor(ginn.cpu(), [2], [1, 2]) #TODO: ditto

        for j, scalar in enumerate(ScalarTypes):
            for t in [a, b, d, e]:
                if i == j:
                    assert t.scalar == scalar
                else:
                    assert t.scalar != scalar

    for i, scalar in enumerate(ScalarTypes):
        a = ginn.Tensor(scalar=scalar)
        b = ginn.Tensor(scalar=scalar, shape=[2])
        d = ginn.Tensor(device=ginn.cpu(), scalar=scalar)
        e = ginn.Tensor(shape=[2], scalar=scalar, device=ginn.cpu())

        for j, scalar_rhs in enumerate(ScalarTypes):
            for t in [a, b, d, e]:
                if i == j:
                    assert t.scalar == scalar_rhs
                else:
                    assert t.scalar != scalar_rhs

    a = ginn.Tensor()
    b = ginn.Tensor([2])
    d = ginn.Tensor(ginn.cpu())
    e = ginn.Tensor(device=ginn.cpu(), shape=[2])


def test_casting():
    for i, Tensor in enumerate(TensorTypes):
        t = Tensor(ginn.cpu())
        t.set([[1, 2, 3], [4, 5, 6]])

        for j, Scalar in enumerate(ScalarTypes):
            if j == i:
                assert t.scalar == Scalar
            else:
                assert t.scalar != Scalar

        assert t.real() == t.cast(ginn.Real)
        assert t.int() == t.cast(ginn.Int)
        assert t.half() == t.cast(ginn.Half)
        assert t.bool() == t.cast(ginn.Bool)

    t = ginn.RealTensor(ginn.cpu(), [2, 3])
    for i, scalar in enumerate(ScalarTypes):
        t_ = t.cast(scalar)
        for j, other_scalar in enumerate(ScalarTypes):
            if j == i:
                assert t_.scalar == other_scalar
            else:
                assert t_.scalar != other_scalar


def test_values():
    a = ginn.RealTensor()
    b = a
    c = ginn.RealTensor()
    d = ginn.RealTensor()

    a.set([1, 2, 3])
    assert a.shape == [3]
    c.set([[1, 2, 3], [4, 5, 6]])
    d.set([[1, 2, 3], [4, 5, 6]])
    assert c.shape == [2, 3]
    assert a == a
    assert a == b
    assert a != c
    assert c == d

    assert a.list() == [1, 2, 3]
    assert c.list() == [1, 4, 2, 5, 3, 6]

    c.set_zero()
    assert c.list() == [0] * 6

    c.set_ones()
    assert c.list() == [1] * 6

    c.set_random()
    assert all([x <= 1 for x in c.list()])
    assert all([x >= -1 for x in c.list()])
    assert all([x != y for (x, y) in zip(c.list()[:-1], c.list()[1:])])


def test_eigen_numpy():
    import numpy as np

    a = ginn.RealTensor([2, 3], [1, 2, 3, 4, 5, 6])
    assert (a.v() == np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)).all()
    assert (a.m() == np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)).all()

    v = a.v()
    v[3] = -4
    assert a.list() == [1, 2, 3, -4, 5, 6]

    m = a.m()
    m[0, 1] = -3
    assert a.list() == [1, 2, -3, -4, 5, 6]


def test_resize():
    # TODO: other devices in this test
    vals = [1, 2, 3, 4, 5, 6]
    a = ginn.RealTensor(ginn.cpu(), [2, 1, 3], vals)

    a.shape = [6]
    assert a.size == 6
    assert a.shape == [6]
    assert a.list() == vals

    a.shape = [3, 2]
    assert a.size == 6
    assert a.shape == [3, 2]
    assert a.list() == vals

    a.shape = [4]
    assert a.size == 4
    assert a.shape == [4]

    a.shape = [2, 2]
    assert a.size == 4
    assert a.shape == [2, 2]
