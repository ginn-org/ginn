import ginn
import pytest


def check(a: ginn.BaseNode, b: ginn.BaseNode):
    ginn.Graph(a).forward()
    ginn.Graph(b).forward()
    assert a.value() == b.value()


def test_dim():
    for scalar in [ginn.Scalar.Real, ginn.Scalar.Half]:
        x = ginn.Random(ginn.cpu(), [3, 2, 1], scalar=scalar)

        d0 = ginn.Dim(x, 0)
        d1 = ginn.Dim(x, 1)
        d2 = ginn.Dim(x, 2)

        ginn.Graph(d0).forward()
        ginn.Graph(d1).forward()
        ginn.Graph(d2).forward()

        assert d0.value() == 3
        assert d1.value() == 2
        assert d2.value() == 1


class TestStack:
    def test_rank_1(self):
        a = ginn.Values([1, 2, 3, 4])
        b = ginn.Values([5, 6, 7, 8])
        c = ginn.Values([9, 10, 11, 12])
        d = ginn.Values([13, 14, 15, 16])
        e = ginn.Values([17, 18, 19, 20])
        f = ginn.Values([21, 22, 23, 24])
        expected = ginn.Values(
            [
                [[1, 5], [9, 13], [17, 21]],
                [[2, 6], [10, 14], [18, 22]],
                [[3, 7], [11, 15], [19, 23]],
                [[4, 8], [12, 16], [20, 24]],
            ]
        )
        check(ginn.Stack([[a, b], [c, d], [e, f]]), expected)

    def test_rank_2(self):
        a = ginn.Values([[1, 2], [3, 4]])
        b = ginn.Values([[5, 6], [7, 8]])
        c = ginn.Values([[9, 10], [11, 12]])
        d = ginn.Values([[13, 14], [15, 16]])
        e = ginn.Values([[17, 18], [19, 20]])
        f = ginn.Values([[21, 22], [23, 24]])
        expected = ginn.Values(
            [
                [[[1, 5], [9, 13], [17, 21]], [[2, 6], [10, 14], [18, 22]]],
                [[[3, 7], [11, 15], [19, 23]], [[4, 8], [12, 16], [20, 24]]],
            ]
        )
        check(ginn.Stack([[a, b], [c, d], [e, f]]), expected)

    def test_errors(self):
        a = ginn.Values([1, 2, 3, 4])
        b = ginn.Values([5, 6, 7, 8])
        c = ginn.Values([9, 10, 11, 12])
        d = ginn.Values([13, 14, 15, 16])
        e = ginn.Values([17, 18, 19])
        f = ginn.Values([21, 22, 23, 24])

        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([[a, b], [c, d], [e, f]])).forward()
        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([[a, b], [c, d], [e]])).forward()
        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([])).forward()
        with pytest.raises(RuntimeError):
            ginn.Graph(ginn.Stack([[]])).forward()


def test_cat():
    a = ginn.Values([[1, 2]])
    b = ginn.Values([[3, 4], [5, 6]])
    c = ginn.Values([[7, 8], [9, 0]])

    cat = ginn.Cat([a, b, c])

    res = ginn.Values([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]])

    check(cat, res)
