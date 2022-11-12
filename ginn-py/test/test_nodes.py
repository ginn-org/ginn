import ginn
import pytest

scalars = pytest.mark.parametrize(
    "scalar",
    [
        ginn.Scalar.Real,
        ginn.Scalar.Half,
        ginn.Scalar.Int,
        ginn.Scalar.Bool,
    ],
)

scalars3 = pytest.mark.parametrize(
    "scalar", [ginn.Scalar.Real, ginn.Scalar.Half, ginn.Scalar.Int]
)

scalars2 = pytest.mark.parametrize("scalar", [ginn.Scalar.Real, ginn.Scalar.Half])

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
        check(ginn.Stack([[a, b], [c, d], [e, f]]), expected)

    @pytest.mark.parametrize(
        "scalar", [ginn.Scalar.Real, ginn.Scalar.Half, ginn.Scalar.Int]
    )
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


@scalars
@devices
def test_rowwise_cat(scalar, dev):
    a = ginn.Values(dev, [[1], [2]]).cast(scalar)
    b = ginn.Values(dev, [[3, 4], [5, 6]]).cast(scalar)
    c = ginn.Values(dev, [[7, 8, 9], [0, 1, 2]]).cast(scalar)

    res = ginn.Values(dev, [[1, 3, 4, 7, 8, 9], [2, 5, 6, 0, 1, 2]]).cast(scalar)

    check(ginn.RowwiseCat([a, b, c]), res)


@scalars
@devices
def test_reshape(scalar, dev):
    W = ginn.Values(dev, [[1, 2, 3, 4, 5, 6]]).cast(scalar)

    col = ginn.Values(dev, [[1], [2], [3], [4], [5], [6]]).cast(scalar)
    mat = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    check(ginn.Reshape(W, shape=[6, 1]), col)
    check(ginn.Reshape(W, shape=[3, 2]), mat)


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


@scalars
@devices
def test_chip(scalar, dev):
    x = ginn.Values(dev, [[1, 2], [3, 4], [5, 6], [7, 8]]).cast(scalar)

    y = ginn.Values(dev, [5, 6]).cast(scalar)
    z = ginn.Values(dev, [2, 4, 6, 8]).cast(scalar)
    check(ginn.Chip(x, 2, 0), y)
    check(ginn.Chip(x, 1, 1), z)


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


class TestBroadcast:
    @scalars
    @devices
    def test_row_broadcast(self, scalar, dev):
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

    @scalars
    @devices
    def test_col_broadcast(self, scalar, dev):
        a = ginn.Values(dev, [[1], [2], [3]]).cast(scalar)
        b = ginn.Values(dev, [[0.1, 1.2, 2.3]]).cast(scalar)

        assert a.shape == [3, 1]
        assert b.shape == [1, 3]

        a3 = ginn.Values(dev, [[1, 1, 1], [2, 2, 2], [3, 3, 3]]).cast(scalar)

        check(ginn.ColBroadcast(a, 3), a3)
        check(ginn.ColBroadcast(a, 1), a)
        check(ginn.ColBroadcast(a, ginn.Dim(b, 1)), a3)
        check(ginn.ColBroadcast(a, ginn.Dim(b, 0)), a)


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

    e = ginn.Values([[1, 12], [2, 15], [3, 18]]).cast(scalar)
    check(ginn.Add([a, b, a]), e)
    check(a + b + a, e)


@scalars3
@devices
def test_add_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    e = ginn.Values([[2, 5], [3, 6], [4, 7]]).cast(scalar)

    for s in [1, 1.0]:
        check(ginn.AddScalar(a, s), e)
        check(a + s, e)
        check(s + a, e)


@scalars3
@devices
def test_subtract_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    e = ginn.Values([[0, -3], [-1, -4], [-2, -5]]).cast(scalar)

    for s in [1, 1.0]:
        check(ginn.SubtractScalar(s, a), e)
        check(s - a, e)

    e2 = ginn.Values([[0, 3], [1, 4], [2, 5]]).cast(scalar)
    for s in [1, 1.0]:
        check(ginn.AddScalar(a, -s), e2)
        check(a - s, e2)


@scalars3
@devices
def test_subtract_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)

    pass


@scalars3
@devices
def test_prod_scalar(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    e = ginn.Values(dev, [[2, 8], [4, 10], [6, 12]]).cast(scalar)

    for s in [2, 2.0]:
        check(ginn.ProdScalar(a, s), e)
        check(a * s, e)
        check(s * a, e)


@scalars3
@devices
def test_cwise_prod(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
    e = ginn.Values(dev, [[-1, 16], [-4, 25], [-9, 36]]).cast(scalar)

    check(ginn.CwiseProd(a, b), e)


class TestCwiseProdAdd:
    @scalars3
    @devices
    def test_regular(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
        c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

        e = ginn.Values(dev, [[0, 12], [-2, 20], [-6, 30]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c), e)

    @scalars3
    @devices
    def test_regular_w_bias(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
        c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

        e = ginn.Values(dev, [[1, 16], [0, 25], [-3, 36]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c, 1), e)
        check(ginn.CwiseProdAdd(a, b, c, 1.0), e)

    @scalars3
    @devices
    def test_broadcast(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [-1, -2, -3]).cast(scalar)
        c = ginn.Values(dev, [4, 5, 6]).cast(scalar)

        e = ginn.Values(dev, [[3, 0], [1, -5], [-3, -12]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c), e)

    @scalars3
    @devices
    def test_broadcast_w_bias(self, scalar, dev):
        a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
        b = ginn.Values(dev, [-1, -2, -3]).cast(scalar)
        c = ginn.Values(dev, [4, 5, 6]).cast(scalar)

        e = ginn.Values(dev, [[4, 4], [3, 0], [0, -6]]).cast(scalar)
        check(ginn.CwiseProdAdd(a, b, c, 1), e)
        check(ginn.CwiseProdAdd(a, b, c, 1.0), e)


@scalars3
@devices
def test_cwise_max(scalar, dev):
    a = ginn.Values(dev, [[1, 4], [2, 5], [3, 6]]).cast(scalar)
    b = ginn.Values(dev, [[-1, 4], [-2, 5], [-3, 6]]).cast(scalar)
    c = ginn.Values(dev, [[1, -4], [2, -5], [3, -6]]).cast(scalar)

    check(ginn.CwiseMax([a, b, c]), a)


@scalars2
@devices
def test_nonlin(scalar, dev):
    W = ginn.Values(dev, [[-1, -2, -3], [4, 5, 6]]).cast(scalar)

    tanhW = ginn.Values(
        [[-0.76159415, -0.96402758, -0.99505475], [0.99932929, 0.99990920, 0.99998771]]
    ).cast(scalar)
    reluW = ginn.Values([[0, 0, 0], [4, 5, 6]]).cast(scalar)
    sigmW = ginn.Values(
        [[0.26894142, 0.11920292, 0.04742587], [0.98201379, 0.99330714, 0.99752737]]
    ).cast(scalar)
    smaxW = ginn.Values(
        [
            [0.00669285, 9.11051194e-04, 1.23394576e-04],
            [0.99330715, 9.99088949e-01, 9.99876605e-01],
        ]
    ).cast(scalar)
    absW = ginn.Values([[1, 2, 3], [4, 5, 6]]).cast(scalar)
    logaW = ginn.Values(
        [[0, 0.69314718, 1.09861229], [1.38629436, 1.60943791, 1.79175947]]
    ).cast(scalar)

    check(ginn.Identity(W), W)
    check(ginn.Tanh(W), tanhW)
    check(ginn.Relu(W), reluW)
    check(ginn.Sigmoid(W), sigmW, eps=1e-3 if W.scalar == ginn.Scalar.Half else 1e-6)
    check(ginn.Softmax(W), smaxW, eps=1e-3 if W.scalar == ginn.Scalar.Half else 1e-6)
    check(ginn.Sqrt(ginn.CwiseProd(W, W)), absW)
    with pytest.raises(RuntimeError):
        check(ginn.Sqrt(W), W)
    check(ginn.Log(absW), logaW)
    with pytest.raises(RuntimeError):
        check(ginn.Log(W), W)
    # TODO: Gelu forward
    # TODO: Gelu2 forward


@scalars2
@devices
def test_nonlin_extreme(scalar, dev):
    x = ginn.Values([[10000.], [-10000.]]).cast(scalar)
    x2 = ginn.Values([[5.], [-float("inf")]]).cast(scalar)

    assert x.shape == [2, 1]
    assert x2.shape == [2, 1]

    tanhx = ginn.Values([[1], [-1]]).cast(scalar)
    sigmoidx = ginn.Values([[1], [0]]).cast(scalar)
    smaxx = ginn.Values([[1], [0]]).cast(scalar)
    smaxx2 = ginn.Values([[1], [1]]).cast(scalar)

    check(ginn.Tanh(x), tanhx)
    check(ginn.Sigmoid(x), sigmoidx)
    check(ginn.Softmax(ginn.Reshape(x, [1, 2])), ginn.Reshape(smaxx2, [1, 2]))
    check(ginn.Softmax(x), smaxx)
    check(ginn.Softmax(x2), smaxx)
