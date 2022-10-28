import ginn
import pytest


def test_cpu():
    d1 = ginn.Cpu()
    assert d1 == d1
    assert d1.id() == d1.id()
    assert d1.precedence() == 0

    d2 = ginn.PreallocCpu(100)  # 100 bytes
    assert d2.size() == 100
    assert d1.id() == d2.id()
    assert d2.precedence() == 1
    assert d2.used() == 0

    t = ginn.RealTensor(d2, [3], [1, 2, 3])
    assert d2.used() == 12

    d2.reset()
    assert d2.used() == 0

    t2 = ginn.RealTensor(ginn.cpu(), [], [0])
    t3 = ginn.RealTensor(ginn.cpu(), [], [0])
    assert t2.dev() == t3.dev()
    assert t2.dev() != d1


@pytest.mark.skipif(ginn.gpus() < 1, reason="requires gpu")
def test_gpu():
    d1 = ginn.Gpu()
    assert d1 == d1
    assert d1.id() == d1.id()
    assert d1.precedence() == 0

    d2 = ginn.PreallocGpu(0, 100)  # 100 bytes
    assert d2.size() == 100
    assert d1.id() == d2.id()
    assert d2.precedence() == 1
    assert d2.used() == 0

    t = ginn.RealTensor(d2, [3], [1, 2, 3])
    assert d2.used() == 12

    d2.reset()
    assert d2.used() == 0

    t2 = ginn.RealTensor(ginn.gpu(), [], [0])
    t3 = ginn.RealTensor(ginn.gpu(), [], [0])
    assert t2.dev() == t3.dev()
    assert t2.dev() != d1


def test_cpu_tensors():
    for scalar, size in [
        (ginn.Scalar.Real, 4),
        (ginn.Scalar.Half, 2),
        (ginn.Scalar.Int, 4),
        (ginn.Scalar.Bool, 1),
    ]:
        dev = ginn.PreallocCpu(100)

        t1 = ginn.Tensor(ginn.cpu(), [2, 3], scalar=scalar)

        assert dev.used() == 0

        t2 = ginn.Tensor(dev, [1, 2], scalar=scalar)
        assert dev.used() == t2.size() * size

        t3 = ginn.Tensor(dev, scalar=scalar)
        assert dev.used() == t2.size() * size

        t4 = t2.copy_to(dev)
        assert dev.used() == 2 * t2.size() * size

        t5 = t1.maybe_copy_to(dev)  # should avoid copy
        assert dev.used() == 2 * t2.size() * size

        t1.move_to(dev)
        assert dev.used() == (2 * t2.size() + t1.size()) * size

        t2.move_to(dev)  # should be a no-op
        assert dev.used() == (2 * t2.size() + t1.size()) * size


def test_cpu_nodes():
    for Data, size in [(ginn.RealData, 4), (ginn.HalfData, 2)]:
        dev = ginn.PreallocCpu(100)

        x = Data(ginn.cpu(), [1, 2])
        y = Data(ginn.cpu(), [1, 2])
        x.has_grad = False

        assert dev.used() == 0

        x.move_to(dev)
        assert dev.used() == x.size() * size

        y.move_to(dev)  # should move both value() and grad() but grad() is empty
        assert dev.used() == 2 * x.size() * size

        z = ginn.Add(x, y)
        g = ginn.Graph(z)
        assert dev.used() == 2 * x.size() * size

        g.forward()
        assert dev.used() == 3 * x.size() * size

        g.reset_grad()  # only z and y have grads
        assert dev.used() == 5 * x.size() * size
