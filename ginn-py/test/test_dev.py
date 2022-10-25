import ginn


def test_dev():
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

    from ginn import cpu

    t2 = ginn.RealTensor(cpu(), [], [0])
    t3 = ginn.RealTensor(cpu(), [], [0])
    assert t2.dev() == t3.dev()
    assert t2.dev() != d1


def test_cpu_tensors():
    dev = ginn.PreallocCpu(100)

    t1 = ginn.RealTensor(ginn.cpu(), [2, 3])

    assert dev.used() == 0

    t2 = ginn.RealTensor(dev, [1, 2])
    assert dev.used() == (t2.size() * 4)

    t3 = ginn.RealTensor(dev)
    assert dev.used() == (t2.size() * 4)

    t4 = t2.copy_to(dev)
    assert dev.used() == (2 * t2.size() * 4)

    t5 = t1.maybe_copy_to(dev)  # should avoid copy
    assert dev.used() == (2 * t2.size() * 4)

    t1.move_to(dev)
    assert dev.used() == ((2 * t2.size() + t1.size()) * 4)

    t2.move_to(dev)  # should be a no-op
    assert dev.used() == ((2 * t2.size() + t1.size()) * 4)


def test_cpu_nodes():
    dev = ginn.PreallocCpu(100)

    x = ginn.RealData(ginn.cpu(), [1, 2])
    y = ginn.RealData(ginn.cpu(), [1, 2])
    x.has_grad = False

    assert dev.used() == 0

    x.move_to(dev)
    assert dev.used() == x.size() * 4

    y.move_to(dev)  # should move both value() and grad() but grad() is empty
    assert dev.used() == 2 * x.size() * 4

    z = ginn.Add(x, y)
    assert dev.used() == 2 * x.size() * 4


'''
y->move_to(dev); // should move both value() and grad(), but grad() is empty
CHECK(dev->used() == (2 * x->size() * sizeof(Scalar)));

auto z = x + y;
Graph g(z);
CHECK(dev->used() == (2 * x->size() * sizeof(Scalar)));

g.forward();
CHECK(dev->used() == (3 * x->size() * sizeof(Scalar)));

g.reset_grad(); // only z and y have grads
CHECK(dev->used() == (5 * x->size() * sizeof(Scalar)));
'''