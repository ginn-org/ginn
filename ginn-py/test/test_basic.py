import ginn


def test_dev():
    d1 = ginn.Cpu()
    assert d1 == d1
    assert d1.id() == d1.id()
    assert d1.precedence() == 0

    d2 = ginn.PreallocCpu(100) # 100 bytes
    assert d2.size() == 100
    assert d1.id() == d2.id()
    assert d2.precedence() == 1
    assert d2.used() == 0

    t = ginn.RealTensor(d2, [3], [1,2,3])
    assert d2.used() == 12

    d2.reset()
    assert d2.used() == 0

    from ginn import cpu
    t2 = ginn.RealTensor(cpu(), [], [0])
    t3 = ginn.RealTensor(cpu(), [], [0])
    assert t2.dev() == t3.dev()
    assert t2.dev() != d1

def test_tensor():
    a = ginn.RealTensor()
    #b = ginn.RealTensor([2]) #this doesn't exist?
    c = ginn.RealTensor([2], [1, 2])
    d = ginn.RealTensor(ginn.cpu())
    e = ginn.RealTensor(ginn.cpu(), [2])
    f = ginn.RealTensor(ginn.cpu(), [2], [1, 2])

    a.set([1, 2, 3])
    assert a.shape() == [3]
    c.set([[1, 2, 3], [4, 5, 6]])
    d.set([[1, 2, 3], [4, 5, 6]])
    assert c.shape() == [2, 3]
    assert a == a
    assert a != c
    assert c == d
