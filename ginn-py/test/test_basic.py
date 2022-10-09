import ginn


def test_dev():
    d1 = ginn.Cpu()
    assert d1.id() == d1.id()
    assert d1.precedence() == 0

    d2 = ginn.PreallocCpu(100) # 100 bytes
    assert d1.id() == d2.id()
    assert d2.precedence() == 1
    assert d2.used() == 0

    if True:
        print("a")
        t = ginn.RealTensor(d2, [3], [1,2,3])
        print("b")
        assert d2.used() == 12
        print("c")
