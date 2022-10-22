import ginn


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

    assert a.list() == [1, 2, 3]
    assert c.list() == [1, 4, 2, 5, 3, 6]
