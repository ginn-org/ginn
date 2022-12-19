## Tensors

<span style="position: absolute; top: 20px; right: 20px;"> Defined in <code><a href="https://github.com/ginn-org/ginn/blob/main/ginn/tensor.h"> \<ginn/tensor.h\> </a></code> </span>

Linear algrebra in Ginn is powered by Eigen. Since `Eigen::Tensor` does not
manage its memory for GPU devices, a `ginn::Tensor` class wraps around it to
simplify device and memory management, and provide various views to the
underlying data as Eigen tensors (or matrices and vectors).

`Tensor` class consists of these core components:
```cpp
template <typename Scalar>
class Tensor {
  Device* dev_ = nullptr;
  Shape shape_ = {0};
  Scalar* data_ = nullptr;
  bool owns_mem_ = true;
};
```
Tensor is a template on the underlying scalar type which can be `Real` (i.e.
`double` or `float`), `Half` (`float16`), `Int`, `bool` etc.

- `dev_`: Non-owned device pointer which is responsible for allocation and
deallocation for the data. This for instance determines if the Tensor is on
CPU or GPU.
- `shape_`: A list of dimensions for the Tensor. Length determines the rank /
order of the Tensor and each element defines the corresponding dimensions.

?> Layout in Ginn is always __column major__, i.e. leftmost index changes quickest.

- `data_`: (Likely owned) pointer to actual storage for the values in the
Tensor. This can be a CPU or GPU pointer depending on what `dev_` is.
- `owns_mem_`: Whether the Tensor owns its memory (`data_`) or not. This will be
`true` in most cases. However in a handful of cases the Tensor can be a view
to another (e.g. reshape operation). In those cases, instead
of doubly allocating the same amount space with same values, `data_` would point
to the other Tensor's `data_`.

## Creating and populating Tensors

Typical way to construct a Tensor is relying on the basic constructor:
```cpp
  Tensor(Device& dev, const Shape& shape);
```
which creates a Tensor on the given device. Since `shape` is already determined,
space is immediately allocated on the given device, however left uninitialized.

For example:
```cpp
Tensor t1(cpu(), Shape{1, 2, 3});
```
constructs a CPU Tensor of rank three with dimensions of `{1, 2, 3}`.
```cpp
Tensor t2(gpu(), {50, 100})
```
constructs a GPU Tensor of rank two with dimensions of `{50, 100}`.

TODO: talk about the nested init list ctor here.

Most flexible way of populating a Tensor is possibly through an Eigen
Tensor or Matrix view (see [section below](#Views-and-operations)) since Eigen itself provides
several initialization mechanisms. For instance, for a CPU
Tensor, a MatrixMap view can be used to perform comma initialization:
```cpp
Tensor<Real> t(cpu(), {3, 2});
t.m() << 0.1, 0.2,
         0.3, 0.4,
         0.5, 0.6;
```
In a similar fashion, classical Eigen indexing can be used:
```cpp
std::vector<Real> some_values(100);
Tensor<Real> t(cpu(), {10, 10});
for (size_t i = 0; i < some_values.size(); i++) {
  t.v()[i] = some_values[i];
}
```

For ease of use, `ginn::Tensor` has some methods that wrap around
`Eigen::Tensor` methods:
- `Tensor::fill(Real c);`: fills the Tensor with the constant value `c`
- `Tensor::set_zero()`: sets all values to zero
- `Tensor::set_ones()`: sets all values to one
- `Tensor::set_random()`: sets each element a random uniform value in `[-1, 1]`.

## Views and operations

To use Eigen operations we need `Eigen::Tensor`-like views to the storage
contained in a `ginn::Tensor`. This is mostly because there is no benefit
to duplicate (by wrapping around) such functionality from Eigen.
To this end, Tensor class has several methods:

```cpp
template <typename Scalar>
class Tensor {
  template <size_t Rank> TensorMap<Scalar, Rank> view();
  TensorMap<Scalar, 2> t();
  MatrixMap<Scalar> m();
  VectorMap<Scalar> v();
};
```

- `view<Rank>()`: Returns an `Eigen::TensorMap<Scalar, Rank>` of specified rank. Shape
of the view is adjusted accordingly to have the specified rank and same size,
by rolling (multiplying) from right to left. If shape needs to grow, it is
appended with `1`. Can be used for both CPU and GPU tensors.
- `t()`: Shorthand for `view<2>()`, a matrix view to the Tensor, since a rank of two is very common.
- `m()`: Returns an `Eigen::MatrixMap<Scalar>` view to the Tensor which works __only
for CPU__. This is to be used when you need operations from the _supported
Eigen modules_ and not the _unsupported Eigen tensor modules_.
- `v()`: Similarly, returns an `Eigen::VectorMap<Scalar>`, and works __only for CPU
Tensors__.

Using these views, you can perform Eigen operations over tensors like the following examples:
```cpp
... = t1.t() + t2.t() * t3.t();
... = (t4.m().transpose() + t5.m().inverse()) * t6.v();
```

Please see Eigen documentation for any possible list of such operations:
- [Classical Eigen](http://eigen.tuxfamily.org/dox/)
- [Eigen unsupported tensor modules](https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html)

## Left-hand side expressions

Since `Eigen::Tensor` operations are evaluated lazily, until the result of the
expression is assigned to another tensor, no evalutaion is performed.
To compute the evaluation and perform the result of the expression, Eigen
needs a device to do that:
```cpp
Tensor<Real> res(cpu(), {2, 3}),
             t1(cpu(), {2, 3}),
             t2(cpu(), {2, 3}),
             t3(cpu(), {2, 3});
t1.set_random();
t2.set_random();
t3.set_random();

Eigen::DefaultDevice cpu_device;
res.t().device(cpu_device) = t1.t() + t2.t() * t3.t();
//                         | \----------v-----------/
//                         | this is NOT a Tensor! it is an
//                         | unevaluated expression object.
//                         V
//           only when operator= is invoked to the r.h.s object
//           actual computation is performed.
```
Or:
```cpp
Tensor res(gpu(), {2, 3}),
       t1(gpu(), {2, 3}),
       t2(gpu(), {2, 3}),
       t3(gpu(), {2, 3});
t1.set_random();
t2.set_random();
t3.set_random();
res.set_random();

Eigen::GpuStreamDevice stream;
Eigen::GpuDevice gpu_device(&stream);
res.t().device(gpu_device) += t1.t() + t2.t() * t3.t();
```

But both bookkeeping of devices, and making a `device()` call each time for
every left-hand side tensor is cumbersome and introduces code clutter.
Since `ginn::Tensor` keeps track of its device, it should be able to make
those calls for us.

To this end, the method `lhs()` can be used to create a temporary object of
type `ginn::Tensor::LhsExpr` that applies the `device()` call upon invocation
of assignment operators:
```cpp
res.lhs() = t1.t() + t2.t() * t3.t();
res.lhs() += t1.t() + t2.t() * t3.t();
```
Similar to `view()`, `lhs()` template parameter can be used to specify 
rank of the tensor on the left-hand side:
```cpp
res.lhs<4>() = t1.view<4>() + t2.view<4>();
```

Finally, instead of remembering to call `lhs()` every time for an assignment,
we can have `operator=` (or `operator+=`) automatically do it for us when a `ginn::Tensor` is on
the left-hand side and an appropriate Eigen expression is on the right. Rank of
the right-hand object is determined at compile-time, therefore can be used as the
appropriate `Rank` parameter for `lhs()`.

With this implemented, eventual user code
looks like this, instead of what is above:

```cpp
res = t1.t() + t2.t() * t3.t();
res += t1.t() + t2.t() * t3.t();
res = t1.view<4>() + t2.view<4>();
```
