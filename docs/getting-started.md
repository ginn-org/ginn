## Build and install

Clone recursively, then build and run tests:

```bash
git clone --recursive https://github.com/ginn-org/ginn.git
cd ginn
make nodestest
./build/tests/nodes # run some tests
```

A c++17 supporting compiler is needed. Tested with **g++9**,
**clang++14** and **nvcc11**.

Since Ginn is header only, putting Ginn (folder `ginn`) and Eigen headers
in your include path should suffice to use it in your project.

You can also build and explore examples:
```bash
make examples -j8
```

GPU builds are enabled with the `GINN_ENABLE_GPU` compile flag and require
nvcc. See the target `cudatests` in `Makefile` as an example. You can attempt
```bash
make cudatests -j8 # or
make cudaexamples -j8
```
and if you have the Cuda toolkit configuration as specified in the `Makefile`
it should work. Or, change the configuration (e.g. target Cuda architectures)
as needed.

## Most basic usage

At its core, Ginn allows you to build dynamic computation graphs by
chaining various types of expressions (nodes). Since each node knows
to compute its own gradients, whole graph becomes automatically
differentiable.

Create some input vector and network weights, which would be equivalent
to leaf nodes in the computation DAG:
```cpp
using namespace ginn;

auto W = Weight<Real>(cpu(), {3, 2});
auto b = Weight<Real>(cpu(), {3, 1});

auto U = Weight<Real>(cpu(), {1, 3});
auto c = Weight<Real>(cpu(), {1, 1});

auto x = Data<Real>(cpu(), {2, 1});
```

Each such node has two core methods, `value()` and `grad()`, both
of which return `ginn::Tensor&`s. `value()` is the actual tensor
values of the node, and `grad()` is the gradient tensor (of something,
that will come later) with respect to the node.

Populate the values of the nodes:

```cpp
W->value().set_random();
b->value().set_zero();
U->value().set_random();
c->value().set_zero();

x->value().v() << 0., 1.; // tiny input to the network
```

Since `set_random()` does uniform initialization in [-1, 1], a more popular
approach would be
```cpp
init::Xavier<Real>().init({W, U});
```
instead.

Then, make up the graph:
```cpp
auto h = Tanh(W * x + b);
auto y = U * h + c;
auto loss = PickNegLogSigmoid(/*predicted*/ y, /*true*/ 1.);
```
`loss`, `y`, `h` are node pointers deriving from the same base class.

Note that at this point, none of these nodes have their `value()`s
computed yet. We have merely constructed the computation graph,
but not evaluated it. If you look at `y->value()`, you will see
an empty tensor.

Construct the `Graph` object spanned by a sink node, which is just
a container of all nodes in topological order:
```cpp
Graph g(loss);
```

Evaluate the expression with
```cpp
g.forward();
```
Now, if you were to query `loss->value()` or `h->value()`, you would
see the result of the computation.

Initialize all the gradients to their appropriate shape (same shape
as value) and reset them to zero:
```cpp
g.reset_grad();
```

Then, compute the gradient of the loss w.r.t. each node:
```cpp
g.backward(1.);
```
At this point, gradients for each of the nodes are computed. You
can safely inspect and use `y->grad()`, `h->grad()`, but more importantly,
the ones for the terminal nodes such as `W->grad()`.

Now that you computed the necessary gradients, you can perform an update:
```cpp
W->value() -= 0.1 * W->grad().t();
b->value() -= 0.1 * b->grad().t();
U->value() -= 0.1 * U->grad().t();
c->value() -= 0.1 * c->grad().t();
```
... but this is too cumbersome, so you would instead have created a
fancy updater and use that instead:
```cpp
ginn::update::Adam<Real> updater(0.1);
updater.update({W, b, U, c});
```
