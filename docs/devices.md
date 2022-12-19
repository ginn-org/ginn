## Devices

<span style="position: absolute; top: 20px; right: 20px;"> Defined in <code><a href="https://github.com/ginn-org/ginn/blob/main/ginn/dev.h"> \<ginn/dev.h\> </a></code> </span>

Devices provides means for memory allocations to Tensors. A device determines
two main things:

 - Where the memory physically resides (i.e. CPU vs GPU), and
 - How the memory allocation is handled.

Devices are defined in header `dev.h`.

In the most basic use case, there is only ever two devices to be used,
`Cpu` and `Gpu`. Both devices handle memory allocation in the typical dynamic
allocation way, i.e. `new`/`malloc` allocation for `Cpu` and `cudaMalloc` for
`Gpu`.

Using these default devices would look like the following:
```cpp
Cpu my_cpu;
Gpu my_gpu;

Tensor<Real> t(my_cpu, Shape{3, 1, 5});
auto d = Data<Half>(my_gpu, Shape{2, 4});
```

If you are repeatedly using default devices, it is cumbersome to keep declaring
and defining them. To avoid this, there are `cpu()` and `gpu()` functions
that return references to statically defined `Cpu` and `Gpu` devices that you
can use:
```cpp
Tensor<Real> t(cpu(), Shape{3, 1, 5});
auto d = Data<Half>(gpu(), Shape{2, 4});
```

## Moving tensors across devices

Typical workflow when working with GPUs involves moving `Gpu` tensors back to
`Cpu` memory to inspect its values as well as moving `Cpu` initialized data
to `Gpu` memory before the start of training.

There are two idiomatic ways of moving a tensor across devices:
 - Assignment operator
 - `move_to()` method

They can be used as follows
```cpp
Tensor<Real> t1(cpu(), Shape{3, 5, 1});
Tensor<Real> t1_gpu(gpu());
t1_gpu = t1; // carry t1's values to t1_gpu on Gpu
assert(t1.dev().type() == CPU);
assert(t1_gpu.dev().type() == GPU);
```
```cpp
Tensor<Real> t(cpu(), Shape{3, 5, 1});
assert(t.dev().type() == CPU);
t.move_to(gpu());
assert(t.dev().type() == GPU);
```

## Moving nodes across devices

Since we almost always work with nodes instead of Tensors directly, we might
want to move nodes instead, for easier use. 

Most nodes are functions of other
nodes, therefore notion of moving them across devices is not meaningful, if, e.g.
inputs to AddNode lie on Gpu 2, AddNode itself will lie on Gpu 2 as well since
the computation will be performed there. Thus, nodes with inputs will lie on
the device that is determined by the inputs (there are exceptions to this).

On the other hand we have terminal nodes without any inputs at the start of a
graph, such as `Data` or `Weight`. It makes perfect sense to be able to move
these across devices by moving their `value()` and `grad()`, therefore
they have a similar `move_to()` method doing just this.

### Transferring during computation

Sometimes you might want to have computation graphs that are spread across
devices. Instead of moving a node to a device and doing all the computation in
there, you might have a workflow to run some of the computation in device A,
then transfer the intermediate result to device B and continue there. That
implies that during backward pass the gradients would downpour in device B
until the transfer point, after which get transferred to device A back and
keep backpropagated using device A.

To enable this, we have the `DeviceTransfer` node that propagates `value()`
to the new device during forward pass and propagates `grad()` to the old
device during backward pass. With this, the notion of moving across devices
becomes a differentiable part of the computation graph.

## Custom devices

In some cases, it might be beneficial to have a custom memory allocation logic
instead of standard heap allocation.

Consider the typical neural network
training setting: We have some permanently stored tensors (for, e.g. weights
of our network) as well as some temporarily created tensors (for, e.g. forward
and backward values for intermediate nodes of our computation graph).

One strategy for repeatedly heap allocating temporary tensors is to preallocate
an initial fixed amount and assign from that pool to each temporary tensor
incrementally:

```
Preallocated pool:
+-----------------------------------------------------------------------------+
|                                                                             |
+-----------------------------------------------------------------------------+
↑
offset
```
```cpp
auto h = Affine(W, x, b);
auto y = Affine(V, h, c);
auto l = PickNegLogSoftmax(y, r);
Graph g(l);
g.forward();
```
would result in following step-by-step allocation:
```
+-----------------------------------------------------------------------------+
|  h |                                                                        |
+-----------------------------------------------------------------------------+
     ↑
     offset
```
```
+-----------------------------------------------------------------------------+
|  h |  y |                                                                   |
+-----------------------------------------------------------------------------+
          ↑
          offset
```
```
+-----------------------------------------------------------------------------+
|  h |  y |  l |                                                              |
+-----------------------------------------------------------------------------+
               ↑
               offset
```
Note that each "allocation" would actually simply be a pointer increment which
is much cheaper. Then
```cpp
g.reset_grad()
```
would also "allocate" gradient tensors for the temporaries:
```
+-----------------------------------------------------------------------------+
|  h |  y |  l | δh | δy | δl |                                               |
+-----------------------------------------------------------------------------+
                              ↑
                              offset
```
Rest of the workflow simply goes:
```cpp
g.backward();
updater.update(g);
```
At this point we no longer need any of the temporary values or gradients.
Before we continue with the next instance $x$ with a fresh computation graph,
we can purge our preallocated memory:
```
+-----------------------------------------------------------------------------+
|                                                                             |
+-----------------------------------------------------------------------------+
↑
offset
```

Devices `PreallocCpu` and `PreallocGpu` do exactly this.
 - They preallocate the specified amount at device construction time
 - A call to `allocate()` is simply a pointer assignment and increment
 - A call to `free()` from any tensor is a no-op, since individual freeing is 
 not possible. Instead, the user is supposed to make a call to `reset()` which
 purges the whole memory by setting the offset to zero.

### How to set which nodes are temporaries?

Let's say we want to use `PreallocCpu` as defined above. We need to have our
weight `W` in the permanent storage, and maybe have our data `x` in the
temporary storage:
```cpp
auto W = Weight<Real>(cpu(), Shape{3, 2});
auto b = Weight<Real>(cpu(), Shape{3});

PreallocCpu tmp_dev(1e8);
auto x = Data<Real>(tmp_dev, Shape{2, 1});
```

Then we perform an operation, e.g. Affine:
```cpp
auto h = Affine(W, x, b);
```
Now, __what is the device of `h`?__

Intuitively, we expect `h` to be temporary as well, since it is a function of
at least one temporary. Therefore, it should be using the device `tmp_dev`.
This is handled by assigning precedence to different devices: `PreallocCpu`
has a higher precedence than `Cpu`, therefore any composite node that has both
inputs from `PreallocCpu` and `Cpu` devices would inherit the `PreallocCpu`
device.

__What if `W`, `x` and `b` are all permanents (on device `cpu()`) but we want to
put `h` in `tmp_dev` regardless and declare it temporary?__

Instead of adding device-aware constructors to each node type, this is
facilitated by having a special `DeviceView` node:
```cpp
auto x_ = DeviceView(x, tmp_dev);
auto h = Affine(W, x_, b);
```
For purposes of computation, `x_` and `x` are exactly the same. In fact, they
also point to the same memory storage, i.e. there is no copy from `x` to `x_`.
In terms of device inheritance however `x_` _looks like_ it has `tmp_dev` as its
device: When `h` queries devices of each of its inputs to inherit the most
precedent one, `x_` will look to be in the `tmp_dev` device, which in turn
will result in `h.dev()` being `tmp_dev`.

See the example, e.g. `sum-lstm.cu.cpp` for an end-to-end usecase.
