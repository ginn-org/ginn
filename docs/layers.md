## Layers

<span style="position: absolute; top: 20px; right: 20px;"> Defined in <code><a href="https://github.com/ginn-org/ginn/blob/main/ginn/layer/layer.h"> \<ginn/layer/layer.h\> </a></code> </span>

Layers provide additional, optional utility for constructing computation graphs
that are similar to PyTorch's `nn.module`. A layer helps with several things:

 - Graph construction: `run()` method of a layer takes in e.g. input node(s)
   and returns new node(s).
 - Containing weights: Something like an `AffineLayer` owns its weights`W` and `b`
   therefore you do not need to explicitly manage these weights yourself.
 - Composition: You can stack layers or combine them in different ways to make
   compound layers that abide by the same interface.

## Interface

Similar to how we work with `Node`s in a shared ownership setting, we will work
with `Layer`s using `std::shared_ptr`s stored in containers, requiring a common
base:

```cpp
class LayerNodeBase {
 private:
  Mode mode_ = Mode::Training;

 public:
  virtual std::vector<BaseNodePtr> weights_() { return {}; }

  template <typename Scalar>
  std::vector<WeightPtr<Scalar>> weights();

  virtual void set_mode(Mode mode) { mode_ = mode; }
  virtual Mode mode() const { return mode_; }
};
```

We view a layer as basically a function with (optional) weights. A layer's type
is parametrized by the input and output types to this function, similar to how
`std::function<Output(Input)>` works:

```cpp
template <typename Func>
class LayerNode;

template <typename Func>
using LayerPtr = std::shared_ptr<LayerNode<Func>>;

template <typename Out, typename In>
class LayerNode<Out(In)> : public LayerNodeBase {
 public:
  using function_t = Out(In);
  using arg_t = In;
  using result_t = Out;

  virtual Out run(const In&) = 0;

  virtual LayerPtr<Out(In)> copy(Copy mode) = 0;
};
```

As mentioned above, `run()` contains the main logic that takes in an input
of type `In` and returns something of type `Out`. For instance, for a typical
`AffineLayer<Scalar>` this would take in a `NodePtr<Scalar>` and return another
`NodePtr<Scalar>`. Something like `CatLayer` can take in a tuple of
nodes instead of just one. In general, `Out` and `In` can be anything
as required by the layer.

In terms of how a layer gets used within user code, it would look something
like this:
```cpp
// AffineLayerNode<Real> : public LayerNode<NodePtr<Real>(NodePtr<Real>)>

auto ff = AffineLayer<Real>(device, dimh, dimx);
// ff is a std::shared_ptr<AffineLayerNode<Real>>

init::Uniform<Real>().init(ff->weights<Real>());
update::Adam<Real> updater(lr);

// Single instance (batch)
auto pass = [&](DataPtr<Real> x, Indices& y, auto& acc, bool train) {
  auto h = ff->run(x);
  auto loss = SomeLoss(h, y);

  auto graph = Graph(loss);
  graph.forward();
  if (train) {
    graph.reset_grad();
    graph.backward(1.);
    updater.update(ff->weights<Real>());
  }
};
```

See `mnist-layers.cu.cpp` for an end-to-end working example.

## Composing layers

### Stacking

Layers can be combined to make bigger layers. The most widely applicable
is stacking two layers on top of each other, which is equivalent to
function composition. This can be done using `operator|`. Underneath,
it is implemented by a `StackLayer` node.
```cpp
auto l1 = AffineLayer<Real>(device, 5, 3);
auto l2 = AffineLayer<Real>(device, 4, 5);
auto mlp = l1 | l2;

auto y = mlp->run(x); // same as l2->run(l1->run(x))
mlp->weights(); // union of weights of both layers
// ... etc
```

Because layers are typed by input and outputs type, resulting stack layer
is typed by the composition function's input and outputs, i.e. input of
the inner (first) layer and the output of the outer (second) layer. And
the intermediate types need to match.

### Putting side by side (joining)

Another use case is combining layers by putting side by side at the same
level. Easiest to explain by example:
```cpp
auto left = SomeLayer(/*...*/); // takes input of type In1 and returns output of type Out1
auto right = OtherLayer(/*...*/); // takes input of type In2 and returns output of type Out2
auto layer = left, right; // operator,() applies the JoinLayer to combine the two

In1 x1;
In2 x2;
auto x = std::make_tuple(x1, x2);
auto y = layer->run(x); // y is now std::tuple<Out1, Out2>
// this is the same as
// auto y = std::make_tuple(left->run(std::get<0>(x)),
//                          right->run(std::get<1>(x)));
```
As an example, this use case appears in `lstm-tag.cpp` when combining a word vector lookup table
layer with a character level RNN layer.

## Functional layer

Some layers don't have any weights, no weight management needs to be involved. In this case,
a layer is barely different from a function. In fact, if you already have a function implemented,
fitting that into the layer interface requires repetitive, reusable boilerplate.

To avoid this work, `FunctionalLayer` takes any `std::function` (anything that can be converted to it)
and converts it to a layer.

Here is an example where we already have a nonlinearity, `Gelu`, working as a function from `NodePtr`
to `NodePtr`, and use `FunctionalLayer` to make it into a Gelu layer:
```cpp
auto gelu_layer = FunctionalLayer<NodePtr<Real>(NodePtr<Real>)>(
    [](const NodePtr<Real>& x) { return Gelu(x); });
mlp = AffineLayer<Real>(dev, 4 * hidden_dim, hidden_dim) |
      gelu_layer |
      AffineLayer<Real>(dev, hidden_dim, 4 * hidden_dim) |
      DropoutLayer<Real>(resid_drop_p, /*inplace*/ true);
```
This snippet is taken from the example `min-gpt.cu.cpp`.

Note that `weights()` of a functional layer always will be the empty `std::vector`, `{}`.

## Containerwise layer

Yet another pattern that appears while working with layers is as follows: Suppose you
already have a layer that operates on `In`, returning `Out`. You want to have the layer
apply to a `Container<In>`, and return `Container<Out>` instead. `ContainerwiseLayer`
allows you to extend any layer such that it now applies to a container of things
instead of things.

Imagine a more concrete example: You have designed a recurrent layer that outputs
a sequence of vectors as `std::vector<NodePtr<Real>>`. You want to apply an affine layer
on top, as the output layer, such that it applies to each timestep and returns
and output for each timestep. However `AffineLayer<Real>` operates on `NodePtr<Real>`, therefore
its not possible to `run(x)` where `x` is a `std::vector<NodePtr<Real>>`.

`ContainerwiseLayer` can be used to create a new layer based on `AffineLayer`:
```cpp
auto rnn = MyRnnLayer(/*...*/); // :Input -> std::vector<NodePtr<Real>>
auto aff = AffineLayer<Real>(dev, out_dim, rnn_dim); // :NodePtr<Real> -> NodePtr<Real>
auto out = ContainerwiseLayer<std::vector>(aff); // :std::vector<NodePtr<Real>> -> std::vector<NodePtr<Real>>

auto model = rnn | out;
Input x;
model->run(x);
```
Observe that the template parameter to `ContainerwiseLayer` itself is a template, in this example
`std::vector` as opposed to `std::vector<NodePtr<Real>>`. That's because container is assumed to
hold the same input type as `aff`, and output container type is determined based on the output type
of `aff`.

Currently supported containers for this transformation are `std::vector` and `ginn::tree::Tree`,
however you can specialize the free template function `clone_empty()` for any container type to add support.
