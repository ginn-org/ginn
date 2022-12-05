__Ginn__: Graph Interface to Neural Networks

A minimalistic, header only neural net library

[![cpu (build & test)](https://github.com/ginn-org/ginn/actions/workflows/cpu.yml/badge.svg?event=push)](https://github.com/ginn-org/ginn/actions/workflows/cpu.yml) [![gpu (build only)](https://github.com/ginn-org/ginn/actions/workflows/gpu.yml/badge.svg?event=push)](https://github.com/ginn-org/ginn/actions/workflows/gpu.yml) 

## Design goals
Ginn is inspired by dynamic computation graph approaches like PyTorch or DyNet. On top of that it strives for the following:
- **Ginn and all its dependencies are header only**, which makes it easy to install. Still, headers are compartmentalized so you can include only what you need. 
- **Minimalism in code.** Readability is emphasized by having small code with minimally functioning building blocks. Hopes to be less aggressive towards newcomers to C++ and easy to extend. (Short) inlined definitions means everything can be seen at one place, also makes it stylistically familiar to python users. It is written with the mindset that a newcomer can quickly pick up the whole framework easily and figure out inner workings. _Every user should be able to become a developer._
- **No centralized flow control.** No central data structures to track a computation graph, graphs are only defined through nodes being connected. This makes it quite simple to handle threadsafety or multithreading parallelism. 
- **No separation between a node in a computation graph and its parameters (or data).**
For example, a `WeightNode` object both implements its node behavior as well as acts as an (owning) container of
the values of its parameters.
- **Small networks or batches are not second class citizens.** Many real life systems have latency
constraints that make it important to have good performance for small networks or
small batch sizes (even completely online).
- **Simple autobatching.** Automatically convert any computation graph to perform
batched operations when it is nontrivial to write batched code, such as TreeLstms.

## Example snippets

### Mnist multilayer perceptron

```cpp
auto [X, Y] = mnist_reader(train_file, bs);
auto [Xt, Yt] = mnist_reader(test_file, bs);

auto W = Weight(dev(), {dimh, dimx});
auto b = Weight(dev(), {dimh});
auto U = Weight(dev(), {dimy, dimh});
auto c = Weight(dev(), {dimy});

std::vector<WeightPtr<Real>> weights = {W, b, U, c};
init::Uniform<Real>().init(weights);
update::Adam<Real> updater(lr);

// Single instance (batch)
auto pass = [&](DataPtr<Real> x, Indices& y, auto& acc, bool train) {
  auto h = Affine<SigmoidOp>(W, x, b);
  auto y_ = Affine(U, h, c);
  auto loss = Sum(PickNegLogSoftmax(y_, y));

  auto graph = Graph(loss);
  graph.forward();
  acc.batched_add(argmax(y_->value(), 0).maybe_copy_to(cpu()), y);

  if (train) {
    graph.reset_grad();
    graph.backward(1.);
    updater.update(weights);
  }
};

std::mt19937 g(seed);

// Single epoch
auto pass_data = [&](auto& X, auto& Y, bool train) {
  metric::Accuracy<Int> acc;
  auto perm = train ? randperm(X.size(), g) : iota(X.size());
  for (size_t j : perm) { pass(X[j], Y[j], acc, train); }
  return 100. * (1. - acc.eval());
};

// Main training loop
std::cout << "TrErr%\tTstErr%\tsecs" << std::endl;
for (size_t e = 0; e < epochs; e++) {
  using namespace ginn::literals;
  timer::tic();
  std::cout << ("{:6.3f}\t"_f, pass_data(X, Y, true)) << std::flush;
  std::cout << ("{:6.3f}\t"_f, pass_data(Xt, Yt, false)) << timer::toc() / 1e6
            << std::endl;
}
```

### Lstm step

```cpp
template <typename DataPtrPair>
State step(const NodePtr<Scalar>& x, const DataPtrPair& past) {
  auto [h_past, c_past] = past;

  auto i = Affine<SigmoidOp>(Wix, x, Wih, h_past, Wic, c_past, bi);
  auto f = Affine<SigmoidOp>(Wfx, x, Wfh, h_past, Wfc, c_past, bf);
  auto g = Affine<TanhOp>(Wcx, x, Wch, h_past, bc);
  auto c = CwiseProd(f, c_past) + CwiseProd(i, g);
  auto o = Affine<SigmoidOp>(Wox, x, Woh, h_past, Woc, c, bo);
  auto h_ = Tanh(c);
  auto h = CwiseProd(o, h_);

  return {h, c};
}
```
