# minGPT

```cpp
// Copyright 2022 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

This example defines a minimal GPT implementation used to train a character
level language model. It uses
[Karpathy's
minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) as
reference implementation.

## Getting started

We will attach a grad-checking mode just because we will end up using some
`InPlace` nodes for smaller memory footprint, and careless use can break
correctness. See [Appendix: Gradcheck](#appendix-gradcheck) for details if
interested.

```cpp
#ifdef GRADCHECK // gradcheck mode for the architecture defined here
#define GINN_DOUBLE_PRECISION
#define CATCH_CONFIG_MAIN
#include <test/testutil.h>
#endif
```

---

Include headers for necessary node and layer types...

```cpp
#include <ginn/init/init.h>
#include <ginn/layer/common.h>
#include <ginn/node/common.h>
#include <ginn/node/inplace.h>
#include <ginn/node/layout.h>
#include <ginn/node/pick.h>
#include <ginn/node/prod.h>
#include <ginn/node/reduce.h>
#include <ginn/update/update.h>
#include <ginn/util/amax.h>
#include <ginn/util/timer.h>

#include <iostream>
#include <string_view>
#include <unordered_set>

namespace ginn {
```

---

Define model parameters in a struct.

```cpp
struct Config {
  Size hidden_dim, n_head, vocab_size, n_layer;
  Real attn_drop_p, resid_drop_p, embd_drop_p;
};
```

---

## Causal self attention

We start with the self attention layer. This will be made up of other layers
such as `AffineLayer` or `DropoutLayer`, therefore it is convenient to have
it derive from a `CombinedLayer` type.

```cpp
struct CausalSelfAttentionLayerNode
    : public CombinedLayerNode<NodePtr<Real>(NodePtr<Real>)> {
  std::shared_ptr<AffineLayerNode<Real>> key, query, value, proj;
  std::shared_ptr<DropoutLayerNode<Real>> attn_drop, resid_drop;

  Size n_head, hidden_dim;

  std::vector<ConstLayerBasePtr> children() const override {
    return {key, query, value, proj, attn_drop, resid_drop};
  }
```

---

Now we can get started with the core logic of self attention. We assume
the input `x` is a $|h| \times B \times T$ tensor where $|h| = nh \cdot
hs$ is the aggregate hidden dim over every attention head, $nh$ is number
of attention heads, $B$ is the batch size and $T$ is the length across
time.

```cpp
  NodePtr<Real> run(const NodePtr<Real>& x) override {
    auto nh = Dim(n_head);
    auto hs = Dim(hidden_dim / n_head);
    auto B = Dim(x, 1);
    auto T = Dim(x, 2);
```

---

We apply `key`, `query`, and `value` affine maps to get $K$, $Q$, and
$V$ matrices. These need to be arranged (reshaped and permuted)
accordingly such that they have the following dimensions:

- $K: T \times hs \times B \times nh$
- $Q: hs \times T \times B \times nh$
- $V: hs \times T \times B \times nh$

```cpp
    auto rearrange = [&](NodePtr<Real> x, const std::vector<Size>& perm) {
      return InPlacePermute(Reshape(x, {hs, nh, B, T}), perm);
    };
    auto k = rearrange(key->run(x), {3, 0, 2, 1});   // T, hs, B, nh
    auto q = rearrange(query->run(x), {0, 3, 2, 1}); // hs, T, B, nh
    auto v = rearrange(value->run(x), {0, 3, 2, 1}); // hs, T, B, nh
```

---

Next is to get the key-query product: $K \times Q / \sqrt{hs}$. Matrix
product logic needs to happen in first two dimensions, and the other two
($B$ and $nh$) are batch dimensions for the sake of the matrix product.
We use `BatchedProd` for this.

```cpp
    Real scale = 1. / ::sqrt(Real(hs->value()));
    NodePtr<Real> att = InPlaceProdScalar(BatchedProd(k, q), scale);
```

Note that `att` is now $T \times T \times B \times nh$.

---

Now we need to define the causal attention mask to disable attending to
future. We do this by defining an upper triangular matrix (of ones), and
feeding it to `InPlaceMask`. Values are masked to $-\infty$ to ensure
zero softmax output later on.

```cpp
    auto mask = Reshape(UpperTri<Real>(x->dev(), T), {T, T, Dim(1), Dim(1)});
    att = InPlaceMask(att, mask, -std::numeric_limits<Real>::infinity());
```

---

Now apply softmax and dropout. Ginn is column major, so `Softmax`
normalizes across the first dimension. This means row determines which
timestep to attend to, and column determines which timestep is doing the
attending. This is why we use an upper triangular mask, column $t$
should attend to rows $t' \leq t$.

```cpp
    att = Softmax(att);
    att = attn_drop->run(att);
```

---

We can finally compute the weighted sum of values by the attention
amount $V \times att$. This is batched similarly to the key-query
product.

Before feeding this to the eventual projection layer, we reshape it such
that (1) first dim is $|h|$ again which matches the expected input
dimensionality for the affine map and (2) resulting `y` is rank three,
similarly to the original input `x`.

```cpp
    NodePtr<Real> y = BatchedProd(v, att);                        // hs,T,nh,B
    y = InPlacePermute(Reshape(y, {hs, T, B, nh}), {0, 3, 2, 1}); // hs,nh,B,T
    y = proj->run(Reshape(y, {Dim(hidden_dim), B, T}));
    y = resid_drop->run(y);

    return y;
  }
```

---

We finished the `run()` method. We can wrap up our class by defining the
necessary constructors. We ignore copy constructor and `copy()` method for
now since for this example we won't need them. They would be useful if we
were doing multi-threaded parallelism and needed tied copies of the
components, or had a reason for deep copies.

As with each layer in Ginn, we make a factory function to get shared
pointers easily to the underlying type.

```cpp
  CausalSelfAttentionLayerNode() = default;
  CausalSelfAttentionLayerNode(DevPtr dev, const Config& params)
      : n_head(params.n_head), hidden_dim(params.hidden_dim) {
    GINN_ASSERT(hidden_dim % n_head == 0,
                "Hidden dims must be a multiple of number of heads!");
    key = AffineLayer<Real>(dev, hidden_dim, params.hidden_dim);
    query = AffineLayer<Real>(dev, hidden_dim, params.hidden_dim);
    value = AffineLayer<Real>(dev, hidden_dim, params.hidden_dim);
    proj = AffineLayer<Real>(dev, hidden_dim, hidden_dim);
    attn_drop = DropoutLayer<Real>(params.attn_drop_p);
    resid_drop = DropoutLayer<Real>(params.resid_drop_p, /*inplace*/ true);
  }
  CausalSelfAttentionLayerNode(const CausalSelfAttentionLayerNode&) =
      default; // TODO

  LayerPtr<NodePtr<Real>(NodePtr<Real>)> copy(Copy) override {
    return nullptr;
  } // TODO
};

GINN_MAKE_LAYER_FACTORY(CausalSelfAttentionLayer);
```

---

## Gpt Block

Next component is a "block" made up of self attention and MLP, with
interleaved layer norm and residual connections.

We will define `mlp` as a composite layer made up of other layers, hence it
is declared as the base type `LayerPtr<NodePtr(NodePtr)>`, which can contain
any derived concrete layer taking in a `NodePtr` and returning a `NodePtr`.

```cpp
struct BlockLayerNode : CombinedLayerNode<NodePtr<Real>(NodePtr<Real>)> {
  std::shared_ptr<LayerNormLayerNode<Real>> ln1, ln2;
  std::shared_ptr<CausalSelfAttentionLayerNode> attn;
  LayerPtr<NodePtr<Real>(NodePtr<Real>)> mlp;

  Size hidden_dim;

  std::vector<ConstLayerBasePtr> children() const override {
    return {ln1, ln2, attn, mlp};
  }
```

---

`run()` method for this component is relatively straightforward. We apply
cross attention `attn` and then feedforward `mlp`. Each of these have a
layer norm preceding them, `ln1` and `ln2` respectively. Furthermore, we
have residual connections skipping over each of these components, those
are implemented with addition, `y = y + ...`, etc.

```cpp
  NodePtr<Real> run(const NodePtr<Real>& x) override {
    auto C = Dim(hidden_dim);
    auto B = Dim(x, 1);
    auto T = Dim(x, 2);

    NodePtr<Real> y = x;
    y = y + attn->run(ln1->run(y));
    y = y + Reshape(mlp->run(Reshape(ln2->run(y), {C, B * T})), {C, B, T});
    return y;
  }
```

There is an indermediate reshaping to a rank two tensor because
`AffineLayer` (which will be a part of `mlp` as we will define later)
requires rank two input tensors.

!> (`TODO(oir)`: Make affine layers work with any rank tensors.)

---

We can define `mlp` within the constructor, since it can be defined purely
as a stacked composition of other layers using the pipe (`|`) operator.

We first define `gelu_layer`. Any function can be made into a (weightless)
layer using `FunctionalLayer<Out(In)>` with the appropriate lambda,
therefore that's what we do here using a lambda that takes in `x` and
returns `Gelu(x)`.

Then we compose `mlp` as successive applications of affine, gelu, another
affine, and dropout layers.

```cpp
  BlockLayerNode() = default;
  BlockLayerNode(DevPtr dev, const Config& params)
      : ln1(LayerNormLayer<Real>(dev, params.hidden_dim)),
        ln2(LayerNormLayer<Real>(dev, params.hidden_dim)),
        attn(CausalSelfAttentionLayer(dev, params)),
        hidden_dim(params.hidden_dim) {
    auto gelu_layer = FunctionalLayer<NodePtr<Real>(NodePtr<Real>)>(
        [](const NodePtr<Real>& x) { return Gelu(x); });
    mlp = AffineLayer<Real>(dev, 4 * params.hidden_dim, params.hidden_dim) |
          gelu_layer |
          AffineLayer<Real>(dev, params.hidden_dim, 4 * params.hidden_dim) |
          DropoutLayer<Real>(params.resid_drop_p, /*inplace*/ true);
  }
```

---

Similar to before, have dummy copy constructor and `copy()` method, since
we won't need them. And define the factory function.

```cpp
  BlockLayerNode(const BlockLayerNode&) = default; // TODO;

  LayerPtr<NodePtr<Real>(NodePtr<Real>)> copy(Copy) override {
    return nullptr;
  } // TODO
};

GINN_MAKE_LAYER_FACTORY(BlockLayer);
```

---

## Gpt Layer

Next is what we call a `GptLayer`. This basically will take in the `Block`s
that were previously defined and stack them. It is entirely made up of other
layers, therefore `blocks` of type `LayerPtr<NodePtr(NodePtr)>` is our only
class attribute.

Constructor starts with a `DropoutLayer`, then adds `n_layer`-many
`BlockLayer`s on top of that. Finally, there is a `LayerNormLayer` and an
`AffineLayer` which will compute the pre-softmax logits as the output layer.

```cpp
struct GptLayerNode : CombinedLayerNode<NodePtr<Real>(NodePtr<Real>)> {
  // std::shared_ptr<DropoutLayerNode> drop;
  LayerPtr<NodePtr<Real>(NodePtr<Real>)> blocks;
  // std::shared_ptr<LayerNormLayerNode> ln_f;
  // std::shared_ptr<AffineLayerNode<>> head;

  std::vector<ConstLayerBasePtr> children() const override { return {blocks}; }

  GptLayerNode() = default;
  GptLayerNode(DevPtr dev, const Config& params) {
    blocks = DropoutLayer<Real>(params.embd_drop_p);
    for (Size i = 0; i < params.n_layer; i++) {
      blocks = blocks | BlockLayer(dev, params);
    }
    blocks = blocks |
             LayerNormLayer<Real>(dev, params.hidden_dim, /*inplace*/ true) |
             AffineLayer<Real>(dev, params.vocab_size, params.hidden_dim);
  }
  GptLayerNode(const GptLayerNode&) = default; // TODO

  LayerPtr<NodePtr<Real>(NodePtr<Real>)> copy(Copy) override {
    return nullptr;
  } // TODO

  NodePtr<Real> run(const NodePtr<Real>& x) override { return blocks->run(x); }
};

GINN_MAKE_LAYER_FACTORY(GptLayer);
```

---

## Gpt Model

We have all of the layer components that we need. However we need some more
stuff and additional bookkeeping in addition to the `GptLayer`, such as
index maps to keep track of char indices or char and positional embeddings.

We define a `GptModel` class to wrap these up with a `GptLayer` itself.
Defining business logic for things like computing overall loss given a raw
text sentence or randomly generating a character sequence is also a good fit
for such a class.

```cpp
struct GptModel {
  IndexMap<char> chars;
  std::unordered_map<char, WeightPtr<Real>> embedding_table;
  std::vector<WeightPtr<Real>> pos_embedding_table;
  decltype(GptLayer()) l;
```

---

In the constructor we initialize embedding tables and the `GptLayer`.
Weights themselves are initialized using Xavier initialization. Being more
clever about initialization (e.g. zero biases or centering layer-norm
multiplier at one) is left for future work.

```cpp
  GptModel(DevPtr dev, Config params, IndexMap<char> cmap, Size len)
      : chars(std::move(cmap)), pos_embedding_table(len) {
    for (auto c : chars.keys()) {
      embedding_table[c] = Weight(dev, {params.hidden_dim});
      init::Xavier<Real>().init(embedding_table[c]);
    }
    for (auto& w : pos_embedding_table) {
      w = Weight(dev, {params.hidden_dim});
      init::Xavier<Real>().init(w);
    }

    l = GptLayer(dev, params);
    init::Xavier<Real>().init(
        l->weights<Real>()); // TODO: init weights more carefully
  }
```

---

`weights()` is a convenience method that returns every weight for easy
iteration over them.

```cpp
  auto weights() {
    std::vector<WeightPtr<Real>> ws;
    for (auto& p : embedding_table) { ws.push_back(p.second); }
    for (auto& w : pos_embedding_table) { ws.push_back(w); }
    ws += l->weights<Real>();
    return ws;
  }
```

---

In `run()` method we define the core logic that takes in a batch of raw
text and computes the overall Gpt encoding, all the way up to the logits
for each timestep.

```cpp
  NodePtr<Real> run(const std::vector<std::string_view>& input) {
    Size batch_size = input.size();
    Size len = input.front().size(); // assuming each has the same len
```

---

We loop over each instance within batch and each timestep of the
instance and collect char and positional embeddings to stack them later.

```cpp
    std::vector<std::vector<NodePtr<Real>>> embeddings(batch_size);
    std::vector<std::vector<NodePtr<Real>>> pos_embeddings(batch_size);

    for (Size b = 0; b < batch_size; b++) {
      embeddings[b].resize(len);
      pos_embeddings[b].resize(len);
      for (Size t = 0; t < len; t++) {
        char c = input[b][t];
        embeddings[b][t] = embedding_table[c];
        pos_embeddings[b][t] = pos_embedding_table[t];
      }
    }
```

---

`Stack` takes in a nested vector of rank-one tensors (vectors) and
returns a rank three tensor which is constructed by stacking the
original tensors. Each tensor is of shape $|h|$ and outer vector
contains $B$ vectors which contain $T$ tensors, therefore resulting
rank-three tensor is of shape $|h| \times B \times T$, which is what
`GptLayer` expects.

!> `TODO:` Ask reviewers. I think the reverse vector dimensions would
make more sense since Ginn is column major and the inner the dim the
more to the left it should be. So vector should probably be $T$ outer
size and $B$ inner size to make the stack $|h| \times B \times T$. OTOH
it is also a bit counterintuitive to not make batch the outer dim.

We add the two stacks because positional embeddings are just added to
the regular char embeddings.

```cpp
    // x: {params.hidden_dim, batch_size, len}
    auto x = Stack(embeddings) + Stack(pos_embeddings);
```

---

Then we can simply feed this input to the `GptLayer` and get back the
output.

```cpp
    auto y = l->run(x);
    return y;
  }
```

---

Another useful method to define is `loss()`. In contrast to `run()`, in
addition to `input`, we have a notion of a gold `label` to compute the
loss with. Since we will be using a language modeling objective, labels
will be in the same index space as input: chars.

```cpp
  NodePtr<Real> loss(const std::vector<std::string_view>& input,
                     const std::vector<std::string_view>& label) {
    Size batch_size = input.size();
    Size len = input.front().size(); // assuming each has the same len
```

---

We start by getting the logits predicted by the model:

```cpp
    auto y = run(input);
```

---

We need the gold labels as an integer tensor with the appropriate shape
($B \times T$) where each entry represents the gold index.

```cpp
    auto labels = Data<Int>(cpu(), Shape{batch_size, len});
    for (Size b = 0; b < batch_size; b++) {
      for (Size t = 0; t < len; t++) {
        char l = label[b][t];
        labels->value().m()(b, t) = chars[l];
      }
    }
```

---

Finally, we can move `labels` to the same device as `y` (likely gpu),
compute the loss as mean log probability assigned to the true class, and
return it.

```cpp
    labels->move_to(y->dev());
    return Mean(PickNegLogSoftmax(y, labels));
  }
```

---

Final method that we will define is `sample()`. This is responsible for
generating random text from the model, given a seeding context `ctx` and a
final length of `len`.

```cpp
  std::string sample(const std::string ctx, size_t len) {
    static std::mt19937 sampling_rng;
```

---

We define a convenience lambda that given a rank-one tensor, samples
from it and returns an integer value.

```cpp
    auto sample = [&](Tensor<Real>& t) {
      auto probs_t = t.maybe_copy_to(cpu()); // has to stay alive for .v()
      auto probs = probs_t.v();
      std::discrete_distribution d(probs.data(), probs.data() + probs.size());
      return d(sampling_rng);
    };
```

---

Starting with the current string `s` we have, we run the model and apply
`Softmax` to get the probabilities. `Chip` extracts the last timestep
probability (last entry in dimension 2). Once we run a forward pass on
this node, we can sample an index and append the corresponding char to
`s`.

```cpp
    std::string s = ctx;
    while (s.size() < len) {
      auto y = run({s});
      auto probs = Softmax(y);
      auto prob = Chip(probs, s.size() - 1, 2);
      Graph(prob).forward();
      Size i = sample(prob->value());
      s += chars.reverse_lookup(i);
    }

    return s;
  }
};

} // namespace ginn
```

Note that there are potential optimizations that we are missing here. When
we work on the next timestep, we could reuse pairwise attentions between all
the previous timesteps ($(t', t'')$ such that $t' < t$ and $t'' < t$) and
compute attentions only between the next timestep and all of the
others ($(t', t)$ such that $t' < t$). Instead, by running `forward` on the
entire network we recompute all of the previously computed attentions in
addition to the new. For the sake of simplicity of this example, we leave
this out of scope.

---

## Main function

Finally we can define the main function which will contain the core training
loop.

```cpp
#ifndef GRADCHECK

int main() {
  using namespace ginn;
#ifdef GINN_ENABLE_GPU
  DevPtr dev = gpu();
#else
  DevPtr dev = cpu();
#endif

  srand(124);
```

---

We load the data as a sequence of chars, into a string. Furthermore, we
construct the index map from data which assigns a unique integer id to
each char, to be used with softmax.

```cpp
  std::string fname = "data/shakespeare.txt";
  std::string data = read(fname);
  std::cout << data.size() << std::endl;

  IndexMap<char> chars(data);
  std::cout << chars.size() << std::endl;
```

---

Define necessary hyperparameters and other configuration, construct the
model and Adam optimizer.

```cpp
  Config params{
      .hidden_dim = 512,
      .n_head = 8,
      .vocab_size = Size(chars.size()),
      .n_layer = 8,
      .attn_drop_p = 0.5,
      .resid_drop_p = 0.5,
      .embd_drop_p = 0.1,
  };

  Size len = 128;
  Size batch_size = 128;
  size_t num_iters = 500;

  GptModel model(dev, params, chars, len);
  update::Adam<Real> updater(6e-4);
```

---

Now we are ready for the training loop. We will use `perm` to index into
data and to shuffle after every epoch.

In the very beginning of the loop body, for every 100th iteration we
print a random sample from the model to make sure as training continues,
samples look more and more reasonable.

```cpp
  std::mt19937 perm_rng(135);
  std::vector<size_t> perm;
  size_t N = data.size() - len - 1;

  for (size_t i = 0; i < num_iters; i++) {
    timer::tic("iter");

    if (i % 100 == 99) { std::cout << model.sample("O", len) << std::endl; }
```

---

Construct input and label batch based on iteration and batch indices.
If this batch happens to roll over to the next epoch, shuffle `perm`.

```cpp
    std::vector<std::string_view> input, label;
    for (Size b = 0; b < batch_size; b++) {
      size_t j = i * batch_size + b;
      if (j % N == 0) {
        std::cout << "Shuffling." << std::endl;
        perm = randperm(N, perm_rng);
      }
      j = perm[j % N];

      input.emplace_back(&data[j], len);
      label.emplace_back(&data[j + 1], len);
    }

    std::cout << "Iter " << i << " ("
              << ((((i + 1) * batch_size) % N) * 100) / N << "% of epoch "
              << (1 + ((i + 1) * batch_size) / N) << "), loss: " << std::flush;
```

---

Get the loss node from the model and apply the forward-backward-update
routine.

```cpp
    auto loss = model.loss(input, label);

    Graph g(loss);
    g.forward();
    std::cout << loss->value().maybe_copy_to(cpu()).v() << std::flush;
    g.reset_grad();
    g.backward(1.);
    updater.update(g);
```

---

After the update, we apply an explicit weight decay step.

```cpp
    for (auto& w : model.weights()) { w->value() -= 1e-6 * w->value().t(); }

    std::cout << " " << timer::toc("iter", timer::HumanReadable) << std::endl;
  }

  // TODO: store the model here after training

  return 0;
}
```

---

## Appendix: Gradcheck

In Ginn, every regular node type is gradient checked for correctness,
which means that any computation graph made out of them should have correct
gradients by construction. However here we use some `InPlace` nodes to have
memory reuse and smaller memory footprint, such as InPlaceProdScalar or
InPlaceMask. These special node types require some assumptions for gradient
correctness, e.g. `::backward()` of the input node to an `InPlace` node
should not use `::value()` in its computation. See [InPlace](inplace.md) for
a full list of requirements and use cases of `InPlace` nodes. If these
assumptions are violated but an `InPlace` node is regardless used, gradients
are likely to be incorrect. We explicitly check gradients on the loss
computation graph here to make sure we are safe in this regard.

```cpp
#else

TEST_CASE("Gradcheck") {
  using namespace ginn;
  DevPtr dev = cpu();

  // Don't forget to enable GINN_DOUBLE_PRECISION
  std::string data = "First Citizen:\n"
                     "Before we proceed any further, hear me speak.\n"
                     "\n"
                     "All:\n"
                     "Speak, speak.\n";

  IndexMap<char> chars(data);
  std::cout << chars.size() << std::endl;

  Config params{
      .hidden_dim = 15,
      .n_head = 3,
      .vocab_size = Size(chars.size()),
      .n_layer = 4,
      .attn_drop_p = 0., // TODO: enable gradchecking for dropout nodes
      .resid_drop_p = 0.,
      .embd_drop_p = 0.,
  };

  Size len = 20;
  Size batch_size = 4;

  GptModel model(dev, params, chars, len);
  init::Xavier<Real>().init(model.weights());

  // construct input and label batch
  std::vector<std::string_view> input, label;
  for (Size b = 0; b < batch_size; b++) {
    input.emplace_back(&data[b], len);
    label.emplace_back(&data[b + 1], len);
  }

  auto loss = model.loss(input, label);

  check_grad(loss, model.weights());
}

#endif
```

---

(Generated with `tools/cpp2md.py` from `examples/min-gpt.cu.cpp`.)
