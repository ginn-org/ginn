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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <ginn/init/init.h>
#include <ginn/metric.h>
#include <ginn/model/lstm.h>
#include <ginn/node/layout.h>
#include <ginn/node/pick.h>
#include <ginn/node/reduce.h>
#include <ginn/update/update.h>
#include <ginn/util/amax.h>
#include <ginn/util/cli2.h>
#include <ginn/util/lookup.h>
#include <ginn/util/timer.h>
#include <ginn/util/util.h>

using namespace ginn;

#ifdef GINN_ENABLE_GPU
DevPtr Dev = gpu();
#else
DevPtr Dev = cpu();
#endif

using Instance = std::pair<std::string, std::string>;
using Instances = std::vector<Instance>;

Instances generate_data(size_t seed = 135,
                        size_t n = 100000,
                        unsigned num_numbers = 3,
                        unsigned max_digits = 7) {
  std::random_device rd;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<unsigned> g_len(1, max_digits);
  std::uniform_int_distribution<unsigned> g_first_digit(1, 9);
  std::uniform_int_distribution<unsigned> g_digit(0, 9);

  Instances X;

  for (size_t i = 0; i < n; i++) {
    std::string x, y;

    for (unsigned j = 0; j < num_numbers; j++) {
      if (j > 0) { x += "+"; }
      unsigned len = g_len(rng);
      for (unsigned i = 0; i < len; i++) {
        unsigned digit = (i == 0 ? g_first_digit(rng) : g_digit(rng));
        x += std::to_string(digit);
      }
    }
    unsigned sum = 0;
    for (const std::string& num : split(x, '+')) { sum += std::stoi(num); }
    y = std::to_string(sum) + ";";

    X.emplace_back(std::move(x), std::move(y));
  }

  return X;
}

std::vector<Instances> batch(const Instances& X, Size bs) {
  auto perm = iota(X.size());

  std::sort(perm.begin(), perm.end(), [&](const size_t a, const size_t b) {
    return (X[a].first.size() + X[a].second.size()) >
           (X[b].first.size() + X[b].second.size());
  });

  size_t num_batches = (X.size() + bs - 1) / bs;

  std::vector<Instances> Xb;
  for (size_t i = 0; i < num_batches; i++) {
    Instances batch;

    size_t max_x_size = 0, max_y_size = 0;
    for (size_t j = i * bs; j < (i + 1) * bs and j < X.size(); j++) {
      auto& [x, y] = X[perm[j]];
      max_x_size = std::max(max_x_size, x.size());
      max_y_size = std::max(max_y_size, y.size());
    }

    for (size_t j = i * bs; j < (i + 1) * bs and j < X.size(); j++) {
      auto [x, y] = X[perm[j]];
      // padding
      x += std::string(max_x_size - x.size(), ';');
      y += std::string(max_y_size - y.size(), ';');
      batch.emplace_back(x, y);
    }

    Xb.push_back(std::move(batch));
  }

  return Xb;
}

int main(int argc, char** argv) {
  std::string model;
  Size xdim = 12;
  Size hdim = 512;
  Size bs = 128;
  Real lr = 1e-4;
  size_t ntra = 100000;
  size_t ndev = 10000;
  size_t epochs = 50;
  size_t seed = 123457;
  size_t mem = 2e9;

  Args args;
  args(Arg(model)
           .name("m,model-file")
           .meta("path")
           .help("file to store model after training")
           .required());
  args(Arg(xdim).name("x,input-dim").help("input (embedding) dimensionn"));
  args(Arg(hdim).name("z,hidden-dim").help("hidden layer dimension"));
  args(Arg(bs).name("b,batch-size").help("batch size"));
  args(Arg(lr).name("l,learning-rate").help("learning rate"));
  args(Arg(ntra).name("t,train-size").help("training set size"));
  args(Arg(ndev).name("d,dev-size").help("dev set size"));
  args(Arg(epochs).name("e,epochs").help("number of epochs to train"));
  args(Arg(seed).name("s,seed").help("random seed"));
  args(Arg(mem).name("M,mem").help("preallocated memory in bytes"));

  args.parse(argc, argv);

  auto XYb = batch(generate_data(seed, ntra), bs);
  auto XYbdev = batch(generate_data(seed + 1, ndev), bs);

  LookupTable<char, WeightPtr<Real>> lt(
      {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ';'},
      [xdim]() { return Weight(Dev, {xdim}); });

  IndexMap<char> outmap(
      {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';'});

  Lstm enc(Dev, hdim, xdim);
  Lstm dec(Dev, hdim, 2 * hdim);

  Size labels = outmap.size();
  auto Wy = Weight(Dev, {labels, hdim});
  auto by = Weight(Dev, {labels});

  auto weights =
      enc.weights() + dec.weights() + lt.weights() + std::vector{Wy, by};

  init::Xavier<Real>().init(weights);
  update::Adam<Real> updater(1e-4);
#ifdef GINN_ENABLE_GPU
  auto device = PreallocGpu(mem);
#else
  auto device = PreallocCpu(mem);
#endif

  auto pass = [&](const Instances& x,
                  metric::Metric<size_t>& m,
                  bool train,
                  bool print) {
    // encode
    Size bs = x.size();
    auto h0 = Zero(device, {hdim, bs});
    auto c0 = Zero(device, {hdim, bs});
    auto state = Lstm<Real>::State{h0, c0};
    for (size_t t = 0; t < x[0].first.size(); t++) {
      std::vector<char> chars(x.size());
      for (size_t i = 0; i < x.size(); i++) { chars[i] = x[i].first[t]; }
      std::vector<NodePtr<Real>> embeddings(chars.size());
      for (size_t i = 0; i < chars.size(); i++) {
        embeddings[i] = DeviceView(lt[chars[i]], device);
      }
      state = enc.step(RowwiseCat(embeddings), state);
    }

    auto code = Cat(state.first, state.second);

    state = Lstm<Real>::State({h0, c0});
    std::vector<NodePtr<Real>> out_seq;

    if (print) {
      std::cout << "\tRandom input:\t" << x[0].first << "\n\t      output:\t"
                << x[0].second << "\n\t   predicted:\t";
    }

    // decode
    for (size_t t = 0; t < x[0].second.size(); t++) {
      state = dec.step(code, state);
      auto out = Affine(Wy, state.first, by);
      out_seq.push_back(out);
      Graph(out).forward();
      auto pred = argmax(out->value(), 0).copy_to(cpu());
      for (size_t i = 0; i < x.size(); i++) {
        m.add(pred.v()[i], outmap[x[i].second[t]]);
      }
      if (print) { std::cout << outmap(pred.v()(0)); }
    }
    if (print) { std::cout << std::endl; }

    std::vector<NodePtr<Real>> losses;
    for (size_t t = 0; t < out_seq.size() and t < x[0].second.size(); t++) {
      std::vector<Int> yi(x.size());
      for (size_t i = 0; i < x.size(); i++) {
        yi[i] = outmap[t < x[i].second.size() ? x[i].second[t] : ';'];
      }
      losses.push_back(PickNegLogSoftmax(out_seq[t], yi));
    }

    auto loss = Add(losses);

    auto graph = Graph(loss);
    graph.forward();

    if (train) {
      graph.reset_grad();
      graph.backward(1.);
      updater.update(graph);
    }

    device.reset();
  };

  std::mt19937 g(seed + 2);
  timer::tic();
  double best_dev = 0;
  for (size_t e = 0; e < epochs; e++) {
    std::cout << "Epoch " << e << std::endl;
    auto perm = randperm(XYb.size(), g);
    metric::Accuracy<size_t> m;
    for (size_t i : perm) { pass(XYb[i], m, true, false); }
    auto tr_acc = m.eval();
    m.clear();
    size_t print_i = rand() % XYbdev.size();
    for (size_t i = 0; i < XYbdev.size(); i++) {
      pass(XYbdev[i], m, false, i == print_i);
    }
    double dev_acc = m.eval();
    std::cout << "Char accuracy (train): " << tr_acc << std::endl
              << "Char accuracy   (dev): " << dev_acc << std::endl;

    if (dev_acc > best_dev) {
      best_dev = dev_acc;
      std::ofstream out(model);
      for (auto& w : weights) { w->value().save(out); }
    }
  }

  std::cout << "Elapsed: " << timer::toc() / 1e6 << "s" << std::endl;

  return 0;
}
