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
#include <unordered_set>

#include <ginn/dev.h>
#include <ginn/util/amax.h>
#include <ginn/util/cli2.h>
#include <ginn/util/fmt.h>
#include <ginn/util/parfor.h>
#include <ginn/util/timer.h>
#include <ginn/util/tree.h>
#include <ginn/util/util.h>
#include <ginn/util/wvec.h>

#include <ginn/init/init.h>
#include <ginn/update/update.h>

#include <ginn/layer/common.h>
#include <ginn/layer/tree.h>
#include <ginn/metric.h>
#include <ginn/model/treelstm.h>

#include <ginn/node/pick.h>

#include <ginn/autobatch.h>

using Content =
    std::pair<unsigned /*sentiment score*/, std::string /*word if leaf*/>;

using namespace ginn;
using namespace ginn::tree;

std::vector<Tree<Content>> read_sstb(const std::string& fname) {
  std::vector<Tree<Content>> trees;
  auto reader = [](std::string_view s) -> Content {
    auto parts = split(std::string(s));
    if (parts.size() == 1) { return {std::stoi(parts[0]), ""}; }
    return {std::stoi(parts.at(0)), parts.at(1)};
  };
  for (auto& line : lines(fname)) {
    trees.push_back(tree::parse<Content>(line, reader));
  }
  return trees;
}

int main(int argc, char** argv) {
  Real lr = 1e-4;
  Real dr = 0.25;
  Real wdr = 0.1;
  Size dim = 100;
  Size wdim = 300;
  size_t layers = 1;
  size_t epochs = 5;
  size_t bs = 30;
  size_t seed = 12347;
  bool wv_tune = false;
  bool autobatch = false;
  size_t mem = 2e9;

  std::string wv_fname, datadir;

  Args args;

  args(Arg(wv_fname)
           .name("w,word-vector-file")
           .meta("path")
           .help("path to pretrained word vector table")
           .required());
  args(
      Arg(datadir)
          .name("d,data-folder")
          .meta("path")
          .help(
              "path to folder containing train, dev & test files in Ptb tree format")
          .required());
  args(Arg(lr).name("l,learning-rate").help("learning rate"));
  args(Arg(dr).name("dropout-rate").help("dropout rate"));
  args(Arg(wdr).name("word-dropout-rate").help("word dropout rate"));
  args(Arg(layers).name("L,layers").help("number of layers"));
  args(Arg(dim).name("D,dim").help("dimensionality of hidden layers"));
  args(Arg(wdim).name("W,word-dim").help("dimensionality of word vectors"));
  args(Arg(wv_tune)
           .name("tune-word-vectors")
           .help("whether to finetune (update) word embeddings"));
  args(Arg(epochs).name("e,epochs").help("number of epochs in training"));
  args(Arg(bs).name("b,batch-size").help("minibatch size to accumulate grads"));
  args(Arg(seed).name("s,seed").meta("n").help("random seed"));
  args(
      Arg(autobatch)
          .name("a,autobatch")
          .help(
              "whether to apply autobatching to minibatched computation graph"));

  args.parse(argc, argv);

  srand(seed);

  auto train = read_sstb(datadir + "/train.txt");
  auto dev = read_sstb(datadir + "/dev.txt");
  auto test = read_sstb(datadir + "/test.txt");

  std::cout << train.size() << " " << dev.size() << " " << test.size()
            << std::endl;

  // optionally sanity check reading
  // print(std::cout, train.front(), [](std::ostream& out, const Content& data)
  // {
  //  out << data.first << " " << data.second;
  // });

#ifdef GINN_ENABLE_GPU
  Device& Dev = gpu();
  PreallocGpu device(mem);
#else
  Device& Dev = cpu();
  PreallocCpu device(mem);
#endif

  std::unordered_set<std::string> wvocab;
  for (auto& n : train + dev) {
    for (auto& c : n) { wvocab.insert(c.second); } // add the words
  }

  std::cout << "Loading word vectors..." << std::flush;
  auto word_table = LookupLayer<WeightPtr<Real>(std::string)>(Dev, wdim, wdr);
  util::load_wvecs(word_table->table, Dev, wv_fname, wvocab);
  word_table->table.unk() = Weight(Dev, {wdim}); // unknown word
  std::cout << " Done. Vocab size: " << word_table->table.size() << std::endl;

  LayerPtr<NodeTree<Real>(NodeTree<Real>)> net;

  for (size_t l = 0; l < layers; l++) {
    Size xdim = (l == 0) ? wdim : dim;
    auto tlstm = TreeLstmLayer<Real>(Dev, 2, dim, xdim, 0);
    if (l == 0) {
      net = tlstm;
    } else {
      net = net | tlstm;
    }
  }

  auto out = Containerwise<Tree>(AffineLayer<Real>(Dev, 5, dim));
  net = net | out;

  std::vector<WeightPtr<Real>> weights;
  weights += net->weights<Real>();
  weights += std::vector<WeightPtr<Real>>{word_table->table.unk()};

  init::Xavier<Real>().init(weights);
  // out->b->value().set_zero();
  word_table->table.unk()->value().set_zero();

  update::Adam<Real> updater(lr);
  updater.init(weights);

  metric::Accuracy<Size> sent, phrase;

  auto zero_wdim = Zero(Dev, {wdim});

  auto pass_batch = [&](auto& instances, bool train) {
    std::vector<NodePtr<Real>> losses;
    std::vector<std::vector<NodePtr<Real>>> ys;

    ys.reserve(instances.size());

    timer::tic("Graph construction");
    for (auto& instance : instances) {
      auto expr_tree = make_like<NodePtr<Real>>(instance);

      for (size_t t = 0; t < expr_tree.size(); t++) {
        if (instance[t].second.empty()) { // internal node
          expr_tree[t] = zero_wdim;
        } else { // leaf node
          word_table->set_mode(Mode::Training);
          expr_tree[t] =
              DeviceView(word_table->run(instance[t].second), device);
        }
      }

      Tree<NodePtr<Real>> h = net->run(expr_tree);

      std::vector<NodePtr<Real>> curr_ys;
      curr_ys.reserve(expr_tree.size());
      for (size_t t = 0; t < expr_tree.size(); t++) {
        NodePtr<Real> y = h[t];
        if (autobatch) { y = Map(y); }
        auto loss = PickNegLogSoftmax(y, instance[t].first);
        curr_ys.push_back(y);
        losses.push_back(loss);
      }
      ys.push_back(std::move(curr_ys));
    }

    auto loss = Add(losses);
    timer::toc("Graph construction");

    GINN_TIME(auto g = Graph(loss));
    if (autobatch) { GINN_TIME(g = Autobatch(g)); }
    GINN_TIME(g.forward());

    for (size_t i = 0; i < instances.size(); i++) {
      auto& instance = instances[i];
      for (size_t t = 0; t < ys.at(i).size(); t++) {
        auto& y = ys[i].at(t);
        Size pred = argmax(y->value());
        phrase.add(pred, instance[t].first);
        if (t == (ys[i].size() - 1)) { sent.add(pred, instance[t].first); }
      }
    }

    if (train) {
      GINN_TIME(g.reset_grad());
      GINN_TIME(g.backward(1.));
      GINN_TIME(updater.update(g));
    }

    return loss->item();
  };

  std::mt19937 g(123457);

  auto run = [&](const auto& X, bool train) {
    auto perm = train ? randperm(X.size(), g) : iota(X.size());

    Real loss = 0.;
    size_t count = 0;
    phrase.clear();
    sent.clear();
    for (size_t i = 0; i < perm.size(); i += bs) {
      std::vector<Tree<Content>> batch;
      for (size_t j = 0; j < bs and (i + j) < perm.size(); j++) {
        batch.push_back(X[perm[i + j]]);
      }
      if (batch.empty()) { continue; }
      loss += pass_batch(batch, train);
      count += batch.size();
      device.reset();
    }

    return std::make_tuple(
        loss / count, phrase.eval() * 100., sent.eval() * 100.);
  };

  Real best_dev = 0, best_test_phr = 0, best_test_sen = 0;

  using namespace ginn::literals;

  std::cout << ("{:22}{:22}Elapsed\n"_f, "Train", "Dev")
            << "Loss  PhrAcc SenAcc   Loss  PhrAcc SenAcc" << std::endl;

  for (size_t e = 0; e < epochs; e++) {
    timer::tic();
    auto [loss, phr_acc, sen_acc] = run(train, true);
    std::cout << ("{:>5.2f}  {:>5.2f}  {:>5.2f}   "_f, loss, phr_acc, sen_acc)
              << std::flush;
    std::tie(loss, phr_acc, sen_acc) = run(dev, false);
    std::cout << ("{:>5.2f}  {:>5.2f}  {:>5.2f}   "_f, loss, phr_acc, sen_acc)
              << timer::toc(timer::HumanReadable) << std::endl;

    if (sen_acc > best_dev) {
      best_dev = sen_acc;
      std::tie(loss, best_test_phr, best_test_sen) = run(test, false);
    }
  }
  std::cout << best_dev << " " << best_test_phr << " " << best_test_sen
            << std::endl;

  timer::print();

  return 0;
}
