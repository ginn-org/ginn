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

#include <iostream>
#include <unordered_set>

#include <tblr.h>

#include <ginn/init/init.h>
#include <ginn/update/update.h>

#include <ginn/node/layout.h>
#include <ginn/node/pick.h>

#include <ginn/metric.h>
#include <ginn/util/amax.h>
#include <ginn/util/cli2.h>
#include <ginn/util/parfor.h>
#include <ginn/util/timer.h>
#include <ginn/util/util.h>

#include "seqtag.h"

using namespace ginn;

DevPtr Dev = cpu();

using Architecture = ginn::SequenceTagger;
using Token = std::tuple<std::string /*word*/, std::string /*label*/>;
using Sequence = std::vector<Token>;
using Sequences = std::vector<Sequence>;

Sequences conll_reader(const std::string& fname) {
  Sequences sents;
  Sequence sent;

  for (const auto& line : lines(fname)) {
    auto v = split(line, '\t');
    if (v.empty()) { // finished a sentence
      if (not sent.empty()) { sents.emplace_back(sent); }
      sent.clear();
    } else {
      sent.emplace_back(v[0], v[1]);
    }
  }
  if (not sent.empty()) { sents.emplace_back(sent); }

  return sents;
}

// Vocabulary extractor
auto vocabs(std::initializer_list<Sequences*> data) {
  std::unordered_set<std::string> words, labels;
  std::unordered_set<char> chars;
  for (Sequences* seqs : data) {
    for (const Sequence& seq : *seqs) {
      for (const auto& [word, label] : seq) {
        words.insert(word);
        labels.insert(label);
        chars.insert(word.begin(), word.end());
      }
    }
  }
  return std::make_tuple(words, labels, chars);
}

int main(int argc, char** argv) {
  size_t seed = 13579;
  size_t num_threads = 1;
  size_t patience = 15, epochs = 25;

  std::string trainset, devset, testset, model_fname, wvecname;
  Size wvecdim = 300;

  Real lr = 1e-4, drop_p = 0.1, word_drop_p = 0.1;
  Size dim_chars = 25, dim = 50, dim_char_lstm = 25;
  size_t layers = 1;
  bool fixed_wvecs = false;
  bool eval = false;

  Args args;

  args(Arg(seed).name("seed").help("random seed"));
  args(Arg(num_threads).name("threads").help("number of threads to use"));

  args(Arg(model_fname)
           .name("m,model-path")
           .meta("path")
           .help("model path to store or load")
           .required());
  args(Arg(trainset)
           .name("t,train-path")
           .meta("path")
           .help("path to training set")
           .required());
  args(Arg(devset)
           .name("d,dev-path")
           .meta("path")
           .help("path to dev set")
           .required());
  args(Arg(testset)
           .name("y,test-path")
           .meta("path")
           .help("path to test set")
           .required());
  args(Arg(wvecname)
           .name("w,embedding-path")
           .meta("path")
           .help("path to pretrained word vector table")
           .required());

  args(Arg(patience)
           .name("n,patience")
           .help("number of epochs without improvement to give up"));
  args(Arg(epochs).name("e,epochs").help("max number of epochs"));
  args(Arg(wvecdim).name("x,word-dim").help("word vector dimensionality"));
  args(Arg(dim_chars).name("c,char-dim").help("char vector dimensionality"));
  args(Arg(dim).name("z,word-lstm-dim").help("word lstm dimensionality"));
  args(Arg(layers).name("L,layers").help("number of layers in word lstm"));
  args(Arg(dim_char_lstm)
           .name("a,char-lstm-dim")
           .help("char lstm dimensionality"));
  args(Arg(lr).name("l,learning-rate").help("learning rate"));
  args(Arg(drop_p).name("p,dropout-rate").help("dropout rate"));
  args(Arg(word_drop_p).name("q,word-dropout-rate").help("word dropout rate"));
  args(Arg(fixed_wvecs)
           .name("f,frozen-embeddings")
           .help("if true, do not update word vectors"));
  args(Flag(eval).name("eval").help("evaluate model without training"));

  args.parse(argc, argv);

  srand(seed);

  auto tra = conll_reader(trainset);
  auto dev = conll_reader(devset);
  auto tst = conll_reader(testset);

  auto [wvocab, label_vocab, cvocab] = vocabs({&tra, &dev, &tst});

  Size dimy = label_vocab.size();

  Architecture::Params params{.layers = layers,
                              .dim = dim,
                              .dim_wvec = wvecdim,
                              .dim_char_lstm = dim_char_lstm,
                              .dim_cvec = dim_chars,
                              .drop_p = drop_p,
                              .word_drop_p = word_drop_p,
                              .fixed_wvecs = fixed_wvecs,
                              .dim_y = dimy};
  Architecture lstm(Dev, params, cvocab, label_vocab);

  if (not eval) {
    init::Xavier<Real>().init(lstm.weights());

    std::cout << "Loading word vectors..." << std::flush;
    lstm.load_wvecs(wvecname, wvocab);
    std::cout << " Done." << std::endl;
  }

  update::Adam<Real> updater(lr);

  std::vector<Architecture> lstms;
  for (size_t i = 0; i < num_threads; i++) {
    Architecture lstm_(Dev);
    lstm_.tie(lstm);
    lstms.push_back(lstm_);
  }

  auto pass = [&](const Sequence& instance,
                  auto& metric,
                  Mode mode,
                  Architecture& lstm_) {
    size_t T = instance.size();
    std::vector<std::string> x, y;
    for (auto& [word, label] : instance) {
      x.push_back(word);
      y.push_back(label);
    }

    NodePtrs<Real> predictions = lstm_.score_words(x, mode);

    if (mode == Mode::Training) {
      NodePtrs<Real> losses;

      for (size_t t = 0; t < T; t++) {
        auto loss = PickNegLogSoftmax(predictions[t], lstm_.label_map()[y[t]]);
        losses.push_back(loss);
      }

      auto root = Add(losses);
      auto graph = Graph(root);
      graph.forward();

      graph.reset_grad();
      graph.backward(1.);
      updater.update(graph);
    } else {
      Graph(Sink(predictions)).forward();
    }

    std::vector<std::string> label_seq(T);
    for (size_t t = 0; t < T; t++) {
      auto y_i = argmax(predictions[t]->value());
      label_seq[t] = lstm_.label_map()(y_i);
    }

    metric.add(label_seq, y);
  };

  std::mt19937 g(seed);

  auto run = [&](const Sequences& X,
                 Mode mode,
                 std::vector<Architecture>& lstms) {
    auto perm = mode == Mode::Training ? randperm(X.size(), g) : iota(X.size());

    metric::SpanF1 fscore;
    parallel_for(
        0,
        perm.size(),
        [&](size_t i, size_t tid) {
          auto j = perm[i];
          pass(X[j], fscore, mode, lstms[tid]);
        },
        num_threads);
    return fscore.eval_all();
  };

  auto sorted_keys = [](const auto& m) {
    std::vector<std::string> keys;
    keys.push_back("all");
    for (const auto& p : m) {
      if (p.first != "all") { keys.push_back(p.first); }
    }
    std::sort(keys.begin() + 1, keys.end());
    return keys;
  };

  auto print = [sorted_keys](const std::string& name, const auto& score) {
    auto t = tblr::Table().layout(tblr::markdown()).precision(3);
    t << name << "Precis"
      << "Recall"
      << "F1" << tblr::endr;
    for (auto& k : sorted_keys(score)) {
      auto [p, r, f] = score.at(k);
      t << k << p << r << f << tblr::endr;
    }
    std::cout << t;
  };

  auto train = [&]() {
    Real best_dev = 0;
    for (size_t p = 1, e = 0; p <= patience and e < epochs; p++, e++) {
      std::cout << "Epoch " << e << std::endl;

      timer::tic();
      auto tr_score = run(tra, Mode::Training, lstms);
      print("Train", tr_score);

      auto dev_score = run(dev, Mode::Inference, lstms);
      print("Dev", dev_score);
      std::cout << timer::toc(timer::HumanReadable) << std::endl;
      std::cout << std::endl;

      if (dev_score["all"].f > best_dev) {
        best_dev = dev_score["all"].f;
        lstm.save(model_fname, 6);
        p = 0; // reset patience
      }
    }
    std::cout << "Done." << std::endl;
  };

  auto evaluate = [&](const std::string& model) {
    Architecture lstm(Dev);
    lstm.load(model);
    std::vector<Architecture> lstms;
    for (size_t i = 0; i < num_threads; i++) {
      lstms.emplace_back(Dev);
      lstms.back().tie(lstm);
    }
    std::cout << model << std::endl;

    print("Dev", run(dev, Mode::Inference, lstms));
    print("Test", run(tst, Mode::Inference, lstms));
  };

  if (not eval) { train(); }
  evaluate(model_fname);

  return 0;
}
