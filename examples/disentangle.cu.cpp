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
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_set>

#include <ginn/init/init.h>
#include <ginn/layer/layers.h>
#include <ginn/layer/tree.h>
#include <ginn/metric.h>
#include <ginn/update/update.h>
#include <ginn/util/commandline.h>
#include <ginn/util/parfor.h>
#include <ginn/util/timer.h>
#include <ginn/util/tree.h>
#include <ginn/util/util.h>
#include <ginn/util/wvec.h>

using Token = std::string; // this could alternatively be an index to the word
using Text = std::vector<Token>;
const std::string PAD_TOKEN = "__PAD__";

struct Utterance {
  std::string timestamp;
  std::string user;
  Text text;
  bool presence = false;
};

struct Chat {
  std::vector<Utterance> utterances;
  std::vector<std::vector<Token>> batched_uttrs;
  std::vector<std::set<size_t>> links_before;
  std::vector<int> past_by_user;
  std::vector<int> future_by_user;
  std::string fname;

  size_t size() const {
    assert(utterances.size() == links_before.size());
    return utterances.size();
  }
};

std::string replace_suffix(std::string s,
                           const std::string& orig,
                           const std::string& subs) {
  size_t orig_s = orig.size();
  assert(s.substr(s.size() - orig_s, s.size()) == orig);
  return s.replace(s.size() - orig_s, orig_s, subs);
}

std::string
replace(std::string s, const std::string& orig, const std::string& subs) {
  size_t b = s.find(orig);
  assert(b != std::string::npos);
  return s.replace(b, orig.size(), subs);
}

Chat read_chat(std::string ufname, const std::string& lfname, size_t skip = 0) {
  Chat chat;
  std::string line;
  ufname = replace_suffix(ufname, ".ascii.txt", ".v1.txt");
  chat.fname = ufname;
  // std::cout << ufname << std::endl;
  std::ifstream in(ufname);
  assert(in);

  for (size_t line_c = 0; std::getline(in, line); line_c++) {
    if (line_c < skip) { continue; }
    if (line.substr(0, 3) == "===") {
      chat.utterances.push_back({"", "", split(line)});
    } else {
      auto parts = split(line);
      assert(parts.size() >= 2);
      assert(parts[0][0] == '[' and parts[0].back() == ']');
      if (parts[1] == "*") {
        // Assume parts[2] is the username
        chat.utterances.push_back({parts[0],
                                   parts[2],
                                   Text(parts.begin() + 3, parts.end()),
                                   /*presence*/ true});
      } else {
        assert((parts[1][0] == '<' and parts[1].back() == '>') or
               (parts[1][0] == '(' and parts[1].back() == ')') or
               (parts[1][0] == '-' and parts[1].back() == '-'));
        chat.utterances.push_back({parts[0],
                                   parts[1].substr(1, parts[1].size() - 2),
                                   Text(parts.begin() + 2, parts.end())});
      }
    }
  }

  chat.links_before.resize(chat.utterances.size());
  for (std::string line : util::ifstream(lfname)) {
    auto parts = split(line);
    assert(parts.size() == 3);
    assert(parts.back() == "-");
    size_t a = std::stoi(parts[0]);
    size_t b = std::stoi(parts[1]);
    if (a < b) { std::swap(a, b); } // ensure a > b
    if (a != b and a > skip and b > skip) {
      chat.links_before.at(a - skip).insert(b - skip);
    } else {
      ; // discarding self links for now...
    }
  }

  size_t s = chat.size();
  size_t max_len = 0;

  for (size_t i = 0; i < s; i++) {
    if (chat.utterances[i].text.size() > max_len) {
      max_len = chat.utterances[i].text.size();
    }
  }

  chat.batched_uttrs = std::vector<std::vector<Token>>(
      max_len, std::vector<Token>(s, PAD_TOKEN));
  for (size_t t = 0; t < max_len; t++) {
    for (size_t i = 0; i < s; i++) {
      if (t < chat.utterances[i].text.size()) {
        chat.batched_uttrs[t][i] = chat.utterances[i].text[t];
      }
    }
  }

  std::unordered_map<std::string, size_t> past_by_user;
  for (size_t t = 0; t < chat.utterances.size(); t++) {
    auto& u = chat.utterances[t];
    if (past_by_user.find(u.user) == past_by_user.end()) {
      chat.past_by_user.push_back(-1);
    } else {
      chat.past_by_user.push_back(past_by_user[u.user]);
    }
    past_by_user[u.user] = t;
  }

  return chat;
}

std::vector<Chat> read_chats(const std::string& datadir,
                             const std::string& listfname,
                             size_t skip = 0) {
  static const size_t ascii_col = 0;
  static const size_t annot_col = 1;
  std::vector<Chat> chats;
  for (std::string line : util::ifstream(datadir + listfname)) {
    auto parts = split(line);
    chats.push_back(read_chat(
        datadir + parts[ascii_col], datadir + parts[annot_col], skip));
  }
  return chats;
}

// For diagnostics
void print(const Chat& chat, std::ostream& out = std::cout) {
  for (size_t i = 0; i < chat.size(); i++) {
    auto& u = chat.utterances[i];
    auto& l = chat.links_before[i];
    out << u.timestamp << "\t" << u.user << "\t";
    for (auto& w : u.text) { out << w << " "; }
    out << "\t";
    for (size_t ix : l) { out << (i - ix) << "\t"; }
    out << std::endl;
  }
}

class Arch {
 public:
  uint dimw = 50;
  uint dimh = 100;
  LookupLayer<Token> look_l;
  BiLstmLayer uttr_l;
  AffineLayer prepool_l;
  TreeLstm chat_f, chat_b;
  AffineLayer out_left, out_right;
  WeightRef biaffine_W, biaffine_b;

  std::vector<WeightRef> weights() {
    return look_l.weights() + uttr_l.weights() + prepool_l.weights() +
           chat_f.weights() /*+ chat_b.weights()*/ + out_left.weights() +
           out_right.weights() + std::vector<WeightRef>{biaffine_W, biaffine_b};
  }

  Device device() { return look_l.dev; }

  void save(std::ostream& out) {
    out << dimw << "\n" << dimh << "\n";
    out << look_l.table.map.i2k.size() << "\n";
    for (Token& w : look_l.table.map.i2k) { out << w << "\n"; }
    for (auto w : weights()) {
      Tensor w_(CPU);
      w_ = w->value();
      w_.save(out);
    }
  }

  void load(std::istream& in) {
    auto dev = CPU;
    Token w;
    size_t s;
    std::getline(in, w);
    dimw = std::stoi(w);
    std::getline(in, w);
    dimh = std::stoi(w);

    look_l = LookupLayer<Token>(dev, dimw, 0.);
    uttr_l.init(dev, dimh, dimw, 0.);
    prepool_l.init(dev, dimh, 2 * dimh);
    chat_f.init(dev, 2, dimh, dimh, TreeLstm::max);
    out_left.init(dev, dimh, dimh);
    out_right.init(dev, dimh, dimh);
    biaffine_W = Weight(dev, dimh, dimh);
    biaffine_b = Weight(dev, dimh, 1);

    std::getline(in, w);
    s = std::stoi(w);
    for (size_t i = 0; i < s; i++) {
      std::getline(in, w);
      look_l.table.insert(w, Weight(CPU, dimw));
    }

    for (auto w : weights()) { w->value().load(in); }
  }

  auto operator()(const Chat& chat, bool train) {
    size_t s = chat.size();
    std::vector<ExprRef> tmp, wvs_;
    for (const auto& w : chat.batched_uttrs) {
      for (auto& w_ : w) {
        auto we = look_l.table[w_];
        assert(we->value().v().size() == dimw);
      }
      auto wvs = Reshape(Cat(look_l(w, train)), Shape{dimw, s});
      wvs_.push_back(wvs);
    }
    auto uh = CwiseMax(prepool_l(uttr_l(wvs_, true, train, s)));

    std::vector<TreeLstm::State> states;
    for (size_t i = 0; i < s; i++) {
      auto e = Slice(uh, Index<2>{0, i}, Index<2>{dimh, 1});
      std::vector<TreeLstm::Child> children;
      if (i > 0) {
        children.emplace_back(
            0, std::get<0>(states.back()), std::get<1>(states.back()));
      }
      int k = chat.past_by_user[i];
      if (k > -1) {
        children.emplace_back(
            1, std::get<0>(states[k]), std::get<1>(states[k]));
      }
      auto s = chat_f.step(e, children);
      states.push_back(s);
    }

    return states;
  }

  auto pairwise(const Chat& chat,
                bool train,
                bool all_neg_pairs,
                size_t start,
                size_t end,
                size_t cap = 100,
                Real neg_sample_p = 0.5,
                Real pos_loss_weight = 1.) {
    auto states = (*this)(chat, train);
    std::vector<ExprRef> losses, neg_losses;
    std::vector<std::pair<ExprRef, short>> outputs;
    std::vector<ExprRef> lefts, rights;
    std::vector<uint> indices;
    std::vector<std::pair<uint, uint>> edges;
    std::vector<Real> weights;

    for (size_t i = start; i < std::min(end, chat.size()); i++) {
      // Positive pairs
      for (size_t j : chat.links_before[i]) {
        auto& h_l = std::get<0>(states[j]);
        auto& h_r = std::get<0>(states[i]);
        lefts.push_back(h_l);
        rights.push_back(h_r);
        indices.push_back(1);
        edges.emplace_back(j, i);
      }
      // Negative pairs
      for (size_t j = std::max(int(0), int(i) - int(cap)); j < i; j++) {
        if (chat.links_before[i].find(j) == chat.links_before[i].end()) {
          if (all_neg_pairs or (float(rand()) / RAND_MAX < neg_sample_p)) {
            auto& h_l = std::get<0>(states[j]);
            auto& h_r = std::get<0>(states[i]);
            lefts.push_back(h_l);
            rights.push_back(h_r);
            indices.push_back(0);
            edges.emplace_back(j, i);
          }
        }
      }
    }
    auto left = Reshape(Cat(lefts), Shape{dimh, lefts.size()});
    auto right = Reshape(Cat(rights), Shape{dimh, rights.size()});
    auto y = ColSum(CwiseProd(
        left, biaffine_W * right + ColBroadcast(biaffine_b, rights.size())));
    return std::tuple<ExprRef,
                      ExprRef,
                      std::vector<uint>,
                      std::vector<std::pair<uint, uint>>>(
        Sum(PickNegLogSigmoid(y, indices)), y, indices, edges);
  }
};

int main(int argc, char** argv) {
#ifdef GINN_ENABLE_GPU
  Device dev = GPU;
  const std::string datadir = "/job/irc-disentanglement/"; // remote path
#else
  Device dev = CPU;
  const std::string datadir =
      "/home/oirsoy/space/data/irc-disentanglement/"; // local path
#endif
  const std::string tr_list = "data/list.ubuntu.train.txt";
  const std::string dev_list = "data/list.ubuntu.dev.txt";
  // const std::string wv_fname = "data/glove-ubuntu.txt";
  const std::string wv_fname = "data/embeddings/w2v/w2v.vec";

  auto tr = read_chats(datadir, tr_list, 800);
  auto dv = read_chats(datadir, dev_list, 800);

  std::unordered_set<Token> vocab;
  for (auto& chat : tr) {
    for (auto& u : chat.utterances) {
      for (auto& w : u.text) { vocab.insert(w); }
    }
  }
  vocab.insert(PAD_TOKEN);
  std::cout << vocab.size() << std::endl;

  Arch arch;
  bool continue_training = false;
  if (continue_training) {
    std::ifstream in("tmp.txt");
    arch.load(in);
  } else {
    arch.dimw = 100;
    arch.dimh = 200;
    arch.look_l = LookupLayer<Token>(dev, arch.dimw, 0.1, vocab);
    arch.uttr_l.init(dev, arch.dimh, arch.dimw, 0.);
    arch.prepool_l.init(dev, arch.dimh, 2 * arch.dimh);
    arch.chat_f.init(dev, 2, arch.dimh, arch.dimh, TreeLstm::max);
    arch.out_left.init(dev, arch.dimh, arch.dimh);
    arch.out_right.init(dev, arch.dimh, arch.dimh);
    arch.biaffine_W = Weight(dev, arch.dimh, arch.dimh);
    arch.biaffine_b = Weight(dev, arch.dimh, 1);

    init::Xavier().init(arch.weights());
    util::load_wvecs(arch.look_l.table, dev, datadir + wv_fname, vocab);
  }

  if (false) {
    // if(true) {
    std::ifstream in("tmp.txt");
    arch.load(in);
    auto print = [](const auto& eval) {
      auto& val = eval.at("Edge");
      std::cout << "Edge F:\t" << val.precision << "\t" << val.recall << "\t"
                << val.f << std::endl;
    };

    auto run = [&](metric::Metric<std::string>& m, const Chat& c) {
      bool train = false;
      ExprRef loss, y;
      std::vector<uint> indices;
      std::vector<std::pair<uint, uint>> edges;
      std::tie(loss, y, indices, edges) =
          arch.pairwise(c, train, !train, 200, 100000, 100);
      auto g = Graph(loss);
      g.forward();
      Tensor y_(CPU);
      y_ = y->value();
      auto yv = y_.v();
      std::cout << indices.size() << " " << c.size() << std::endl;
      uint offset = 800;
      auto preds_fname = replace_suffix(replace(c.fname, "/dev/", "/dev_pred/"),
                                        ".v1.txt",
                                        ".annotation.txt");
      std::cout << preds_fname << std::endl;
      std::ofstream pout(preds_fname);
      std::map<size_t, size_t> outgoing_count;
      for (size_t i = 0; i < indices.size(); i++) {
        bool pred = bool(yv[i] > 0.);
        m.add(pred ? "Edge" : "NoEdge", bool(indices[i]) ? "Edge" : "NoEdge");
        if (pred) {
          pout << edges[i].first + offset << " " << edges[i].second + offset
               << " -\n";
          outgoing_count[edges[i].second + offset]++;
        } else {
          outgoing_count[edges[i].second + offset];
        }
      }
      for (auto& p : outgoing_count) {
        if (p.second == 0) { pout << p.first << " " << p.first << " -\n"; }
      }
    };
    /* dev eval */ {
      timer::tic();
      metric::F1<std::string> f("_Micro", "_Macro");
      for (const auto& c : dv) { run(f, c); }
      print(f.eval_all());
      std::cout << timer::toc() / 1'000'000. << std::endl;
    }
    return 0;
  }

  update::Adam updater(/*lr*/ 1e-4);

  auto print = [](const auto& eval) {
    auto& val = eval.at("Edge");
    std::cout << "Edge F:\t" << val.precision << "\t" << val.recall << "\t"
              << val.f << std::endl;
  };

  auto run =
      [&](metric::Metric<std::string>& m, const Chat& c, bool train = false) {
        ExprRef loss, y;
        std::vector<uint> indices;
        std::vector<std::pair<uint, uint>> edges;
        std::tie(loss, y, indices, edges) =
            arch.pairwise(c, train, true, 200, 100000, 100);
        auto g = Graph(loss);
        g.forward();
        if (train) {
          g.reset_grad();
          g.backward(1.);
          updater.update(g);
        }
        Tensor y_(CPU);
        y_ = y->value();
        auto yv = y_.v();
        for (size_t i = 0; i < indices.size(); i++) {
          bool pred = bool(yv[i] > 0.);
          m.add(pred ? "Edge" : "NoEdge", bool(indices[i]) ? "Edge" : "NoEdge");
        }
      };

  size_t epochs = 50;
  std::vector<size_t> perm(tr.size());
  std::iota(perm.begin(), perm.end(), 0);
  Real best_dev = -1.;
  for (size_t e = 0; e < epochs; e++) {
    std::random_shuffle(perm.begin(), perm.end());
    /* train */ {
      timer::tic();
      metric::F1<std::string> f("_Micro", "_Macro");
      for (size_t i = 0; i < perm.size(); i++) {
        if (i % 10 == 0) {
          std::cout << i << " / " << perm.size() << std::endl;
        }
        run(f, tr[perm[i]], true);
      }
      print(f.eval_all());
      std::cout << timer::toc() / 1'000'000. << std::endl;
    }
    /* dev eval */ {
      timer::tic();
      metric::F1<std::string> f("_Micro", "_Macro");
      for (const auto& c : dv) { run(f, c, false); }
      auto eval = f.eval_all();
      print(eval);
      Real f1 = eval["Edge"].f;
      if (f1 > best_dev) {
        best_dev = f1;
        std::ofstream out("tmp-w2v.txt");
        arch.save(out);
      }
      std::cout << timer::toc() / 1'000'000. << std::endl;
    }
  }

  return 0;
}
