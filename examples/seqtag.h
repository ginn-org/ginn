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

#ifndef GINN_EXAMPLES_SEQTAG_H
#define GINN_EXAMPLES_SEQTAG_H

#include <ginn/layer/common.h>
#include <ginn/util/indexmap.h>
#include <ginn/util/wvec.h>

namespace ginn {

class SequenceTagger {
 public:
  struct Params {
    size_t layers = 1;
    Size dim = 50;
    Size dim_wvec = 300;
    Size dim_char_lstm = 25;
    Size dim_cvec = 25;
    Real drop_p = 0.3;
    Real word_drop_p = 0.1;
    bool fixed_wvecs = false;
    Size dim_y;
  };

  template <typename T>
  using Set = std::unordered_set<T>;

  using Word = std::string;
  using Words = std::vector<Word>;
  using Chars = std::vector<char>;
  using CharsAndWord = std::tuple<Chars, Word>;

  // private:
  Device* dev_;
  Params params_;

  std::shared_ptr<LookupLayerNode<WeightPtr<Real>(char)>> char_table_;
  std::shared_ptr<LookupLayerNode<WeightPtr<Real>(Word)>> word_table_;

  LayerPtr<NodePtrs<Real>(Words)> model_;

  bool use_char_model_;
  IndexMap<std::string> label_map_;

 public:
  SequenceTagger(Device& dev) : dev_(&dev) {}
  SequenceTagger(Device& dev,
                 Params params,
                 Set<char>& char_vocab,
                 Set<std::string>& label_vocab)
      : dev_(&dev), params_(params) {
    init(char_vocab, label_vocab);
  }

  void init(const Set<char>& char_vocab, const Set<std::string>& label_vocab) {
    auto& p = params_;

    auto basecast = FunctionalLayer<NodePtr<Real>(WeightPtr<Real>)>(
        [](const WeightPtr<Real> x) -> NodePtr<Real> { return x; });

    char_table_ =
        LookupLayer<WeightPtr<Real>(char)>(*dev_, p.dim_cvec, 0.1, char_vocab);
    auto ctr_char_table = Containerwise<std::vector>(char_table_ | basecast);

    LayerPtr<NodePtrs<Real>(NodePtrs<Real>)> char_model;
    if (p.dim_char_lstm > 0 and p.dim_cvec > 0) {
      char_model =
          BiLstmLayer<Real>(*dev_, p.dim_char_lstm, p.dim_cvec, true, 0.1);
      use_char_model_ = true;
    } else {
      p.dim_char_lstm = p.dim_cvec = 0;
      use_char_model_ = false;
    }

    word_table_ =
        LookupLayer<WeightPtr<Real>(Word)>(*dev_, p.dim_wvec, p.word_drop_p);

    LayerPtr<NodePtrs<Real>(NodePtrs<Real>)> word_model;
    for (size_t l = 0; l < p.layers; l++) {
      Size in_dim = (l == 0) ? p.dim_wvec + 2 * p.dim_char_lstm : 2 * p.dim;
      auto lstm = BiLstmLayer<Real>(*dev_, p.dim, in_dim, false, p.drop_p);
      if (l == 0) {
        word_model = lstm;
      } else {
        word_model = word_model | lstm;
      }
    }
    word_model = word_model | Containerwise<std::vector>(
                                  AffineLayer<Real>(*dev_, p.dim_y, 2 * p.dim));

    LayerPtr<NodePtr<Real>(Word)> word_repr;
    if (use_char_model_) {
      auto extract_chars = FunctionalLayer<CharsAndWord(std::string)>(
          [](const Word& word) -> CharsAndWord {
            return {Chars(word.begin(), word.end()), word};
          });
      auto item = FunctionalLayer<NodePtr<Real>(NodePtrs<Real>)>(
          [](const NodePtrs<Real>& in) {
            GINN_ASSERT(in.size() == 1);
            return in[0];
          });

      word_repr = extract_chars |
                  ((ctr_char_table | char_model | item), word_table_) |
                  CatLayer<Real>();
    } else {
      word_repr = word_table_ | basecast;
    }

    model_ = Containerwise<std::vector>(word_repr) | word_model;

    label_map_ = IndexMap<std::string>(label_vocab);
  }

  void tie(SequenceTagger& other) {
    dev_ = other.dev_;
    params_ = other.params_;

    use_char_model_ = other.use_char_model_;
    label_map_ = other.label_map_;

    char_table_ = nullptr;
    word_table_ = nullptr;
    model_ = other.model_->copy(Copy::Tied);
  }

  void load_wvecs(const std::string& fname,
                  const std::unordered_set<std::string>& word_vocab) {
    util::load_wvecs(
        word_table_->table, *dev_, fname, word_vocab, params_.fixed_wvecs);
    word_table_->table.unk() = Weight(*dev_, {params_.dim_wvec}); // unk word
  }

  const auto& label_map() const { return label_map_; }

  std::vector<WeightPtr<Real>> weights() const {
    return model_->weights<Real>();
  }

  void save(const std::string& fname, size_t precision = 6) {
    std::ofstream out(fname);
    GINN_ASSERT(out);
    out << std::setprecision(precision);

    out << params_.layers << "\n"
        << params_.dim << "\n"
        << params_.dim_wvec << "\n"
        << params_.dim_char_lstm << "\n"
        << params_.dim_cvec << "\n"
        << params_.drop_p << "\n"
        << params_.word_drop_p << "\n"
        << int(params_.fixed_wvecs) << "\n"
        << params_.dim_y << "\n";

    for (const std::string& y : label_map_.keys()) { out << y << " "; }
    out << "\n";
    if (params_.dim_cvec > 0) {
      for (char c : char_table_->table.keys()) { out << c << " "; }
      out << "\n";
    }

    out << word_table_->table.size() << "\n";
    for (const std::string& w : word_table_->table.keys()) { out << w << "\n"; }

    for (auto w : weights()) { w->value().save(out); }
  };

  void load(const std::string& fname) {
    std::ifstream in(fname);
    GINN_ASSERT(in);
    std::string line;

    params_.layers = getline<size_t>(in);
    params_.dim = getline<Size>(in);
    params_.dim_wvec = getline<Size>(in);
    params_.dim_char_lstm = getline<Size>(in);
    params_.dim_cvec = getline<Size>(in);
    params_.drop_p = getline<Real>(in);
    params_.word_drop_p = getline<Real>(in);
    params_.fixed_wvecs = getline<unsigned>(in);
    params_.dim_y = getline<Size>(in);

    init({}, {});

    label_map_.clear();
    std::getline(in, line);
    for (std::string y : split(line)) { label_map_.insert(y); }

    if (params_.dim_cvec > 0) {
      std::vector<char> chars;
      std::getline(in, line);
      for (size_t i = 0; i < line.size(); i++)
        if (i % 2 == 0) { chars.push_back(line[i]); }
      for (auto c : chars) {
        char_table_->table.insert(c, Weight(*dev_, {params_.dim_cvec}));
      }
    }

    size_t words = getline<size_t>(in);
    for (size_t i = 0; i < words; i++) {
      std::string w = getline<std::string>(in);
      word_table_->table.insert(w, Weight(*dev_, {params_.dim_wvec}));
    }

    for (auto w : weights()) { w->value().load(in); }
  };

  NodePtrs<Real> score_words(const std::vector<std::string>& x, Mode mode) {
    model_->set_mode(mode);
    return model_->run(x);
  }
};

} // namespace ginn

#endif