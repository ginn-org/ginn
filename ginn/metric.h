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

#ifndef GINN_METRIC_H
#define GINN_METRIC_H

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ginn/def.h>
#include <ginn/except.h>

namespace ginn {
namespace metric {

template <typename LabelType>
class Metric {
 private:
  std::mutex access_;

  virtual void
  add_(const LabelType& pred, const LabelType& tru, double weight) = 0;

 public:
  void add(const LabelType& pred, const LabelType& tru, double weight = 1.) {
    std::lock_guard l(access_);
    add_(pred, tru, weight);
  }
  // Batched version //TODO: Weighted version of batched
  template <typename Preds, typename Trus>
  void batched_add(const Preds& preds, const Trus& trus) {
    GINN_ASSERT((size_t)preds.size() == (size_t)trus.size());
    auto p = preds.begin();
    auto t = trus.begin();
    while (p != preds.end()) { add(*p++, *t++); }
  }
  virtual Real eval() const = 0;
  virtual void clear() = 0;
};

template <typename T>
class Accuracy : public Metric<T> {
 public:
  double match = 0, count = 0;

  void add_(const T& a, const T& b, double weight = 1.) override {
    match += weight * (a == b);
    count += weight;
  }
  Real eval() const override { return Real(match / count); }
  void clear() override { count = match = 0; }
};

template <typename T>
class MSE : public Metric<T> {
 private:
  long double se = 0.;
  long double count = 0;

 public:
  void add_(const T& a, const T& b, double weight = 1.) override {
    long double d = a - b;
    se += d * d * weight;
    count += weight;
  }
  Real eval() const override { return Real(se / count); }
  void clear() override { se = count = 0; }
};

struct F1Value {
  Real precision, recall, f;
};

template <typename Label>
class F1 : public Metric<Label> {
 public:
  const Label all, macro;
  std::unordered_set<Label> tags{all};
  std::unordered_map<Label, double> pred_counts, tru_counts, match_counts;

  F1(const Label& a_all, const Label& a_macro) : all(a_all), macro(a_macro) {}

  void add_(const Label& pred, const Label& tru, double weight = 1.) {
    tags.insert(pred);
    tags.insert(tru);
    pred_counts[pred] += weight;
    pred_counts[all] += weight;
    tru_counts[tru] += weight;
    tru_counts[all] += weight;
    if (pred == tru) {
      match_counts[all] += weight;
      match_counts[pred] += weight;
    }
  }

  std::map<Label, F1Value> eval_all() const {
    std::map<Label, F1Value> scores;

    for (const auto& tag : tags) {
      scores[tag].precision =
          (pred_counts[tag] == 0 ? 1.
                                 : Real(match_counts[tag]) / pred_counts[tag]);
      scores[tag].recall =
          (tru_counts[tag] == 0 ? 1.
                                : Real(match_counts[tag]) / tru_counts[tag]);
      scores[tag].f = (2 * scores[tag].precision * scores[tag].recall) /
                      (scores[tag].precision + scores[tag].recall);
      if (scores[tag].precision == 0. and scores[tag].recall == 0.) {
        scores[tag].f = 0.;
      }

      if (tag != all) {
        scores[macro].precision += scores[tag].precision;
        scores[macro].recall += scores[tag].recall;
        scores[macro].f += scores[tag].f;
      }
    }

    scores[macro].precision /= (tags.size() - 1);
    scores[macro].recall /= (tags.size() - 1);
    scores[macro].f /= (tags.size() - 1);

    return scores;
  }

  Real eval() const override {
    auto scores = eval_all();
    return scores[all].f;
  }

  void clear() override {
    pred_counts.clear();
    tru_counts.clear();
    match_counts.clear();
  }
};

class SpanF1 : public Metric<std::vector<std::string>> {
 public:
  const std::string all = "all";
  std::unordered_set<std::string> tags{all};
  std::unordered_map<std::string, size_t> pred_counts, tru_counts, match_counts;

  typedef std::pair<Real, Real> Span;

  // TODO: weight is missing
  void add_(const std::vector<std::string>& pred,
            const std::vector<std::string>& tru,
            double /*weight*/ = 1.) override {

    auto extract_spans = [this](const std::vector<std::string>& seq) {
      std::unordered_map<std::string, std::vector<Span>> spans;

      size_t left, right;
      for (left = 0; left < seq.size(); left++) {
        if (seq[left][0] == 'B' or seq[left][0] == 'I') {
          std::string tag = seq[left].substr(2, seq[left].size());
          tags.insert(tag);

          for (right = left + 1; right <= seq.size(); right++) {
            if (right == seq.size() or
                (seq[right] != "I_" + tag and seq[right] != "I-" + tag))
              break;
          }
          spans[tag].push_back(Span(left, right));
          left = right - 1;
        }
      }
      return spans;
    };

    auto pred_spans = extract_spans(pred);
    auto tru_spans = extract_spans(tru);

    for (const auto& tag : tags) {
      pred_counts[tag] += pred_spans[tag].size();
      pred_counts[all] += pred_spans[tag].size();
      tru_counts[tag] += tru_spans[tag].size();
      tru_counts[all] += tru_spans[tag].size();
      for (const auto& pred_span : pred_spans[tag]) { // quadratic in sentence
        for (const auto& tru_span : tru_spans[tag]) { // should be mostly okay
          if (pred_span == tru_span) {
            match_counts[tag]++;
            match_counts[all]++;
          }
        }
      }
    }
  }

  std::unordered_map<std::string, F1Value> eval_all() const {
    std::unordered_map<std::string, F1Value> scores;

    auto div = [](Real a, Real b, Real def) { return b != 0. ? a / b : def; };

    for (const auto& tag : tags) {
      auto& [precision, recall, f] = scores[tag];
      precision = div(match_counts.at(tag), pred_counts.at(tag), 1.);
      recall = div(match_counts.at(tag), tru_counts.at(tag), 1.);
      f = div(2 * precision * recall, precision + recall, 0.);
    }

    return scores;
  }

  Real eval() const override {
    auto scores = eval_all();
    return scores[all].f;
  }

  void clear() override {
    pred_counts.clear();
    tru_counts.clear();
    match_counts.clear();
  }
};

} // namespace metric
} // namespace ginn

#endif
