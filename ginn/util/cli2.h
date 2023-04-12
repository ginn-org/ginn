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

// Tiny command line parsing utilities

#ifndef GINN_UTIL_CLI_H
#define GINN_UTIL_CLI_H

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <tblr/tblr.h>

#include "fmt.h"
#include "sto.h"
#include "tty.h"
#include "util.h"

namespace ginn {

class Args;
template <typename T>
class Arg;

namespace internal {

template <typename Formatter>
std::string form(Formatter f, const std::string& s) {
  if constexpr (std::is_same_v<Formatter, fmt::terminal_color>) {
    return fmt::format(fmt::fg(f), "{}", s);
  } else {
    return fmt::format(f, "{}", s);
  }
}

class ArgBase {
 public:
  virtual ~ArgBase() {}
  ArgBase() = default;

  struct DefinitionError : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  static void ensure(bool predicate, const std::string& msg) {
    if (not predicate) { throw DefinitionError(msg.c_str()); }
  }

 protected:
  using Color = fmt::terminal_color;
  using Emphasis = fmt::emphasis;

  std::string color(Color c, const std::string& s) const {
    return color_ ? form(c, s) : s;
  }
  std::string emphasis(Emphasis e, const std::string& s) const {
    return color_ ? form(e, s) : s;
  }

  std::string help_; // description of the option in the helpstr
  std::string short_name_, long_name_; // names of the option
  std::string meta_;                   // value name in the helpstr
  bool required_ = false;              // required vs optional

  bool parsed_ = false; // is this already parsed
  bool color_ = false;

  void parse(const std::string& value) {
    parse_(value);
    parsed_ = true;
  }

  virtual std::string value() const = 0; // value stored in var, as a string
  virtual std::string help() const { return help_; }
  std::string names() const {
    if (short_name_.empty() and long_name_.empty()) { return ""; }
    if (short_name_.empty()) { return color(Color::green, "--" + long_name_); }
    if (long_name_.empty()) { return color(Color::cyan, "-" + short_name_); }
    return color(Color::cyan, "-" + short_name_) + ", " +
           color(Color::green, "--" + long_name_);
  }
  std::string usage() const {
    std::string s = names();
    if (not is_flag()) {
      if (not s.empty()) { s += " "; }
      std::string meta = color(Color::blue, "<" + meta_ + ">");
      s += meta;
      if (is_nary()) {
        s += emphasis(Emphasis::faint, " [" + meta) +
             emphasis(Emphasis::faint, "... ]");
      }
    }
    return s;
  }

  virtual bool is_flag() const { return false; }
  virtual bool is_nary() const { return false; }
  virtual std::unique_ptr<ArgBase> clone() const = 0;
  virtual void parse_(const std::string& value) = 0;

  friend class ginn::Args;
};

template <typename T>
class Validator {
 private:
  bool throw_;    // whether to warn vs throw
  bool is_range_; // is this a range (interval) or a set
  std::vector<T> choices_;

  std::string str() const {
    if (is_range_) { return fmt::format("[{}, {}]", choices_[0], choices_[1]); }
    return fmt::format("{{{}}}", fmt::join(choices_, ", "));
  }

  void warn_or_throw(const std::string& name, const T& val) const {
    std::string adj = throw_ ? "required" : "suggested";
    std::string noun = is_range_ ? "range" : "set";
    std::string msg = fmt::format(
        "Value {} for {} is not in {} {}: {}", val, name, adj, noun, str());
    if (throw_) { throw std::runtime_error(msg); }
    std::cerr << "Warning: " + msg << std::endl;
  }

  Validator(bool throws, bool is_range, std::vector<T> choices)
      : throw_(throws), is_range_(is_range), choices_(std::move(choices)) {
    ArgBase::ensure(not std::is_same_v<T, bool>,
                    "Ranges and choices are disabled for booleans!");
    if (is_range_) {
      assert(choices_.size() == 2);
      ArgBase::ensure(choices_[0] <= choices_[1],
                      "left <= right should hold for ranges!");
    }
  }

 public:
  void check(const std::string& name, const T& x) const {
    bool invalid = is_range_ ? not(choices_[0] <= x and x <= choices_[1])
                             : std::find(choices_.begin(), choices_.end(), x) ==
                                   choices_.end();
    if (invalid) { warn_or_throw(name, x); }
  }

  template <typename Dest>
  friend class TypedArgBase;
};

// Helper to determine inner type if std::vector
template <typename T>
struct MaybeValueType {
  using type = T;
};

template <typename T>
struct MaybeValueType<std::vector<T>> {
  using type = T;
};

// Define default meta strings for various types
template <typename T>
inline std::string default_meta() {
  if constexpr (std::is_same_v<T, bool>) { return "true|false"; }
  if constexpr (std::is_floating_point_v<T>) { return "x"; }
  if constexpr (std::is_integral_v<T>) { return "n"; } // WARN: covers char
  return "";
}

template <typename T>
class TypedArgBase : public ArgBase {
 protected:
  using value_t = typename MaybeValueType<T>::type;

  T& dest_;
  std::vector<Validator<value_t>> validators_;

  TypedArgBase(T& dest) : dest_(dest) { this->meta_ = default_meta<T>(); }
  std::string help() const override {
    std::string paren;
    if (not required_) { paren = color(Color::blue, "default: " + value()); }
    for (auto& v : validators_) {
      if (not paren.empty()) { paren += ", "; }
      auto c = v.throw_ ? Color::red : Color::yellow;
      std::string adj = v.throw_ ? "required" : "suggested";
      paren += color(c, fmt::format("{} in {}", adj, v.str()));
    }
    if (not paren.empty()) { paren = " (" + paren + ")"; }

    return help_ + paren;
  }

 public:
  auto& name(const std::string& s) {
    auto namesv = split(s, ',');
    ensure(not namesv.empty(), "Option name cannot be empty!");
    ensure(namesv.size() <= 2,
           "Option names can be one short and one long at most! "
           "E.g. \"o,option\" or \"o\" or \"option\".");
    if (namesv.size() == 2) { // ether specify "o,option",
      short_name_ = namesv[0];
      long_name_ = namesv[1];
      ensure(short_name_.size() == 1 and long_name_.size() > 1,
             "Multiple form option names should be first short then long! "
             "E.g. \"o,option\".");
    } else { // or "o" only or "option" only.
      // TODO: using `long_name_` this way is confusing in terms of var name
      long_name_ = namesv[0];
    }
    return *this;
  }
  auto& help(const std::string& s) {
    help_ = s;
    return *this;
  }
  auto& meta(const std::string& s) {
    ensure(not is_flag(), "No meta to display for flags!");
    meta_ = s;
    return *this;
  }
  auto& required(bool r = true) {
    ensure(not is_flag(), "Flags are always optional!");
    required_ = r;
    return *this;
  }
  auto& suggest_range(value_t left, value_t right) {
    validators_.push_back(Validator<value_t>(false, true, {left, right}));
    return *this;
  }
  auto& require_range(value_t left, value_t right) {
    validators_.push_back(Validator<value_t>(true, true, {left, right}));
    return *this;
  }
  template <typename Container>
  auto& suggest_choices(const Container& vals) {
    validators_.push_back(Validator<value_t>(
        false, false, std::vector<value_t>(vals.begin(), vals.end())));
    return *this;
  }
  auto& suggest_choices(std::initializer_list<value_t> vals) {
    return suggest_choices(std::vector<value_t>(vals));
  }
  template <typename Container>
  auto& require_choices(const Container& vals) {
    // TODO: nvcc doesn't like leaving out template arg "value_t" in the
    // following. why?
    validators_.push_back(Validator<value_t>(
        true, false, std::vector<value_t>(vals.begin(), vals.end())));
    return *this;
  }
  auto& require_choices(std::initializer_list<value_t> vals) {
    return require_choices(std::vector<value_t>(vals));
  }

  friend class ginn::Args;
};

} // namespace internal

template <typename T>
class Arg : public internal::TypedArgBase<T> {
 public:
  Arg(T& dest) : internal::TypedArgBase<T>(dest) {}

 protected:
  std::string value() const override { return fmt::to_string(this->dest_); }
  std::unique_ptr<internal::ArgBase> clone() const override {
    return std::make_unique<Arg<T>>(*this);
  }

  void parse_(const std::string& value) override {
    this->dest_ = sto::sto<T>(value);
    for (auto& v : this->validators_) { v.check(this->names(), this->dest_); }
  }
};

template <>
class Arg<bool> : public internal::TypedArgBase<bool> {
 public:
  Arg(bool& dest) : internal::TypedArgBase<bool>(dest) {}

  std::string value() const override { return (dest_ ? "true" : "false"); }
  std::unique_ptr<internal::ArgBase> clone() const override {
    return std::make_unique<Arg<bool>>(*this);
  }

 protected:
  void parse_(const std::string& value) override {
    dest_ = sto::sto<bool>(value);
  }
};

class Flag : public Arg<bool> {
 public:
  using Arg<bool>::Arg;

 protected:
  bool is_flag() const override { return true; }
  std::string help() const override {
    return help_ + " (" + color(Color::blue, "flag") + ")";
  }
  std::unique_ptr<internal::ArgBase> clone() const override {
    return std::make_unique<Flag>(*this);
  }

  void parse_(const std::string&) override { dest_ = true; }
};

template <typename T>
class Arg<std::vector<T>> : public internal::TypedArgBase<std::vector<T>> {
 public:
  Arg(std::vector<T>& dest) : internal::TypedArgBase<std::vector<T>>(dest) {}

 protected:
  std::string value() const override {
    return fmt::format("{}", fmt::join(this->dest_, ", "));
  }
  std::unique_ptr<internal::ArgBase> clone() const override {
    return std::make_unique<Arg<std::vector<T>>>(*this);
  }

  bool is_nary() const override { return true; }

  bool first_parse_ = true;

  void parse_(const std::string& value) override {
    T arg = sto::sto<T>(value);
    for (auto& v : this->validators_) { v.check(this->names(), arg); }
    if (first_parse_) {
      this->dest_.clear(); // clear default values
      first_parse_ = false;
    }
    this->dest_.push_back(arg);
  }
};

class Args {
 public:
  enum class Color { Auto, Always, Never };
  struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };
  using ArgBase = internal::ArgBase;

  Args(std::string name = "", Color color = Color::Auto)
      : program_name_(std::move(name)), color_(color) {
    add_help(true);
  }
  Args(Color color, std::string name = "") : Args(name, color) {}

 private:
  std::vector<std::unique_ptr<ArgBase>> positional_args_, named_args_;
  std::unordered_map<std::string, size_t> name2i_;
  bool has_help_ = false;
  std::string program_name_;
  Color color_{Color::Auto};

  static void ensure(bool predicate, const std::string& msg) {
    if (not predicate) { throw ParseError(msg.c_str()); }
  }

  static std::optional<std::string> parse_name(const std::string& value) {
    std::string name;
    if (value.size() >= 2 and value[0] == '-' and value[1] == '-') {
      name = value.substr(2, value.size());
      ensure(not name.empty(), "Prefix `--` not followed by an option!");
      return name;
    }
    if (value.size() >= 1 and value[0] == '-') {
      name = value.substr(1, value.size());
      ensure(not name.empty(), "Prefix `-` not followed by an option!");
      ensure(name.size() == 1,
             "Options prefixed by `-` have to be short names! " +
                 fmt::format("Did you mean `--{}`?", name));
      return name;
    }
    return {};
  }

  std::optional<std::string> parse_known_name(const std::string& value) {
    auto n = parse_name(value);
    if (n and name2i_.find(*n) != name2i_.end()) { return n; }
    return {};
  }

 public:
  void parse(const std::vector<std::string>& argv) {
    using namespace ginn::literals;
    size_t i = 0, positional_i = 0;
    std::string help = helpstr(); // get help string before parsing anything

    size_t argc = argv.size();
    while (i < argc) {
      if (auto name = parse_name(argv[i])) { // a named argument (option)
        if (has_help_ and (name == "?" or name == "h" or name == "help")) {
          std::cout << help << std::flush;
          exit(0);
        }
        ensure(name2i_.find(*name) != name2i_.end(),
               "Unexpected option: " + *name);
        auto& opt = *named_args_.at(name2i_.at(*name));
        ensure(not opt.parsed_, ("Option `{}` is multiply given!"_f, *name));
        if (opt.is_flag()) {
          opt.parse("");
          i++;
        } else if (opt.is_nary()) {
          size_t j = i + 1;
          // keep parsing until something looks like an option name
          for (; j < argc and not parse_known_name(argv[j]); j++) {
            opt.parse(argv[j]);
          }
          i = j;
        } else {
          ensure((i + 1) < argc, ("Option `{}` is missing value!"_f, *name));
          opt.parse(argv[i + 1]);
          i += 2;
        }
      } else { // a positional argument
        ensure(positional_i < positional_args_.size(),
               "Unexpected positional argument: " + argv[i]);
        auto& arg = *positional_args_[positional_i];
        if (arg.is_nary()) {
          size_t j = positional_i;
          // keep parsing until something looks like an option name
          for (; j < argc and not parse_known_name(argv[j]); j++) {
            arg.parse(argv[j]);
          }
          positional_i = i = j;
        } else {
          arg.parse(argv[i]);
          i++;
          positional_i++;
        }
      }
    }

    // check if all required args are parsed
    for (auto& arg : positional_args_) {
      ensure(
          not arg->required_ or arg->parsed_,
          ("Required positional argument {} is not provided!"_f, arg->usage()));
    }
    for (auto& arg : named_args_) {
      if (arg->required_) {
        ensure(arg->parsed_,
               ("Required option `{}` is not provided!"_f, arg->usage()));
      }
    }
  }

  void parse(int argc, char** argv) {
    if (program_name_.empty()) { program_name_ = argv[0]; }
    parse(std::vector<std::string>(argv + 1, argv + argc));
  }

  void add_name(const std::string& name) {
    ArgBase::ensure(name2i_.find(name) == name2i_.end(),
                    "Option `" + name + "` is multiply defined!");
    name2i_[name] = named_args_.size();
  }

  auto& add(const internal::ArgBase& arg) {
    if (arg.is_flag()) { // flag
      auto& dest = dynamic_cast<const Flag&>(arg).dest_;
      ArgBase::ensure(not dest,
                      "Optional boolean flags need to default to false, "
                      "since the action is `store true`!");
      ArgBase::ensure(not arg.long_name_.empty(),
                      "Flag is missing option name!");
    }
    if (arg.long_name_.empty()) { // positional arg
      positional_args_.push_back(arg.clone());
    } else { // named arg / option
      add_name(arg.long_name_);
      if (not arg.short_name_.empty()) { add_name(arg.short_name_); }
      named_args_.push_back(arg.clone());
    }
    return *this;
  }

  auto& operator()(const internal::ArgBase& arg) { return add(arg); }

  // add helpstring flag (-?, -h, --help)
  void add_help(bool has_help) {
    const static std::vector<std::string> names({"?", "h", "help"});
    if (has_help) {
      for (auto& name : names) { add_name(name); }
    }
    has_help_ = has_help;
  };

  std::string helpstr() const {
    using namespace tblr;
    auto table = []() {
      return Table().widths({0, 50}).multiline(Space).layout(indented_list());
    };

    // set coloring
    bool color = color_ == Color::Always or
                 (color_ == Color::Auto and internal::istty());
    for (auto& arg : positional_args_) { arg->color_ = color; }
    for (auto& arg : named_args_) { arg->color_ = color; }

    std::stringstream ss;

    ss << "Usage:\n  " << (program_name_.empty() ? "program" : program_name_);
    for (auto& arg : positional_args_) { ss << " " << arg->usage(); }
    ss << " options\n";

    if (not positional_args_.empty() or not named_args_.empty()) {
      ss << "\nwhere ";
    }

    auto add_rows = [](Table& t, auto& args) {
      for (auto& arg : args) { t << arg->usage() << arg->help() << endr; }
    };

    if (not positional_args_.empty()) {
      auto t = table();
      add_rows(t, positional_args_);
      ss << "positional arguments are:\n" << t << "\n";
    }

    std::vector<ArgBase*> required_opts, optional_opts;
    for (auto& arg : named_args_) {
      (arg->required_ ? required_opts : optional_opts).push_back(&*arg);
    }

    if (not required_opts.empty()) {
      auto t = table();
      add_rows(t, required_opts);
      ss << "required options are:\n" << t << "\n";
    }

    if (not optional_opts.empty() or has_help_) {
      auto t = table();
      if (has_help_) {
        using C = ArgBase::Color;
        using internal::form;
        std::string usage =
            color ? (form(C::cyan, "-?") + ", " + form(C::cyan, "-h") + ", " +
                     form(C::green, "--help"))
                  : "-?, -h, --help";
        t << usage << "print this message and exit" << endr;
      }
      add_rows(t, optional_opts);
      ss << "optional options are:\n" << t << "\n";
    }
    return ss.str();
  }
};

} // namespace ginn

#endif
