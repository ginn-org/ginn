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

// Test <ginn/util/*> in this file

#define CATCH_CONFIG_MAIN // so that Catch is responsible for main()

#include <catch.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include <ginn/node/data.h>
#include <ginn/util/amax.h>
#include <ginn/util/cli2.h>
#include <ginn/util/indexmap.h>
#include <ginn/util/tensorio.h>
#include <ginn/util/timer.h>
#include <ginn/util/tree.h>

using namespace ginn;

template <typename T>
T to(const std::string& s) {
  T thing;
  std::stringstream(s) >> thing;
  return thing;
}

template <>
bool to(const std::string& s) {
  static std::unordered_map<std::string, bool> map{
      {"True", 1}, {"true", 1}, {"1", 1}, {"False", 0}, {"false", 0}, {"0", 0}};
  return map.at(s);
}

template <typename T>
T to(const std::optional<std::string>& s, T def) {
  if (s) { return to<T>(*s); }
  return def;
}

inline auto operator"" _m(const char* s, size_t) {
  return Catch::Message(std::string(s));
}

TEST_CASE("Basic args", "[args]") {
  float mana;
  int energy = -3;
  std::string spell = "teleport";
  bool focus = false;

  Args args("cast_spell", Args::Color::Never);
  args.add(Arg(mana)
               .name("m,mana")
               .meta("x")
               .help("amount of mana needed for the spell")
               .required());
  args.add(
      Arg(energy).name("e,energy").meta("n").help("energy cost of the spell"));
  args.add(Arg(spell).name("s,spell").meta("name").help("name of the spell"));
  args.add(Arg(focus)
               .name("f,focus")
               .meta("true|false")
               .help("whether focusing is needed for the spell"));

  SECTION("Definition errors") {
    bool dummy = false;
    SECTION("Too many options") {
      std::string val = GENERATE("d,du,dummy", "d,u,du,dum,dummy");
      CHECK_THROWS_MATCHES(
          args.add(Arg(dummy).name(val)),
          Arg<bool>::DefinitionError,
          "Option names can be one short and one long at most! "
          "E.g. \"o,option\" or \"o\" or \"option\"."_m);
    }
    SECTION("Repeated name") {
      std::string val =
          GENERATE("m", "mana", "f", "focus", "e", "energy", "s", "spell");
      CHECK_THROWS_MATCHES(
          args.add(Arg(dummy).name(val)),
          Arg<bool>::DefinitionError,
          Catch::Message("Option `" + val + "` is multiply defined!"));
    }
    SECTION("Repeated name after help") {
      // args.add_help();
      std::string val = GENERATE("h", "?", "help");
      CHECK_THROWS_MATCHES(
          args.add(Arg(dummy).name(val)),
          Arg<bool>::DefinitionError,
          Catch::Message("Option `" + val + "` is multiply defined!"));
    }
    // SECTION("Repeated name before help") {
    //  std::string val = GENERATE("h", "?", "help");
    //  args.add(Arg(dummy).name(val));
    //  CHECK_THROWS_MATCHES(
    //      args.add_help(),
    //      Arg<bool>::DefinitionError,
    //      Catch::Message("Option `" + val + "` is multiply defined!"));
    //}
    SECTION("Flag with true default") {
      dummy = true;
      CHECK_THROWS_MATCHES(args.add(Flag(dummy)),
                           Arg<bool>::DefinitionError,
                           "Optional boolean flags need to default to false, "
                           "since the action is `store true`!"_m);
    }
    SECTION("Empty flag name") {
      CHECK_THROWS_MATCHES(args.add(Flag(dummy)),
                           Arg<bool>::DefinitionError,
                           "Flag is missing option name!"_m);
    }
    SECTION("Flag with meta") {
      CHECK_THROWS_MATCHES(args.add(Flag(dummy).name("dummy").meta("dum")),
                           Arg<bool>::DefinitionError,
                           "No meta to display for flags!"_m);
    }
    SECTION("Required flag") {
      CHECK_THROWS_MATCHES(args.add(Flag(dummy).name("dummy").required()),
                           Arg<bool>::DefinitionError,
                           "Flags are always optional!"_m);
    }
  }

  SECTION("Parse errors") {
    SECTION("Missing required") {
      CHECK_THROWS_MATCHES(
          args.parse({}),
          Args::ParseError,
          "Required option `-m, --mana <x>` is not provided!"_m);
    }
    SECTION("Unexpected option") {
      CHECK_THROWS_MATCHES(
          args.parse({"-n", "42"}), Args::ParseError, "Unexpected option: n"_m);
    }
    SECTION("Unexpected positional") {
      CHECK_THROWS_MATCHES(args.parse({"n", "42"}),
                           Args::ParseError,
                           "Unexpected positional argument: n"_m);
    }
    SECTION("Incorrect dash") {
      CHECK_THROWS_MATCHES(
          args.parse({"-mana", "3"}),
          Args::ParseError,
          "Options prefixed by `-` have to be short names! Did you mean `--mana`?"_m);
    }
    SECTION("Missing value for option") {
      CHECK_THROWS_MATCHES(args.parse({"--mana"}),
                           Args::ParseError,
                           "Option `mana` is missing value!"_m);
    }
    SECTION("Broken values") {
      SECTION("float") {
        std::string val = GENERATE("a", "3.0a", "x10", "...", "1.2 2.3");
        CHECK_THROWS(args.parse({"--mana", val}));
      }
      SECTION("int") {
        std::string val = GENERATE("a", "x10", "...", "1e2", "3 -5");
        CHECK_THROWS(args.parse({"-m", "1.0", "-e", val}));
      }
      SECTION("bool") {
        std::string val = GENERATE("a", "00", "yes", "-1", "true false");
        CHECK_THROWS(args.parse({"-m", "1.0", "-f", val}));
      }
    }
  }

  SECTION("Parsing") {
    CHECK(energy == -3);
    CHECK(spell == "teleport");
    CHECK(focus == false);

    using OptStr = std::optional<std::string>;

    std::string mana_opt = GENERATE("-m", "--mana");
    auto mana_str = GENERATE(as<OptStr>(), "-3", "3.0", "1e2", ".4");
    std::string energy_opt = GENERATE("-e", "--energy");
    auto energy_str = GENERATE(as<OptStr>(), "-2", "3", "+4", OptStr{});
    std::string spell_opt = GENERATE("-s", "--spell");
    auto spell_str = GENERATE(as<OptStr>(), "fireball", "revive", OptStr{});
    std::string focus_opt = GENERATE("-f", "--focus");
    auto focus_str = GENERATE(
        as<OptStr>(), "True", "true", "1", "False", "false", "0", OptStr{});

    std::vector<std::string> cli_args;
    if (mana_str) { cli_args.insert(cli_args.end(), {mana_opt, *mana_str}); }
    if (energy_str) {
      cli_args.insert(cli_args.end(), {energy_opt, *energy_str});
    }
    if (spell_str) { cli_args.insert(cli_args.end(), {spell_opt, *spell_str}); }
    if (focus_str) { cli_args.insert(cli_args.end(), {focus_opt, *focus_str}); }

    args.parse(cli_args);

    CHECK(mana == to<float>(mana_str, 4.0));
    CHECK(energy == to<int>(energy_str, -3));
    CHECK(spell == to<std::string>(spell_str, "teleport"));
    CHECK(focus == to<bool>(focus_str, false));
  }

  SECTION("Help") {
    std::string expected =
        "Usage:\n"
        "  cast_spell options\n"
        "\n"
        "where required options are:\n"
        "  -m, --mana <x>   amount of mana needed for the spell               \n"
        "\n"
        "optional options are:\n"
        "  -?, -h, --help             print this message and exit                       \n"
        "  -e, --energy <n>           energy cost of the spell (default: -3)            \n"
        "  -s, --spell <name>         name of the spell (default: teleport)             \n"
        "  -f, --focus <true|false>   whether focusing is needed for the spell          \n"
        "                             (default: false)                                  \n"
        "\n";
    CHECK(args.helpstr() == expected);
  }
}

template <typename T>
std::vector<T> to_v(const std::vector<std::string>& v,
                    const std::vector<T>& def) {
  if (v.empty()) { return def; }
  std::vector<T> rv(v.size());
  std::transform(v.begin(), v.end(), rv.begin(), [&](const std::string& s) {
    return to<T>(s);
  });
  return rv;
}

TEST_CASE("N-ary options", "[args]") {
  std::vector<double> mana;
  std::vector<unsigned> energy{2};
  std::vector<std::string> spell{"teleport"};
  std::vector<bool> focus{false};

  Args args(Args::Color::Never);
  args.add(Arg(mana)
               .name("m,mana")
               .meta("x")
               .help("amount of mana needed for the spell")
               .required());
  args.add(
      Arg(energy).name("e,energy").meta("n").help("energy cost of the spell"));
  args.add(Arg(spell).name("s,spell").meta("name").help("name of the spell"));
  args.add(Arg(focus)
               .name("f,focus")
               .meta("true|false")
               .help("whether focusing is needed for the spell"));

  using Strs = std::vector<std::string>;

  SECTION("Parsing") {
    CHECK(energy == std::vector<unsigned>{2});
    CHECK(spell == std::vector<std::string>{"teleport"});
    CHECK(focus == std::vector<bool>{false});

    std::string mana_opt = GENERATE("-m", "--mana");
    auto mana_str = GENERATE(Strs{"-3"},
                             Strs{"-3", "3.0"},
                             Strs{"-3", "3.0", "1e2"},
                             Strs{"-3", "3.0", "1e2", ".4"});
    std::string energy_opt = GENERATE("-e", "--energy");
    auto energy_str =
        GENERATE(Strs{}, Strs{"2"}, Strs{"2", "3"}, Strs{"2", "3", "+4"});
    std::string spell_opt = GENERATE("-s", "--spell");
    auto spell_str =
        GENERATE(Strs{}, Strs{"fireball"}, Strs{"fireball", "revive"});
    std::string focus_opt = GENERATE("-f", "--focus");
    auto focus_str = GENERATE(
        Strs{}, Strs{"true"}, Strs{"true", "0"}, Strs{"true", "0", "False"});

    std::vector<std::string> cli_args;
    auto maybe_add = [&](const auto& strs, const auto& opt) {
      if (not strs.empty()) {
        cli_args.push_back(opt);
        cli_args.insert(cli_args.end(), strs.begin(), strs.end());
      }
    };
    maybe_add(mana_str, mana_opt);
    maybe_add(energy_str, energy_opt);
    maybe_add(spell_str, spell_opt);
    maybe_add(focus_str, focus_opt);

    args.parse(cli_args);

    CHECK(mana == to_v<double>(mana_str, {}));
    CHECK(energy == to_v<unsigned>(energy_str, {2}));
    CHECK(spell == to_v<std::string>(spell_str, {"teleport"}));
  }

  SECTION("Parsing empty") {
    CHECK(energy == std::vector<unsigned>{2});
    CHECK(spell == std::vector<std::string>{"teleport"});
    CHECK(focus == std::vector<bool>{false});

    // TODO: what is the right behavior?
  }
}

TEST_CASE("Mixed args", "[args]") {
  using Strs = std::vector<std::string>;
  std::string a, b, c;
  Strs d;

  Args args(Args::Color::Never);

  args.add(Arg(a).required());
  args.add(Arg(b).required(false));
  args.add(Arg(c).name("c"));
  args.add(Arg(d).name("ds"));

  SECTION("All") {
    SECTION("Positionals first") {
      args.parse({"aaa", "bbb", "-c", "ccc", "--ds", "d", "dd", "ddd"});
    }
    SECTION("Positionals second") {
      args.parse({"-c", "ccc", "aaa", "bbb", "--ds", "d", "dd", "ddd"});
    }
    SECTION("Positionals mixed") {
      args.parse({"aaa", "-c", "ccc", "bbb", "--ds", "d", "dd", "ddd"});
    }
    SECTION("Option ends nary") {
      args.parse({"aaa", "bbb", "--ds", "d", "dd", "ddd", "-c", "ccc"});
    }
    SECTION("Option ends nary 2") {
      args.parse({"aaa", "--ds", "d", "dd", "ddd", "-c", "ccc", "bbb"});
    }
    SECTION("Option ends nary 3") {
      args.parse({"--ds", "d", "dd", "ddd", "-c", "ccc", "aaa", "bbb"});
    }
    CHECK(b == "bbb");
  }
  SECTION("One omitted") {
    SECTION("Positional first") {
      args.parse({"aaa", "-c", "ccc", "--ds", "d", "dd", "ddd"});
    }
    SECTION("Positional second") {
      args.parse({"-c", "ccc", "aaa", "--ds", "d", "dd", "ddd"});
    }
    SECTION("Option ends nary") {
      args.parse({"aaa", "--ds", "d", "dd", "ddd", "-c", "ccc"});
    }
    SECTION("Option ends nary 2") {
      args.parse({"--ds", "d", "dd", "ddd", "-c", "ccc", "aaa"});
    }
    CHECK(b == "");
  }
  CHECK(a == "aaa");
  CHECK(c == "ccc");
  CHECK(d == Strs{"d", "dd", "ddd"});
}

// RAII utility class for tmp files
class TempFile {
 private:
  std::filesystem::path path_;

  static std::string randname() {
    static std::mt19937 rng(time(0));
    std::stringstream ss;
    ss << std::hex << rng();
    return ss.str();
  }

 public:
  TempFile() {
    auto tmpdir = std::filesystem::temp_directory_path();
    path_ = tmpdir / randname();
    std::ofstream out(path_);
  }
  ~TempFile() { std::filesystem::remove(path_); }

  const auto& path() const { return path_; }
};

TEST_CASE("File lines", "[util]") {
  TempFile tmp;
  std::ofstream(tmp.path()) << GENERATE("a\nb\nc\n", "a\nb\nc");
  std::vector<std::string> v;

  SECTION("Loop by lines") {
    for (const auto& line : lines(tmp.path())) { v.push_back(line); }
  }
  SECTION("Read lines") { v = readlines(tmp.path()); }
  CHECK(v == std::vector<std::string>{"a", "b", "c"});
}

TEST_CASE("Filestream lines", "[util]") {
  TempFile tmp;
  std::ofstream(tmp.path()) << GENERATE("a\nb\nc\n", "a\nb\nc");
  std::ifstream in(tmp.path());
  std::vector<std::string> v;

  SECTION("Loop by lines") {
    for (const auto& line : lines(in)) { v.push_back(line); }
  }
  SECTION("Read lines") { v = readlines(in); }
  CHECK(v == std::vector<std::string>{"a", "b", "c"});
}

TEST_CASE("Filestream skip lines", "[util]") {
  TempFile tmp;
  std::ofstream(tmp.path()) << GENERATE("a\nb\nc\n", "a\nb\nc");
  std::ifstream in(tmp.path());
  std::vector<std::string> v;

  std::string _;
  size_t skip_count = GENERATE(1, 2, 3, 4);
  for (size_t i = 0; i < skip_count; i++) { std::getline(in, _); }

  SECTION("Loop by lines") {
    for (const auto& line : lines(in)) { v.push_back(line); }
  }
  SECTION("Read lines") { v = readlines(in); }
  std::vector<std::string> expected{"a", "b", "c"};
  for (size_t i = 0; i < skip_count; i++) {
    if (not expected.empty()) { expected.erase(expected.begin()); }
  }
  CHECK(v == expected);
}

TEST_CASE("Basic trees", "[tree]") {
  using namespace ginn::tree;

  /* String */ {
    auto t = parse<std::string>("(a (b (b 1) (b 2)) (c))");
    std::ostringstream ss;
    print(ss, t);
    // TODO: Space chars after a and b are bothersome :/
    CHECK(ss.str() == "╌╌a \n"
                      "  ├─b \n"
                      "  │ ├─b 1\n"
                      "  │ └─b 2\n"
                      "  └─c\n");
    std::vector<std::string> sorted(t.begin(), t.end());
    CHECK(sorted == std::vector<std::string>{"b 1", "b 2", "b ", "c", "a "});
    auto t2 = clone_empty<int>(t);
    std::ostringstream ss2;
    print(ss2, t2);
    CHECK(ss2.str() == "╌╌0\n"
                       "  ├─0\n"
                       "  │ ├─0\n"
                       "  │ └─0\n"
                       "  └─0\n");
  }
  /* Int */ {
    auto t = parse<int>("(1 (2 (3) (-4)) (5))");
    std::ostringstream ss;
    print(ss, t);
    CHECK(ss.str() == "╌╌1\n"
                      "  ├─2\n"
                      "  │ ├─3\n"
                      "  │ └─-4\n"
                      "  └─5\n");
    std::vector<int> sorted(t.begin(), t.end());
    CHECK(sorted == std::vector<int>{3, -4, 2, 5, 1});
  }
  /* Empty */ {
    auto t = parse<float>("");
    CHECK(t.size() == 0);
    CHECK(std::vector<float>{} == std::vector<float>(t.begin(), t.end()));
  }

  CHECK_THROWS(parse<int>("(1 (2 (3) (-4)) (5)"));
  CHECK_THROWS(parse<int>("1 (2 (3) (-4)) (5)"));
  CHECK_THROWS(parse<int>("((("));
  CHECK_THROWS(parse<int>(")))"));
}

TEST_CASE("Custom reader printer", "[tree]") {
  using Type = std::pair<std::string, float>;
  auto reader = [](std::string_view s) -> Type {
    auto parts = split(std::string(s));
    return {parts.at(0), std::stof(parts.at(1))};
  };

  auto t = tree::parse<Type>("(a 1.1 (b 2.2 (b 3.3) (b 4.4)) (c 5.5 (d 6.6)))",
                             reader);
  CHECK(t.size() == 6);

  auto printer = [](std::ostream& out, const Type& data) {
    out << data.first << ":" << data.second;
  };
  std::ostringstream ss;
  tree::print(ss, t, printer);
  CHECK(ss.str() == "╌╌a:1.1\n"
                    "  ├─b:2.2\n"
                    "  │ ├─b:3.3\n"
                    "  │ └─b:4.4\n"
                    "  └─c:5.5\n"
                    "    └─d:6.6\n");
}

// TODO: This test stack overflows because of shared pointer destructor chain,
//   even though parsing itself is non-recursive. Fix. Note that this will
//   happen for computation graphs too.
// TEST_CASE("Tall tree", "[tree]") {
//  size_t height = 1'000'000;
//  std::string s;
//  for (size_t i = 0; i < height; i++) { s += "(0"; }
//  for (size_t i = 0; i < height; i++) { s += ")"; }
//  std::cout << s.size() << std::endl;
//  {
//    auto t = tree::parse<int>(s);
//    std::cout << "x" << std::endl;
//    CHECK(t.size() == height);
//    std::cout << "y" << std::endl;
//  }
//  std::cout << "z" << std::endl;
//}

TEMPLATE_TEST_CASE("std::vector argmax", "[amax]", int, short, float, double) {
  SECTION("Basic") {
    std::vector<TestType> v{2, -3, 7, 5};
    auto am = argmax(v);
    CHECK(am == 2);
  }
  SECTION("Multiple") {
    std::vector<TestType> v{7, -3, 7, 5};
    auto am = argmax(v);
    CHECK((am == 0 or am == 2));
  }
  SECTION("Extreme") {
    TestType max, min;
    if constexpr (std::is_floating_point_v<TestType>) {
      max = std::numeric_limits<TestType>::infinity();
      min = -max;
    } else {
      max = std::numeric_limits<TestType>::max();
      min = std::numeric_limits<TestType>::lowest();
    }
    std::vector<TestType> v{2, min, max, 5};
    auto am = argmax(v);
    CHECK(am == 2);
  }
}

TEST_CASE("VectorMap argmax", "[amax]") {
  SECTION("Basic") {
    Tensor<Real> t(Shape{4}, {2, -3, 7, 5});
    auto am = argmax(t.v());
    CHECK(am == 2);
  }
  SECTION("Multiple") {
    Tensor<Real> t(Shape{4}, {7, -3, 7, 5});
    auto am = argmax(t.v());
    CHECK((am == 0 or am == 2));
  }
  SECTION("Extreme") {
    Real max = std::numeric_limits<Real>::infinity();
    Real min = -max;
    Tensor<Real> t(Shape{4}, {2, min, max, 5});
    auto am = argmax(t.v());
    CHECK(am == 2);
  }
}

TEST_CASE("MatrixMap argmax", "[amax]") {
  SECTION("Basic") {
    //  2  1  3
    // -3  3 -1
    //  7 -5  0
    Tensor<Real> t(Shape{3, 3}, {2, -3, 7, 1, 3, -5, 3, -1, 0});
    RowVector<Int> am = argmax(t.m());
    RowVector<Int> expected(3);
    expected << 2, 1, 0;

    CHECK(am == expected);
  }
  SECTION("Multiple") {
    //  7  1 -2
    // -3  1 -1
    //  7 -5 -1
    Tensor<Real> t(Shape{3, 3}, {7, -3, 7, 1, 1, -5, -2, -1, -1});
    RowVector<Int> am = argmax(t.m());

    CHECK((am[0] == 0 or am[0] == 2));
    CHECK((am[1] == 0 or am[1] == 1));
    CHECK((am[2] == 1 or am[2] == 2));
  }
  SECTION("Extreme") {
    Real max = std::numeric_limits<Real>::infinity();
    Real min = -max;
    //  2 -∞  3
    // -3  3 -1
    //  ∞ -5  0
    Tensor<Real> t(Shape{3, 3}, {2, -3, max, min, 3, -5, 3, -1, 0});
    RowVector<Int> am = argmax(t.m());
    RowVector<Int> expected(3);
    expected << 2, 1, 0;

    CHECK(am == expected);
  }
}

TEST_CASE("Tensor flat argmax", "[amax]") {
  Shape s = GENERATE(Shape{4}, Shape{2, 2});

  SECTION("Basic") {
    Tensor<Real> t(s, {2, -3, 7, 5});
    auto am = argmax(t);
    CHECK(am == 2);
  }
  SECTION("Multiple") {
    Tensor<Real> t(s, {7, -3, 7, 5});
    auto am = argmax(t);
    CHECK((am == 0 or am == 2));
  }
  SECTION("Extreme") {
    Real max = std::numeric_limits<Real>::infinity();
    Real min = -max;
    Tensor<Real> t(s, {2, min, max, 5});
    auto am = argmax(t);
    CHECK(am == 2);
  }
}

TEST_CASE("Tensor axiswise argmax", "[amax]") {
  SECTION("Basic") {
    //  3  1  2
    // -3  3 -1
    //  7 -5  0
    Tensor<Real> t(Shape{3, 3}, {3, -3, 7, 1, 2, -5, 3, -1, 0});
    using Index = TensorMap<Real, 1>::Index;
    Tensor<Index> row(Shape{3}, {2, 1, 0});
    Tensor<Index> col(Shape{3}, {0, 1, 0});

    CHECK(argmax(t, 0) == row);
    CHECK(argmax(t, 1) == col);
  }
  SECTION("Extreme") {
    //  3  1  2
    // -3  3 -∞
    //  ∞ -5  0
    Real inf = std::numeric_limits<Real>::infinity();
    Tensor<Real> t(Shape{3, 3}, {3, -3, inf, 1, 2, -5, 3, -inf, 0});
    using Index = TensorMap<Real, 1>::Index;
    Tensor<Index> row(Shape{3}, {2, 1, 0});
    Tensor<Index> col(Shape{3}, {0, 1, 0});

    CHECK(argmax(t, 0) == row);
    CHECK(argmax(t, 1) == col);
  }
  // SECTION("Nan") {
  //  //  3  1  2
  //  // -3  N -1
  //  //  7 -5  0
  //  Real nan = std::numeric_limits<Real>::signaling_NaN();
  //  Tensor<Real> t(Shape{3,3}, {3, -3, 7, 1, nan, -5, 3, -1, 0});
  //  // What is the correct expected behavior here?
  //}
}

TEMPLATE_TEST_CASE("Basic indexmap", "[indexmap]", std::string, int) {
  std::set<TestType> s;
  if constexpr (std::is_same_v<TestType, std::string>) {
    s = {"some", "words", "go", "here", "now"};
  } else {
    s = {10, 100, 27, -3};
  }

  IndexMap<TestType> m(s);

  CHECK(m.size() == s.size());
  auto it = s.begin();
  for (size_t i = 0; i < s.size(); i++) {
    CHECK(m.has(*it));
    CHECK(m[*it] == i);
    it++;
  }

  it = s.begin();
  for (size_t i = 0; i < m.size(); i++) {
    CHECK(m(i) == *it);
    it++;
  }

  auto x = *s.begin();
  m.insert(x);
  CHECK(m.size() == s.size());

  auto val = x + x;
  m.insert(val);
  CHECK(m.size() == (s.size() + 1));
  CHECK(m(m.size() - 1) == val);
  CHECK(m[val] == (m.size() - 1));
  CHECK(m.has(val));

  m.clear();
  CHECK(m.size() == 0);
  for (auto& x : s) { CHECK(not m.has(x)); }
}

// Won't test actual timing but side functionality.
TEST_CASE("Basic timer", "[timer]") {
  using namespace ginn::timer;
  using namespace ginn::timer::internal; // Don't do this 🥲

  tic("apple");
  tic("banana");
  tic("orange");
  toc("banana");
  toc("apple");
  toc("orange");

  CHECK(totals.size() == 3);
  for (auto s : {"apple", "orange", "banana"}) {
    CHECK(has(totals, s));
    CHECK(count.at(s) == 1);
    CHECK(totals[s] <= total);
  }
  // "apple" should strictly contain "banana"
  CHECK(totals["banana"] <= totals["apple"]);

  std::ostringstream ss;
  timer::print(ss);
  std::cout << ss.str() << std::endl;

  for (auto s : {"[apple]", "[orange]", "[banana]"}) {
    CHECK(has(ss.str(), s));
  }
  CHECK(startswith(ss.str(),
                   "Timing:\n"
                   "  Name       time   #       %   relative____________\n"));
}

TEST_CASE("Output stream tensor", "[io]") {
  SECTION("Tensor") {
    std::ostringstream ss;
    SECTION("From values") {
      auto x =
          Values<3>({{{1, 7}, {4, 10}}, {{2, 8}, {5, 11}}, {{3, 9}, {6, 12}}});
      ss << x->value();
    }
    SECTION("From flat storage") {
      Tensor t(cpu(), {3, 2, 2});
      t.set(std::vector<Real>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
      ss << t;
    }
    std::string expected = "{{{  1,   7}, {  4,  10}},\n"
                           " {{  2,   8}, {  5,  11}},\n"
                           " {{  3,   9}, {  6,  12}}}";
    CHECK(ss.str() == expected);
  }
  SECTION("Matrix") {
    std::ostringstream ss;
    SECTION("From values") {
      auto x = Values<2>({{1, 4}, {2, 5}, {3, 6}});
      ss << x->value();
    }
    SECTION("From flat storage") {
      Tensor t(cpu(), {3, 2});
      t.set(std::vector<Real>{1, 2, 3, 4, 5, 6});
      ss << t;
    }
    std::string expected = "{{ 1,  4},\n"
                           " { 2,  5},\n"
                           " { 3,  6}}";
    CHECK(ss.str() == expected);
  }
  SECTION("Vector") {
    std::ostringstream ss;
    SECTION("From values") {
      auto x = Values<1>({1, 2, 3});
      ss << x->value();
    }
    SECTION("From flat storage") {
      Tensor t(cpu(), {3});
      t.set(std::vector<Real>{1, 2, 3});
      ss << t;
    }
    std::string expected = "{ 1,  2,  3}";
    CHECK(ss.str() == expected);
  }
}
