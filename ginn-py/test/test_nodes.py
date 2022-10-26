import ginn


def test_dim():
    """
    TEMPLATE_TEST_CASE("Dim", "[layout]", Real, Int, Half, void) {
      BaseNodePtr x;
      if constexpr(std::is_same_v<TestType, void>) {
        x = Random(Dev, {3, 2, 1});
      } else {
        x = Random<TestType>(Dev, {3, 2, 1});
      }
      auto d0 = Dim(x, 0);
      auto d1 = Dim(x, 1);
      auto d2 = Dim(x, 2);
      SECTION("Basic") {
        Graph(d0).forward();
        Graph(d1).forward();
        Graph(d2).forward();
        CHECK(d0->value() == 3);
        CHECK(d1->value() == 2);
        CHECK(d2->value() == 1);
      }
    }
    """
    pass
