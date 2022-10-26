#ifndef GINN_PY_UTIL_PY_H
#define GINN_PY_UTIL_PY_H

namespace ginn {
namespace python {

template <typename ... Scalars, typename F>
void for_each(F f) {
  (f(Scalars()), ...);
}

}
}
#endif
