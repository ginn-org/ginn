#ifndef GINN_UTIL_TYPENAME_H
#define GINN_UTIL_TYPENAME_H

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <cstdlib>
#include <memory>
#include <string>

namespace ginn {

// https://stackoverflow.com/a/20170989
template <class T>
std::string type_name() {
  using TR = typename std::remove_reference<T>::type;
#ifndef _MSC_VER
  auto f = abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr);
#else
  auto f = nullptr;
#endif
  std::unique_ptr<char, void (*)(void*)> own(f, std::free);
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const_v<TR>) { r += " const"; }
  if (std::is_volatile_v<TR>) { r += " volatile"; }
  if (std::is_lvalue_reference_v<T>) {
    r += "&";
  } else if (std::is_rvalue_reference_v<T>) {
    r += "&&";
  }
  return r;
}

} // namespace ginn

#endif
