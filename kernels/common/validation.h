#pragma once

#include <climits>
#include <initializer_list>

namespace quixicore_cpu::detail {

inline bool valid_product(std::initializer_list<long long> dimensions) {
  long long product = 1;
  for (const long long dimension : dimensions) {
    if (dimension <= 0 || product > LLONG_MAX / dimension) {
      return false;
    }
    product *= dimension;
  }
  return true;
}

template <typename... Pointers>
inline bool all_nonnull(Pointers... pointers) {
  return ((pointers != nullptr) && ...);
}

}  // namespace quixicore_cpu::detail
