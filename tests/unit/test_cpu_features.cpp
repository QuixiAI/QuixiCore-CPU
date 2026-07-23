#include <iostream>

#include "quixicore_cpu/cpu_features.h"

// Tests must fail in any build configuration, so no assert()/NDEBUG.
#define REQUIRE(cond)                                                     \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ \
                << '\n';                                                  \
      return 1;                                                           \
    }                                                                     \
  } while (0)

int main() {
  const auto& features = quixicore_cpu::cpu_features();
  const auto& again = quixicore_cpu::cpu_features();
  REQUIRE(&features == &again);

#if defined(__aarch64__) || defined(_M_ARM64)
  REQUIRE(features.neon);
  REQUIRE(!features.avx2);
  REQUIRE(!features.avx512f);
  REQUIRE(!features.amx_tile);
  if (features.fp16) REQUIRE(features.neon);
#endif

#if defined(__x86_64__) || defined(_M_X64)
  REQUIRE(!features.neon);
  REQUIRE(!features.sve);
  REQUIRE(!features.sme);
  REQUIRE(!features.fp16);
  if (features.avx512_vnni) {
    REQUIRE(features.avx512f);
  }
  if (features.amx_int8 || features.amx_bf16) {
    REQUIRE(features.amx_tile);
  }
#endif

  const auto hints = quixicore_cpu::runtime_feature_hints();
  REQUIRE(!hints.empty());
  for (const auto& hint : hints) {
    REQUIRE(!hint.name.empty());
    REQUIRE(hint.source == "runtime detection");
  }

  return 0;
}
