#include <algorithm>
#include <iostream>
#include <string_view>

#include "quixicore_cpu/backend.h"

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
  const auto metadata = quixicore_cpu::backend_metadata();

  REQUIRE(metadata.backend == "cpu");
  REQUIRE(metadata.name == "QuixiCore CPU");
  REQUIRE(metadata.repo == "QuixiAI/QuixiCore-CPU");
  REQUIRE(metadata.umbrella == "QuixiAI/QuixiCore");
  REQUIRE(metadata.contract == "v0.1");
  REQUIRE(metadata.status == "experimental");
  REQUIRE(metadata.targets.size() == 2);

  const auto supported = quixicore_cpu::supported_kernel_families();
  REQUIRE(supported.empty());
  REQUIRE(!quixicore_cpu::is_kernel_family_supported("quant_gemv"));

  const auto planned = quixicore_cpu::planned_kernel_families();
  REQUIRE(std::find(planned.begin(), planned.end(),
                    std::string_view{"quant_gemv"}) != planned.end());
  REQUIRE(std::find(planned.begin(), planned.end(),
                    std::string_view{"quant_gemm"}) != planned.end());

  const auto features = quixicore_cpu::compile_time_feature_hints();
  REQUIRE(!features.empty());

  return 0;
}
