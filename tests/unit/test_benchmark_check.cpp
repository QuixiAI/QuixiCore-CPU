#include <cmath>
#include <iostream>
#include <limits>

#include "harness/case.h"

#define REQUIRE(cond)                                                     \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ \
                << '\n';                                                  \
      return 1;                                                           \
    }                                                                     \
  } while (0)

int main() {
  {
    qcb::CheckResult check;
    qcb::check_value(check, 1.0 + 5e-6, 1.0, qcb::kFp32Tolerance);
    REQUIRE(check.passed);
    REQUIRE(check.finite);
  }
  {
    qcb::CheckResult check;
    qcb::check_value(check, 1.0 + 5e-5, 1.0, qcb::kFp32Tolerance);
    REQUIRE(!check.passed);
    REQUIRE(check.finite);
  }
  {
    qcb::CheckResult check;
    qcb::check_value(check, 5e-7, 0.0, qcb::kFp32Tolerance);
    REQUIRE(check.passed);
    REQUIRE(std::isfinite(check.max_rel_err));
  }
  {
    qcb::CheckResult check;
    qcb::check_value(check, std::numeric_limits<double>::quiet_NaN(), 0.0,
                     qcb::kFp32Tolerance);
    REQUIRE(!check.passed);
    REQUIRE(!check.finite);
  }
  return 0;
}
