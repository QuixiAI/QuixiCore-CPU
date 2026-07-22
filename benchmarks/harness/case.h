#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qcb {

enum class Preset { kSmoke, kQuick, kComprehensive };

struct BuildCtx {
  Preset preset;
  int threads;
};

struct Tolerance {
  double atol;
  double rtol;
};

// Mirrored from the umbrella registry/tolerances.yaml. Cases may use a
// stricter operation-specific tolerance when their oracle excludes storage
// quantization error (for example, qgemv against exactly dequantized weights).
inline constexpr Tolerance kFp32Tolerance{1e-6, 1e-5};
inline constexpr Tolerance kQuantizedTolerance{0.03, 0.03};

struct CheckResult {
  bool passed = true;
  bool finite = true;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
};

inline void check_value(CheckResult& result, double actual, double expected,
                        Tolerance tolerance) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    result.passed = false;
    result.finite = false;
    result.max_abs_err = std::numeric_limits<double>::infinity();
    result.max_rel_err = std::numeric_limits<double>::infinity();
    return;
  }
  const double diff = std::fabs(actual - expected);
  const double ref_abs = std::fabs(expected);
  result.max_abs_err = std::max(result.max_abs_err, diff);
  // Keep the reporting metric finite near zero; pass/fail still uses the
  // standard elementwise atol + rtol*|ref| predicate below.
  const double rel = diff / std::max(ref_abs, tolerance.atol);
  result.max_rel_err = std::max(result.max_rel_err, rel);
  if (diff > tolerance.atol + tolerance.rtol * ref_abs) {
    result.passed = false;
  }
}

// Runtime half of a case. The closures own the buffers, so destroying the
// body after timing bounds peak memory to one live case at a time.
//
// Thunk contract: every thunk must route its output through
// do_not_optimize() (or call clobber()) so the compiler cannot delete or
// hoist the measured work.
struct CaseBody {
  std::function<void()> target;
  std::vector<std::pair<std::string, std::function<void()>>> baselines;
  std::function<CheckResult()> check;  // null when no oracle applies
};

// Metadata half of a case; cheap to build eagerly.
struct CaseDecl {
  std::string kernel;
  std::string variant;
  std::vector<std::pair<std::string, long long>> shape;  // ordered named dims
  std::string dtype = "f32";
  std::string format;       // empty -> JSON null; quant format name later
  std::string notes;
  std::string skip_reason;  // non-empty -> skip row, make stays null
  double bytes_moved = 0.0;   // > 0 -> gbps reported
  double weight_bytes = 0.0;  // > 0 -> weight_gbps reported (quant decode)
  double flops = 0.0;         // > 0 -> gflops reported
  std::function<CaseBody()> make;  // allocates and fills buffers on demand
};

using CaseBuilder = void (*)(const BuildCtx&, std::vector<CaseDecl>&);

struct KernelEntry {
  const char* name;
  CaseBuilder build;
};

// Explicit table in registry.cpp; adding a kernel is one line there.
const std::vector<KernelEntry>& kernel_registry();

}  // namespace qcb
