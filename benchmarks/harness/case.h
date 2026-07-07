#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace qcb {

enum class Preset { kSmoke, kQuick, kComprehensive };

struct BuildCtx {
  Preset preset;
  int threads;  // recorded in outputs; v1 cases are single-threaded
};

struct CheckResult {
  double max_abs_err;
  double max_rel_err;  // max|diff| / (max|ref| + 1e-9), the repo convention
};

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
