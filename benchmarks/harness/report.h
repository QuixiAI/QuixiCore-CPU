#pragma once

#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/env_info.h"
#include "harness/timing.h"

namespace qcb {

struct BaselineResult {
  std::string name;
  bool ok = false;
  double ms = 0.0;
  std::string error;
};

struct CaseResult {
  CaseDecl decl;       // metadata copy; make is not used here
  std::string status;  // "ok" | "skip" | "error"
  std::string skip_reason;
  std::string error;
  bool checked = false;
  CheckResult check;
  TimingResult timing{0.0, 0.0, 0.0, 0.0, 0};
  std::vector<BaselineResult> baselines;
  double gbps = 0.0;
  double weight_gbps = 0.0;
  double gflops = 0.0;
};

struct RunMeta {
  EnvInfo env;
  std::string timestamp;
  std::string preset;
  int warmup = 0;
  int iters = 0;
  double min_sample_ms = 0.0;
  bool check = true;
  int threads = 1;
  std::vector<std::string> kernels;
  double wall_s = 0.0;
};

// One results.jsonl line (no trailing newline). Field order mirrors the
// Metal harness schema-1 rows.
std::string result_row_json(const CaseResult& result);

// Pretty-printed run.json contents.
std::string run_json(const RunMeta& meta);

// Markdown summary table for the whole run.
std::string summary_md(const RunMeta& meta,
                       const std::vector<CaseResult>& results);

}  // namespace qcb
