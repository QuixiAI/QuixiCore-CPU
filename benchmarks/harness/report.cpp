#include "harness/report.h"

#include <cmath>
#include <cstdio>
#include <string>

#include "harness/json_writer.h"

namespace qcb {
namespace {

double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

std::string fixed(double v, const char* fmt) {
  char buf[48];
  std::snprintf(buf, sizeof buf, fmt, v);
  return std::string(buf);
}

std::string shape_str(const CaseDecl& decl) {
  std::string out;
  for (const auto& [name, value] : decl.shape) {
    (void)name;
    if (!out.empty()) {
      out += "x";
    }
    out += std::to_string(value);
  }
  return out;
}

const BaselineResult* best_baseline(const CaseResult& result) {
  const BaselineResult* best = nullptr;
  for (const auto& baseline : result.baselines) {
    if (baseline.ok && (best == nullptr || baseline.ms < best->ms)) {
      best = &baseline;
    }
  }
  return best;
}

}  // namespace

std::string result_row_json(const CaseResult& result) {
  const CaseDecl& decl = result.decl;
  JsonWriter w;
  w.begin_obj();
  w.key("schema");
  w.val(1);
  w.key("kernel");
  w.val(decl.kernel);
  w.key("variant");
  w.val(decl.variant);
  w.key("shape");
  w.begin_obj();
  for (const auto& [name, value] : decl.shape) {
    w.key(name);
    w.val(value);
  }
  w.end_obj();
  w.key("dtype");
  w.val(decl.dtype);
  w.key("format");
  if (decl.format.empty()) {
    w.null();
  } else {
    w.val(decl.format);
  }
  w.key("status");
  w.val(result.status);
  w.key("notes");
  if (result.error.empty()) {
    w.val(decl.notes);
  } else {
    w.val(decl.notes.empty() ? result.error : decl.notes + "; " + result.error);
  }
  if (!result.skip_reason.empty()) {
    w.key("skip_reason");
    w.val(result.skip_reason);
  }
  if (result.status == "ok") {
    if (result.checked) {
      w.key("max_abs_err");
      w.val(result.check.max_abs_err);
      w.key("max_rel_err");
      w.val(result.check.max_rel_err);
    }
    w.key("target_ms");
    w.val(result.timing.ms);
    w.key("target_p20_ms");
    w.val(result.timing.p20_ms);
    w.key("target_p80_ms");
    w.val(result.timing.p80_ms);
    w.key("target_cv");
    w.val(round4(result.timing.cv));
    w.key("batch");
    w.val(result.timing.batch);
    if (!result.baselines.empty()) {
      w.key("baselines");
      w.begin_obj();
      for (const auto& baseline : result.baselines) {
        w.key(baseline.name);
        w.begin_obj();
        if (baseline.ok) {
          w.key("ms");
          w.val(baseline.ms);
          w.key("speedup");
          w.val(result.timing.ms > 0.0 ? baseline.ms / result.timing.ms : 0.0);
        } else {
          w.key("error");
          w.val(baseline.error);
        }
        w.end_obj();
      }
      w.end_obj();
    }
    if (result.gbps > 0.0) {
      w.key("gbps");
      w.val(result.gbps);
    }
    if (result.weight_gbps > 0.0) {
      w.key("weight_gbps");
      w.val(result.weight_gbps);
    }
    if (result.gflops > 0.0) {
      w.key("gflops");
      w.val(result.gflops);
    }
  }
  w.end_obj();
  return w.str();
}

std::string run_json(const RunMeta& meta) {
  JsonWriter w;
  w.begin_obj();
  w.key("schema");
  w.val(1);
  w.key("git");
  w.val(meta.env.git_label);
  w.key("timestamp");
  w.val(meta.timestamp);
  w.key("backend");
  w.val(meta.env.backend);
  w.key("repo");
  w.val(meta.env.repo);
  w.key("contract");
  w.val(meta.env.contract);
  w.key("os");
  w.val(meta.env.os);
  w.key("arch");
  w.val(meta.env.arch);
  w.key("cpu_model");
  w.val(meta.env.cpu_model);
  w.key("logical_cores");
  w.val(meta.env.logical_cores);
  w.key("physical_cores");
  if (meta.env.physical_cores < 0) {
    w.null();
  } else {
    w.val(meta.env.physical_cores);
  }
  if (meta.env.perf_cores >= 0) {
    w.key("perf_cores");
    w.val(meta.env.perf_cores);
    w.key("eff_cores");
    w.val(meta.env.eff_cores);
  }
  w.key("memory_bytes");
  if (meta.env.memory_bytes < 0) {
    w.null();
  } else {
    w.val(meta.env.memory_bytes);
  }
  w.key("compiler");
  w.val(meta.env.compiler);
  w.key("build_type");
  w.val(meta.env.build_type);
  w.key("cpu_features");
  w.begin_arr();
  for (const auto& feature : meta.env.cpu_features) {
    w.val(feature);
  }
  w.end_arr();
  w.key("threads");
  w.val(meta.threads);
  w.key("affinity_policy");
  w.val("none (OS scheduler default)");
  w.key("frequency_policy");
  w.val("OS default; turbo and power management not pinned");
  w.key("preset");
  w.val(meta.preset);
  w.key("warmup");
  w.val(meta.warmup);
  w.key("iters");
  w.val(meta.iters);
  w.key("min_sample_ms");
  w.val(meta.min_sample_ms);
  w.key("check");
  w.val(meta.check);
  w.key("kernels");
  w.begin_arr();
  for (const auto& kernel : meta.kernels) {
    w.val(kernel);
  }
  w.end_arr();
  w.key("wall_s");
  w.val(round4(meta.wall_s));
  w.end_obj();
  return w.str();
}

std::string summary_md(const RunMeta& meta,
                       const std::vector<CaseResult>& results) {
  std::string out;
  out += "# QuixiCore CPU Bench Summary\n\n";
  out += "- `" + meta.env.git_label + "` · " + meta.env.cpu_model + " · " +
         meta.env.os + " · " + meta.env.compiler + "\n";
  out += "- preset `" + meta.preset + "` · warmup " +
         std::to_string(meta.warmup) + " · iters " +
         std::to_string(meta.iters) + " · threads " +
         std::to_string(meta.threads) + "\n\n";
  out +=
      "| kernel | variant | shape | ms | best baseline | base ms | speedup |"
      " GB/s | W-GB/s | GFLOP/s | rel err |\n";
  out +=
      "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|\n";
  for (const auto& result : results) {
    const CaseDecl& decl = result.decl;
    out += "| " + decl.kernel + " | " + decl.variant + " | " +
           shape_str(decl) + " | ";
    if (result.status != "ok") {
      out += "_" + result.status + "_";
      if (!result.skip_reason.empty()) {
        out += " (" + result.skip_reason + ")";
      }
      out += " | | | | | | | |\n";
      continue;
    }
    out += fixed(result.timing.ms, "%.4f") + " | ";
    const BaselineResult* best = best_baseline(result);
    if (best != nullptr) {
      out += best->name + " | " + fixed(best->ms, "%.4f") + " | " +
             fixed(result.timing.ms > 0.0 ? best->ms / result.timing.ms : 0.0,
                   "%.2f") +
             " | ";
    } else {
      out += " | | | ";
    }
    out += (result.gbps > 0.0 ? fixed(result.gbps, "%.1f") : "") + " | ";
    out += (result.weight_gbps > 0.0 ? fixed(result.weight_gbps, "%.1f") : "") +
           " | ";
    out += (result.gflops > 0.0 ? fixed(result.gflops, "%.1f") : "") + " | ";
    out += (result.checked ? fixed(result.check.max_rel_err, "%.2e") : "") +
           " |\n";
  }
  return out;
}

}  // namespace qcb
