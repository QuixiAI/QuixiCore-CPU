// quixicore_cpu_bench: QuixiCore CPU benchmark harness.
//
// Mirrors the QuixiCore-Metal harness conventions (case registry, presets,
// adaptive-batch timing, correctness-before-timing, results.jsonl/run.json/
// summary.md outputs) as a native CPU binary. See benchmarks/README.md and
// perf/perf.md.

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/env_info.h"
#include "harness/report.h"
#include "harness/timing.h"
#include "quixicore_cpu/threading.h"

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace qcb {
namespace {

struct Options {
  Preset preset = Preset::kQuick;
  std::string preset_name = "quick";
  std::vector<std::string> kernels;  // empty -> all
  bool list = false;
  int warmup = 3;
  int iters = 20;
  double min_sample_ms = 2.0;
  bool check = true;
  int threads = 1;
  std::string out_dir;
};

void print_usage(std::ostream& os) {
  os << "usage: quixicore_cpu_bench [options]\n"
        "  --preset smoke|quick|comprehensive   case selection (default"
        " quick)\n"
        "  --kernel <name>[,<name>...] | all    kernels to run (default"
        " all)\n"
        "  --list                               list kernels and case"
        " counts\n"
        "  --warmup N                           warmup calls (default 3)\n"
        "  --iters N                            timed samples (default 20)\n"
        "  --min-sample-ms X                    min time per sample (default"
        " 2.0)\n"
        "  --no-check                           skip the correctness"
        " oracle\n"
        "  --threads N                          total parallelism for"
        " threaded kernels (default 1)\n"
        "  --out-dir PATH                       output directory (default"
        " perf/results/<date>/<time>-<preset>)\n"
        "  --help\n";
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> parts;
  std::string::size_type start = 0;
  while (start <= value.size()) {
    const auto comma = value.find(',', start);
    const auto end = comma == std::string::npos ? value.size() : comma;
    if (end > start) {
      parts.push_back(value.substr(start, end - start));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return parts;
}

int list_kernels() {
  for (const auto& entry : kernel_registry()) {
    long long counts[3] = {0, 0, 0};
    const Preset presets[3] = {Preset::kSmoke, Preset::kQuick,
                               Preset::kComprehensive};
    for (int p = 0; p < 3; ++p) {
      std::vector<CaseDecl> decls;
      entry.build(BuildCtx{presets[p], 1}, decls);
      counts[p] = static_cast<long long>(decls.size());
    }
    std::printf("%-14s smoke:%lld quick:%lld comprehensive:%lld\n", entry.name,
                counts[0], counts[1], counts[2]);
  }
  return 0;
}

std::vector<const KernelEntry*> select_kernels(
    const std::vector<std::string>& requested) {
  const auto& registry = kernel_registry();
  std::vector<const KernelEntry*> selected;
  if (requested.empty()) {
    for (const auto& entry : registry) {
      selected.push_back(&entry);
    }
    return selected;
  }
  for (const auto& name : requested) {
    const KernelEntry* found = nullptr;
    for (const auto& entry : registry) {
      if (name == entry.name) {
        found = &entry;
        break;
      }
    }
    if (found == nullptr) {
      std::cerr << "unknown kernel: " << name << "\navailable:";
      for (const auto& entry : registry) {
        std::cerr << " " << entry.name;
      }
      std::cerr << "\n";
      return {};
    }
    selected.push_back(found);
  }
  return selected;
}

CaseResult run_case(const CaseDecl& decl, const Options& opts) {
  CaseResult result;
  result.decl = decl;
  result.decl.make = nullptr;

  if (!decl.skip_reason.empty() || !decl.make) {
    result.status = "skip";
    result.skip_reason =
        decl.skip_reason.empty() ? "no case body" : decl.skip_reason;
    return result;
  }

  try {
    CaseBody body = decl.make();
    if (opts.check && body.check) {
      result.check = body.check();
      result.checked = true;
    }
    result.timing =
        time_thunk(body.target, opts.warmup, opts.iters, opts.min_sample_ms);
    for (const auto& [name, fn] : body.baselines) {
      BaselineResult baseline;
      baseline.name = name;
      try {
        baseline.ms =
            time_thunk(fn, opts.warmup, opts.iters, opts.min_sample_ms).ms;
        baseline.ok = true;
      } catch (const std::exception& e) {
        baseline.error = e.what();
      }
      result.baselines.push_back(std::move(baseline));
    }
    result.status = "ok";
    const double seconds = result.timing.ms / 1e3;
    if (seconds > 0.0) {
      if (decl.bytes_moved > 0.0) {
        result.gbps = decl.bytes_moved / seconds / 1e9;
      }
      if (decl.weight_bytes > 0.0) {
        result.weight_gbps = decl.weight_bytes / seconds / 1e9;
      }
      if (decl.flops > 0.0) {
        result.gflops = decl.flops / seconds / 1e9;
      }
    }
  } catch (const std::bad_alloc&) {
    result.status = "skip";
    result.skip_reason = "allocation failed";
    result.baselines.clear();
  } catch (const std::exception& e) {
    result.status = "error";
    result.error = e.what();
  }
  return result;
}

void print_progress(const CaseResult& result) {
  const std::string label = result.decl.kernel + "/" + result.decl.variant;
  if (result.status == "ok") {
    std::printf("%-32s %10.4f ms  (batch %d)", label.c_str(),
                result.timing.ms, result.timing.batch);
    if (result.checked) {
      std::printf("  rel_err %.2e", result.check.max_rel_err);
    }
    std::printf("\n");
  } else {
    std::printf("%-32s %s%s%s\n", label.c_str(), result.status.c_str(),
                result.skip_reason.empty() ? "" : ": ",
                result.skip_reason.c_str());
  }
  std::fflush(stdout);
}

int run(const Options& opts) {
#if defined(__APPLE__)
  // Bias scheduling away from efficiency cores for less noisy samples.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const auto wall_start = std::chrono::steady_clock::now();

  const auto selected = select_kernels(opts.kernels);
  if (selected.empty()) {
    return 2;
  }
  quixicore_cpu::set_num_threads(opts.threads);

  std::string out_dir = opts.out_dir;
  if (out_dir.empty()) {
    out_dir = std::string(QUIXICORE_CPU_BENCH_REPO_DIR) + "/perf/results/" +
              format_now("%Y-%m-%d") + "/" + format_now("%H%M%S") + "-" +
              opts.preset_name;
  }
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "cannot create out dir " << out_dir << ": " << ec.message()
              << "\n";
    return 1;
  }

  std::ofstream jsonl(std::filesystem::path(out_dir) / "results.jsonl");
  if (!jsonl) {
    std::cerr << "cannot open results.jsonl in " << out_dir << "\n";
    return 1;
  }

  RunMeta meta;
  meta.env = collect_env_info();
  meta.timestamp = format_now("%Y-%m-%dT%H:%M:%S");
  meta.preset = opts.preset_name;
  meta.warmup = opts.warmup;
  meta.iters = opts.iters;
  meta.min_sample_ms = opts.min_sample_ms;
  meta.check = opts.check;
  meta.threads = opts.threads;
  for (const auto* entry : selected) {
    meta.kernels.emplace_back(entry->name);
  }

  std::printf("quixicore_cpu_bench · %s · %s · preset %s\n",
              meta.env.git_label.c_str(), meta.env.cpu_model.c_str(),
              meta.preset.c_str());

  std::vector<CaseResult> results;
  bool any_error = false;
  const BuildCtx ctx{opts.preset, opts.threads};
  for (const auto* entry : selected) {
    std::vector<CaseDecl> decls;
    entry->build(ctx, decls);
    for (const auto& decl : decls) {
      CaseResult result = run_case(decl, opts);
      any_error = any_error || result.status == "error";
      print_progress(result);
      jsonl << result_row_json(result) << "\n";
      jsonl.flush();
      results.push_back(std::move(result));
    }
  }

  meta.wall_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wall_start)
                    .count();

  std::ofstream(std::filesystem::path(out_dir) / "run.json")
      << run_json(meta) << "\n";
  std::ofstream(std::filesystem::path(out_dir) / "summary.md")
      << summary_md(meta, results);

  std::printf("wrote %s\n", out_dir.c_str());
  return any_error ? 1 : 0;
}

int bench_main(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    bool has_inline_value = false;
    if (arg.rfind("--", 0) == 0) {
      const auto eq = arg.find('=');
      if (eq != std::string::npos) {
        value = arg.substr(eq + 1);
        arg = arg.substr(0, eq);
        has_inline_value = true;
      }
    }
    const auto need_value = [&]() {
      if (has_inline_value) {
        return true;
      }
      if (i + 1 < argc) {
        value = argv[++i];
        return true;
      }
      std::cerr << "missing value for " << arg << "\n";
      return false;
    };

    try {
      if (arg == "--help" || arg == "-h") {
        print_usage(std::cout);
        return 0;
      } else if (arg == "--list") {
        opts.list = true;
      } else if (arg == "--no-check") {
        opts.check = false;
      } else if (arg == "--preset") {
        if (!need_value()) return 2;
        if (value == "smoke") {
          opts.preset = Preset::kSmoke;
        } else if (value == "quick") {
          opts.preset = Preset::kQuick;
        } else if (value == "comprehensive") {
          opts.preset = Preset::kComprehensive;
        } else {
          std::cerr << "unknown preset: " << value << "\n";
          return 2;
        }
        opts.preset_name = value;
      } else if (arg == "--kernel") {
        if (!need_value()) return 2;
        if (value != "all") {
          opts.kernels = split_csv(value);
        }
      } else if (arg == "--warmup") {
        if (!need_value()) return 2;
        opts.warmup = std::max(0, std::stoi(value));
      } else if (arg == "--iters") {
        if (!need_value()) return 2;
        opts.iters = std::max(1, std::stoi(value));
      } else if (arg == "--min-sample-ms") {
        if (!need_value()) return 2;
        opts.min_sample_ms = std::max(0.0, std::stod(value));
      } else if (arg == "--threads") {
        if (!need_value()) return 2;
        opts.threads = std::max(1, std::stoi(value));
      } else if (arg == "--out-dir") {
        if (!need_value()) return 2;
        opts.out_dir = value;
      } else {
        std::cerr << "unknown option: " << arg << "\n";
        print_usage(std::cerr);
        return 2;
      }
    } catch (const std::exception&) {
      std::cerr << "invalid value for " << arg << ": " << value << "\n";
      return 2;
    }
  }

  if (opts.list) {
    return list_kernels();
  }
  return run(opts);
}

}  // namespace
}  // namespace qcb

int main(int argc, char** argv) { return qcb::bench_main(argc, argv); }
