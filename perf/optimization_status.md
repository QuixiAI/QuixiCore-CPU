# Optimization Status

No CPU kernel optimization runs have been recorded yet. The log currently
contains benchmark-harness bring-up evidence only, which implements no kernel
and makes no performance claim.

Every future kernel implementation, routing change, benchmark change, or
performance claim must add a focused optimization entry here.

## Required Entry Fields

- Kernel, operation, dtype or quant format, and shape set.
- Correctness command and result.
- Baseline measurement.
- Candidate or current measurement.
- Hardware model, core count, memory configuration, and ISA target.
- Operating system, compiler, runtime, and thread settings.
- Command line, git commit or working-tree label, warmups, iterations, median,
  and variance or min/max.
- Keep or reject decision.

## Log

| Date | Kernel | Dtype / Format | Shape Set | Target | Baseline | Candidate | Decision | Evidence |
|---|---|---|---|---|---:|---:|---|---|
| 2026-07-07 | harness (system probes) | f32 | mem_triad ladder + quant_matmul m=1 | aarch64 baseline flags | n/a | n/a | harness validated; no kernel, no speedup claim | `perf/results/2026-07-07/021238-quick/` |

## 2026-07-07: Benchmark harness bring-up (system probes only)

Status: landed.
Current implementation: `quixicore_cpu_bench` harness with two
system-characterization probes (`mem_triad`, `sgemv_naive`). No contract
kernel exists or is claimed; no speedup is claimed.
Current public route: n/a (probes call harness-local loops, not library
kernels).
References inspected: QuixiCore-Metal `perf/bench_kernels.py` (timing
discipline, row schema, presets), umbrella `docs/benchmarking.md` and
`registry/benchmark-shapes.yaml`.
Correctness: in-harness float64 oracle per case. mem_triad max rel err
0.00e+00 (exact in f32 for the chosen fills); sgemv_naive max rel err
1.7e-06 - 2.6e-06 across shapes, within fp32 contract tolerance
(rtol 1e-5).
Baseline: this run establishes the machine reference points below; there is
no prior measurement to compare against and no comparison is made.
Experiments: none (bring-up only).
Decision: harness validated end-to-end (correctness oracle, adaptive-batch
timing, JSONL/run.json/summary outputs, gitignored results, ctest smoke on
all CI platforms). No kernel implemented, no performance claim.
Open questions: affinity pinning and multi-threaded probes deferred until
threaded kernels exist.
Raw results: `perf/results/2026-07-07/021238-quick/` (git-ignored; summary
below).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags (no `-march=native`).
- Command: `scripts/bench --preset quick` (warmup 3, iters 20,
  min_sample_ms 2.0, threads 1, QoS user-interactive).
- Working-tree label: `0220bc1-dirty`.

| case | variant | median ms | CV | GB/s | GFLOP/s | max rel err |
|---|---|---:|---:|---:|---:|---|
| mem_triad | ws_96KiB | 0.0004 | 0.071 | 246.3 | | 0.00e+00 |
| mem_triad | ws_1536KiB | 0.0132 | 0.176 | 118.9 | | 0.00e+00 |
| mem_triad | ws_24MiB | 0.2146 | 0.059 | 117.3 | | 0.00e+00 |
| mem_triad | ws_192MiB | 1.7480 | 0.060 | 115.2 | | 0.00e+00 |
| sgemv_naive | N2048_K2048 | 2.0521 | 0.040 | 8.2 | 4.1 | 1.72e-06 |
| sgemv_naive | N4096_K4096 | 8.7347 | 0.016 | 7.7 | 3.8 | 1.79e-06 |
| sgemv_naive | N8192_K8192 | 35.7706 | 0.028 | 7.5 | 3.8 | 2.44e-06 |
| sgemv_naive | N16384_K4096 | 34.9894 | 0.012 | 7.7 | 3.8 | 2.61e-06 |

Reading (context, not a claim): single-thread DRAM triad plateaus at
~115-117 GB/s; the naive sgemv runs at ~7.7 GB/s effective, an order of
magnitude under the roofline because strict-FP scalar accumulation is
latency-bound. That gap is the headroom future GEMV work is judged
against.

