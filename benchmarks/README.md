# Benchmarks

`quixicore_cpu_bench` is the native CPU benchmark harness. It mirrors the
QuixiCore-Metal harness conventions (case registry, presets, adaptive-batch
timing, correctness-before-timing, JSONL/JSON/Markdown outputs) as a compiled
C++ binary calling directly into the library — no bindings, no third-party
dependencies.

## Quick Start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/quixicore_cpu_bench --list
./build/quixicore_cpu_bench --preset quick
# or, from any directory:
scripts/bench --preset quick
```

Always benchmark a `Release` build. On Windows, run
`build\Release\quixicore_cpu_bench.exe` directly.

## Command Line

```text
--preset smoke|quick|comprehensive   case selection (default quick)
--kernel <name>[,<name>...] | all    kernels to run (default all)
--list                               list kernels and per-preset case counts
--warmup N                           warmup calls (default 3)
--iters N                            timed samples (default 20)
--min-sample-ms X                    min time per sample (default 2.0)
--no-check                           skip the correctness oracle
--threads N                          total worker count (default 1)
--out-dir PATH                       default perf/results/<date>/<time>-<preset>/
```

Exit code: 0 when every case is `ok` or `skip`, 1 when any case errors,
2 on usage errors.

## Presets And Shape Provenance

Shape values come from the umbrella `registry/benchmark-shapes.yaml`
(contract v0.1) and are hardcoded verbatim in `harness/shapes.h` with a
provenance comment — the same no-YAML-parsing convention the Metal harness
uses. Update them by hand when the registry changes. Local exploratory
shapes are allowed per umbrella `docs/benchmarking.md`, but contract
compatibility is measured only against the registry shapes.

Presets follow the Metal ladder: `smoke` is a CI-sized sanity sweep,
`quick` is the default working set, `comprehensive` is the full grid
(largest case allocates a 1 GiB weight; allocation failure produces a
`skip` row, not an error).

## Output Layout

Each run writes three files to `perf/results/YYYY-MM-DD/HHMMSS-<preset>/`
(contents git-ignored):

- `run.json` — environment and invocation: git label, OS, arch, CPU model,
  core counts, memory, compiler, build type, runtime-detected
  `cpu_features`, thread/affinity/frequency policy, preset, warmup, iters,
  kernel list, wall time.
- `results.jsonl` — one schema-1 row per case: `kernel`, `variant`,
  `shape{}`, `dtype`, `format`, `status` (`ok`/`skip`/`error`), `notes`,
  `check_passed`, `max_abs_err`, `max_rel_err`, `target_ms`, `target_p20_ms`,
  `target_p80_ms`, `target_cv`, `batch`, `baselines{name:{ms,speedup}}`,
  and `gbps` / `weight_gbps` / `gflops` when applicable.
- `summary.md` — human-readable table of the same rows.

## Current Cases

The harness includes two system/reference probes, the existing q8_0 GEMV and
RMSNorm cases, and a representative cross-family contract batch. Every checked
case must pass its finite, elementwise tolerance gate before timing starts; a
failed oracle produces an `error` row and a nonzero process exit:

- `mem_triad` — STREAM-triad bandwidth probe (`a[i] = b[i] + s*c[i]`,
  single-thread f32) over a working-set ladder from 96 KiB to 768 MiB,
  crossing L1/L2/SLC into DRAM. Establishes the memory roofline that
  memory-bound kernels (quantized decode) are judged against. Carries a
  `memcpy` baseline; note memcpy moves 2n·4 bytes vs triad's counted 3n·4
  (STREAM convention; write-allocate traffic not counted).
- `sgemv_naive` — plain-loop scalar f32 GEMV on `quant_matmul` m=1 shapes
  plus `decode_small` hidden sizes. This is the reference semantics future
  ISA/quantized GEMV variants must beat; auto-vectorization at baseline
  Release flags is part of the definition.
- `qgemv` — public q8_0 GEMV on the contract `quant_matmul` m=1 shapes,
  checked against float64 accumulation over exactly dequantized weights.
  Contract-compatible f32-activation variants are timed; activation-quantized
  math is exposed only through the separate `qgemv_w8a8` operation.
- `qgemv_formats` — q4_0 weight-only GEMV and q4_0/q8_0 W8A8 GEMV at the
  registry N4096 K4096 decode shape. W8A8 correctness is checked against
  independently dequantized weights and blockwise-int8 activations.
- `rms_norm` — public f32 RMSNorm over `decode_small` shapes plus an R512
  throughput stress shape, checked against a float64 oracle.
- `contract_ops` — portable f32 softmax, causal attention, MoE top-k routing,
  Mamba selective scan, and q8_0 quantized GEMM. Each target has an independent
  scalar or decomposed baseline and a correctness gate; these are
  representative performance paths for the sibling-port batch, not an
  ISA-tuning claim.

## Adding A Kernel Case

1. Add `cases/<op>.cpp` with a builder
   `void build_<op>_cases(const BuildCtx&, std::vector<CaseDecl>&)` that
   emits one `CaseDecl` per shape (× quant format later). Fill
   `bytes_moved` / `weight_bytes` / `flops` per the formulas in
   `perf/perf.md`, attach baselines per the three-baseline rule, and give
   every case a float64 oracle.
2. Register it: one line in `harness/registry.cpp`.
3. Add the source to `benchmarks/CMakeLists.txt`.
4. Never compile harness or `_ref`/naive case sources with
   `quixicore_cpu_add_isa_sources()` — baseline flags only. ISA-variant
   *kernels* live in the library and are reached through dispatch; the
   harness measures the public entry point.
5. Skip, don't fail: unsupported format/ISA on the running machine emits a
   `CaseDecl` with `skip_reason` set.

Every thunk must route its output through `qcb::do_not_optimize()` so the
compiler cannot delete the measured work.

## Measurement Notes

Timing uses `std::chrono::steady_clock` with the discipline documented in
`perf/perf.md`: warmup by call count and wall time, then `iters` samples of
adaptively batched calls sized to `min_sample_ms` (median/p20/p80/CV
reported). Runs record thread, affinity, and frequency policy in `run.json`
as required by umbrella `docs/benchmarking.md`. Recorded evidence goes to
`perf/optimization_status.md` / `perf/baseline_status.md` per the repo
performance gate.
