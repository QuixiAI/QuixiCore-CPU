# Optimization Status

Running notebook of focused optimization runs on CPU kernel paths. Every
kernel implementation, routing change, benchmark change, or performance
claim must add an entry here.

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
| 2026-07-07 | rms_norm | f32 | decode_small (R1-R4, H2048/H4096) + R512 stress | aarch64 NEON | 2.26 us | 0.52 us | keep neon variant (4.3-4.6x over ref) | `perf/results/2026-07-07/024347-quick/` |
| 2026-07-07 | quant_gemv | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 NEON DotProd | 4.314 ms | 0.301 ms | keep dotprod variant (14.4x, 51% of DRAM roofline) | `perf/results/2026-07-07/023619-quick/` |
| 2026-07-07 | quant_gemv | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 baseline flags | 4.319 ms | 4.441 ms | reject multi-acc candidate; keep plain loop as ref | `perf/results/2026-07-07/022305-quick/` |

## 2026-07-07: rms_norm f32 reference + NEON variant

Status: landed.
Current implementation: `kernels/norms/rms_norm_ref.cpp` (scalar, float64
sum-of-squares accumulation — a near-oracle reference) and
`kernels/norms/rms_norm_neon.cpp` (f32 four-accumulator sum of squares +
vectorized scale pass; NEON is baseline on aarch64, no extra build flags).
Current public route: `quixicore_cpu::rms_norm` via
`src/dispatch/rms_norm.cpp`; resolves `neon` on aarch64, `ref` elsewhere;
`QUIXICORE_CPU_RMS_NORM_VARIANT` override.
References inspected: ggml rms_norm (float64-accumulating scalar
reference precedent).
Correctness: `tests/correctness/test_rms_norm.cpp` — float64 oracle at
umbrella fp32 tolerance (public and neon < 1e-5, measured ~2e-7; ref
< 1e-6), shapes with vector tails (H1, H7, H777), zero-row finiteness,
public-vs-variant bit-exact, determinism (norms family policy).
Baseline: `ref` scalar via direct call (the `ref_scalar` bench baseline).
Experiments: single candidate (NEON f32 structure above). Result:
4.34-4.58x over ref on all shapes. The ref pays for scalar float64
accumulation; the NEON f32 pairwise-style sums keep error at ~2e-7 while
vectorizing fully.
Decision: keep. Decode shapes run at 94-97 GB/s (cache-resident, vs
~246 GB/s in-cache triad); R512xH4096 at 63.8 GB/s.
Open questions: batch>1 threading once the thread pool exists; fused
residual-add variant if the contract adds it.
Raw results: `perf/results/2026-07-07/024347-quick/` (git-ignored; table
below).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags throughout (NEON is aarch64 baseline).
- Command: `scripts/bench --preset quick --kernel rms_norm` (warmup 3,
  iters 20, min_sample_ms 2.0, threads 1).
- Working-tree label: `1eb4984-dirty`.

| shape | neon us | CV | vs ref | GB/s | rel err | decision |
|---|---:|---:|---:|---:|---|---|
| R1 H2048 | 0.25 | 0.046 | 4.43x | 97.1 | 7.5e-08 | keep |
| R1 H4096 | 0.52 | 0.056 | 4.34x | 94.4 | 1.3e-07 | keep |
| R4 H4096 | 2.03 | 0.031 | 4.40x | 72.7 | 1.2e-07 | keep |
| R512 H4096 | 263.33 | 0.052 | 4.58x | 63.8 | 2.2e-07 | keep |

## 2026-07-07: quant_gemv q8_0 NEON DotProd variant

Status: landed.
Current implementation: `kernels/quantization/qgemv_dotprod.cpp`
(`q8_0_gemv_dotprod`): activations quantized to int8 blocks once per call
(d = amax/127, round-to-nearest, grow-only thread-local scratch), then per
32-element block two `vdotq_s32` int8x16 dot products accumulated as
`float32x4` lanes with the combined weight*activation scale
(`vmlaq_n_f32`), two independent vector accumulators per row, scalar tail
for odd block counts. Compiled with `-march=armv8.2-a+dotprod` via
`quixicore_cpu_add_isa_sources()`; baseline build untouched.
Current public route: `quixicore_cpu::quant_gemv` dispatch resolves
`dotprod` when `cpu_features().dotprod` is true, else `ref`;
`QUIXICORE_CPU_QGEMV_VARIANT` still forces either.
References inspected: llama.cpp `ggml_vec_dot_q8_0_q8_0` NEON structure
(activation quantization + sdot + per-block scale fmla).
Correctness: `test_quant_gemv` extended — public entry bit-exact vs the
direct variant call; tight float64 oracle over dequantized weights AND
dequantized quantized activations < 1e-5 (int dot is exact; only f32
scale/accumulate rounds); contract oracle vs original f32 weights < 3e-2
on every shape (measured 2.8e-3 - 6.5e-3); bit-exact determinism. The
f32-activation oracle does not apply to this variant (activation
quantization error dominates on tiny outputs) and is asserted only for
`ref`.
Baseline: `ref` scalar via the same public route (previous entry).
Experiments: single candidate (the structure above). Result: 14.35-14.45x
over `ref` on all three shapes; 43.98-44.61x over the decomposed
dequant+sgemv path.
Decision: keep. 59.3-59.4 effective weight-GB/s = 51% of the same
machine's mem_triad single-thread DRAM roofline (~115-117 GB/s); the
remaining gap is per-call activation quantization plus non-streaming
access, to be revisited with i8mm/threading, not with more scalar work.
Open questions: i8mm (smmla) variant; multi-thread row partitioning; x86
VNNI equivalent blocked on hardware.
Raw results: `perf/results/2026-07-07/023619-quick/` (git-ignored; table
below).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release; this
  TU `-march=armv8.2-a+dotprod`, everything else baseline flags.
- Command: `scripts/bench --preset quick --kernel qgemv` (warmup 3,
  iters 20, min_sample_ms 2.0, threads 1).
- Working-tree label: `1092837-dirty`.

| shape (m=1) | dotprod ms | CV | vs ref | vs dequant_sgemv | W-GB/s | GFLOP/s | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| N4096 K4096 | 0.3006 | 0.020 | 14.35x | 43.98x | 59.3 | 111.6 | keep |
| N8192 K8192 | 1.2010 | 0.018 | 14.45x | 44.61x | 59.4 | 111.8 | keep |
| N16384 K4096 | 1.2012 | 0.025 | 14.45x | 44.15x | 59.4 | 111.7 | keep |
| 2026-07-07 | harness (system probes) | f32 | mem_triad ladder + quant_matmul m=1 | aarch64 baseline flags | n/a | n/a | harness validated; no kernel, no speedup claim | `perf/results/2026-07-07/021238-quick/` |

## 2026-07-07: quant_gemv q8_0 scalar reference bring-up

Status: landed.
Current implementation: `kernels/quantization/qgemv_ref.cpp`
(`q8_0_gemv_ref`): plain per-block loop, f32 accumulation, single
accumulator, scale applied once per 32-element block. GGUF-compatible
`q8_0` packing (`34` bytes / 32 weights, fp16 scale + int8).
Current public route: `quixicore_cpu::quant_gemv(QuantFormat::kQ8_0, ...)`
via `src/dispatch/quant_gemv.cpp` (variant `ref`; env override
`QUIXICORE_CPU_QGEMV_VARIANT`).
References inspected: llama.cpp q8_0 block layout and quantization
semantics; QuixiCore-Metal qgemv bench conventions.
Correctness: `tests/correctness/test_quant_gemv.cpp` — fp16 roundtrip,
argument validation, pack/unpack bound (< 1% of amax), float64 oracles
(vs dequantized weights < 1e-4; vs original weights < 3e-2 = umbrella
quantized tolerance), bit-exact determinism. In-harness oracle max rel err
2.4e-07 - 2.6e-07 on the measured shapes.
Baseline: plain single-accumulator loop (shipped as `ref`).
Experiments: candidate = manual 4-way multi-accumulator split (hypothesis:
f32 accumulation chain is latency-bound like `sgemv_naive`). Result: the
candidate measured 1-3% SLOWER on every shape — Apple clang already
auto-vectorizes the plain int8→f32 loop, and the manual split obstructs it.
Decision: reject the multi-accumulator candidate; keep the plain loop as
the reference. The rejected variant is preserved as the `scalar_multiacc`
bench baseline for reproducibility.
Open questions: NEON/dotprod variant is the obvious next step — the ref
runs at ~4.1 weight-GB/s against a ~115 GB/s single-thread DRAM roofline
(mem_triad, same machine), ~28x headroom.
Raw results: `perf/results/2026-07-07/022305-quick/` (git-ignored; table
below). Earlier A/B with candidate as target:
`perf/results/2026-07-07/022143-quick/`.

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags.
- Command: `scripts/bench --preset quick --kernel qgemv` (warmup 3,
  iters 20, min_sample_ms 2.0, threads 1).
- Working-tree label: `cb47aff-dirty`.

| shape (m=1) | ref ms | CV | vs scalar_multiacc | vs dequant_sgemv | W-GB/s | GFLOP/s | rel err | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| N4096 K4096 | 4.3189 | 0.014 | 1.03x | 3.05x | 4.1 | 7.8 | 2.39e-07 | keep plain loop |
| N8192 K8192 | 17.2801 | 0.022 | 1.01x | 3.09x | 4.1 | 7.8 | 2.55e-07 | keep plain loop |
| N16384 K4096 | 17.1908 | 0.011 | 1.03x | 3.08x | 4.1 | 7.8 | 2.39e-07 | keep plain loop |

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

