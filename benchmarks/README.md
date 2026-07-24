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

- `mem_triad` — STREAM-triad bandwidth probe (`a[i] = b[i] + s*c[i]`, f32)
  using the configured `--threads` count over a working-set ladder from 96 KiB
  to 768 MiB,
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
- `qgemv_formats` — q4_0, q4_k, q5_k, q6_k, iq4_nl, iq4_xs, iq2_xxs,
  iq2_xs, iq3_xxs, and iq1_s weight-only GEMV plus q4_0/q8_0 W8A8 GEMV.
  The non-Q8 paths use the registry N4096 K4096 decode shape (N1024 for the
  larger block-decode format sweep) and independent element-decode oracles.
- `rms_norm` — public f32 RMSNorm over `decode_small` shapes plus an R512
  throughput stress shape, checked against a float64 oracle.
- `contract_ops` — portable f32 softmax, causal attention, MoE top-k routing,
  Mamba selective scan, and q8_0 quantized GEMM. Each target has an independent
  scalar or decomposed baseline and a correctness gate; these are
  representative performance paths for the sibling-port batch, not an
  ISA-tuning claim.
- `ported_ops` — fused RMSNorm-add plus dynamic group-int8 quantization,
  checked against the same public operations composed with preallocated
  intermediate storage. This records focused evidence for the extended port
  batch without asserting an optimized performance tier.
- `colibri_ops` — row-scaled W8A32, prequantized IDOT, INT4 f32/W4A8,
  heap top-p, threshold top-k, and expert-union quantized MoE. Each case
  compares the ported CPU route with the scalar, full-sort, or per-row
  formulation it replaces.
- `prerequisites` — Q4_0, Q4_K, Q6_K, and IQ4_XS QGEMM through a
  runtime-selected CPU row panel and warmed caller-owned workspace, checked
  and timed against canonical-layout QGEMM. Both M16 prefill and the Q4_0 M128
  reuse case are covered; weight preparation is outside the timed region.
- `quant_lifecycle` — importance-aware IQ authoring, canonical Q8_0/Q8_1/Q8_K
  activation packing, and the quantized-activation GEMV contract route. Unlike
  reusable-weight compute cases, conversion is deliberately inside timing.
- `quant_import` — canonical INT4/U4/INT8/FP8/FP4/MX/NV/BitNet conversion,
  AWQ/GPTQ/SmoothQuant checkpoint normalization, and canonical-to-CPU panel
  preparation. Conversion and preparation are deliberately inside timing.
- `base_q` — direct packed BaseQ2/3/4/5/6/8 GEMV at M=1 and tiled GEMM at
  M=16, using BF16 affine group metadata and FP32 activation/output storage.
  Every case checks against independently materialized weights and an FP64
  projection oracle; the same-run dense comparator excludes dequantization.
- `quant_gate_up` — paired canonical gate/up projection for representative
  INT4, FP8, MXFP4, and BitNet weights at M=1/16/128. The target owns both
  traversals under one CPU schedule and is checked against independent FP64
  oracles; the comparator makes two prepared projection calls.
- `quant_swiglu` — the same paired projection with direct FP32 SwiGLU output,
  compared with paired gate/up output plus a separate activation pass.
- `quant_swiglu_quant` — direct group-A8 packing from paired SwiGLU at
  M=1/16/128. The same-binary comparator runs the fused typed-output SwiGLU
  kernel and standalone canonical quantizer; correctness uses decoded
  canonical output. The broader A4/A8/FP8/MX/NV matrix is exercised by
  `test_quant_projection_matrix`.
- `quant_qkv` — unequal Nq/Nk/Nv canonical Q/K/V projection at M=1/16/128
  for representative INT4, FP8, MXFP4, and BitNet weights. FP32 cases plus
  direct FP16/BF16 activation-storage representatives compare one CPU schedule
  with three public prepared projection calls; focused correctness uses three
  independent FP64 oracles across all 11 layouts.
- `quant_qkv_rope_kv` — single-token canonical Q/K/V projection with
  split-half RoPE and direct cache-slot insertion for representative INT4,
  FP8, MXFP4, and BitNet weights. FP32 plus direct FP16/BF16 activation cases
  compare against F4 QKV followed by explicit RoPE/cache writes. The focused
  matrix covers every canonical layout and mixed FP16/BF16 cache storage.
- `quant_norm` — fused RMSNorm/LayerNorm residual add with direct canonical
  INT4/INT8/FP4/FP8/MXFP4/MXFP8/NVFP4 activation packing. The same-binary
  comparator materializes a preallocated normalized tensor before calling the
  standalone canonical quantizer; focused correctness also covers independent
  FP16/BF16 and mixed storage boundaries.
- `quant_embedding` — selected-row canonical embedding gather and weighted
  embedding-bag reduction for representative INT4, FP8, MXFP4, and BitNet
  tables. The same-binary comparator dequantizes the complete table first;
  the target decodes only selected rows and stores direct FP32/FP16/BF16.
- `quant_lm_head` — prepared canonical argmax, top-k, exact top-p, packed-mask,
  sparse-candidate, and beam selection for representative INT4, FP8, MXFP4,
  and BitNet heads. Selection state is streamed with deterministic tie rules;
  exact top-p retains one vocabulary row, never rows-by-vocabulary logits.
- `quant_moe` — prepared canonical expert projection, dual-quantized
  activation projection, fused SwiGLU, and SwiGLU with direct next-activation
  packing for sorted or unsorted token routes. Comprehensive cases cover
  INT4, FP8 E4M3, MXFP4/MXFP8, and NVFP4 with eight experts. Comparators use
  one prepared GEMV per token, dequantize activations once before grouped
  projection, or materialize fused SwiGLU before standalone quantization.
- `quant_cache_attention` — typed E4M3FN/E5M2, canonical MXFP8, TurboQuant,
  and BitNet a4.8 KV3 cache codecs plus direct online paged attention.
  TurboQuant cases cover 2/4/8-bit packed K and centroid-coded rotated V;
  KV3 cases cover signed symmetric FP16-scale and unsigned affine FP32-scale
  streams. Attention is compared with full FP32 cache materialization; typed
  cache I/O is compared with explicit FP32 staging and conversion.
- `q8_kv` — Metal-compatible Q8_0 cache scatter/gather with signed-int8 codes
  and one raw FP16 scale per 32 values, functional cache-block copy, and
  direct compressed-cache paged attention. Codec/copy checks use independent
  equal-work contract oracles; attention compares with complete FP32 cache
  materialization.
- `rotary_extended` — explicit-position partial RoPE, three-axis M-RoPE, and
  fused positioned Q/K RMSNorm+RoPE over packed QKV. Cases cover split and
  adjacent layouts, shared/per-batch positions, unrotated tails, and both
  ordinary and multimodal fused paths.
- `gdn` — functional varlen GatedDeltaNet recurrence and short convolution,
  mixed-projection QKV preparation, decay/beta preparation, gated RMSNorm, and
  split sigmoid/value multiplication. Every case has an independent scalar
  contract baseline; quick shapes are measured at one and 16 threads.
- `lora` — direct F16-adapter LoRA with FP16 rounding after both low-rank
  projections and fused scale/base addition. Quick cases cover M=1/4/8/64 at
  K=N=4096, R=16 using BF16 activation/base/output storage.
- `basert_aux` — BF16 per-channel calibration absmax with optional running
  state, scalar-bounds value clipping, and Gemma final-logit softcap over
  BaseRT vocabulary shapes.
- `basert_embedding` — BF16 token/type gather-add and arbitrary-mask terminal
  mean-pool+RMSNorm+L2 normalization.
- `basert_vision` — BF16 NHWC/NTHWC patch extraction and projection,
  learned/factorized positions, regular/coordinate pooling, and Gemma/Qwen
  2-D vision RoPE.
- `basert_audio` — BF16 NWC Whisper-style general convolution,
  symmetric/causal Conformer depthwise convolution, cross-attention, and
  learned relative attention.
- `llama_parity` — convolution, recurrent GLA, DSV4 hyper-connections, and the
  complete unary/GLU selectors (softplus and OpenAI SwiGLU throughput cases)
  from the exhaustive llama.cpp CPU operation audit.
- `optimization_plan` — focused before/after cases for blocked dense GEMM,
  fused Q4_0 SwiGLU projection, fused QKV+RoPE+KV insertion, streaming
  quantized LM-head argmax, radix-2 FFT convolution, recurrent Mamba2,
  grouped Q4_0 MoE SwiGLU, and online paged attention. Each baseline is the
  replaced scalar, materialized, or asymptotically slower formulation in the
  same binary.
- `float_storage` — FP16 and BF16 storage round trips plus typed BF16 softmax.
  Scalar, manual-unroll, explicit-staging, threaded, and runtime-gated native
  conversion routes provide the three-pass dtype optimization evidence.
- `quant_formats`, `quant_activation`, `quant_gemv_matrix`,
  `quant_gemm_matrix`, `quant_fusions`, `quant_gate_up`, `quant_swiglu`,
  `quant_swiglu_quant`, `quant_qkv`, `quant_qkv_rope_kv`, `quant_norm`,
  `quant_embedding`, `quant_lm_head`, `quant_moe`, `quant_cache_attention`, `quant_serving`,
  `quant_kv`, and `bitnet_matrix` —
  full-matrix milestone families. Most entries still relabel
  M0 checked references; `quant_gemv_matrix` and `quant_gemm_matrix` also
  contain M2 canonical prepared-panel cases at M=1/16/128 with a same-binary
  predecoded scalar baseline. M=1 covers all 11 non-cache canonical layouts;
  M=1/16/128 also cover ten direct packed weight/activation pairs spanning
  W4A4/W4A8/W8A8, FP8/MX/NV, and BitNet A8/A4 with an independent dual-decode
  oracle and same-binary dequantized scalar GEMV baseline. M=1 and M=16 include
  direct FP16 and BF16 activation-storage cases for all 11 layouts; M=128 keeps
  four INT4/FP8/MXFP4/BitNet representatives. Every typed case is checked
  against a rounded-input FP64 oracle. Comprehensive multi-row runs retain four
  representatives to bound suite time. `quant_fusions` additionally contains
  M1/16/128 canonical bias+SiLU/ReLU2 epilogues for INT4, FP8, MXFP4, and
  BitNet, including direct FP16/BF16 output representatives. Its same-binary
  baseline is projection followed by a preallocated post-op pass. F2 gate/up
  cases cover the same M ladder with a two-prepared-call comparator. F3 direct
  SwiGLU and activation-quantization cases cover the same representative
  weights without full gate/up or floating-point quantization intermediates.
  F4 Q/K/V cases use unequal head rows, mixed direct storage, one scheduling
  region, and selected shared activation traversal; RoPE/KV insertion remains
  a separate F5 family. F5 projects directly into Q and the chosen ordinary
  floating cache slot, with no separate raw Q/K/V staging allocation;
  compressed cache formats remain M4.
  F6/F7 norm cases pack canonical activations from row-local normalized
  scratch and retain only per-row statistics plus bounded worker storage.
  A registered placeholder is not by itself a support or speedup claim.

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
