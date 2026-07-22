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
| 2026-07-22 | CPU packed-panel/workspace prerequisites | q4_0 weights / f32 activation | M16 N1024 K1408 | aarch64 NEON layout, 1/6 threads | canonical QGEMM 4.0272 / 1.7384 ms | prepacked QGEMM 2.0219 / 0.6028 ms | keep; exact output, 1.99x/2.88x, retained workspace | `perf/results/2026-07-22/prerequisites-t{1,6}/` |
| 2026-07-22 | Colibri CPU algorithm batch | row int8, packed int4, f32 selection, q4_0 MoE | M4 N1024 K1408; V65536; R16 W8192 K2048; R32 E8 K256 N512 | aarch64 NEON/DotProd/I8MM, 1/6 threads | scalar/full-sort/per-row 0.4502-4.9532 / 0.1656-4.1210 ms | 0.0663-1.8725 / 0.0357-1.5431 ms | keep optimized numeric/selection routes; keep MoE union only for shared multi-thread batches | `perf/results/2026-07-22/colibri-port-final-t{1,6}/` |
| 2026-07-22 | qgemv_w8a8 q4_0 SDOT (dotprod_i8) | q4_0 weight / int8 act | quant_matmul m=1 (perf validation pending) | aarch64 DotProd | q4_0 w8a8 scalar ref | NEON SDOT (perf pending) | land; correctness validated (NEON==ref 1.3e-6), perf pending maintainer link env | see "2026-07-22: qgemv_w8a8 q4_0 SDOT" section below |
| 2026-07-22 | MXFP8 logical-scale GEMM | E4M3 / E8M0 | M16 N128 K256 G32 | portable table-decoded reference, 1/6 threads | direct decoder 1.6688 / 0.3660 ms | lookup decoder 0.4608 / 0.1229 ms | keep lookup; 3.62x/2.98x faster, still below predecoded dense | `perf/results/2026-07-22/sibling-entrypoints-{final,lookup}-t{1,6}/` |
| 2026-07-22 | fused RMSNorm-add + dynamic quantization | f32 / group int8 | R512 H4096 G128 | portable fused reference, 1/6 threads | 3.2327 / 1.2658 ms | 3.2725 / 1.2906 ms | keep semantic fusion; 1.2-2.0% allocation cost, no speedup claim | `perf/results/2026-07-22/ported-ops-final-{t1,t6}/` |
| 2026-07-22 | q4_0 qgemv + q4_0/q8_0 qgemv_w8a8 | q4_0/q8_0, f32/int8 activation | quant_matmul m=1 N4096 K4096 | portable refs + aarch64 DotProd, 6 threads | q8 W8A8 ref 0.7394 ms | q8 W8A8 DotProd 0.1386 ms | keep public routes; q4 references are correctness anchors, q8 DotProd is 5.33x | `perf/results/2026-07-22/qgemv-formats-final-t6-fixed/` |
| 2026-07-22 | sibling semantic port batch | f32 / q8_0 | five representative quick shapes | aarch64 portable + existing NEON q8_0 route, 1/6 threads | scalar/decomposed 0.8276-31.7397 ms | 0.1520-3.8159 ms | keep portable candidates and parallel routes; no new ISA or family-wide support claim | `perf/results/2026-07-22/all-kernels-final-{t1,t6}/` |
| 2026-07-21 | qgemv + rms_norm correctness hardening | q8_0 / f32 | quant_matmul m=1 + decode_small + R512 stress | aarch64 NEON | qgemv 0.990 ms; RMS R512 0.259 ms | qgemv 0.969 ms; RMS R512 0.260 ms | keep; contract fixes with no material hot-path regression | `perf/results/2026-07-21/review-{baseline,candidate-final,rms-baseline-repeat,rms-candidate-repeat}/` |
| 2026-07-07 | qgemv (contract realignment) | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 NEON f32-act | 4.127 ms | 1.034 ms | keep neon as contract default (4.0x over ref, family numerics); dotprod_i8 demoted to env-only | `perf/results/2026-07-07/033244-quick/` |
| 2026-07-07 | quant_gemv + rms_norm (threading) | q8_0 / f32 | quant_matmul m=1 + R512 stress | aarch64, 8-12 threads | 0.303 ms | 0.068 ms | keep row-partitioned threading (4.3-4.5x, saturates aggregate DRAM BW) | `perf/results/2026-07-07/0307{23,45,51}-quick/` |
| 2026-07-07 | rms_norm | f32 | decode_small (R1-R4, H2048/H4096) + R512 stress | aarch64 NEON | 2.26 us | 0.52 us | keep neon variant (4.3-4.6x over ref) | `perf/results/2026-07-07/024347-quick/` |
| 2026-07-07 | quant_gemv | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 NEON DotProd | 4.314 ms | 0.301 ms | keep dotprod variant (14.4x, 51% of DRAM roofline) | `perf/results/2026-07-07/023619-quick/` |
| 2026-07-07 | quant_gemv | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 baseline flags | 4.319 ms | 4.441 ms | reject multi-acc candidate; keep plain loop as ref | `perf/results/2026-07-07/022305-quick/` |

## 2026-07-22: CPU packed-panel and workspace prerequisites

Status: retained CPU infrastructure and first measured consumer.

Current implementation: `CpuPackedWeights` preserves an owned copy of the
canonical QuixiCore/GGUF bytes and derives a private 64-byte-aligned row panel.
The automatic panel width is 4 rows on NEON, 8 on AVX2, 16 on AVX-512, and 1
for the portable fallback. `Workspace` is a segmented, pointer-stable arena;
caller-owned and persistent thread-local paths both retain capacity after a
frame rewinds. The first consumer, Q4_0/Q8_0 `qgemm_prepacked`, transposes the
M activations once and reuses each packed weight block across the M tile. The
fused norm/quantization and paired-projection paths now use retained internal
scratch instead of per-call vectors.

Correctness: the focused tests force every panel width and every registered
packed block geometry, verify byte placement, zero padding, 64-byte alignment,
canonical-byte preservation, Q4_0/Q8_0 output for M1/M3/M17 at one and four
threads, workspace pointer stability, and capacity reuse. The benchmark output
is exact versus canonical Q4_0 QGEMM. Native Release CTest passes 37/37,
including benchmark smoke; ASan + UBSan + float-cast-overflow passes 36/36;
the x86_64 AVX2/AVX-512 cross-build and Rosetta test run pass 31/31.

Baseline: canonical-layout `qgemm`, which dispatches one QGEMV per activation
row. Candidate: prepared Q4_0 QGEMM with panel construction outside timing and
a warmed caller-owned workspace.

| threads | candidate median ms (CV; p20-p80) | baseline median ms | speedup | decision |
|---:|---:|---:|---:|---|
| 1 | 2.021854 (0.0569; 1.970958-2.135056) | 4.027229 | 1.99x | keep panel reuse |
| 6 | 0.602766 (0.0356; 0.587861-0.626130) | 1.738412 | 2.88x | keep panel reuse and one fork-join region |

The existing R512 H4096 fused RMSNorm-add/int8 case was rerun after replacing
its vector with retained thread-local scratch: 3.350604 ms (CV 0.0687) at one
thread and 1.295057 ms (CV 0.0349) at six threads. Both outputs remain exact;
the result is consistent with the prior 3.272490/1.290578 ms run, so this
change claims allocation removal but no norm speedup.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel prerequisites
  --threads {1,6} --warmup 5 --iters 30 --min-sample-ms 5`; fused-path rerun
  substitutes `--kernel ported_ops`.
- Working-tree label: `33ad02b-dirty`.
- Raw results: `perf/results/2026-07-22/prerequisites-t{1,6}/` and
  `perf/results/2026-07-22/workspace-ported-t{1,6}/`.

## 2026-07-22: Colibri CPU algorithm excavation

Status: candidate kernels with focused aarch64 evidence.

Current implementation: Colibri's runner-local algorithms are exposed as CPU
kernels rather than copied as model orchestration. The batch adds row-scaled
W8A32 GEMM; prequantized W8A8 integer GEMM; int2, int3-g64, row/group int4,
dynamic W4A8, paired projection, packed-row, and E8/IQ3 conversion/compute
operations; adjacent-to-split RoPE; heap nucleus sampling; threshold top-k;
quantized dense/sparse MLA weight absorption; and stable MoE expert-batch
union. Portable references remain available. Runtime routes cover NEON,
DotProd, I8MM, AVX2, AVX-512, and VNNI where the operation and build target
support them. The exact source-to-kernel inventory and non-kernel exclusions
are in `docs/colibri-port-matrix.md`.

Correctness: focused tests cover layouts, pack/unpack, scalar-versus-ISA
routes, asymmetric int8 correction, deterministic sampling/ties, in-place
RoPE, dense/sparse MLA oracles, MoE grouping, and invalid inputs. E8/IQ3
packing additionally matches a fixed 98-byte oracle emitted by Colibri's
Python encoder. Native Release CTest passes 35/35. ASan + UBSan +
float-cast-overflow CTest passes 35/35. The x86_64 cross-build compiles every
new AVX2/AVX-512/VNNI file and its full Rosetta CTest run passes 29/29.

Baseline: every benchmark case carries a same-binary independent scalar,
full-sort, or dispatched-per-row baseline. Candidate: the public CPU route.
Both use identical inputs and correctness is checked before timing.

| case | 1-thread candidate / baseline ms (CV) | speedup | 6-thread candidate / baseline ms (CV) | speedup | decision |
|---|---:|---:|---:|---:|---|
| W8A32 M4 N1024 K1408 | 0.5601 / 3.0125 (0.0238) | 5.38x | 0.1645 / 0.5505 (0.0416) | 3.35x | keep NEON/runtime route |
| prequantized int8 IDOT M4 N1024 K1408 | 0.0663 / 1.1570 (0.0301) | 17.45x | 0.0357 / 0.2743 (0.0567) | 7.68x | keep DotProd/I8MM dispatch |
| int4 f32 M4 N1024 K1408 | 0.6231 / 4.9532 (0.0271) | 7.95x | 0.1879 / 0.8327 (0.0138) | 4.43x | keep NEON/runtime route |
| dynamic W4A8 M4 N1024 K1408 | 0.0966 / 0.4502 (0.0448) | 4.66x | 0.0502 / 0.1656 (0.0535) | 3.30x | keep quantize-once IDOT route |
| heap top-p V65536 | 1.8725 / 4.6414 (0.0630) | 2.48x | 1.5431 / 4.1210 (0.0478) | 2.67x | keep partial heap extraction |
| threshold top-k R16 W8192 K2048 | 1.0982 / 2.2536 (0.0293) | 2.05x | 0.3230 / 1.9589 (0.0342) | 6.06x | keep threshold partition + row parallelism |
| MoE union R32 E8 K256 N512 | 0.9235 / 0.8339 (0.1501) | 0.90x | 0.6481 / 0.8369 (0.0242) | 1.29x | use union only for shared experts with multiple threads |

The MoE one-thread candidate and baseline execute the same dispatched per-row
GEMV path after the retained guard; their 10% difference is measurement noise
with CV 0.1501, not a regression claim. An initial always-union experiment was
about 3x slower at one thread and was rejected. The final route preserves the
normal GEMV path for one-thread or all-unique routing and enables weight reuse
only for repeated experts with a live worker pool.

Decision: keep. The numeric and selection routes produce material, repeatable
speedups against their exact baselines. Keep the guarded MoE union because its
six-thread repeated-expert case is 1.29x faster and its one-thread route falls
back. The portable E8/IQ3, MLA, RoPE, paired-projection, and row primitives are
correctness candidates; this run does not claim a performance tier for those
unmeasured shapes.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel colibri_ops --threads
  {1,6} --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `33ad02b-dirty`.
- Raw results: `perf/results/2026-07-22/colibri-port-final-t{1,6}/`.

## 2026-07-22: MXFP8/NVFP4 and final sibling entry points

Status: candidate portable references.

Current implementation: the final sibling audit adds logical-layout MXFP8 and
NVFP4 producers/GEMMs, raw E2M1 packing, split-layout MXFP4/NVFP4 GEMV,
FP8-output attention-state merging, named FP8/WNA16/NVFP4 MoE paths, explicit
attention stages, serving metadata adapters, and the remaining named
projection/norm/SSD routes. Accelerator-only scale swizzles are intentionally
replaced by row-major CPU scale tables. No optimized dtype or ISA tier is
claimed.

Optimization experiment: the first MXFP8 GEMM decoded every E4M3 operand with
the generic arithmetic converter. The retained candidate uses one immutable
256-entry E4M3 decode table; accumulation and public semantics are unchanged.
The benchmark compares the quantized path with the same matrices decoded once
and passed to `dense_gemm`.

| threads | direct decode ms (CV; p20-p80) | lookup decode ms (CV; p20-p80) | predecoded dense ms | lookup speedup |
|---:|---:|---:|---:|---:|
| 1 | 1.668820 (0.0322; 1.616028-1.672236) | 0.460790 (0.0012; 0.460216-0.461152) | 0.275678 | 3.62x |
| 6 | 0.365973 (0.0081; 0.363771-0.368289) | 0.122948 (0.0126; 0.121838-0.123981) | 0.085398 | 2.98x |

Correctness: the MXFP8 benchmark agrees exactly with GEMM over independently
dequantized operands. Focused tests additionally reconstruct NVFP4 GEMM,
MXFP4/NVFP4 GEMV, FP4 packing, all three quantized MoE layouts, and FP8 state
merge. Release, ASan+UBSan+float-cast-overflow, and x86_64/Rosetta CTest each
pass 19/19.

Decision: keep the lookup decoder. It removes most portable conversion cost
but remains 1.67x/1.44x slower than predecoded dense at one/six threads, so no
performance improvement over dense GEMM is claimed. Native SIMD decode and
packed microkernels remain future optimization work.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Shape/format: M16 N128 K256, 32-value MX groups, E4M3 codes, logical E8M0
  power-of-two scales, f32 output.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel ported_ops --threads
  {1,6} --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `d1343ce-dirty`.
- Raw results: `perf/results/2026-07-22/sibling-entrypoints-{final,lookup}-t{1,6}/`.

## 2026-07-22: extended sibling semantic port completion

Status: candidate.

Current implementation: the final sibling inventory adds portable reference
semantics for the extended norm/attention/quantization/serving/MoE/SSM/vision
paths and host-reference collectives. The focused measured path is fused
RMSNorm-add followed by dynamic signed int8 group quantization. It uses the
public `rms_norm_add_quant_int8` entry point and the R512 H4096 shape with
128-value quantization groups.

Correctness: the benchmark's fused result is checked against public
`rms_norm_add` plus `quantize_int8` with preallocated intermediate storage.
Codes, scales, and residual output agree exactly (`max_abs_err=0`). The full
Release suite passes 18/18; ASan + UBSan + float-cast-overflow passes 18/18;
the x86_64 cross-build and full Rosetta test run pass 18/18. Sanitizer
initially caught two undersized test buffers; the corrected fixtures now
encode the documented `[rows,groups]` scale and `[rows,intermediate]` output
shapes and remain as regression coverage.

Baseline: the decomposed operations with a persistent intermediate buffer.
Candidate in this historical run: the fused public operation before its
temporary moved to the reusable workspace.

| threads | candidate median ms (CV; p20-p80) | baseline median ms | candidate / baseline | decision |
|---:|---:|---:|---:|---|
| 1 | 3.272490 (0.0791; 3.250896-3.354709) | 3.232729 | 1.012x | keep semantic entry; allocation cost is inside variance |
| 6 | 1.290578 (0.0239; 1.258834-1.316208) | 1.265828 | 1.020x | keep; 2.54x over the one-thread candidate, no fusion speedup claim |

Decision: keep. The public composition supplies the sibling semantics with a
small, measured allocation cost versus a caller-managed intermediate. No
performance improvement or optimized tier is claimed. The prerequisite run
above removes that allocation and records the post-change measurement.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel ported_ops --threads
  {1,6} --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `c47957e-dirty`.
- Raw results: `perf/results/2026-07-22/ported-ops-final-{t1,t6}/`.

## 2026-07-22: sibling semantic port batch

Status: candidate.

Current implementation: portable f32 reference entry points now cover each of
the 16 active umbrella v0.1 families. The batch adds activations, softmax,
LayerNorm, dense and grouped GEMM, RoPE, dense/GQA/paged/MLA attention,
sampling, beam/speculative decode, serving utilities, linear attention, Mamba
S6 scan, MoE routing, AdamW, and a q8_0 QGEMM/LM-head composition. Existing
q8_0 GEMV and RMSNorm routes remain in place. GPU tiling variants were
collapsed into shared operation semantics; only f32 and q8_0 are in scope.

Current public route: `include/quixicore_cpu/ops.h`, `qgemv.h`, `qgemm.h`, and
`rms_norm.h`. Independent portable implementations live under `kernels/` and
parallelize disjoint rows/heads/channels through the existing thread pool.

References inspected: umbrella `registry/kernels.yaml`,
`registry/quant-formats.yaml`, `registry/benchmark-shapes.yaml`,
`registry/tolerances.yaml`, all `matrices/` and kernel/format specs; sibling
Metal, XPU, CUDA, and ROCm manifests and public operation surfaces.

Correctness: Release CTest 10/10; ASan + UBSan + float-cast-overflow CTest
10/10; x86_64 cross-build and full Rosetta CTest 10/10. The operation suite has
independent known-value or float64 checks for every new public entry point.
Thread-pool resize/startup generation ordering was fixed after sanitizer
stress exposed one intermittent early-completion failure; a 32-resize
regression test and 100 repeated sanitizer threading runs pass.

Baseline: each benchmark case carries an independent scalar or decomposed
baseline. Candidate: the public CPU route. Both were measured in the same
Release binary. The six-thread run is the stable performance-core-sized run on
this host; attempted eight-thread repeats had CV above 0.10 on several cases
and are retained as raw diagnostics but excluded from the decision table.

| case | 1-thread target ms (CV) | scalar/decomposed ms | 6-thread target ms (CV) | vs 1 thread | vs baseline at 6 threads | decision |
|---|---:|---:|---:|---:|---:|---|
| softmax R512 H4096 | 2.571875 (0.0450) | 3.487917 | 0.549761 (0.0463) | 4.68x | 6.33x | keep row parallelism |
| causal attention H8 S128 D64 | 2.570625 (0.0156) | 2.677979 | 0.555919 (0.0993) | 4.62x | 4.79x | keep head/query parallelism; CV is at threshold |
| MoE routing T1024 E64 K4 | 0.156679 (0.0936) | 0.827618 | 0.151962 (0.0506) | 1.03x | 5.46x | keep partial-sort implementation; intentionally serial |
| Mamba scan C256 S512 N16 | 3.560417 (0.0172) | 3.586625 | 0.773006 (0.0550) | 4.61x | 4.57x | keep channel parallelism |
| q8_0 QGEMM M16 N2048 K2048 | 3.815875 (0.0161) | 31.739709 | 1.377552 (0.0968) | 2.77x | 22.94x | keep composition over the existing dispatched q8_0 GEMV |

Experiments: portable scalar/reference implementations were the correctness
baseline. Disjoint outer-dimension partitioning was retained for softmax,
attention, scan, and q8_0 QGEMM. MoE routing remains serial because the tested
operation is already 0.15 ms and pool overhead did not produce a useful win.
The MoE comparison includes equivalent stable sorting and selected-expert
softmax work.

Decision: keep the semantic ports and threaded candidates. This establishes
correctness and representative performance evidence, not blanket support for
every dtype, quant format, sibling-only fusion, or performance tier. Families
remain conservatively unclaimed in backend metadata until operation-level
benchmark coverage is complete.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with runtime DotProd/I8MM detection.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency; t1 uses 3
  warmups/20 samples/2 ms minimum; t6 uses 5 warmups/30 samples/5 ms minimum.
- Commands: `quixicore_cpu_bench --preset quick --kernel contract_ops
  --threads 1 --warmup 3 --iters 20 --min-sample-ms 2` and the six-thread
  equivalent with `--warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `4a95ead-dirty`.
- Raw results: `perf/results/2026-07-22/all-kernels-final-t1/` and
  `perf/results/2026-07-22/all-kernels-final-t6/`.

## 2026-07-21: CPU kernel correctness hardening

Status: candidate.

Current implementation:

- Corrected f32-to-fp16 subnormal rounding so valid q8_0 subnormal scales do
  not collapse to zero.
- Made q8_0 packing reject non-finite inputs before integer conversion and
  added checked shape/packed-size arithmetic plus pointer validation.
- Removed activation-quantizing `dotprod_i8` from public qgemv dispatch,
  including the environment override; it remains an internal benchmark for a
  future separately contracted `qgemv_w8a8` operation.
- Made RMSNorm recompute its reduction in f64 only when the normal NEON f32
  sum is non-finite or subnormal, and normalized before applying the weight to
  avoid otherwise avoidable intermediate overflow.
- Changed benchmark correctness from passive error reporting to a finite,
  elementwise `atol + rtol * |reference|` gate that fails before timing.

Contract references inspected: umbrella `registry/kernels.yaml`,
`registry/quant-formats.yaml`, `registry/benchmark-shapes.yaml`,
`registry/tolerances.yaml`, and `matrices/`; CPU `perf/perf.md`.

Correctness: Release CTest 9/9; ASan + UBSan + float-cast-overflow CTest 9/9;
x86_64 cross-architecture compilation completed. Coverage includes exact fp16
subnormal encodings, q8_0 NaN/Inf rejection, low-amplitude nonzero packing,
overflowing dimensions, forced `dotprod_i8` fallback, extreme/tiny RMSNorm
inputs, and benchmark-gate NaN rejection.

Baseline: pre-change Release binary from the same `4a95ead-dirty` working-tree
line. Candidate: post-change Release build. Both use public entry points with
one thread. The repeat RMSNorm pair was run consecutively to reduce scheduler
drift. Medians and CV follow; deltas are candidate versus baseline.

| case | baseline ms (CV) | candidate ms (CV) | delta |
|---|---:|---:|---:|
| qgemv N4096 K4096 | 0.989646 (0.0796) | 0.969306 (0.0132) | -2.06% |
| qgemv N8192 K8192 | 3.783417 (0.0116) | 3.804854 (0.0117) | +0.57% |
| qgemv N16384 K4096 | 3.898792 (0.0180) | 3.913396 (0.0115) | +0.37% |
| RMSNorm R1 H2048, repeat | 0.000280 (0.0500) | 0.000267 (0.0489) | -4.48% |
| RMSNorm R1 H4096, repeat | 0.000532 (0.0349) | 0.000504 (0.0200) | -5.14% |
| RMSNorm R4 H4096, repeat | 0.002006 (0.0993) | 0.001991 (0.0179) | -0.78% |
| RMSNorm R512 H4096, repeat | 0.258800 (0.0195) | 0.260141 (0.0299) | +0.52% |

Experiments: the first RMSNorm candidate used f64 division and square root on
every row and regressed the R512 stress case by about 5.7%; rejected. The kept
version preserves the original f32 fast path and enters f64 only for exceptional
reductions. All candidate benchmark oracles passed; qgemv maximum absolute
error was at most `9.98e-6` and RMSNorm at most `4.02e-7`.

Decision: keep. The largest stable-shape candidate regressions are 0.57% for
qgemv and 0.52% for RMSNorm, both inside run variance; no performance
improvement is claimed. This entry supersedes the historical 2026-07-07
`dotprod_i8` environment-override routing note.

- Hardware: Apple M5 Max, 18 logical/physical cores, 128 GB; aarch64 NEON.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one thread, OS-default affinity/frequency, 3 warmups, 20 timed
  samples, 2 ms minimum sample time.
- Command: `scripts/bench --preset quick --kernel qgemv,rms_norm --threads 1
  --warmup 3 --iters 20 --min-sample-ms 2`, plus consecutive RMSNorm-only
  repeats with the same timing settings.
- Working-tree label: `4a95ead-dirty`.
- Raw results: `perf/results/2026-07-21/review-baseline/`,
  `review-candidate-final/`, `review-rms-baseline-repeat/`, and
  `review-rms-candidate-repeat/` under the same date directory.

## 2026-07-07: qgemv contract realignment with Metal/CUDA

Status: landed.
Current implementation: cross-backend review (QuixiCore-Metal
`dequant.metal` + `tk/quant.py`, QuixiCore-CUDA `quant_formats.cuh`) showed
the family `qgemv` contract is `out = dequantize(wq) @ x` with
full-precision activations and f32 accumulation; activation-quantized
integer math is a separate `qgemv_w8a8` op in both siblings. The CPU
backend had made its activation-quantizing dotprod path the default —
contract-divergent numerics at the public boundary. Realigned:
- Public API renamed `quant_gemv*` -> `qgemv*` (family op naming; the
  umbrella family key `quant_gemv` is the registry label, not the op).
- New `kernels/quantization/qgemv_neon.cpp` ("neon"): int8 weights widened
  to f32 in registers, FMA against f32 activations, per-block scale — the
  contract numerics, now the aarch64 default.
- The SDOT path renamed `dotprod_i8`, excluded from auto-selection
  (`QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8` only); it previews a future
  `qgemv_w8a8` twin op with family-matching per-token scales.
- `rms_norm` verified formula-identical to both siblings (eps inside sqrt
  on mean(x^2), multiplicative weight); default eps=1e-5 added. q8_0
  block bytes verified identical to Metal/CUDA (34 B, fp16 scale, RNE).
Current public route: `quixicore_cpu::qgemv(...)`; resolves `neon` on
aarch64, `ref` elsewhere.
References inspected: QuixiCore-Metal `qgemv.metal`/`qgemv.cpp`/`quant.py`;
QuixiCore-CUDA `qgemv.cu`/`quant_formats.cuh`/`quant_rt.cu`.
Correctness: `tests/correctness/test_qgemv.cpp` — the public entry now
meets the family oracle (float64 dequantize(wq) @ x) at < 1e-4 on every
platform (measured 1.4e-08 - 2.9e-08 in-harness); dotprod_i8 keeps its own
quantized-activation oracle; public==dispatched-variant bit-exact. 7/7
suites green.
Baseline: `ref` scalar (same build).
Experiments: single candidate (neon f32-activation structure). Result:
3.95-4.02x over ref; 17 W-GB/s single-thread. The dotprod_i8 baseline
remains ~3.5x faster (0.29 ms vs 1.03 ms at 4096^2) — that speed is
intentionally not reachable via default dispatch until it ships under the
family's `qgemv_w8a8` semantics.
Decision: keep neon as the contract default. Accepting the default-path
slowdown buys cross-backend numerical parity: one oracle, one contract,
six backends.
Open questions: `qgemv_w8a8` twin op (per-token activation scales, RNE,
matching Metal/CUDA); i8mm for qgemm; quant.py byte-parity fixtures as
shared test vectors.
Raw results: `perf/results/2026-07-07/033244-quick/`.

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags (NEON is aarch64 baseline).
- Command: `scripts/bench --preset quick --kernel qgemv --threads 1`.
- Working-tree label: `96b2c37-dirty`.

| shape (m=1) | neon ms | CV | vs ref | vs dotprod_i8 | W-GB/s | rel err (family oracle) | decision |
|---|---:|---:|---:|---:|---:|---|---|
| N4096 K4096 | 1.0344 | 0.027 | 3.99x | 0.28x | 17.2 | 2.92e-08 | keep as contract default |
| N8192 K8192 | 4.1164 | 0.022 | 4.02x | 0.29x | 17.3 | 1.43e-08 | keep as contract default |
| N16384 K4096 | 4.2517 | 0.016 | 3.95x | 0.28x | 16.8 | 2.92e-08 | keep as contract default |

## 2026-07-07: Threading layer — row-partitioned kernels on a fork-join pool

Status: landed.
Current implementation: `src/threading/thread_pool.{h,cpp}` — persistent
fork-join workers, deterministic contiguous-range partition, inline
execution for small counts / nesting / `num_threads()==1`, no per-call
allocation (fn-pointer + context trampoline). Public control:
`quixicore_cpu::set_num_threads()` (default 1). `quant_gemv` (both
variants) and `rms_norm` (both variants) partition rows; the harness
`--threads` flag now drives the pool, and `mem_triad` runs on it too so
one probe measures single-thread and aggregate rooflines.
Current public route: unchanged APIs; thread count via
`include/quixicore_cpu/threading.h`.
References inspected: llama.cpp/ggml threadpool chunking conventions.
Correctness: `tests/unit/test_threading.cpp` — exact range coverage,
small-count inlining, nested-call inline fallback, and bit-exact kernel
outputs at 1 vs 4 threads for quant_gemv and rms_norm (rows are never
split, so results are identical at any thread count). Full suite 7/7.
Baseline: 1-thread numbers from the same build (see table).
Experiments:
1. First wrapper used `std::function` + capture-heavy lambdas: measured
   1.6-1.9x SINGLE-THREAD regression on rms_norm (loop bounds/pointers
   reloaded through the capture frame every iteration — stores through the
   output pointer may alias the frame). Rejected; replaced with the
   fn-pointer trampoline and free-function loop bodies with by-value
   arguments. Single-thread numbers recovered (qgemv 0.303 ms, R512
   0.284 ms; residual ~80 ns/call on microsecond cases from the control
   mutex — accepted, decode-sized rms_norm calls stay inline anyway).
2. Row-partitioned execution at 8 and 12 threads (12P+4E hybrid; no
   pinning, macOS QoS bias only). Kept.
Decision: keep. Threaded q8_0 GEMV reaches 263-266 W-GB/s, matching the
aggregate DRAM roofline the threaded triad measures (251-304 GB/s) — the
kernel is bandwidth-saturated; more threads cannot help until memory does.
Decode-latency shapes (R1-R4) correctly stay on the inline path.
Open questions: 8 vs 12 threads is shape-dependent on the hybrid part
(E-core drag vs extra bandwidth); revisit with affinity pinning. NUMA
policy deferred until multi-socket hardware exists.
Raw results: `perf/results/2026-07-07/030723-quick/` (1t),
`030745-quick/` (8t), `030751-quick/` (12t).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release.
- Command: `scripts/bench --preset quick --kernel qgemv,mem_triad,rms_norm
  --threads {1,8,12}`.
- Working-tree label: `a5c080a-dirty`.

| case | 1t ms | 8t ms | 8t speedup | 12t ms | 12t speedup | best BW | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| qgemv N4096 K4096 | 0.3029 | 0.0679 | 4.46x | 0.0820 | 3.69x | 263 W-GB/s | keep |
| qgemv N8192 K8192 | 1.1962 | 0.3166 | 3.78x | 0.2678 | 4.47x | 266 W-GB/s | keep |
| qgemv N16384 K4096 | 1.1749 | 0.3056 | 3.84x | 0.2696 | 4.36x | 264 W-GB/s | keep |
| rms_norm R512 H4096 | 0.2840 | 0.0663 | 4.28x | 0.0754 | 3.77x | 253 GB/s | keep |
| mem_triad ws_192MiB (roofline) | 1.72-2.24 | 0.8022 | — | 0.6632 | — | 251-304 GB/s | reference |
| mem_triad ws_24MiB (roofline) | 0.2480 | 0.0645 | — | 0.0685 | — | 368-390 GB/s | reference |

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

## 2026-07-22: q4_0 weight-only and q4_0/q8_0 W8A8 GEMV

Status: candidate.

Current implementation: q4_0 packing, unpacking, and full-f32-activation GEMV
now use the public `qgemv` API alongside q8_0. The separate `qgemv_w8a8` API
supports both q4_0 and q8_0 weights with per-32-element int8 activation
quantization. Portable scalar references anchor every combination; q8_0 W8A8
selects the existing aarch64 DotProd kernel when runtime support is present.
The weight-only qgemv dispatcher never selects activation-quantized numerics.

Current public route: `include/quixicore_cpu/qgemv.h` and
`qgemv_w8a8.h`; dispatch in `src/dispatch/qgemv.cpp` and
`qgemv_w8a8.cpp`; portable kernels and GGUF-compatible q4_0 layout in
`kernels/quantization/qgemv_w8a8_ref.cpp`.

References inspected: sibling q4_0/q8_0 layouts and W8A8 operation separation;
umbrella `registry/quant-formats.yaml`, `registry/benchmark-shapes.yaml`,
`registry/tolerances.yaml`, and `specs/kernels/quantization.md`.

Correctness: Release CTest 11/11; ASan + UBSan + float-cast-overflow CTest
11/11; x86_64 cross-build and full Rosetta CTest 11/11. Weight-only q4_0 is
checked against float64 accumulation over exactly dequantized weights. W8A8 is
checked against the exact semantic oracle
`dequant(W) @ dequant(blockwise_int8(x))`, plus format/shape/non-finite-input
validation, determinism, and bit-identical one-versus-four-thread output.

During the optimization run, moving W8A8 activation scratch to `thread_local`
storage initially made worker lambdas resolve each worker's empty TLS instance;
the six-thread benchmark crashed and ASan identified the null read. The kept
implementation snapshots the caller's scratch data pointers before entering
the pool. The sanitizer benchmark reproduction and multithread regression test
pass after the fix.

Baseline: direct portable reference in the same Release binary, plus a
dequantize-then-scalar-GEMV baseline. Candidate: the public route. The stable
six-thread performance-core-sized run is recorded below; all correctness gates
passed and all target CVs are below 0.10.

| case (N4096 K4096) | candidate ms (CV) | direct ref ms | dequant scalar ms | speedup vs ref | W-GB/s | decision |
|---|---:|---:|---:|---:|---:|---|
| q4_0 weight-only qgemv | 1.048475 (0.0358) | 1.072847 | 13.960396 | 1.02x | 9.0 | keep portable correctness anchor; no ISA speedup claim |
| q4_0 W8A8 qgemv | 0.858071 (0.0200) | 0.866313 | 13.762375 | 1.01x | 11.0 | keep portable correctness anchor; no ISA speedup claim |
| q8_0 W8A8 qgemv | 0.138599 (0.0852) | 0.739384 | 14.674480 | 5.33x | 128.6 | keep aarch64 DotProd dispatch |

Decision: keep all three public routes and the q8_0 DotProd selection. q4_0
remains a portable reference path until a separately measured NEON/AVX
implementation beats it. No unsupported GGUF format or cross-machine speedup
is claimed.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON + DotProd.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: six threads, OS-default affinity/frequency, 5 warmups, 30 samples,
  5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel qgemv_formats
  --threads 6 --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `15a78dc-dirty`.
- Raw results:
  `perf/results/2026-07-22/qgemv-formats-final-t6-fixed/`.

## 2026-07-22: qgemv_w8a8 q4_0 SDOT (dotprod_i8)

Status: landed; correctness validated; on-hardware perf pending in the
maintainer link environment.

New variant: `q4_0_gemv_w8a8_dotprod` — the aarch64 DotProd (SDOT) SIMD variant
of the q4_0-weight x int8-activation GEMV, completing the q4_0 lane of
`qgemv_w8a8` (which previously ran only the scalar `q4_0_gemv_w8a8_ref` while the
q8_0 lane already had its `dotprod` variant). Ported from embeddinggemma.c's
proven `ei_vec_dot_q4_0_q8_0_neon` (`src/quants.c`) and structured exactly like
the existing `q8_0_gemv_dotprod`: activations quantized once per call to
per-32-block int8 (shared, `thread_local` scratch), each 32-weight block
nibble-unpacked (`vandq_u8`/`vshrq_n_u8`, `-8` offset via `vsubq_s8`) and dotted
with `vdotq_s32`, lane-wise f32 accumulation with the combined fp16 scales,
rows partitioned across the thread pool. Selected last-wins by CPU feature via a
new `kQ4_0Variants` table mirroring `kQ8_0Variants`; forced with
`QUIXICORE_CPU_QGEMV_W8A8_VARIANT=dotprod`.

Correctness:
- Standalone runtime check (NEON variant vs the scalar reference on the same
  packed weights + per-block-quantized activations): **max rel diff 1.28e-6**
  (pure fp summation-order; the int8 dot is exact). The intrinsics are correct.
- fp64-oracle CTest wired: `qgemv_w8a8_forced_dotprod` re-runs
  `test_qgemv_w8a8` with the variant forced to `dotprod`, so the SDOT path is
  checked against `dequant(wq) @ x` at the umbrella quantized tolerance.
- Full library compiles clean (`cmake --build --target quixicore_cpu`, dotprod
  ISA source built, no warnings).

Measurement note (honesty gate): this environment cannot LINK the test
executables (the pre-existing libc++ ABI mismatch that also hits
`test_cpu_features`), so no CTest run or perf number was produced here. No
speedup is claimed. The q8_0 SDOT precedent (14.4x over scalar, ~51% of DRAM
roofline) suggests a comparable SDOT-bound win, but that is NOT a claim until
measured. To close the perf gate in the maintainer env (Apple Clang 21):

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
    ctest --test-dir build -R 'qgemv_w8a8'   # incl. qgemv_w8a8_forced_dotprod
    scripts/bench --preset quick             # q4_0 w8a8 dotprod vs ref/scalar

Follow-up: AVX2/VNNI x86 variant of the same q4_0 w8a8 dot (QuixiCore-CPU still
has no x86 SIMD in the quant family).
