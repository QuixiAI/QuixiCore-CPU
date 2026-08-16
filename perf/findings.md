# CPU Established Findings — Do Not Re-Derive

Distilled from `perf/optimization_status.md` through 2026-07-25. Treat as
current truth until re-measured; every entry names its date and notebook
entry so it can be challenged with new data.

## Environment anchor

- **Apple M4 Max** — 16 logical cores (12P+4E), 128 GB, macOS 26.5.1, Apple
  Clang 21.0.0, CMake Release, baseline arch flags. Host for the 2026-07-07
  bring-up class: harness validation, q8_0 GEMV ref/DotProd/neon, rms_norm,
  threading layer, and the qgemv contract realignment.
- **Apple M5 Max** — 18 physical/logical cores (6 performance, 12 efficiency),
  128 GB, macOS 26.5.2, Apple Clang 21.0.0. Primary AArch64 host from
  2026-07-21 onward: correctness hardening, all M0-M5 canonical quant work,
  M3 fusions, M4 compressed caches, BaseQN, Q8_0 KV, rotary, GDN, LoRA, and
  BaseRT waves.
- **AMD EPYC 7702** — one socket/NUMA node, 64 physical cores with SMT off,
  256 MiB aggregate L3, 1.0 TiB memory; AVX2/FMA/F16C only (no
  AVX-512/VNNI/AMX); Ubuntu Linux 6.8.0-134-generic, GCC 13.3.0. Native x86
  evidence for the 2026-07-23/24 M2/M3/M4, BaseQN, rotary, Q8_0 KV, GDN,
  LoRA, and first BaseRT entries. AVX2-class findings only.
- **Intel Xeon Gold 6454S (Sapphire Rapids)** — AVX2, AVX-512F/BW/VL/DQ,
  AVX-512 VNNI, AMX tile/int8/bf16 detected; GCC 15.2.0, Linux x86-64.
  Host for the 2026-07-24 BaseRT waves 2-4, the AVX-512/VNNI dispatch
  closure, and the 2026-07-25 INT8_MIN-safe IDOT entry. The only host where
  AVX-512-class findings are measured.

Cross-machine values are never compared as speedups in the notebook; treat
every finding below as specific to the machine class named in its entry.

## Wins

| finding | effect | date | notebook entry |
|---|---|---|---|
| NEON DotProd q8_0 activation-quantized GEMV | 14.35x-14.45x over scalar ref; 59.3-59.4 W-GB/s = 51% of the same machine's 1-thread DRAM roofline | 2026-07-07 | quant_gemv q8_0 NEON DotProd variant |
| rms_norm NEON f32 four-accumulator sum of squares | 4.34x-4.58x over the f64-accumulating scalar ref; 94-97 GB/s on decode shapes | 2026-07-07 | rms_norm f32 reference + NEON variant |
| Row-partitioned fork-join thread pool | 4.28x-4.47x at 8-12 threads; threaded q8_0 GEMV reaches 263-266 W-GB/s, matching the 251-304 GB/s aggregate triad roofline | 2026-07-07 | Threading layer — row-partitioned kernels on a fork-join pool |
| qgemv `neon` f32-activation contract default | 3.95x-4.02x over scalar ref at family numerics (17 W-GB/s) | 2026-07-07 | qgemv contract realignment with Metal/CUDA |
| q8_0 W8A8 DotProd public route | 5.33x over the portable ref at 6 threads; 128.6 W-GB/s | 2026-07-22 | q4_0 weight-only and q4_0/q8_0 W8A8 GEMV |
| Colibri integer numeric routes | int8 IDOT 17.45x/7.68x (t1/t6); W8A32 5.38x/3.35x; int4-f32 7.95x/4.43x; dynamic W4A8 4.66x/3.30x over scalar | 2026-07-22 | Colibri CPU algorithm excavation |
| Heap top-p and threshold top-k selection | 2.48x/2.67x and 2.05x/6.06x (t1/t6) over full-sort baselines | 2026-07-22 | Colibri CPU algorithm excavation |
| Guarded MoE expert-batch union | 1.29x at 6 threads for repeated experts; one-thread/all-unique routing falls back to per-row GEMV | 2026-07-22 | Colibri CPU algorithm excavation |
| Prepacked Q4_0 QGEMM row panels + retained workspace | 1.99x/2.88x (t1/t6) over canonical per-row QGEMV, exact output | 2026-07-22 | CPU packed-panel and workspace prerequisites |
| Native FP16/BF16 conversion routes | FP16 17.60x/38.77x and BF16 3.78x/10.77x (t1/t6) over the scalar pass; typed softmax dispatch is neutral | 2026-07-22 | Log (universal floating storage, three passes) |
| Selector hoist + source-exact direct sigmoid | softplus 2.6758→2.4371 ms; OpenAI SwiGLU 2.4515→1.2074 ms (2.05x) at one thread | 2026-07-22 | exhaustive llama operation and quant lifecycle closure |
| Direct packed NEON block dots, all 23 non-Q4_0/Q8_0 stored GGUF formats | 4.68x-45.22x over the same-binary element decoder at M1 N1024 K4096 | 2026-07-22 | complete llama.cpp stored-quant CPU paths |
| Three-pass K/IQ GEMV refinement | K formats 6.3x-8.7x and IQ formats 4.0x-31.1x over pass 0 (t1) | 2026-07-22 | three-pass optimization sweep |
| Blocked dense GEMM 4x32 output tiles | 8.80x/30.85x (t1/t6) over scalar dense | 2026-07-22 | planned P0-P2 CPU kernel batch |
| Radix-2 FFT convolution and recurrent Mamba2 | FFT 32.85x/29.79x; Mamba 38.45x/53.83x vs direct baselines (106.57x/115.19x vs source expansion) | 2026-07-22 | planned P0-P2 CPU kernel batch / three-pass optimization sweep |
| Online paged attention and online MLA (no score materialization) | paged 1.90x/4.67x; MLA 1.20x/4.12x (t1/t6) over materialized baselines | 2026-07-22 | three-pass optimization sweep |
| Fused RMS-add-int8 row-local NEON route | 5.02x/7.21x (t1/t6) over the decomposed preallocated path | 2026-07-22 | three-pass optimization sweep |
| Direct-bit FP8/MX encoders | E4M3 96.39x, E5M2 68.89x, MXFP8 87.10x, NVFP4 5.32x over the pass-1 linear search | 2026-07-22 | M1 canonical lifecycle and checkpoint ingestion |
| MXFP8 GEMM 256-entry lookup decoder | 3.62x/2.98x (t1/t6) over arithmetic decode; still 1.67x/1.44x slower than predecoded dense | 2026-07-22 | MXFP8/NVFP4 and final sibling entry points |
| K1 canonical weight-only GEMV NEON block dots | pass0→final INT4 8.68x, MXFP4 9.17x, BitNet 4.83x, FP8 2.11x; final 1.10x-5.63x over same-run dequantized scalar | 2026-07-23 | M2 canonical weight-only GEMV block dots (K1) |
| M2 shared-framework 32-row tiles + contiguous output lanes | M16/M128 2.21x-2.53x / 2.48x-2.56x over same-run predecoded scalar GEMM (t1) | 2026-07-23 | M2 canonical universal projection (K1-K15 shared framework) |
| Dual-quant GEMV direct packed-pair block dots | pass0→final 11.22x-21.88x; integer/FP4/BitNet routes 3.78x-5.95x over dequantized scalar | 2026-07-23 | M2 canonical dual-quant GEMV block dots (K4-K14) |
| Dual-quant GEMM decode-once activation/output panels | 2.16x-14.98x over pass 0 at M16; non-FP8 pairs 4.61x-8.64x over dequantized scalar | 2026-07-23 | M2 canonical dual-quant GEMM panels (K5-K14) |
| Typed (FP16/BF16) GEMV conversion-reuse output panels | 1.86x-17.13x over scalar conversion pass 0; 1.11x-5.22x over dequantized scalar | 2026-07-23 | M2 canonical FP16/BF16 GEMV panels (K1-K15) |
| Typed GEMM four-panel activation-reuse groups | 1.16x-2.90x (M16) and 1.43x-2.91x (M128) over pass 0; 2.08x-3.23x over dequantized scalar | 2026-07-23 | M2 canonical FP16/BF16 GEMM panel groups (K1-K3) |
| Dual-FP8 eight-output-panel activation-decode reuse | GEMV 1.93x-2.05x over the per-row FP8 route; all three FP8 pairs now beat dequantized scalar (GEMV 1.09x-1.35x, GEMM M16 1.10x-1.25x) | 2026-07-23 | M2 dual-FP8 GEMV/GEMM output-panel reuse (K10-K12) |
| x86 AVX2/FMA panel FMA + four-panel groups + direct dual dots | weight routes 1.58x-3.69x and dual routes 1.00x-4.49x over the portable pass 1 | 2026-07-23 | M2 x86 canonical M16/M128 projection (K1-K15) |
| x86 AVX2/F16C typed GEMV panel groups | 1.19x-2.34x over pass 0; FP32 E4M3/E5M2 panel traversal 1.31x-1.32x | 2026-07-23 | M2 x86 canonical typed GEMV panels (K1-K15) |
| BaseQN group-metadata hoist + 16-row tile reuse + FP32 specialization | 2.22x-2.58x (M1) and 5.24x-6.13x (M16) over per-element decode | 2026-07-23 | Metal BaseQN semantic port and direct projection |
| F2 paired gate/up: shared traversal + paired M1 row dots | M1 2.06x-10.15x over pass 1 on Apple; stable EPYC M16/M128 1.04x-1.17x over two prepared calls | 2026-07-23 | M3 canonical paired gate/up projection (F2) |
| F3 fused SwiGLU + activation quantization panels | Apple up to 2.188x vs fused SwiGLU + standalone quantizer; FP8 M16/M128 6.9202/53.2569→2.7201/20.5146 ms via 32-row panels | 2026-07-23 | M3 canonical fused SwiGLU and activation quantization (F3) |
| F6/F7 norm-add-quant full-row reuse, threaded | 16-thread 1.41x-5.71x (Apple) and 1.77x-6.52x (EPYC) over preallocated composition; RMS INT4/INT8 t1 1.51x/1.36x | 2026-07-23 | M3 canonical RMSNorm/LayerNorm add-quant (F6/F7) |
| S1/S2 direct selected-row embedding decode | 2.57x-25.69x t1 and 3.86x-83.77x t16 on Apple (removal of full-table dequantization, not equal-work throughput) | 2026-07-23 | M3 canonical embedding and embedding bag (S1/S2) |
| S3/S4 streaming LM head, format-selected traversal | structured routes 3.37x-130.40x; top-k 1.14x-2.96x on Apple; EPYC multi-row top-k 2.04x-3.85x | 2026-07-23 | M3 canonical streaming LM head (S3/S4) |
| S5-S7 grouped MoE, one shared threaded dispatch | Apple FP8 grouping 2.01x-2.35x t1; 16-thread 3.18x-8.72x; EPYC grouped route 3.68x-6.49x t1 and 3.56x-8.30x t16 | 2026-07-23 | M3 canonical grouped MoE weight-only path (S5/S6/S7 partial) |
| S5 dual compact packed-pair dots | compact pairs 1.54x-2.03x over per-token projection on Apple | 2026-07-23 | M3 canonical MoE dual activation and next-layer quantization |
| M4 FP8 direct paged attention (decode tables + scale hoist) | 1.22x-1.78x / 6.47x-8.35x (Apple t1/t16); 3.83x-5.09x / 13.12x-28.99x (EPYC) over FP32 cache materialization | 2026-07-23 | M4 typed FP8 cache and direct online paged attention (A1/A2) |
| M4 MXFP8 direct paged attention | 1.95x-2.10x / 8.32x-15.57x (Apple); 1.53x-1.57x / 4.71x-7.13x (EPYC); pass 2 tables 5.53x-5.71x over scalar decode | 2026-07-23 | M4 canonical MXFP8 cache and online attention (A3/A4) |
| TurboQuant fixed-bit extraction + direct attention | 1.29x-1.54x t1 and 3.00x-3.28x t16 (Apple); 6.28x-7.99x EPYC t16 over materialization | 2026-07-23 | M4 canonical TurboQuant codec and online attention (A5-A8) |
| KV3 24-bit packet reuse | one-thread attention another 1.77x-2.03x over pass 3; 16-thread 2.14x-3.02x over materialization (t1 remains 0.87x-0.97x, no claim) | 2026-07-23 | M4 BitNet a4.8 KV3 codec and online attention (A9/A10) |
| Specialized portable Q4_0 reference + AVX2 dispatch | 7.9x over the generic element reference; AVX2 another 3.33x (EPYC, t1) | 2026-07-23 | M5 portable fallback and GGUF Q4 routing |
| BaseQN LM-head four-row fused selection | 1.40x-2.12x t1 and 1.46x-1.69x t16 over materialized composition on Apple | 2026-07-23 | Metal BaseQN LM-head argmax drift closure |
| BaseQN MoE output tiles + direct paired SwiGLU | 1.23x/1.46x over row-wise pass 1; removes the gate/up tensor | 2026-07-23 | Metal BaseQN grouped expert projection and SwiGLU |
| Fused positioned/multimodal QK norm/RoPE specialization | 1.80x/1.40x (Apple) and 1.08x/1.35x (EPYC) over pass 1 | 2026-07-23 | positioned RoPE, M-RoPE, and positioned QK norm/RoPE |
| Q8_0 KV one-load scatter groups, scale hoist, 32-row attention tiles | gather 11.63x/7.64x and attention 4.14x-4.33x/4.59x-4.64x over pass 1 (Apple/EPYC) | 2026-07-24 | Metal Q8_0 KV-cache codec, block copy, and paged attention |
| GDN collision-safe scheduling + serial crossover + fixed K4 | selected threaded cases 1.83x-10.27x (Apple) and 3.88x-9.95x (EPYC) over equal-work scalar; serial K4 convolution 1.12x/1.29x | 2026-07-24 | Metal GatedDeltaNet and split sigmoid gate |
| Direct F16-adapter LoRA with bounded stack low-rank | 2.29x-2.38x / 2.60x-16.64x (Apple t1/t16) and 1.73x-1.74x / 2.06x-12.87x (EPYC) over scalar two-matmul composition | 2026-07-24 | direct F16-adapter LoRA |
| BaseRT hybrid storage selection | large calibration 7.87x (Apple) / 3.88x (EPYC); masked pool 2.57x/2.42x; EPYC depthwise 2.16x, pass1→pass3 | 2026-07-24 | live Metal BaseRT auxiliary, embedding, vision, and audio batch |
| Fused patch projection + four-way FP32 attention dot | projection 16.1148→2.0294 ms; cross-attention 1.2042/0.2461→0.2670/0.0675 ms (Apple t16) | 2026-07-24 | published BaseRT patch projection and cross-attention |
| K5-interior causal depthwise + online relative softmax | 0.2925→0.2028 ms and 1.4445→0.9989 ms (Apple t16) | 2026-07-24 | CPU ports for published position and relative-audio kernels |
| BF16 bit-preserving clip, interior row copies, four-way 3-D projection | clip 0.2069→0.0963 ms; projection 0.2776→0.1513 ms; Qwen RoPE 0.3512→0.2086 ms (Apple t16) | 2026-07-24 | CPU ports for Qwen temporal vision and scalar value clipping |
| Sapphire Rapids AVX-512/VNNI selected routes | W8A32 1.91x, INT8 1.19x, W4A8 1.34x over forced AVX2 (same host) | 2026-07-24 | Sapphire Rapids AVX-512/VNNI dispatch closure |
| INT8_MIN-safe SIMD matrix pre-scan | full int8 activation domain accepted with no measured regression (0.112040 vs 0.121324 ms baseline; no-regression claim only) | 2026-07-25 | INT8_MIN-safe x86 IDOT and exact AVX-512 gating |

Reversals recorded once, final direction:

- **dotprod_i8 as public qgemv default → reversed.** The activation-quantizing
  SDOT path was the default (2026-07-07, quant_gemv q8_0 NEON DotProd
  variant), then demoted because the family contract is
  `dequantize(wq) @ x` with full-precision activations (2026-07-07, qgemv
  contract realignment); the environment override was removed entirely during
  hardening (2026-07-21, CPU kernel correctness hardening). Final direction:
  `neon` f32-activation is the qgemv default; the SDOT kernel lives under
  the separately contracted `qgemv_w8a8`.
- **Dual-FP8 M1/M16 slower than dequantized scalar → reversed.** The first
  packed FP8 block dots were 0.61x-0.62x the dequantized-scalar baseline
  (2026-07-23, M2 canonical dual-quant GEMV block dots); eight-output-panel
  activation-decode reuse reversed this to 1.09x-1.35x GEMV and 1.10x-1.25x
  GEMM M16 (2026-07-23, M2 dual-FP8 GEMV/GEMM output-panel reuse). Final
  direction: FP8-family routes use grouped output panels and beat the
  comparator.
- **FP8 encoder chain superseded twice.** Linear search → binary
  representable-value table → direct IEEE bit rounding; only the last is
  retained (2026-07-22, M1 canonical lifecycle and checkpoint ingestion).
- **`blocked_ref` GGUF fallback → force-only.** Once a fallback candidate, it
  measured 86.2722 ms vs 11.7684 ms for the specialized portable Q4_0
  reference and is now reachable only by force (2026-07-23, M5 portable
  fallback and GGUF Q4 routing).

## Rejected — with the reason, so they are not retried

**Manual multi-accumulator/unroll of compiler-vectorized loops — REJECTED
(2026-07-07, 2026-07-22).** The q8_0 scalar GEMV 4-way accumulator split
measured 1-3% slower on every shape because Apple clang already
auto-vectorizes the plain int8→f32 loop and the manual split obstructs it
(quant_gemv q8_0 scalar reference bring-up). The same failure repeated for
the FP16/BF16 conversion manual unroll (Log, universal floating storage), the
unary-selector four-way unroll (softplus 2.4371→2.5798 ms), and the OpenAI
SwiGLU four-way unroll (1.2074→1.4535 ms) (exhaustive llama operation and
quant lifecycle closure). Rule: on Apple clang, do not hand-unroll or
hand-split a loop the compiler already vectorizes; measure before assuming
latency-bound accumulation.

**std::function-based thread-pool wrapper — REJECTED (2026-07-07).**
Capture-heavy lambdas made loop bounds/pointers reload through the capture
frame every iteration (stores through the output pointer may alias the
frame), a 1.6-1.9x single-thread regression on rms_norm. Replaced with a
fn-pointer + context trampoline and free-function loop bodies with by-value
arguments (Threading layer). Rule: hot-path parallel-for interfaces must not
route through std::function or capture frames.

**Unconditional f64 RMSNorm division/sqrt — REJECTED (2026-07-21).** Doing
the row division and square root in f64 on every row regressed the R512
stress case by about 5.7%; the kept version enters f64 only when the f32
reduction is non-finite or subnormal (CPU kernel correctness hardening).
Rule: pay for exceptional-value handling only on the exceptional path.

**Blocked f32-scratch GGUF decode as the optimized default — REJECTED
(2026-07-22).** Decoding a whole block to aligned f32 scratch then dotting
was only 0.93x-1.20x versus element decode; kept solely as a portable
fallback while the direct packed NEON dot became the default (complete
llama.cpp stored-quant CPU paths). Related: the TQ1_0 branch/division decode
regressed and was restructured into fixed 160/80/16-value base-3 sections.
Rule: dot directly from packed bytes; a float staging block is not an
optimization.

**Dense GEMM retiling family — REJECTED (2026-07-22).** Float dense
accumulation and direct generic panel accumulation failed tolerances; the
64-column dense tile, 8-row dense tile, and 16-row panel unroll all
regressed. The original 4x32 dense tile plus pass-2 four-row NEON panel
accumulation was kept (three-pass optimization sweep). Rule: the 4x32 tile is
the measured sweet spot on this core; re-tiling needs new evidence.

**Paged-attention value-update SIMD — REJECTED (2026-07-22).** Advanced
paged-attention value-update vectorization regressed while online score
tiles and the NEON f32 score dot won (three-pass optimization sweep).

**Fused-projection load-sharing candidates — REJECTED (2026-07-22).** Paired
Q/K RoPE-row traversal and the shared activation-load pair dot were both
neutral and removed; only the pass-1 gate/up semantic fusion stayed
(three-pass optimization sweep). Rule: fusion earns its place by removing
materialization, not by hoping shared loads pay.

**Streaming LM-head 64-token scheduling grain — REJECTED (2026-07-22).**
The grain retune was neutral and reverted; vocabulary-parallel selection
plus multi-row packed-block reuse were kept (three-pass optimization sweep).

**Grouped-MoE shared float dequantization / activation reload / output
tiles — REJECTED (2026-07-22).** Shared float decode regressed about 2x;
activation-load reuse and two-output task tiles were neutral or regressive.
Only the pass-1 paired gate/up block dot survived (three-pass optimization
sweep).

**FFT normalization fusion and NEON Mamba state update — REJECTED
(2026-07-22).** Both regressed; complex-f32 batch-one FFT and the plain
float recurrent state were kept (three-pass optimization sweep).

**Always-on MoE expert-batch union — REJECTED (2026-07-22).** The unguarded
union was about 3x slower at one thread; the retained route unions only
repeated experts with a live worker pool and otherwise falls back to per-row
GEMV (Colibri CPU algorithm excavation). Rule: batching optimizations need a
thread/duplication guard.

**Weight-major GEMM cell traversal — REJECTED (2026-07-23).** Keeping one
prepared weight row hot across M produced strided output stores that
regressed contiguous-output pairs by as much as 41% despite small
MXFP4/BitNet gains (M2 canonical dual-quant GEMM panels). Rule: keep output
lanes contiguous; activation rows outermost, decoded activation blocks
reused across output panels.

**Output-panel scheduling grain 2 — REJECTED (2026-07-23).** Only 7 of 22
typed cases improved, the mean ratio regressed, and max CV rose from 0.3575
to 0.5005; grain 1 was restored (M2 canonical FP16/BF16 GEMV panels). Rule:
grain-1 output-panel tasks unless a specific shape proves otherwise.

**Blanket x86 FP32 output-panel traversal — REJECTED (2026-07-23).** The
panel traversal helped only FP32 E4M3/E5M2 (1.31x-1.32x) and regressed the
integer, microscale, FP4, NVFP4, and BitNet FP32 layouts, which stay on the
portable row route; dispatch narrowed to format-selective (M2 x86 canonical
typed GEMV panels). In the same wave, the pass-2 one-panel FP32 route
improved M16 but regressed M128 and was replaced by four-panel groups (M2
x86 canonical M16/M128 projection). Rule: x86 route promotion is per-format,
decided by measurement, never blanket.

**Direct AVX2 W8A8 GEMM — REJECTED (2026-07-23).** The AVX2 W8A8 route
regressed 4.4258→5.8649 ms at M16; the final format selector restores the
4.4237 ms portable panel route (M2 x86 canonical M16/M128 projection). Rule:
a working portable route is the bar; an ISA route that loses stays out of
dispatch.

**Blanket plane-major M1 gate/up traversal — REJECTED on Apple
(2026-07-23).** Plane-major M=1 traversal regressed Apple MXFP4 and is
selected on x86 only, where switching between two packed matrices is the
cost to avoid; the first x86 M1 row interleave was itself 0.62x-1.01x and
was replaced by the plane-local/paired-panel selection (M3 canonical paired
gate/up projection). Rule: locality trades differ per architecture; encode
them in the selector, not the kernel.

**F3 generic M1 panel route and forced x86 INT4 panel — REJECTED
(2026-07-23).** The generic M1 panel route for activation-quantizing SwiGLU
regressed against mature row dots and was replaced by shape/format
selection; a forced INT4 panel experiment on x86 was rejected (M3 canonical
fused SwiGLU and activation quantization).

**Blanket shared-activation QKV traversal — REJECTED (2026-07-23).** Pass-2
sharing regressed FP8 M16 on Apple and BF16 plus multithreaded large-M FP32
on x86; the final selector shares only where measured to win, and Apple
typed QKV stays on the mature M2 typed panel kernels (M3 canonical
unequal-head Q/K/V projection).

**F5 generic decoded paired panels — REJECTED (2026-07-23).** Generic paired
panels (0.3351-0.5857 ms vs 0.0799-0.3132 ms) and the group-4 generic-panel
follow-up lose the mature format-specific row dots for typed Apple routes;
NEON typed pair panels and FP32 pair tiles were kept instead (M3 canonical
QKV plus RoPE plus typed KV write). Rule: a generic decode loop does not
replace format-specialized dots even inside a fusion.

**Global multi-row LM-head panels on Apple — REJECTED (2026-07-23).**
Multi-row prepared panels won for FP8 but lost to direct row dots for
INT4/MXFP4/BitNet; retained for FP8 only via format-selected traversal (M3
canonical streaming LM head).

**MoE compact token selector and direct packet dots on x86 — REJECTED
(2026-07-23).** The Apple compact token-row selector reached 410.5366 ms on
EPYC; EPYC direct packet dots ran 104.3126-382.5582 ms (0.09x-0.29x) and
were replaced by a decode-once workspace selector. The FP8 output-panel
group-16 sweep (26.6631 ms) lost to group 8, and the pass-3 materializing
threaded x86 output experiment regressed INT4/MXFP4 (M3 canonical grouped
MoE weight-only path; M3 canonical MoE dual activation and next-layer
quantization). Rule: Apple-tuned selectors are hypotheses on x86; measure
before porting the policy.

**M4 attention score tile 32 — REJECTED (2026-07-23).** Widening the online
score tile from 16 to 32 regressed FP8 attention (0.4036-0.8354 vs
0.3949-0.8290 ms); tile 16 retained (M4 typed FP8 cache and direct online
paged attention).

**Codec-only threading at one thread and parallel full-plane KV copy —
REJECTED (2026-07-24).** Q8_0 codec scheduling alone regressed at one thread
(pass 2), and the parallel full-plane-copy candidate (0.0516 ms Apple /
0.0906 ms EPYC) lost to the serial optimized library copy (0.0160/0.0300 ms
at one thread) at the quick shape (Metal Q8_0 KV-cache codec, block copy,
and paged attention). Rule: sub-100-us copies stay serial.

**GGUF generic AVX-512 promotion — REJECTED (2026-07-24).** With forced
AVX-512, 19 of 23 GGUF format medians trail forced AVX2 on Sapphire Rapids,
because the direct quant kernels already delegate to their AVX2 block dots
and only the generic decoded fallback widens; automatic GGUF dispatch stays
AVX2 while W8A32/INT8/W4A8 keep their measured AVX-512/VNNI wins (Sapphire
Rapids AVX-512/VNNI dispatch closure). Rule: AVX-512 promotion is
per-kernel, justified by a same-host forced A/B.

**Four-FP32-accumulator LoRA up-projection on Apple — REJECTED
(2026-07-24).** The fixed-R16 four-accumulator pass helped EPYC but was
slower than the inline sequential path on Apple (0.1204 vs 0.1088 ms at M1)
and is selected on x86 only (direct F16-adapter LoRA).

**Per-element direct half conversion for compute-heavy BaseRT routes —
REJECTED (2026-07-24).** Direct same-type storage tripled general audio
convolution cost (Apple 3.3980→10.9954 ms) versus bulk SIMD conversion;
direct typed access is kept only for sparse gathers/copies while
transcendental, interpolation, and general-convolution paths keep the bulk
adapter (live Metal BaseRT batch). The wave-3 repeats (direct conversion for
factorized position, coordinate pool, causal depthwise) and the pool
channel-tiling experiment were likewise rejected (CPU ports for published
position and relative-audio kernels); wave-2/3 also rejected direct half
conversion and coordinate channel tiling (baseline notes, 2026-07-24). Rule:
bandwidth-shaped ops convert in bulk; compute-shaped ops must not pay
per-element conversion.

**Per-SIMD-block INT8_MIN check — REJECTED (2026-07-25).** Checking for
`-128` inside every dot block slowed the common path (0.1284 ms vs the
0.1121 ms one-scan structure); one ISA-vectorized matrix-level pre-scan
preserves the original hot loop and routes only matrices containing `-128`
to the safe path (INT8_MIN-safe x86 IDOT). Rule: hoist domain checks to a
single pre-scan; never put a rare-case branch in the hot dot loop.

**Slower standalone rotary rewrites — REJECTED (2026-07-23).** Pass-3
standalone RoPE rewrites that lost to the existing routes were dropped; only
the fused specializations and fastest standalone selections were kept
(positioned RoPE, M-RoPE, and positioned QK norm/RoPE).

**Scalar ternary unpack in K1 pass 1 — REJECTED alone (2026-07-23).** The
output-row traversal alone regressed BitNet (0.2760→0.3164 ms) and was not
accepted until the vector shift/zip unpack landed (M2 canonical weight-only
GEMV block dots).

**Threaded-only GDN without a serial crossover — REJECTED as sole route
(2026-07-24).** Row/state scheduling alone left one-thread
recurrence/convolution paying task-index overhead; a direct one-thread
crossover and fixed-K4 history loop were required (Metal GatedDeltaNet and
split sigmoid gate). Rule: every threaded route needs a measured serial
crossover for small work.

## Patterns and generalized rules

- Dot directly from packed bytes and never materialize a dequantized weight
  matrix or float staging block; the single most repeated win, from
  4.68x-45.22x on stored GGUF formats to every canonical M2 route
  (2026-07-22, complete llama.cpp stored-quant CPU paths).
- Decode or convert each activation block once and reuse it across grouped
  output panels (2/4/8 panels); this converted every FP8 and typed-input
  loss into a win (2026-07-23, M2 dual-FP8 GEMV/GEMM output-panel reuse).
- Hoist scales, zero points, and group metadata out of element loops once
  per block or group; it is the first pass that pays in nearly every ladder
  (2026-07-23, Metal BaseQN semantic port and direct projection).
- 256-entry lookup tables beat arithmetic bit decode for FP8-class formats
  on the decode side (2026-07-22, MXFP8/NVFP4 and final sibling entry
  points), while direct IEEE bit manipulation beats search and tables on the
  encode side (2026-07-22, M1 canonical lifecycle and checkpoint ingestion).
- Online max/sum recurrence with a bounded score tile of 16 replaces
  materialized attention scores and selection state everywhere it was tried
  (2026-07-23, M4 typed FP8 cache and direct online paged attention).
- Manual unrolling and multi-accumulator splits of loops Apple clang already
  auto-vectorizes lose (2026-07-07, quant_gemv q8_0 scalar reference
  bring-up).
- A mechanism measured on one architecture is only a hypothesis on the
  other: LoRA accumulators are x86-only, plane-major traversal is x86-only,
  the compact MoE selector is Apple-only, W8A8 stays portable on EPYC
  (2026-07-24, direct F16-adapter LoRA; 2026-07-23, M3/M2 x86 entries).
- ISA route promotion is per-kernel and per-format, decided by same-host
  forced A/B, never blanket (2026-07-24, Sapphire Rapids AVX-512/VNNI
  dispatch closure).
- Keep output lanes contiguous; strided stores from weight-major traversal
  regress up to 41% (2026-07-23, M2 canonical dual-quant GEMM panels).
- Threaded kernels need collision-safe scheduling with a deterministic
  serial fallback for duplicate slots, and a measured one-thread crossover
  for small work (2026-07-24, Metal GatedDeltaNet and split sigmoid gate).
- Fusions are materialization/allocation wins and are frequently
  throughput-neutral at one thread; claim the memory guarantee, not a
  speedup (2026-07-23, M3 canonical fused projection epilogue (F1)).
- Ratios produced by removing full-table or full-tensor materialization are
  not equal-work throughput improvements and must not be quoted as such
  (2026-07-23, M3 canonical embedding and embedding bag (S1/S2)).
- Bandwidth saturation bounds threading: once a kernel matches the
  aggregate triad roofline, more threads cannot help until memory does
  (2026-07-07, Threading layer — row-partitioned kernels).
- Route tiny shapes serially or inline: 0.15 ms MoE routing stays serial,
  decode-sized rms_norm stays inline, sub-100-us copies stay serial
  (2026-07-22, sibling semantic port batch; 2026-07-24, Metal Q8_0
  KV-cache codec, block copy, and paged attention).
- Hoist rare-domain checks into one SIMD matrix pre-scan instead of
  per-block branches in the hot loop (2026-07-25, INT8_MIN-safe x86 IDOT).
- Bulk SIMD storage conversion for bandwidth-shaped ops; direct typed access
  only for sparse gathers and copies (2026-07-24, live Metal BaseRT batch).
- Contract semantics outrank a measured speedup: the roughly 3.5x-faster
  dotprod_i8 path was demoted from public qgemv dispatch to preserve
  cross-backend numerics (2026-07-07, qgemv contract realignment with
  Metal/CUDA).

## Open contradictions

- The M2 x86 canonical typed GEMV panels entry (2026-07-23) closes with
  "native M16/128 three-pass records remain open" while the M2 x86 canonical
  M16/M128 projection entry (2026-07-23), which precedes it in the
  append-only notebook, already records that three-pass evidence. One of the
  two closing notes is stale. Resolve by checking the raw artifact
  timestamps under `perf/results/2026-07-23/x86-m2-*` versus
  `x86-canonical-*` and recording which run actually came second.
- Direct compressed-cache attention beats materialization at one thread for
  FP8 (1.22x-1.78x), MXFP8 (1.95x-2.10x), and TurboQuant (1.29x-1.54x) but
  loses for BitNet KV3 (0.87x-0.97x Apple, 0.46x-0.48x EPYC) (2026-07-23,
  M4 A1-A10 entries). Whether KV3 should take a one-thread crossover to a
  gather-then-attend path is unmeasured. Resolve with a same-binary KV3 t1
  A/B of direct versus gather+FP32 attention across S and D.
- Direct typed cache I/O versus staged conversion is unresolved per-case:
  staged F16C/AVX2 conversion is faster for some x86 typed FP8 I/O cases and
  the Apple F16 staged comparator wins at 16 threads, yet the direct routes
  are retained for bounded memory (2026-07-23, M4 typed FP8 cache A1/A2).
  Resolve with a per-case direct-versus-staged selection matrix at t1/t16 on
  both machines.
- 8 versus 12 threads is shape-dependent on the M4 Max hybrid part (qgemv
  N4096 best at 8 threads, N8192/N16384 at 12) and was left to OS QoS
  (2026-07-07, Threading layer). Resolve with affinity pinning on an
  Apple-silicon host.
- BaseQN grouped SwiGLU on EPYC is 0.82x its predecoded comparator while
  grouped GEMM is 1.12x on the same shape (2026-07-23, Metal BaseQN grouped
  expert projection and SwiGLU); the x86 SwiGLU epilogue direction is
  unresolved. Resolve with a dedicated x86 pass on the paired epilogue.
