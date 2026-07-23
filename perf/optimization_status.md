# Optimization Status

Running notebook of focused optimization runs on CPU kernel paths. Every
kernel implementation, routing change, benchmark change, or performance
claim must add an entry here.

Every future kernel implementation, routing change, benchmark change, or
performance claim must add a focused optimization entry here.

## 2026-07-22: M1 canonical lifecycle and checkpoint ingestion

Status: keep; M1 lifecycle exit gate complete. This does not claim M2 packed
projection support.

Implementation: `include/quixicore_cpu/quant_import.h` and
`kernels/quantization/quant_import_ref.cpp` add an owned canonical tensor,
FP32/FP16/BF16 packers for integer, FP8, FP4, MXFP8, MXFP4, NVFP4, and BitNet,
and import adapters for AWQ, GPTQ v1/v2 plus act-order, AutoRound target
formats, SmoothQuant/AZP, and checked BitNet I2_S blocks. Canonical bytes remain
unchanged when `CpuPackedWeights::prepare` builds version-2 row panels and
aligned side tables. Generic GGUF QGEMM rejects these panels until M2 installs
format-specific consumers.

References inspected:

- AutoRound `6ff414b15728f97848936551021d0b180dd40320`, especially
  `auto_round/export/export_to_awq/utils.py` and the AutoGPTQ/GGUF exporters;
- GPTQ `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` and T-MAC
  `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`, for packed zeros, `g_idx`, and
  act-order semantics;
- SmoothQuant `c61476d728e42ae0d8a35e7e78494edcac3237b5`, for per-channel
  weights and activation metadata; and
- BitNet `16da220ae2b510caff437d403288882687f44ae5`, for canonical I2_S and
  rebuildable TL preparation boundaries.

Correctness: `test_quant_import` checks exact AWQ lane reversal and decoded
outputs, GPTQ v1 zero+1, GPTQ-v2 symmetric fields, non-identity `g_idx` and
act-order, every AutoRound target layout, SmoothQuant row sums/AZP, BitNet
reserved codes, all FP8 scale topologies, all canonical prepared layouts, and
FP16/BF16 input parity. `test_quantization` checks every finite FP8 code,
signed identity, every adjacent midpoint tie, and 20,000 deterministic random
values against an exhaustive nearest-value oracle. Full Release CTest is the
landing gate below. The Release build passes 46/46 CTest targets. A focused
ASan/UBSan build compiled the complete library but hit the repository's known
Apple libc++ ABI link mismatch while linking `test_quant_import`; no sanitizer
pass is claimed from that build.

Three optimization passes, all R/N128 K1024, one thread:

| format | pass 1 linear search ms | pass 2 binary table ms | pass 3 direct bits ms | final/pass 1 |
|---|---:|---:|---:|---:|
| FP8 E4M3FN | 34.8737 | 1.2458 | 0.3618 | 96.39x |
| FP8 E5M2 | 25.1908 | 1.1670 | 0.3656 | 68.89x |
| MXFP8 | 35.8365 | 1.2718 | 0.4114 | 87.10x |
| NVFP4 | 2.6581 | 0.6741 | 0.5000 | 5.32x |

Pass 1 used the original exact but O(127)-per-element FP8 search. Pass 2 kept
exact rounding with a monotonic representable-value table and binary search.
Pass 3 directly rounds IEEE FP32 exponent/significand bits and is retained;
the independent exhaustive test oracle prevents the optimization from defining
its own correctness. Non-FP8 pack/import cases stayed within ordinary run
variance, so no unsupported speedup is claimed for them.

- Hardware: Apple M5 Max, 18 logical/physical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON/DotProd/I8MM available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one thread, OS-default affinity/frequency, high-QoS harness hint,
  steady-state buffers; final 5 warmups, 30 samples, 10 ms minimum samples.
- Working-tree label: `7f61388-dirty`.
- Pass artifacts: `perf/results/2026-07-22/m1-pass1-t1-fixed/`,
  `perf/results/2026-07-22/m1-pass2-t1/`,
  `perf/results/2026-07-22/m1-pass3-t1/`; stable final
  `perf/results/2026-07-22/m1-pass3-final2-t1/`.
- Decision: retain direct bit rounding and the complete canonical lifecycle;
  reject the original linear encoder and supersede the binary-search pass.

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
| 2026-07-22 | M0 full-matrix benchmark registration | f32 plus Q4_0/Q4_K/Q6_K, INT8, MXFP4/MXFP8, NVFP4, BitNet b1.58 | 23 pass-0 decode/prefill/fusion/serving/cache cases | Apple M5 Max, aarch64 NEON/DotProd/I8MM, 1 thread | same-run scalar, dequantized, materialized, or decomposed references: 0.0414-14.2072 ms where applicable | registered current routes: 0.0874-10.1409 ms; CV 0.0103-0.0775 | keep all eight executable M0 families; no new support/speedup claim; MXFP8 and BitNet results explicitly retain optimization gaps | `perf/results/2026-07-22/m0-pass0-t1/` |
| 2026-07-22 | universal floating storage, three passes | FP16/BF16 storage, FP32 accumulation | round trip N1048576; BF16 softmax R256 H4096 | Apple M5 Max, aarch64 FP16, 1/6 threads | scalar pass: FP16 1.6856/1.7549, BF16 0.5976/0.6010 ms; explicit softmax 1.4096/0.2985 ms | final FP16 0.0961/0.0466, BF16 0.1569/0.0548, typed softmax 1.4028/0.2995 ms | keep reusable typed adapter, compiler-vectorized BF16, threading, NEON FP16; manual unroll rejected; typed dispatch is performance-neutral | `perf/results/2026-07-22/float-storage-{pass123,native-final}-t{1,6}/` |
| 2026-07-22 | llama unary/GLU selector closure | f32 | all 22 unary and six GLU modes; softplus/OAI SwiGLU N1048576 | portable/aarch64, 1/6 threads | unary 2.6758/0.5109; OAI GLU 2.4515/0.6971 ms | unary final 2.5420/0.5358; OAI GLU final 1.1954/0.2811 ms | keep selector hoist and source-exact direct sigmoid; reject both manual unrolls; OAI 2.05x/2.48x over pass 1 | `perf/results/2026-07-22/{unary-pass*,glu-pass*,activation-final}-t{1,6}/` |
| 2026-07-22 | exhaustive llama operation parity | f32 | Conv C16 O32 S32; GLA B4 T64 H8 D32; DSV4 T4096 | portable/aarch64, 1/6 threads | serial conv 1.9942 ms; one-thread GLA/DSV4 0.2882/0.1385 ms | six-thread 0.7870/0.0772/0.0364 ms | keep portable semantics and outer-task parallelism; 2.53x/3.73x/3.80x | `perf/results/2026-07-22/parity-final-t{1,6}/` |
| 2026-07-22 | complete quant authoring/activation lifecycle | seven IQ formats; Q8_0/Q8_1/Q8_K | IQ R1 K4096; Q8 R64 K4096; Q4_KxQ8_K N1024 K4096 | portable/aarch64, 1/6 threads | one-thread 11.1633-75.6794 / 0.1509-0.3383 / 1.4484 ms | six-thread 2.8692-20.0169 / 0.0515-0.0991 / 0.2876 ms | keep block-parallel authoring/activation paths; weighted work matches uniform overhead | `perf/results/2026-07-22/parity-final-t{1,6}/` |
| 2026-07-22 | complete llama stored-format GEMV sweep | 23 non-Q4_0/Q8_0 stored formats / f32 activation | M1 N1024 K4096 | aarch64 NEON, 1 thread | same-run element decode 14.1884-17.3736 ms | direct packed block dot 0.3178-3.0547 ms | keep all direct routes; 4.68x-45.22x, CV 0.0079-0.0759 | `perf/results/2026-07-22/llama-quants-final-neon-t1-stable/` |
| 2026-07-22 | GGUF three-pass GEMV sweep | Q4_K/Q5_K/Q6_K and IQ formats / f32 | M1 N1024 K4096 | aarch64 NEON, 1/6 threads | pass-0 8.3503-14.7039 / 1.7567-2.9806 ms | final 0.3137-2.5965 / 0.0890-0.5164 ms | keep direct packed-block dots and IQ4 table lookup; 4.0x-31.1x / 4.3x-22.5x over pass 0 | `perf/results/2026-07-22/opt3-{pass0,final}-t{1,6}/` |
| 2026-07-22 | dense/quantized GEMM three-pass sweep | f32; Q4_0/K, Q6_K, IQ4_XS | M16/M128 N256/512/1024 K512/1024/1408 | aarch64 NEON, 1/6 threads | pass-0 dense 0.3282 / 0.1066 ms; Q4_K/Q6_K M16 1.3853-1.6778 / 0.3207-0.3802 ms | final dense 0.3319 / 0.0964 ms; Q4_K/Q6_K 0.9457-1.2375 / 0.2695-0.3599 ms | reject dense retile; keep generic panel SIMD and small-IQ4 canonical bypass | same |
| 2026-07-22 | paged attention + MLA three-pass sweep | f32 | paged B2 HQ8 HKV2 S512 D64; MLA B2 H8 S512 DL64 DR16 | aarch64 NEON, 1/6 threads | pass-0 paged 0.1935 / 0.0562 ms; materialized MLA 0.2437 / 0.2453 ms | final paged 0.1126 / 0.0457 ms; online MLA 0.2033 / 0.0595 ms | keep online score tiles, SIMD f32 score dot, and request/head-parallel MLA; reject value-update SIMD | `perf/results/2026-07-22/{opt3-final,opt3-mla-final}-t{1,6}/` |
| 2026-07-22 | fused decode projections three-pass sweep | Q4_0 / f32 | SwiGLU N1024 K1408; QKV HQ8 HKV2 D64 K1408 | aarch64 NEON, 1/6 threads | pass-0 0.2074/0.0774 and 0.0635/0.0335 ms | final 0.2094/0.0780 and 0.0677/0.0351 ms | keep paired gate/up semantic fusion; reject Q/K pairing and activation-load-sharing candidates; no new speedup claim | same |
| 2026-07-22 | streaming LM-head three-pass sweep | Q4_0 / f32 | R2 V8192 H1024 | aarch64 NEON, 1/6 threads | pass-0 1.2227 / 0.3927 ms | final 0.8795 / 0.2353 ms | keep vocabulary-parallel selection and multi-row packed decode reuse; reject grain retune | same |
| 2026-07-22 | grouped MoE three-pass sweep | Q4_0 / f32 | R64 E8 K256 I512 | aarch64 NEON, 1/6 threads | pass-0 1.2931 / 0.3810 ms | final 1.2636 / 0.3387 ms | keep paired gate/up block dot; reject shared float decode, activation reload, and output-tile candidates | same |
| 2026-07-22 | norm-add-quant three-pass sweep | f32 / group int8 | R512 H4096 G128 | aarch64 NEON, 1/6 threads | pass-0 1.0311 / 0.2664 ms | final 0.6351 / 0.1804 ms | keep four-way sumsq, 16-byte quant pack, and four-way maximum reduction | same |
| 2026-07-22 | FFT + Mamba three-pass sweep | f32 | B1 H2 L1024; B1 H2 L128 D32 | aarch64/portable, 1/6 threads | pass-0 FFT 0.0546/0.0569; Mamba 0.0973/0.0714 ms | final FFT 0.0550/0.0407; Mamba 0.0356/0.0342 ms | keep complex-f32/batch-one FFT and float recurrent state; reject normalization fusion and Mamba NEON | same |
| 2026-07-22 | GGUF SIMD GEMV | Q4_0/K, Q5_K, Q6_K, IQ formats / f32 | M1 N4096 K4096; K/IQ N1024 | aarch64 NEON, 1/6 threads; x86 variants compile/test | scalar/element decode 2.8578 / 0.5892 ms Q4_0; 13.8664-15.1367 / 2.4908-3.6649 ms K/IQ | 1.1953 / 0.2798 ms Q4_0; 8.4295-14.6360 / 1.7739-3.1180 ms K/IQ | keep dispatch; Q4_0 2.39x/2.11x, every K/IQ format improves 1.03x-1.64x | `perf/results/2026-07-22/qgemv-allformats-final-t{1,6}/` |
| 2026-07-22 | blocked dense/quantized GEMM | f32; Q4_0/K, Q6_K, IQ4_XS | M16/M128 N256/512/1024 K512/1024/1408 | aarch64 NEON, 1/6 threads | scalar dense 2.9337 / 3.0997 ms; canonical quantized 0.6082-14.1906 ms | dense 0.3336 / 0.1005 ms; packed quantized 0.3313-7.4150 ms | keep; dense 8.80x/30.85x, Q4_0 M128 1.84x/2.94x, K/IQ 6.75x-9.04x; bypass panel for small Q4_0 | `perf/results/2026-07-22/{optimization-plan-complete,prepacked-allformats-final2}-t{1,6}/` |
| 2026-07-22 | online attention, fused decode, streaming selection, tiled MoE | f32 / Q4_0 | focused decode/prefill shapes | aarch64 NEON, 1/6 threads | materialized/decomposed 0.0397-1.7029 ms at 6 threads | 0.0391-0.4526 ms at 6 threads | keep online attention (2.96x), fused SwiGLU (1.31x), fused QKV+RoPE+KV (1.02x), and grouped MoE (3.76x) at six threads; LM-head tiling is allocation-bounded and neutral/noisy | `perf/results/2026-07-22/optimization-plan-complete-t{1,6}/` |
| 2026-07-22 | fused norm-add-quant | f32 / group int8 | R512 H4096 G128 | aarch64 NEON, 1/6 threads | decomposed preallocated 3.4495 / 1.3100 ms | row-local fused 1.0248 / 0.2545 ms | keep; 3.37x/5.15x and no rows-by-hidden temporary | `perf/results/2026-07-22/norm-fused-neon-t{1,6}/` |
| 2026-07-22 | FFT convolution and recurrent Mamba2 | f32 | B1 H2 L1024; B1 H2 L128 D32 | portable radix-2/recurrent, 1/6 threads | direct convolution 1.8173 / 1.8128 ms; source expansion 3.7921 / 3.9839 ms | FFT 0.0553 / 0.0608 ms; Mamba 0.0986 / 0.0740 ms | keep; 29.79x-53.83x with direct fallback for non-power-of-two/small FFT lengths | `perf/results/2026-07-22/optimization-plan-complete-t{1,6}/` |
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

## 2026-07-22: exhaustive llama operation and quant lifecycle closure

Status: retained portable-reference coverage. The pinned llama.cpp CPU
inventory now has 105/105 operation symbols classified: 89 public numerical
mappings, seven validated view/layout adapters, and nine non-numerical enum
markers, callback hooks, or pool selectors. Nested drift tables additionally
cover all 22 numerical `GGML_UNARY_OP_*` modes and all six numerical
`GGML_GLU_OP_*` modes; `GGML_OP_SILU_BACK` maps to its exact derivative rather
than to a gated derivative. The quant inventory has canonical
pack, unpack, and f32 GEMV coverage for all 25 stored formats, plus public
Q8_1/Q8_K activation intermediates. These counts are enforced by
`scripts/check_parity_manifest.py`; ISA performance tiers remain separate.

New operation paths include scalar arithmetic/reductions/layout transforms;
im2col/col2im, 2-D/3-D/depthwise/transposed convolution and pooling/backward;
relative-position and window transforms; GLA; RWKV6/RWKV7; and all three DSV4
hyper-connection stages. Recurrent state is explicit and parallel work is
partitioned by sequence/head. Convolution partitions independent output
channels; DSV4 partitions tokens.

The quant lifecycle pass added exact Q2_K/Q3_K/Q4_K/Q5_K/Q6_K and IQ4_NL/XS
packers, weighted IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ2_S/IQ1_S/IQ1_M packers,
canonical Q8_0/Q8_1/Q8_K activation layouts, and a quantized-activation GEMV
contract route. During sanitizer validation, the new IQ3_S encoder exposed an
eight-byte sign-scratch overwrite; its region scratch is now correctly sized
for all 64 sign codes and the complete sanitizer suite passes.

Three focused implementation/optimization passes were retained:

| pass | change | measured outcome |
|---|---|---|
| 1 | direct portable operation recurrences and multi-seed IQ lattice search | correctness baseline; IQ2_S R1 K256 was 3.8177 ms |
| 2 | remove repeated codebook-extrema scans and use llama-compatible f32 convolution accumulation | IQ2_S fell to 2.7280 ms; convolution became bit-exact to the independent f32 oracle |
| 3 | block/row/token/output-channel parallel regions with hoisted strides and state-local reuse | retained; all final quick cases pass their embedded oracle |

The nested unary selector received its own three passes: (1) a direct
per-element selector, (2) one selector dispatch per call with template-fixed
element loops, and (3) manual four-way unrolling. Pass 2 improved softplus from
2.6758/0.5109 ms to 2.4371/0.4640 ms at one/six threads. Pass 3 regressed to
2.5798/0.5158 ms and was rejected. The retained code's independent final repeat
was 2.5420/0.5358 ms (CV 0.0134/0.0463), exact to the scalar oracle.

OpenAI SwiGLU also received three passes: (1) the shared overflow-stable
sigmoid helper, (2) llama's source-exact direct exponential expression, and
(3) manual four-way unrolling. Pass 2 reduced N1048576 from
2.4515/0.6971 ms to 1.2074/0.2814 ms at one/six threads. Pass 3 regressed to
1.4535/0.3384 ms and was rejected. The retained final repeat measured
1.1954/0.2811 ms (CV 0.0126/0.0586), bit-exact to the independent expression.

Final Release medians:

| path | one thread ms | six threads ms | scaling / baseline |
|---|---:|---:|---:|
| seven weighted IQ packers, R1 K4096 | 11.1633-75.6794 | 2.8692-20.0169 | 3.78x-3.91x thread scaling |
| Q8_0/Q8_1/Q8_K pack, R64 K4096 | 0.1509/0.1791/0.3383 | 0.0515/0.0596/0.0991 | 2.93x/3.01x/3.41x |
| Q4_K x Q8_K GEMV, N1024 K4096 | 1.4484 | 0.2876 | 5.04x; exact versus explicit unpack |
| conv2d, C16 O32 S32 K3 | 3.7463 | 0.7870 | 2.53x versus 1.9942 ms serial baseline at six threads |
| GLA, B4 T64 H8 D32 | 0.2882 | 0.0772 | 3.73x thread scaling |
| DSV4 combine, T4096 | 0.1385 | 0.0364 | 3.80x thread scaling |
| unary softplus, N1048576 | 2.5420 | 0.5358 | 4.65x over same-run serial baseline at six threads |
| OpenAI SwiGLU, N1048576 | 1.1954 | 0.2811 | 4.59x over same-run serial baseline at six threads |

Correctness and portability: native aarch64 Release CTest passes 43/43;
ASan+UBSan+float-cast-overflow passes 42/42 with leak detection disabled because
Apple's ASan runtime does not support it; x86_64 Release compiles AVX2,
AVX-512, VNNI, and AMX source sets and passes 38/38 under Rosetta. The parity
checker also verifies that every mapped operation names a real public CPU
function, derives the complete live quant-type inventory from llama's
`enum ggml_type`, rejects unary or GLU selector drift, and maps all 66
operation-level plus 29 quant-family entries published by the pinned sibling
manifests. CUDA's pinned manifest remains family-level, which is recorded as a
source-metadata limit rather than inferred operation-level evidence.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; native aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 3 warmups,
  10 timed samples, 5 ms minimum sample; correctness enabled.
- Command: `quixicore_cpu_bench --preset quick --kernel
  quant_lifecycle,llama_parity --threads {1,6} --warmup 3 --iters 10
  --min-sample-ms 5`.
- Working-tree label: `7f61388-dirty`.
- Raw results: `perf/results/2026-07-22/parity-final-t{1,6}/`; exploratory
  passes are retained under `quant-lifecycle-pass*`, `llama-parity-pass*`, and
  `unary-pass{1,2,3}-t{1,6}`. The nested-selector final repeat is
  `perf/results/2026-07-22/activation-final-t{1,6}/`; OpenAI SwiGLU passes are
  retained under `glu-pass{1,2,3}-t{1,6}`.

## 2026-07-22: complete llama.cpp stored-quant CPU paths

Status: retained for canonical decode/GEMV and superseded for authoring by the
complete lifecycle entry above. Compute and authoring cover all 25 stored
llama.cpp weight formats. Q8_1 and Q8_K remain activation/dot-partner layouts,
not stored-weight enum members, but now have public activation pack/unpack APIs.

References inspected at implementation time: llama.cpp `2beefef68`, vLLM
Marlin `4b594b4aa`, and QuixiCore-Metal `bc968fc`. The CPU application of those
algorithms keeps canonical GGUF blocks at the API boundary, expands scales once
per block/sub-block, performs packed table/nibble dots directly, and leaves
ISA-specific panel packing behind `CpuPackedWeights`.

Coverage added in this pass:

- Canonical layouts and element decoders for Q1_0, Q2_0, IQ3_S, IQ2_S,
  IQ1_M, and TQ1_0; NVFP4 is now the canonical 64-value/36-byte four-scale
  layout. UE4M3 is decoded as unsigned, and IQ1_M uses the signed CPU grid
  ordering rather than the GPU-transposed lookup table.
- Canonical public packers for Q1_0, Q2_0, Q4_1, Q5_0, Q5_1, MXFP4, NVFP4,
  and TQ1_0, alongside Q4_0, Q8_0, and TQ2_0. The subsequent lifecycle pass
  added Q2_K through Q6_K, IQ4_NL/XS, and all seven importance-sensitive IQ
  encoders. `qgemv_pack_weighted` accepts an explicit importance matrix;
  `qgemv_pack` supplies uniform importance. Exact golden bytes pin every
  deterministic non-IQ layout, while IQ output is checked through canonical
  decode, error bounds, determinism, and llama.cpp interoperability.
- Direct aarch64 block-dot routes for every stored llama format. Classic Q4/Q5
  and FP4 use NEON nibble/table lookup; K, IQ, and ternary formats consume
  packed fields without a 256-float decode scratch block.
- `qgemv_formats` now has a focused case for every stored format that was not
  already covered by the dedicated Q4_0/Q8_0 cases.

Three measured passes:

| pass | candidate | result |
|---|---|---|
| 1 | element-at-a-time canonical decoder | correctness/performance baseline |
| 2 | decode one complete block to aligned f32 scratch, then blocked dot | 0.93x-1.20x versus element decode; useful portable fallback, rejected as the optimized default |
| 3 | direct packed NEON dot with block-local scale/table reuse | retained for all formats; TQ1_0 was further restructured into fixed 160/80/16-value base-3 sections after the branch/division version regressed |

Primary stable result at one thread, M1 N1024 K4096: 0.3178-3.0547 ms versus
14.1884-17.3736 ms for the same-binary independent element decoder,
4.68x-45.22x faster. All target CVs are 0.0079-0.0759 and all embedded
oracles pass; maximum reported relative error is 9.37e-4, within the benchmark
tolerance. Six-thread artifacts are retained as scaling context, but the
single-thread run is the performance claim because its complete format sweep
passes the handbook variance gate.

Correctness and portability:

- Native Release CTest: 38/38.
- ASan + UBSan + float-cast-overflow CTest: 37/37.
- x86_64 cross-build and Rosetta CTest, including AVX2/AVX-512 sources: 32/32.
- Focused canonical pack/unpack, arbitrary-payload block-dot, and forced-ref
  tests: 7/7.

Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
efficiency), 128 GB, aarch64 NEON. OS/toolchain: macOS 26.5.2, Apple Clang
21.0.0 (`clang-2100.1.1.101`). Primary command:
`QUIXICORE_CPU_GGUF_GEMV_VARIANT=neon quixicore_cpu_bench --preset quick
--kernel qgemv_formats --threads 1 --warmup 5 --iters 20 --min-sample-ms 30`.
Working-tree label: `1d3dc15-dirty`. Raw pass artifacts are
`perf/results/2026-07-22/llama-quants-pass{1-ref,2-blocked,3-neon}/`; the stable
retained artifact is
`perf/results/2026-07-22/llama-quants-final-neon-t1-stable/`.

## 2026-07-22: three-pass optimization sweep

Status: completed. Each of the eight planned kernel families received three
measured optimization passes. A pass is an experiment, not an obligation to
land code: candidates that missed correctness tolerances or regressed the
focused benchmark were removed. The final tree contains only the retained
pieces below and preserves the public contract and canonical GGUF bytes.

| family | pass 1 | pass 2 | pass 3 | final decision |
|---|---|---|---|---|
| GGUF SIMD GEMV | direct packed dots for Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS | direct packed dots for IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S | NEON table lookup/dot for IQ4_NL and IQ4_XS | keep all three |
| dense + quantized GEMM | float dense accumulation and direct generic panel accumulation failed tolerances and were rejected | 64-column dense tile rejected; four-row NEON accumulation retained for generic panels | 8-row dense tile and 16-row panel unroll rejected; small IQ4 uses the now-faster canonical GEMV route | keep the original 4x32 dense tile plus pass-2 panel work |
| paged attention + MLA | 16-score blockwise online softmax for advanced paged attention | NEON f32 score dot | vector primitives for plain attention/MLA retained; advanced value-update SIMD rejected | keep online tiles and score/MLA vectorization |
| fused decode projections | paired Q4 gate/up block traversal | paired Q/K RoPE-row traversal was neutral and rejected | shared activation-load pair dot was neutral and rejected | keep pass-1 gate/up fusion; no new QKV speedup claim |
| streaming LM-head | one vocabulary-parallel Q4 argmax region | reuse each decoded Q4 block across hidden rows | 64-token scheduling grain was neutral and reverted | keep passes 1-2 |
| grouped MoE | paired Q4 gate/up block dot | shared float dequantization regressed about 2x; activation-load reuse was also rejected | two-output task tiles were neutral/regressive | keep pass-1 pair dot |
| fused norm-add-quant | four-way f32 sum-of-squares accumulation | 16-code int8 narrowing/store | four-way normalized-maximum accumulation | keep all three |
| FFT + Mamba | complex-f32 FFT and projection fused into recurrent state update | one fork-join for batch-one FFT; f32 Mamba state | inverse-normalization fusion and NEON Mamba state update regressed and were rejected | keep passes 1-2 |

Pass-0 to final retained measurements:

| path | pass 0 ms (1 / 6 threads) | final ms (1 / 6 threads) | result |
|---|---:|---:|---|
| Q4_K/Q5_K/Q6_K GEMV N1024 K4096 | 9.7727-14.7039 / 2.0937-2.9806 | 1.5332-2.1639 / 0.3278-0.4313 | 6.3x-8.7x / 6.4x-8.6x |
| IQ GEMV N1024 K4096 | 8.3503-11.2778 / 1.7567-2.3323 | 0.3137-2.5965 / 0.0890-0.5164 | 4.0x-31.1x / 4.3x-22.5x |
| Q4_K/Q6_K panel GEMM M16 N256 K1024 | 1.3853-1.6778 / 0.3207-0.3802 | 0.9457-1.2375 / 0.2695-0.3599 | 1.36x-1.47x / 1.06x-1.19x |
| streaming LM-head R2 V8192 H1024 | 1.2227 / 0.3927 | 0.8795 / 0.2353 | 1.39x / 1.67x |
| online paged attention B2 HQ8 HKV2 S512 D64 | 0.1935 / 0.0562 | 0.1126 / 0.0457 | 1.72x / 1.23x |
| online MLA B2 H8 S512 DL64 DR16 | materialized 0.2437 / 0.2453 | 0.2033 / 0.0595 | 1.20x / 4.12x |
| grouped MoE R64 E8 K256 I512 | 1.2931 / 0.3810 | 1.2636 / 0.3387 | 1.02x / 1.12x |
| fused RMS-add-int8 R512 H4096 G128 | 1.0311 / 0.2664 | 0.6351 / 0.1804 | 1.62x / 1.48x |
| FFT B1 H2 L1024 | 0.0546 / 0.0569 | 0.0550 / 0.0407 | parity / 1.40x |
| recurrent Mamba B1 H2 L128 D32 | 0.0973 / 0.0714 | 0.0356 / 0.0342 | 2.73x / 2.09x |

The final same-binary independent baselines show 6.83x-45.24x at one thread
and 5.35x-30.41x at six threads for the listed K/IQ GEMVs; 1.42x/1.46x for
streaming LM-head; 1.90x/4.67x for paged attention; 5.02x/7.21x for fused
norm-add-int8; 32.89x/44.92x for FFT; 106.57x/115.19x for recurrent Mamba;
and 1.20x/4.12x for online MLA. Fused projections remain
semantic/materialization wins: the final run
is at parity at one thread and noisy from 0.90x to 1.21x at six threads, so no
new projection speedup is asserted.

- Correctness: final native Release CTest passes 38/38; ASan + UBSan +
  float-cast-overflow passes 37/37; and the x86_64 cross-build, including the
  AVX2/AVX-512 sources, passes 32/32 under Rosetta. Every timed case's embedded
  oracle passed. Final maximum relative errors are 1.19e-5 or lower for K/IQ
  GEMV, 3.39e-4 for paged attention, 3.46e-4 for Mamba, and 3.03e-2 for
  dequantized int8 norm output.
- Hardware: Apple M5 Max, 18 logical/physical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups, 30
  timed samples, 10 ms minimum sample. Final target CV range is
  0.0047-0.1260 at one thread and 0.0160-0.1600 at six threads.
- Command: `quixicore_cpu_bench --preset quick --kernel
  qgemv_formats,prerequisites,optimization_plan,ported_ops --threads {1,6}
  --warmup 5 --iters 30 --min-sample-ms 10`.
- Working-tree label: `1d3dc15-dirty`.
- Raw results: `perf/results/2026-07-22/opt3-pass0-t{1,6}/`,
  `opt3-pass1-fixed-t{1,6}/`, `opt3-pass2-fixed-t{1,6}/`,
  `opt3-pass3-fixed-t{1,6}/`, and `opt3-final-t{1,6}/` beneath the same date
  directory. MLA evidence is in `opt3-mla-final-t{1,6}/`. Rejected pass
  artifacts are retained beside them.

## 2026-07-22: planned P0-P2 CPU kernel batch

Status: implemented and retained with focused correctness, portability, and
performance evidence. Public API semantics and GGUF byte layouts are unchanged;
all packing, tiling, scratch storage, and ISA selection remain internal.

Implementation summary:

- GGUF weight-only GEMV now dispatches Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL,
  IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS, and IQ1_S through runtime-selected
  blocked/NEON/AVX2/AVX-512 functions. Q4_0 has a direct packed-nibble SIMD
  path; the retained NEON route now also evaluates every listed K/IQ format
  directly from its packed block, including table-lookup SIMD dots for IQ4.
- Dense GEMM uses 4-by-32 output tiling. Prepared quantized GEMM reuses each
  packed weight block across the activation-row tile for every listed GGUF
  format. Q4_0/Q8_0/IQ4 M<64 retains the faster canonical GEMV route; larger
  M and K formats use the panel kernel.
- Basic, advanced, FP8, absorbed-MLA, and paged-attention paths use online
  max/sum recurrence and do not materialize score vectors. Output rescaling is
  only performed when a new running maximum is observed.
- Gate/up+SwiGLU and Q/K/V projections share one scheduling region and consume
  quantized rows directly. The activation route no longer stores the full
  gate/up projection pair. A single-token route projects Q/K/V, rotates Q/K,
  and writes K/V directly to the requested cache slot without K/V tensors.
- Quantized LM-head argmax/top-k scans 4096-vocabulary tiles while retaining
  only selection state. Grouped MoE forms expert row groups and produces
  SwiGLU output directly without a rows-by-2I projection.
- Norm-add-quant is row-local and two-pass, with per-worker group scratch and
  NEON reductions/conversion. It no longer allocates a normalized tensor.
- Power-of-two FFT convolution uses an iterative radix-2 complex transform;
  short or non-power-of-two lengths keep the direct oracle. Mamba2 carries the
  recurrent D-by-D state instead of expanding every prior source position.

Correctness and portability:

- Native Release: 38/38 CTest cases pass, including benchmark smoke.
- ASan + UBSan + float-cast-overflow: 37/37 CTest cases pass.
- x86_64 cross-build: AVX2 and AVX-512 GGUF sources compile and 32/32 Rosetta
  CTest cases pass.
- Focused tests cover forced GGUF reference dispatch, all new packed formats,
  M16/M65 panel thresholds, FFT L64 versus direct convolution, multi-step
  Mamba2 versus source expansion, fused projection, norm/quantization,
  fused QKV+RoPE+KV insertion (including bounds), grouped MoE, and online
  attention/MLA entry points.

Measured decisions on Apple M5 Max:

| path | 1-thread candidate / baseline ms | speedup | 6-thread candidate / baseline ms | speedup | decision |
|---|---:|---:|---:|---:|---|
| Q4_0 GEMV N4096 K4096 | 1.1953 / 2.8578 | 2.39x | 0.2798 / 0.5892 | 2.11x | keep direct SIMD decode |
| K/IQ GEMV N1024 K4096 | 8.4295-14.6360 / 13.8664-15.1367 | 1.03x-1.64x | 1.7739-3.1180 / 2.4908-3.6649 | 1.05x-1.40x | keep block decode + SIMD dot |
| Q4_0 packed GEMM M128 N1024 K1408 | 7.4150 / 13.6738 | 1.84x | 1.6954 / 4.9766 | 2.94x | keep M>=64 panel route |
| Q4_K/Q6_K/IQ4_XS GEMM M16 | 1.4007-1.6641 / 9.8004-14.1906 | 6.97x-8.53x | 0.3313-0.3883 / 2.4374-3.5107 | 6.75x-9.04x | keep panel route |
| blocked dense M16 N512 K512 | 0.3336 / 2.9337 | 8.80x | 0.1005 / 3.0997 | 30.85x | keep tiled/parallel route |
| fused Q4_0 SwiGLU N1024 K1408 | 0.2107 / 0.2186 | 1.04x | 0.0687 / 0.0896 | 1.31x | keep fusion; removes intermediate |
| fused QKV+RoPE+KV HQ8 HKV2 D64 K1408 | 0.0801 / 0.0815 | 1.02x | 0.0391 / 0.0397 | 1.02x | keep fusion; removes Q/K/V staging and cache pass |
| streaming LM-head R2 V8192 H1024 | 1.2631 / 1.3157 | 1.04x | 0.4444 / 0.4067 | 0.92x | keep bounded-memory route; no speedup claim |
| grouped MoE R64 E8 K256 I512 | 1.3072 / 1.2815 | 0.98x | 0.4526 / 1.7029 | 3.76x | keep grouping for threaded batches |
| online paged attention B2 HQ8 HKV2 S512 D64 | 0.1942 / 0.2118 | 1.09x | 0.0753 / 0.2233 | 2.96x | keep one-pass online recurrence |
| fused RMS-add-int8 R512 H4096 G128 | 1.0248 / 3.4495 | 3.37x | 0.2545 / 1.3100 | 5.15x | keep row-local NEON route |
| FFT convolution B1 H2 L1024 | 0.0553 / 1.8173 | 32.85x | 0.0608 / 1.8128 | 29.79x | keep radix-2 route |
| recurrent Mamba2 B1 H2 L128 D32 | 0.0986 / 3.7921 | 38.45x | 0.0740 / 3.9839 | 53.83x | keep recurrent state |

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups, 30
  timed samples. The consolidated optimization batch uses a 20 ms minimum
  sample; GGUF, packed-GEMM, and norm sweeps use 5 ms.
- Commands: `quixicore_cpu_bench --preset quick --kernel
  {optimization_plan,qgemv_formats,prerequisites,ported_ops} --threads {1,6}
  --warmup 5 --iters 30 --min-sample-ms {20,5}`.
- Working-tree label: `1970aa1-dirty`.
- Raw results: `perf/results/2026-07-22/optimization-plan-complete-t{1,6}/`,
  `qgemv-allformats-final-t{1,6}/`, `prepacked-allformats-final2-t{1,6}/`, and
  `norm-fused-neon-t{1,6}/` beneath the same date directory.

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
