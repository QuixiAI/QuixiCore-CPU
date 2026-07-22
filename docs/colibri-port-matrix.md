# Colibri CPU Algorithm Port Matrix

Inventory date: 2026-07-22. This audit covers the reusable CPU compute logic
in `.reference/colibri/c/quant.h`, `colibri.c`, `olmoe.c`, and `sample.h`, plus
the E8/IQ3 conversion algorithm in `c/tools/iq3_pack.py`. Colibri organizes
most of this logic inside a model runner; the CPU backend exposes it as
independent kernels while preserving QuixiCore validation, layouts, threading,
and runtime ISA dispatch.

The word "ported" below means that the algorithm has a public CPU operation,
correctness evidence, and—where it changes a hot path—a focused benchmark. It
does not mean that Colibri's model runner, file formats, or server API became
part of the QuixiCore kernel contract.

## Quantized compute and conversion

| Colibri algorithm | CPU kernel surface | Implementation notes |
|---|---|---|
| `matmul` | `dense_gemm` | Existing f32 GEMM covers the same row-major projection. |
| `matmul_q` | `qgemm_w8a32` | Row-scaled int8 weights with f32 activations; portable, NEON, AVX2, and AVX-512 routes. |
| `qrow_i8`, `dot_i8i8`, `matmul_q_idot` | `quantize_int8`, `int8_gemm` | Prequantized W8A8 route with exact integer accumulation, asymmetric zero-point correction, and DotProd/I8MM/AVX2/AVX-512 VNNI dispatch. |
| `matmul_i4`, `matmul_i4_grouped` | `lowbit_gemm(kInt4Row/kInt4Group)` | Per-row and per-group packed int4 f32-activation kernels with NEON, AVX2, and AVX-512 inner loops where available. |
| `dot_i4i8`, `matmul_i4_idot` | `lowbit_gemm_w8a8` | Dynamic activation quantization is performed once per input row; DotProd, I8MM, AVX2, and AVX-512 VNNI routes retain exact integer-dot semantics. |
| `matmul_i4_pair`, `expert_gate_up` | `lowbit_gemm_pair`, `qgemv_up_gate` | Gate/up projections share one scheduling pass. The portable paired low-bit loop remains the exact fallback. |
| `matmul_i2` | `lowbit_gemm(kInt2Row)` | Adjacent 2-bit fields use the same implicit signed offset as Colibri. |
| `matmul_i3` | `lowbit_gemm(kInt3Group64)` | Dual-plane 24-byte blocks and one f32 scale per 64 values. |
| `quantize_rows`, `pack_int4`, `pack_int3_g64`, `pack_int2` | `lowbit_pack`, `lowbit_unpack` | Conversion-time producers and exact decoders for every Colibri integer layout. |
| `qt_matvec_rows` | `qgemv_rows`, `lowbit_gemv_rows` | Computes only requested packed rows; used by candidate vocabularies and absorbed projections. |
| `qt_addrow`, `axpy_i4f_avx512` | `qgemv_axpy_row`, `lowbit_axpy_row` | Accumulates one packed row without materializing a dense matrix. |
| E8/IQ3 block decode and `matmul_e8` | `e8iq3_unpack`, `e8iq3_gemm` | Consumes Colibri's 98-byte/256-value lattice block, which is deliberately distinct from GGUF IQ3_XXS. |
| `iq3_pack.py` nearest-grid encoder | `e8iq3_pack` | Implements sign-parity repair, RMS super-scale, exhaustive subscale search, nearest published grid lookup, and packed sign/scale words. A fixed Colibri-produced byte oracle guards compatibility. |
| `e8_fwht`, `e8_rot_rows` | `e8iq3_rotate` | Deterministic signed block FWHT, forward and inverse, with in-place support. |

The older `olmoe.c` dense/int8 implementation is a subset of the first three
rows and does not introduce an additional packed format or mathematical
operation.

## Attention, selection, sampling, and MoE

| Colibri algorithm | CPU kernel surface | Implementation notes |
|---|---|---|
| `rmsnorm`, `layernorm`, `softmax`, `siluf` | Existing norm, softmax, and activation APIs | The CPU contract already supplied the same operations; no duplicate Colibri-specific entry point is needed. |
| `rope_interleave` | `rope_interleaved_to_split` | Applies adjacent-pair RoPE and writes split real/imaginary halves; exact in-place operation is supported. |
| MLA weight absorption in `attention_rows` | `quantized_mla_decode_absorbed` | Projects non-RoPE queries into latent space, attends paged latent/RoPE caches, and projects the mixed latent vector into values without reconstructing per-token K/V. It accepts every supported GGUF `QuantFormat`. |
| DSA sparse MLA path | `quantized_mla_decode_absorbed_sparse` | Consumes selected logical token positions through the same block table. Indexer projection/norm/RoPE are compositions of existing projection and norm kernels. |
| `partial_select_desc` | `threshold_topk_indices` | Median-of-three threshold selection is average linear time; source order and lowest-index threshold ties are deterministic. |
| `dist_build` max-heap nucleus construction | `top_p_sample` | Replaces a full vocabulary sort with partial heap extraction while preserving the prior higher-logit/lower-token ordering and seeded result. Non-finite logits remain rejected by the QuixiCore API contract. |
| MoE expert batch union | `moe_grouped_qgemm`, `moe_grouped_qswiglu` | Stable expert-major grouping reuses each packed output row across routed tokens in one pool dispatch. Decode/all-unique/one-thread cases retain the normal ISA-dispatched per-row GEMV path. |
| dense MLP, attention projection, embedding row lookup | Existing projection, activation, attention, and quantized-embedding APIs | These are compositions of kernels above rather than distinct numerical primitives. |

## Deliberate non-kernel scope

The audit also classified the remaining Colibri code so omissions are
explicit:

- CUDA and Metal backends, loader bridges, device residency, and multi-GPU
  pipelines are accelerator/runtime policy, not CPU tensor algorithms.
- Safetensors loading, mmap/pread/io_uring, expert tiering, cache heat,
  prefetch, NUMA placement, RAM sizing, and KV persistence are storage or
  memory-management systems.
- Tokenization, Unicode tables, grammar/schema parsing, stop strings, HTTP
  serving, telemetry, profiling, and CLI code are application/runtime logic.
- N-gram/MTP draft orchestration and model-layer scheduling call existing
  speculative, projection, attention, and cache kernels; they do not define a
  new standalone tensor result in the QuixiCore contract.
- Colibri tests and benchmark fixtures are evidence for its implementation,
  not additional kernels. Their numerical ideas were used to construct the
  independent CPU tests and benchmark baselines.

Within the audited source set, every reusable CPU tensor, quantization,
selection, sampling, and routed-projection algorithm is represented by the
kernel surfaces above. This is an algorithm-coverage statement, not a blanket
performance-tier or dtype-support claim.

## Evidence

Focused correctness lives in `test_lowbit`, `test_lowbit_w8a8`,
`test_int8_gemm`, `test_w8a32_gemm`, `test_colibri_algorithms`,
`test_mla_absorb`, and `test_moe_batch_union`. The native Release and
sanitizer suites pass 35/35; the x86_64 cross-build/Rosetta suite passes 29/29.

`colibri_ops` benchmarks W8A32, prequantized IDOT, int4 f32, dynamic W4A8,
heap top-p, threshold top-k, and shared-expert MoE against scalar, full-sort,
or per-row baselines. Exact commands, medians, variance, hardware, and keep
decisions are recorded in `perf/optimization_status.md`.
