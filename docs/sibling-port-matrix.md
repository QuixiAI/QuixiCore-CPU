# Sibling Kernel Port Matrix

This matrix records the semantic CPU ports derived from the union of
QuixiCore-Metal, QuixiCore-XPU, QuixiCore-CUDA, and QuixiCore-ROCm, reconciled
against the umbrella QuixiCore v0.1 registry. GPU launch strategies, tile
variants, and backend-specific fusions are not separate CPU APIs: one portable
operation implements the shared behavior.

`Implemented` below means a public CPU entry point, deterministic correctness
coverage, and a successful Release/sanitizer/x86_64 test matrix. `Benchmarked`
also has a focused Release run. Neither label expands the stated dtype or
quant-format scope.

## Active umbrella families

| Umbrella family | CPU operation(s) | Scope | Evidence state |
|---|---|---|---|
| `norms` | `rms_norm`, `layer_norm` | f32 | Implemented; RMSNorm benchmarked |
| `softmax` | `softmax` | f32 | Benchmarked |
| `activations` | `gelu`, `gelu_backward`, `silu`, `glu` | f32; erf/tanh GELU and SwiGLU/GeGLU/ReGLU/GLU | Implemented |
| `causal_attention` | `attention`, `rope` | f32; MHA/GQA, causal or non-causal, NeoX RoPE | Attention benchmarked |
| `paged_attention` | `paged_attention` | f32; paged KV cache, optional sliding window | Implemented |
| `mla_decode` | `mla_decode` | f32 latent-cache decode | Implemented |
| `quant_gemv` | `qgemv`, `qgemv_w8a8` | f32 activation with GGUF q8_0/q4_0 weights; optional blockwise int8 activation | Benchmarked |
| `quant_gemm` | `qgemm` | f32 activation, GGUF q8_0 weights | Benchmarked |
| `quantized_lm_head` | `quantized_lm_head_argmax` | f32 activation, GGUF q8_0 weights | Implemented |
| `sampling` | `argmax_sample`, `sample_categorical`, `top_k_sample`, `top_p_sample`, `min_p_sample` | f32 logits | Implemented |
| `beam_search` | `beam_search_step` | f32 logits and cumulative scores | Implemented |
| `speculative_decode` | `speculative_verify` | linear draft verification | Implemented |
| `mamba_ssd` | `selective_scan` | f32 Mamba S6 recurrence | Benchmarked |
| `moe_routing` | `moe_route_topk` | f32, stable expert-id tie break | Benchmarked |
| `grouped_moe_gemm` | `grouped_gemm` | f32 grouped dense GEMM | Implemented |
| `optimizers` | `adamw` | f32 fused in-place AdamW | Implemented |

Correctness lives in `tests/correctness/test_ops.cpp`,
`tests/correctness/test_qgemv.cpp`, and
`tests/correctness/test_rms_norm.cpp`. Focused measurements and exact commands
are recorded in `perf/optimization_status.md`; raw output is under
`perf/results/2026-07-22/all-kernels-final-{t1,t6}/`.

## Sibling-only shared utilities

The sibling inventory also exposed reusable operations outside the 16 active
umbrella family keys. Their portable f32 CPU candidates are:

- dense row-major GEMM and non-causal linear attention;
- embedding lookup and KV-cache scatter/gather;
- deterministic dropout, cross entropy, and Hadamard transform.

These are CPU extensions until the umbrella registry assigns them contract
entries.

## Deliberate non-claims

This batch does not claim fp16, bf16, FP8, FP4/MX, ternary, or GGUF formats
other than q8_0 and q4_0. It also does not claim GPU-specific tile variants,
tensor-core/MFMA/AMX implementations, distributed collectives, vision kernels,
attention/norm backward passes, FFT convolution, quantized MoE fusions, or the
many repo-local composite epilogues. Those require explicit shared semantics,
CPU-appropriate APIs, correctness fixtures, and focused benchmark evidence
before they can be advertised.
