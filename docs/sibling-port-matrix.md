# Sibling Kernel Port Matrix

Inventory date: 2026-07-22. The source inventory is the union of the
top-level model, training, serving, quantization, and distributed operations
under `QuixiCore-Metal`, `QuixiCore-XPU`, `QuixiCore-CUDA`, and
`QuixiCore-ROCm`, reconciled with the umbrella registries and matrices.

The CPU port is semantic. A GPU repository may expose separate symbols for a
tile size, architecture, pipeline stage, cache layout, partition, or reduction.
Those symbols collapse to one CPU operation when their host-observable result
is the same. Backend programming primitives, tests, demos, educational GEMM
stages, and hardware instructions are not model kernels and are outside this
inventory. The result is a portable f32 reference surface, not a claim of
fp16/bf16 ABI support or an optimized performance tier.

At this inventory point every distinct top-level sibling operation semantic is
represented by a public CPU entry point or by an explicitly documented
composition below. Correctness tests cover the public candidates. Benchmark
coverage remains deliberately narrower and is indexed in `perf/`.

## Operation mapping

| Sibling area | CPU semantic surface | Notes |
|---|---|---|
| Activations and elementwise | `gelu`, `gelu_backward`, `silu`, `glu`, `glu_backward`, `add`, `softmax` | Erf/tanh GELU and SwiGLU/GeGLU/ReGLU/plain GLU modes are explicit. GPU elementwise launch variants collapse by operation. |
| Norms and norm-quant | `rms_norm`, `layer_norm`, both backward paths, add/residual seams, residual-next, int8/FP8 norm-add quantization | Metal split/fused backward symbols produce one combined CPU result. AddNorm FP8 token/group variants use the generic group-size API. |
| Dense matmul and projections | `dense_gemm`, `dense_gemm_ex`, `linear_epilogue`, `decode_swiglu`, `gemm_gate_residual`, `complex_gemm`, `grouped_gemm` | CUDA/ROCm architecture, transpose, tile, staged, Flux, and educational variants collapse to these mathematical operations and epilogues. |
| Quantized matmul | `qgemv`, W8A8 GEMV, fused up/gate/QKV GEMV, `qgemm`, backward-input, epilogues, int8/AZP, BitNet W2A8, FP8 scaled/block-scale, act-order, MXFP8/NVFP4 quantize+GEMM, split-layout MXFP4/NVFP4 GEMV, raw FP4x2 conversion, and quantized LM-head operations | Packed weight decode is shared across GEMV/GEMM/MoE/embedding. Accelerator scale swizzles become logical row-major scale tables at the CPU boundary. |
| Dense and variable attention | dense/GQA attention forward/backward, explicit prep/DQ/DKV stages, forward LSE/sinks/softcap, window and biased/Swin attention, varlen attention, Q/K norm+RoPE, table RoPE | Causal/noncausal, MHA/GQA, multiwarp, and GPU head-dimension variants share these entries. Worklist/pack/regather symbols remain CPU scheduling details. |
| Decode attention and caches | paged, advanced, staged-equivalent, xcache-layout, FP8 paged, f32/FP8-prefix cascade, f32/FP8-output state merge, dense/sparse FP8 MLA, fused cache decode, quantized dense attention, RoPE+KV insert | Partition/reduce and direct/staged routes are CPU scheduling details. `paged_attention_xcache` adapts the sibling split/transposed cache layout. |
| Sampling and logit processing | argmax/categorical/top-k/top-p/min-p/typical-p; token/bad-word/repetition processors; top-k/top-p renorm; quadratic, nsigma, top-a, epsilon, eta, XTC, skew, no-repeat-ngram, and DRY transforms | Partial/reduction symbols in the LM-head and sampler implementations collapse into final selection APIs. |
| Beam and speculative decode | beam step/remap/copy metadata; linear/tree/ragged rejection verification; recovered-token sampling; compact/KV metadata; EAGLE metadata | Padded and ragged host-observable results are explicit. Tree-building and EAGLE launch stages remain separate where callers consume their metadata. |
| Embedding and multimodal serving | lookup/backward, explicit sorted backward, multimodal source-map build/merge, pooled RMS-L2, quantized embedding and embedding-bag | Atomic and sorted GPU accumulation paths produce the same deterministic CPU result. |
| MoE | top-k/grouped/scored routing, permute/padded and per-LoRA schedules, gather/finalize and backward, grouped GEMM/SwiGLU and backward, generic packed grouped GEMM/SwiGLU, named FP8/WNA16/NVFP4 grouped GEMMs | Rectangular, padded, quantized, and fused GPU launch variants map to row-shaped expert ids and logical metadata. Distributed MoE dispatch is `all_to_all` plus the same grouped operation. |
| Linear attention | normalized/unnormalized, causal, decayed, Based, Hedgehog, and GatedDeltaNet recurrence | CUDA/ROCm chunk KV/scan/output phases are combined in the final recurrence/attention API. |
| State-space and convolution | selective scan, stateful varlen/APC scan, Mamba2 forward/backward, SSD decode, direct circular `fft_convolution` | SSD chunk phases and row/column backward launches collapse into full results. The direct convolution is the portable correctness oracle for FFT implementations. |
| Training and utility kernels | cross entropy forward/backward, dense/top-k KD-KL and fused KD+CE, deterministic dropout forward/backward, Hadamard/FWHT rotate, packbits/segment-packbits/permute/tau-tail, masked AdamW | Layout/bit utilities grouped by Metal's `Marginal` primitive are explicit CPU functions. |
| Vision and sparse serving | patch-merge LayerNorm, space-to-depth norm+linear, edge MLP, indexer quant/paged gather, vertical/slash sparse conversion, MInference block mask | Packed D32 Swin attention is an explicit adapter over biased attention semantics. |
| Collectives and parallel composites | all-reduce, all-gather, reduce-scatter, all-to-all, broadcast, reduce, GEMM+AR, AG+GEMM, GEMM+RS | Host-reference buffers make every rank explicit. Ring/Ulysses attention and MoE dispatch are compositions of these collectives with attention or grouped MoE; network transport and overlap strategy are not CPU kernel semantics. |

## Quantization formats

The public format enum and decoder cover the packed formats found across the
sibling quantization trees:

- GGUF: q4_0, q4_1, q5_0, q5_1, q8_0, q2_K through q6_K, IQ4_NL, IQ4_XS,
  IQ2_XXS, IQ2_XS, IQ3_XXS, and IQ1_S.
- Integer/custom: U4B8, U4, HQQ, BitNet ternary, and TQ2_0.
- Floating/microscaled: E4M3/E5M2 FP8, block/raw FP8, E2M1 FP4, MXFP8,
  NVFP4, MXFP4, and both MXFP6 encodings.
- Runtime quantization: symmetric/asymmetric int8, signed grouped int4,
  per-row/per-group FP8 with optional power-of-two scales, fake quantization,
  fused SiLU/gate quantization, MXFP8 and NVFP4 producers, raw E2M1 packing,
  ternary statistics, and TurboQuant KV coding.

`qgemv_pack` deliberately authors only q8_0, q4_0, and TQ2_0. The other
entries consume sibling/GGUF packed bytes and have exact decoder coverage;
this distinction avoids claiming a packer that the sibling contract does not
provide.

## Semantic collapses

| Sibling implementation symbols | CPU representation |
|---|---|
| `AttnBwdPrep`, `AttnBwdDQ`, `AttnBwdDKV` | explicit staged calls plus one `attention_backward` convenience result |
| paged/MLA partition and reduction stages | one paged/MLA decode call |
| LM-head partial/top-k/top-p/argcat reducers | final LM-head candidate, sample, beam, or constrained call |
| linear-attention and SSD chunk KV/scan/output phases | final attention/recurrence call |
| architecture/tile/direct/staged GEMM variants | one dense or quantized GEMM semantic plus explicit epilogue |
| CUDA `fused_layernorm` and `fused_rotary` launchers | `dropout` + `layer_norm_add`, and `rope_table`, respectively |
| CUDA `attn_q` and staged paged/MLA launchers | `quantized_attention` or one final paged/MLA decode call |
| CUDA sampler/beam/EAGLE spelling aliases | the typed sampling, beam-search, speculative, and EAGLE metadata entries |
| CUDA/ROCm ring or Ulysses attention | collective composition plus `attention` |
| CUDA/ROCm distributed MoE dispatch | `all_to_all` plus MoE gather/grouped GEMM/finalize |
| Metal atomic versus sorted embedding backward | deterministic `embedding_backward` |

## Evidence

Correctness is split across the focused executables under
`tests/correctness/`: the core contract tests plus `extended_ops`,
`gguf_formats`, `quantization`, `projection_ops`, `parallel_sampling`,
`serving_metadata`, `remaining_ops`, and `sibling_entrypoints`. These cover known-value or
independently decomposed oracles, invalid arguments, quantized byte layouts,
and the semantic collapses above.

`contract_ops` records representative performance evidence for softmax,
attention, MoE routing, selective scan, and q8_0 QGEMM. `ported_ops` records a
focused fused norm-add/int8-quant run and MXFP8 logical-scale GEMM against a
predecoded dense baseline. Exact
commands, hardware, medians, variance, and decisions live in
`perf/optimization_status.md`; this matrix makes no family-wide speed claim.
