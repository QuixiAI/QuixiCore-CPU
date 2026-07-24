# Sibling Kernel Port Matrix

Inventory date: 2026-07-24. The source inventory is the union of the
top-level model, training, serving, quantization, and distributed operations
under `QuixiCore-Metal`, `QuixiCore-XPU`, `QuixiCore-CUDA`, and
`QuixiCore-ROCm`, reconciled with the umbrella registries and matrices.

The CPU port is semantic. A GPU repository may expose separate symbols for a
tile size, architecture, pipeline stage, cache layout, partition, or reduction.
Those symbols collapse to one CPU operation when their host-observable result
is the same. Backend programming primitives, tests, demos, educational GEMM
stages, and hardware instructions are not model kernels and are outside this
inventory. Native kernels retain FP32 accumulation, while the universal
storage adapter accepts FP16/BF16 inputs and outputs for every floating
tensor. That is storage coverage, not a claim that every operation has native
half arithmetic or an optimized performance tier.

At this inventory point every distinct top-level sibling operation semantic is
represented by a public CPU entry point or by an explicitly documented
composition below. Correctness tests cover the public candidates. Benchmark
coverage remains deliberately narrower and is indexed in `perf/`.

## Operation mapping

| Sibling area | CPU semantic surface | Notes |
|---|---|---|
| Activations and elementwise | `unary` (all 22 llama selectors), `gelu`, `gelu_backward`, `silu`, `silu_backward`, `glu`, `glu_backward`, `swiglu_oai`, split `sigmoid_mul` and its analytic backward, scalar-bounds `value_clip`, arithmetic/trigonometric elementwise operations, reductions, sorting, repeat/pad/roll, strided set, and `softmax` | Tanh/erf/quick GeGLU, SwiGLU, ReGLU, plain GLU, split Qwen output gating, infinite-bound clipping, and the clamped OpenAI SwiGLU rule are explicit. GPU elementwise launch variants collapse by operation. |
| Norms and norm-quant | `rms_norm`, `layer_norm`, both backward paths, add/residual seams, residual-next, int8/FP8 norm-add quantization | Metal split/fused backward symbols produce one combined CPU result. AddNorm FP8 token/group variants use the generic group-size API. |
| Dense matmul and projections | `dense_gemm`, `dense_gemm_ex`, `linear_epilogue`, `decode_swiglu`, `gemm_gate_residual`, `complex_gemm`, `grouped_gemm`, and `lora_apply_direct_f16` | CUDA/ROCm architecture, transpose, tile, staged, Flux, and educational variants collapse to these mathematical operations and epilogues. Published Metal LoRA semantics map to raw FP16 adapters with FP16 rounding after both projections, optional base fusion, rank 1-256, and FP32/FP16/BF16 activation/output storage. |
| Quantized matmul | `qgemv`, W8A8 GEMV, fused up/gate/QKV GEMV, `qgemm`, backward-input, epilogues, int8/AZP, BitNet W2A8, FP8 scaled/block-scale, act-order, MXFP8/NVFP4 quantize+GEMM, split-layout MXFP4/NVFP4 GEMV, canonical BaseQ2/3/4/5/6/8 dequant/GEMV/GEMM/embedding/fused consumers/LM-head/grouped-MoE, raw FP4x2 conversion, and quantized LM-head operations | Packed weight decode is shared across GEMV/GEMM/MoE/embedding. Accelerator scale swizzles become logical row-major scale tables at the CPU boundary. BaseQ preserves Metal's little-endian bitstream, symmetric/affine rule, group sizes 32/64/128, BF16/F16/E8M0/E4M3 scale storage constraints, output-storage rounding before argmax, lower-token tie breaking, and the existing 32-row padded expert schedule. |
| Dense and variable attention | dense/GQA attention forward/backward, independent-length cross-attention, learned relative audio attention, explicit prep/DQ/DKV stages, forward LSE/sinks/softcap, window and biased/Swin attention, varlen attention, Q/K norm+RoPE, table RoPE, explicit-position partial RoPE, Gemma/Qwen 2-D vision RoPE, and three-axis M-RoPE | Causal/noncausal, MHA/GQA, multiwarp, and GPU head-dimension variants share these entries. Cross-attention preserves per-batch key lengths, optional score bias, automatic/explicit scale, score softcap, and D64/D128/D256. Relative attention preserves chunk/left/right geometry, learned per-dimension query scaling, relative shifts, lengths, and optional softcap. Positioned routes preserve shared/per-batch indices, local-axis/global-split layout, split/adjacent pairing, unrotated tails, and multimodal axis selection. Worklist/pack/regather symbols remain CPU scheduling details. |
| Decode attention and caches | paged, advanced, staged-equivalent, xcache-layout, typed E4M3/E5M2 FP8 and canonical MXFP8 cache insert/gather and direct online paged attention, canonical per-32-scale Q8_0 codec/functional block copy/direct attention, TurboQuant typed codec/query transform/direct packed-cache attention, BitNet a4.8 KV3 typed codec/direct attention, f32/FP8-prefix cascade, f32/FP8-output state merge, dense/sparse FP8 MLA, fused cache decode, quantized dense attention, RoPE+KV insert | Partition/reduce and direct/staged routes are CPU scheduling details. `paged_attention_xcache` adapts the sibling split/transposed cache layout. Q8_0 preserves separate signed-int8 code planes and raw FP16 scale planes; sparse negative blocks zero-fill or are skipped. FP8 bytes, interleaved MXFP8 blocks, TurboQuant packed K/rotated V, and low-bit-first KV3 remain canonical while FP32/FP16/BF16 boundaries share direct cores. TurboQuant canonical K is not transformed; V stays in the signed-FWHT domain until one final inverse transform. KV3 signedness, zero-point mode, group size, and FP16/FP32 scale encoding remain explicit. |
| Sampling and logit processing | argmax/categorical/top-k/top-p/min-p/typical-p; token/bad-word/repetition processors; top-k/top-p renorm; final-logit softcap; quadratic, nsigma, top-a, epsilon, eta, XTC, skew, no-repeat-ngram, and DRY transforms | Partial/reduction symbols in the LM-head and sampler implementations collapse into final selection APIs. Published final-logit softcap is distinct from attention score capping and retains FP32/FP16/BF16 storage. |
| Beam and speculative decode | beam step/remap/copy metadata; linear/tree/ragged rejection verification; recovered-token sampling; compact/KV metadata; EAGLE metadata | Padded and ragged host-observable results are explicit. Tree-building and EAGLE launch stages remain separate where callers consume their metadata. |
| Embedding and multimodal serving | lookup/backward, token+type lookup/add, explicit sorted backward, multimodal source-map build/merge, prefix-length and arbitrary-mask pooled RMS-L2, quantized embedding and embedding-bag | Atomic and sorted GPU accumulation paths produce the same deterministic CPU result. Invalid token/type ids contribute zero; an empty masked pool emits zero. Published BaseRT preparation routes retain FP32/FP16/BF16 storage. |
| MoE | top-k/grouped/scored routing, permute/padded and per-LoRA schedules, gather/finalize and backward, grouped GEMM/SwiGLU and backward, canonical dual-quantized grouped projection, direct next-activation packing, generic packed grouped GEMM/SwiGLU, named FP8/WNA16/NVFP4 grouped GEMMs | Rectangular, padded, quantized, and fused GPU launch variants map to row-shaped expert ids and logical metadata. Sorted expert panels share one CPU dispatch; x86 may decode an activation packet once into workspace when that beats direct packet dots. Distributed MoE dispatch is `all_to_all` plus the same grouped operation. |
| Linear attention | normalized/unnormalized, causal, decayed, Based, Hedgehog, GatedDeltaNet, GLA, and RWKV6/RWKV7 recurrence; varlen GDN recurrence, depthwise short convolution, QKV preparation, decay/beta preparation, and gated RMSNorm | CUDA/ROCm chunk KV/scan/output phases are combined in the final recurrence/attention API. GDN preserves functional FP32 state/history pools, slot mapping, D64/D128 heads, MQA/GQA mapping, and FP32/F16/BF16 tensor boundaries. Aliased slots retain request order; independent state rows and channels use the CPU pool. |
| State-space and convolution | selective scan, stateful varlen/APC scan, Mamba2 forward/backward, SSD decode, direct circular `fft_convolution`, DSV4 hyper-connections, im2col/col2im, 2-D/3-D/depthwise/transposed convolution, NWC audio convolution, symmetric depthwise LightConv+SiLU, causal left-only depthwise convolution, and pooling/backward | SSD chunk phases and row/column backward launches collapse into full results. The direct convolution is the portable correctness oracle for FFT implementations. Audio routes preserve `[B,T,C]`, `[O,K,C]`/`[C,K]`, stride/padding/dilation, optional bias, typed storage, and explicit symmetric versus causal geometry. |
| Training and utility kernels | cross entropy forward/backward, dense/top-k KD-KL and fused KD+CE, deterministic dropout forward/backward, Hadamard/FWHT rotate, packbits/segment-packbits/permute/tau-tail, masked AdamW | Layout/bit utilities grouped by Metal's `Marginal` primitive are explicit CPU functions. |
| Vision and sparse serving | NHWC/NTHWC 2-D/3-D patch-token extraction and learned projection, learned-position bilinear resize, factorized x/y position gather, regular and arbitrary-coordinate token pooling, Gemma/Qwen 2-D RoPE, patch-merge LayerNorm, space-to-depth norm+linear, edge MLP, relative-position/window transforms, timestep embeddings, convolution/pooling, indexer quant/paged gather, vertical/slash sparse conversion, MInference block mask | Packed D32 Swin attention is an explicit adapter over biased attention semantics. BaseRT vision routes preserve padding/stride/temporal patch order, `[O,KH,KW,C]` or `[O,KT,KH,KW,C]` projection weights, half-pixel/aligned-corner interpolation, invalid-token zeroing, coordinate masks, local-axis/global-split RoPE, partial ceil-mode pool windows, and FP32/FP16/BF16 storage. |
| Collectives and parallel composites | all-reduce, all-gather, reduce-scatter, all-to-all, broadcast, reduce, GEMM+AR, AG+GEMM, GEMM+RS | Host-reference buffers make every rank explicit. Ring/Ulysses attention and MoE dispatch are compositions of these collectives with attention or grouped MoE; network transport and overlap strategy are not CPU kernel semantics. |

## Quantization formats

The public format enum and decoder cover the packed formats found across the
sibling quantization trees:

- GGUF/llama.cpp: Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K
  through Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ1_S, IQ1_M,
  IQ4_NL, IQ4_XS, TQ1_0, and TQ2_0.
- Integer/custom: U4B8, U4, HQQ, and BitNet ternary.
- Floating/microscaled: E4M3/E5M2 FP8, block/raw FP8, E2M1 FP4, MXFP8,
  NVFP4, MXFP4, and both MXFP6 encodings.
- Runtime quantization: symmetric/asymmetric int8, signed grouped int4,
  per-row/per-group FP8 with optional power-of-two scales, fake quantization,
  fused SiLU/gate quantization, MXFP8 and NVFP4 producers, raw E2M1 packing,
  ternary statistics, and TurboQuant KV coding.

`qgemv_pack` authors every stored llama.cpp/GGUF format above, using uniform
importance for the seven importance-sensitive IQ encoders. The explicit
`qgemv_pack_weighted` entry accepts a row-major calibration/importance matrix.
Q8_1 and Q8_K remain activation intermediates rather than stored-weight enum
members and are exposed through `quant_activation_pack`/`unpack`.

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
`tests/correctness/`: the core contract tests plus `base_q`, `extended_ops`,
`gguf_formats`, `quantization`, `projection_ops`, `parallel_sampling`,
`serving_metadata`, `remaining_ops`, and `sibling_entrypoints`. These cover known-value or
independently decomposed oracles, invalid arguments, quantized byte layouts,
and the semantic collapses above.

`contract_ops` records representative performance evidence for softmax,
attention, MoE routing, selective scan, and q8_0 QGEMM. `ported_ops` records a
focused fused norm-add/int8-quant run and MXFP8 logical-scale GEMM against a
predecoded dense baseline. `quant_lifecycle` covers every importance-aware IQ
encoder plus Q8 activation layouts, and `llama_parity` covers convolution, GLA,
DSV4, and the complete unary/GLU selectors. `gdn` covers all five live GDN
operations plus split sigmoid multiplication against independent scalar
contracts on Apple AArch64 and native x86-64. `lora` covers the newer
source-tree-only direct F16-adapter route at decode and small-prefill shapes.
`basert_aux`, `basert_embedding`, `basert_vision`, and `basert_audio` cover the
published calibration, output, BERT preparation, 2-D/3-D vision preparation,
position/RoPE, audio front-end, and relative-attention routes across 31 quick
cases and both native Apple and x86-64 hosts.
Exact
commands, hardware, medians, variance, and decisions live in
`perf/optimization_status.md`; this matrix makes no family-wide speed claim.

`include/quixicore_cpu/float_storage.h` is the dtype seam. FP32 buffers remain
zero-copy; FP16/BF16 buffers are decoded before the wrapped operation and
encoded only after success. Exact aliases share scratch, allowing in-place
optimizers and transforms. ARM FP16 and x86 F16C conversion are runtime gated;
all other machines retain the portable bit-exact conversion path.

The separate algorithm excavation from the Colibri CPU runner is inventoried
in `docs/colibri-port-matrix.md`.
