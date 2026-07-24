# Full Quantization Matrix Implementation Plan

Plan date: 2026-07-22.

This document is the execution plan for complete CPU inference coverage of:

- NVFP4, MXFP4, and MXFP8;
- FP8 E4M3 and E5M2, including W8A16 and quantized-activation paths;
- INT4 W4A16, W4A8, and W4A4;
- INT8 W8A16 and W8A8;
- AWQ, GPTQ, SmoothQuant, AutoRound, and TurboQuant; and
- BitNet b1.58 and BitNet a4.8.

It describes work that remains to be implemented. It does not assert that any
unmeasured format or performance tier is already supported.

## Definition of complete

A format or scheme is complete only when every applicable layer below is
complete:

1. Its canonical serialized layout and numerical behavior are specified.
2. Checkpoint import or quantization can produce those canonical bytes.
3. Pack, unpack, and dequantization have deterministic correctness evidence.
4. Decode GEMV and prefill GEMM have portable reference implementations.
5. FP32, FP16, and BF16 activation storage is accepted with FP32 accumulation.
6. Required CPU ISA routes are runtime-dispatched without changing semantics.
7. Transformer fusions, embeddings, LM head, and MoE use packed weights without
   materializing a complete dequantized matrix.
8. Any advertised KV-cache format has direct cache insertion, gathering, and
   attention consumption.
9. Focused correctness tests cover aligned, ragged, zero, saturated, non-finite,
   and invalid inputs.
10. Every performance-sensitive kernel completes three measured optimization
    passes and records kept and rejected experiments in
    `perf/optimization_status.md`.
11. Benchmark artifacts exist under `perf/results/`, and accepted baselines are
    indexed in `perf/baseline_status.md`.
12. The CPU parity ledger and sibling matrix are updated only after the evidence
    above exists.

Inference is the primary scope. Direct packed backward-input kernels are part of
the final completeness phase, but weight-gradient quantization, quantization-
aware training, and calibration algorithms remain outside the CPU runtime.

## Contract and repository constraints

Before changing a public format or operation, re-read:

- `../registry/kernels.yaml`;
- `../registry/quant-formats.yaml`;
- `../registry/benchmark-shapes.yaml`;
- `../registry/tolerances.yaml`;
- `../matrices/`;
- `../specs/formats/mx-formats.md`;
- `../specs/formats/fp8.md`;
- `../specs/formats/fp4.md`;
- `../specs/formats/bitnet.md`;
- `../specs/formats/integer.md`;
- `../specs/formats/schemes.md`;
- `../specs/formats/turboquant.md`; and
- the relevant kernel spec under `../specs/kernels/`.

M0 expanded the former umbrella outlines with exact rounding, saturation,
scale sharing, block sizes, NaN/Infinity behavior, zero-point conventions, and
accumulation behavior. CPU-private panels, interleaves, LUTs, and ISA layouts
must never become serialized QuixiCore formats.

The following invariants apply throughout the work:

- Existing numeric values in `QuantFormat` remain stable; new values append.
- Canonical bytes remain portable and owned by `CpuPackedWeights`.
- Prepared CPU panels are rebuildable caches, not model artifacts.
- FP16/BF16 are storage types. Unless a contract says otherwise, products and
  reductions accumulate in FP32.
- An offline algorithm name does not automatically imply a new dot product.
  AWQ, GPTQ, and AutoRound share grouped low-bit compute where their emitted
  layouts and metadata permit it.
- Quantization error is evaluated separately from kernel error. Kernel
  correctness compares against dequantization of the exact emitted codes.
- ISA sources compile with isolated flags and are entered only after runtime
  hardware and operating-system checks.

## Sibling parity rule

The target is semantic parity with QuixiCore-Metal, QuixiCore-CUDA,
QuixiCore-ROCm, and QuixiCore-XPU, not a literal copy of accelerator launch
variants. Track parity by the tuple:

```text
public operation x weight format x activation storage/format x execution mode
    x fused epilogue x cache mode x correctness tier x benchmark tier
```

For every supported sibling tuple, the CPU backend must provide either a direct
kernel or a documented composition that preserves the umbrella contract. GPU
warps, threadgroups, tensor-core fragments, and device-local layouts translate
to CPU-native SIMD lanes, cache-resident panels, matrix instructions, and thread
scheduling. Accelerator-private packing must not leak into the shared API.

Parity is closed only when the tuple has a portable oracle, runtime dispatch,
focused correctness evidence, and the required three-pass optimization record.
A symbol with an untested scalar body does not close a parity cell. Conversely,
multiple sibling launch specializations may map to one shape-dispatched CPU
kernel family when they expose the same public semantics.

At the start of each milestone, regenerate a sibling inventory from all four
repositories and reconcile it against `registry/kernels.yaml` and the CPU parity
ledger. Classify every unmatched sibling entry as one of: implement directly,
compose from existing CPU primitives, accelerator-private/non-contract, or
blocked by an umbrella specification change. Only the first two classifications
count as CPU parity.

## Existing CPU foundation

The implementation should extend the following code rather than create a
parallel quantization subsystem:

| Existing surface | Current role | Planned use |
|---|---|---|
| `include/quixicore_cpu/qgemv.h` | `QuantFormat`, canonical packed GEMV, selected rows, AXPY, fused decode projections | Extend authoring and direct packed dots without changing existing format values |
| `include/quixicore_cpu/qgemm.h` | Quantized GEMM, W8A8, FP8, MXFP8, NVFP4, act-order, BitNet, LM head | Add missing GEMV twins, dual-operand modes, and prepared dispatch |
| `include/quixicore_cpu/quantization.h` | Integer/FP8 quantizers, MXFP8/NVFP4, BitNet, TurboQuant | Add missing canonical packers, MXFP4, A4, and cache-consuming operations |
| `include/quixicore_cpu/float_storage.h` | Universal FP32/FP16/BF16 storage adapter | Correctness fallback; hot quantized kernels must eventually consume F16/BF16 directly |
| `include/quixicore_cpu/packed_weights.h` | Canonical bytes plus CPU-private row panels | Generalize into versioned, format-aware prepared panels |
| `kernels/quantization/gguf_ref.cpp` | Canonical block metadata and element/block decode | Portable numerical oracle for weight-only formats |
| `kernels/quantization/qgemv_gguf_*.cpp` | Existing packed SIMD GEMV routes | Home for direct weight-only block dots where practical |
| `kernels/quantization/qgemm_panel.cpp` | Reusable multi-row prepared panel | Replace generic per-element decode with format-specialized microkernels |
| `kernels/quantization/int8_gemm_*.cpp` | Native symmetric/asymmetric W8A8 | SmoothQuant core and model for integer dispatch |
| `kernels/quantization/lowbit_*.cpp` | INT2/3/4 weight-only GEMM and paired projection | Base for W4A16 and selected-row operations |
| `kernels/quantization/lowbit_w8a8_*.cpp` | Symmetric row-scaled W4A8 | Extend to grouped and affine layouts |
| `kernels/quantization/microscale_ref.cpp` | MXFP8, MXFP4, and NVFP4 references | Split into portable codecs and ISA compute routes |
| `kernels/quantization/qgemm_extended_ref.cpp` | FP8, act-order, BitNet, epilogue references | Retain as oracle while removing allocations and scalar hot loops |
| `kernels/quantization/qgemv_fused_ref.cpp` | Gate/up and QKV fusions | Drive all new formats through shared block-dot interfaces |
| `kernels/serving/serving_quant_ref.cpp` | Packed embedding and embedding-bag | Add direct block/vector decode implementations |
| `kernels/quantization/lm_head_ref.cpp` | Streaming packed LM-head selection | Add format-specialized multi-row dot tiles |
| `kernels/moe/moe_extended_ref.cpp` | Generic packed MoE and named FP8/WNA16/NVFP4 seams | Add prepared expert tiles and fused SwiGLU |
| `kernels/quantization/turboquant_ref.cpp` | TurboQuant encode/decode oracle | Add compressed-cache attention rather than decode-before-attention |

Current scalar semantics are valuable correctness anchors, but they are not an
optimized support claim. In particular, generic `qgemm` still calls GEMV once
per input row, the generic prepared panel decodes unsupported formats to float,
and several microscale and BitNet entry points are scalar references.

## Reference snapshot and routing

Record these revisions with any port derived from the cloned reference trees.
Paths are relative to this repository unless otherwise noted.

| Reference | Revision | Use in this plan | Primary routes to inspect |
|---|---|---|---|
| BitNet | `16da220ae2b510caff437d403288882687f44ae5` | Ternary layouts, LUT construction, prepared transforms, CPU scheduling | `.reference/BitNet/src/ggml-bitnet-lut.cpp`, `.reference/BitNet/src/ggml-bitnet-mad.cpp`, `.reference/BitNet/include/ggml-bitnet.h`, `.reference/BitNet/utils/codegen_tl1.py`, `.reference/BitNet/utils/codegen_tl2.py` |
| T-MAC | `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee` | CPU LUT GEMV/GEMM for 1/2/4-bit weights and GPTQ ingestion | `.reference/T-MAC/python/t_mac/ops/qgemm.py`, `.reference/T-MAC/python/t_mac/intrins/lut_ctor.py`, `.reference/T-MAC/python/t_mac/model_utils.py`, `.reference/T-MAC/deploy/tuned/` |
| TorchAO | `1580a9ec67e01d818cff40c3ab8610a44d31c851` | MX/NV numerical formats, FP8 scaling, INT4 packing, CPU low-bit behavior | `.reference/ao/torchao/prototype/mx_formats/`, `.reference/ao/torchao/quantization/quantize_/workflows/int4/`, `.reference/ao/torchao/csrc/cpu/` |
| AutoRound | `6ff414b15728f97848936551021d0b180dd40320` | Export metadata and adapters for AWQ/GPTQ/FP8/MXFP4/MXFP8/NVFP4 | `.reference/auto-round/auto_round/formats.py`, `.reference/auto-round/auto_round/export/export_to_autoround/`, `.reference/auto-round/auto_round/export/export_to_awq/`, `.reference/auto-round/auto_round/export/export_to_autogptq/` |
| CUTLASS | `2802e228c23b8c09f946a3a46e56df35939d34e2` | Cross-check FP8/FP4/MX/NV encoding and scaling, not CPU scheduling | `.reference/cutlass/include/cutlass/float8.h`, `.reference/cutlass/tools/library/src/reference/block_scaled_gemm_*`, `.reference/cutlass/examples/72_blackwell_narrow_precision_gemm/` |
| GPTQ | `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` | Original group quantization and checkpoint field semantics | `.reference/gptq/quant.py`, `.reference/gptq/gptq.py`, `.reference/gptq/quant_cuda_kernel.cu` |
| KleidiAI | `13cd35993d8439143aff1e756a862d366acded0d` | ARM packing and W4A8/W4A16 microkernel structure | `.reference/kleidiai/kai/ukernels/matmul/`, especially `matmul_clamp_*qai8*_*qsi4*`, and `.reference/kleidiai/kai/ukernels/matmul/pack/` |
| LLM-AWQ | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | Groupwise affine U4, calibration-produced scales/zeros, W4A16/W8A8 semantics | `.reference/llm-awq/awq/quantize/quantizer.py`, `.reference/llm-awq/awq/quantize/qmodule.py`, `.reference/llm-awq/awq/kernels/csrc/quantization_new/` |
| SmoothQuant | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | Offline smoothing and W8A8 metadata semantics | `.reference/smoothquant/smoothquant/smooth.py`, `.reference/smoothquant/smoothquant/calibration.py`, `.reference/smoothquant/examples/export_int8_model.py` |
| QuixiCore Metal | `bc968fc3215e` | Shared operation semantics, format coverage, and fused boundaries | `../QuixiCore-Metal/include/metal/ops/warp/register/tile/dequant.metal`, `../QuixiCore-Metal/kernels/quantization/`, `../QuixiCore-Metal/kernels/attention/attn_q/`, `../QuixiCore-Metal/kernels/moe/moe/` |
| QuixiCore CUDA | `7519b317c901` | CUDA operation inventory, tensor-core paths, quantized fusions, and shape variants | `../QuixiCore-CUDA/kernels/`, `../QuixiCore-CUDA/include/` |
| QuixiCore ROCm | `16c7fd636b1d` | HIP operation inventory, AMD format routes, wave-level fusions, and shape variants | `../QuixiCore-ROCm/kernels/`, `../QuixiCore-ROCm/include/` |
| QuixiCore XPU | `67c70fe4dc0c` | XPU operation inventory, subgroup paths, quantized fusions, and shape variants | `../QuixiCore-XPU/kernels/`, `../QuixiCore-XPU/include/` |
| vLLM Marlin | local checkout | AWQ/GPTQ repacking, act-order, WNA16 and MoE layout ideas | `/Users/eric/vllm/csrc/libtorch_stable/quantization/marlin/`, `/Users/eric/vllm/csrc/libtorch_stable/moe/marlin_moe_wna16/` |
| llama.cpp | local checkout | Portable CPU quant blocks, paired activation formats, ISA dispatch, AMX/KleidiAI integration | `/Users/eric/llama.cpp/ggml/src/ggml-quants.c`, `/Users/eric/llama.cpp/ggml/src/ggml-cpu/quants.c`, `/Users/eric/llama.cpp/ggml/src/ggml-cpu/repack.cpp`, `/Users/eric/llama.cpp/ggml/src/ggml-cpu/amx/` |

Reference code supplies algorithms, layouts, and benchmark ideas. Before
transcribing code, verify its license and record whether the result is a clean
implementation, an adapted implementation, or a byte-compatible port.

## Architecture of the implementation

The implementation is divided into four layers.

```text
external checkpoint
    -> scheme-specific importer
    -> canonical QuixiCore packed bytes + metadata
    -> CPU-private prepared panel/LUT
    -> runtime-dispatched GEMV/GEMM/fused microkernel
```

### Canonical layouts

Canonical layouts are stable, portable, serializable, and sufficient for
reference decode. Existing `QuantFormat` blocks remain the canonical weight-
only representation. Split-code APIs such as dual-operand MXFP8 and NVFP4 need
their scale layout specified at the public boundary before consolidation.

Canonical metadata must describe, where applicable:

- element encoding and signedness;
- K block or group size;
- packed byte order;
- weight and activation scale granularity;
- zero-point granularity and bias convention;
- global and local scale order;
- act-order permutation or GPTQ `g_idx`;
- static versus dynamic activation scaling; and
- any sparse mask or centroid table.

### Scheme importers

Importers convert external model artifacts to canonical bytes. They are not hot
inference kernels. The initial adapters are:

- `import_awq_u4` for affine groupwise U4;
- `import_gptq_u4` for symmetric/asymmetric GPTQ and optional act-order;
- `import_autoround` for AutoRound INT, FP8, MX, and NV exports;
- `import_smoothquant_w8a8` for per-channel weights and activation metadata;
- `import_bitnet` for canonical ternary and compatible I2_S/TL layouts; and
- raw canonical import for FP8, MXFP8, MXFP4, and NVFP4.

No public algorithm-specific GEMM is added when an importer can normalize the
artifact to a shared compute representation.

### Prepared CPU layouts

`CpuPackedWeights` remains the owner of canonical bytes and an optional prepared
representation. Extend its private description to include:

- prepared-layout version;
- row and K tile;
- scale/zero-point side-table offsets;
- optional row sums;
- optional act-order map;
- optional LUT or sign planes;
- required ISA; and
- preparation cost and byte size for diagnostics.

Prepared formats are selected from format, operation shape, activation mode,
and ISA. A decode-oriented M=1 layout may differ from a prefill M>=16 layout.
Preparation must be deterministic, bounded, cacheable, and excluded from steady-
state kernel timing while still receiving its own conversion benchmark.

### Microkernel interface

Use internal microkernel interfaces rather than branching on the format inside
the innermost element loop. The interfaces should cover:

- one packed weight block times F32/F16/BF16 activation;
- one packed weight block times A8 or A4 activation plus combined scales;
- a row tile times an M tile for blocked GEMM;
- paired output panels for gate/up;
- selected rows and row AXPY; and
- optional bias and activation epilogues.

Reference, ARM, and x86 implementations share validation and dispatch. Format
selection is hoisted before parallel regions.

## Requested format and scheme matrix

The matrix below defines the work required for each requested name. “Shared”
means the scheme reuses the listed primitive after import; it does not require a
separate arithmetic kernel.

| Requested coverage | Canonical/import work | Activation work | Core projection | Fused/serving work | KV/attention work |
|---|---|---|---|---|---|
| NVFP4 | E2M1 values, E4M3 local scale per 16, global scale, 1-D/2-D scale modes | Dynamic/static NVFP4 quantization from F32/F16/BF16 | Weight-only W4A16 GEMV/GEMM and dual NVFP4 GEMV/GEMM | Epilogue, gate/up, QKV, embedding, LM head, MoE | Optional only when selected as a cache format; do not infer from weight support |
| MXFP4 | E2M1 values plus E8M0 scale per 32 | Add MXFP4 activation quantize/dequantize | Weight-only W4A16 and dual MXFP4 W4A4 GEMV/GEMM | Epilogue, gate/up, QKV, embedding, LM head, MoE | Not required unless added to the umbrella cache contract |
| MXFP8 | E4M3 values plus E8M0 scale per 32 | Complete direct F32/F16/BF16 quantization | Weight-only W8A16 and dual MXFP8 GEMV/GEMM | Epilogue, gate/up, QKV, embedding, LM head, MoE | MXFP8 cache insertion, gather, and direct paged attention |
| FP8 E4M3 | Raw and scaled canonical forms; tensor/row/group/block scale modes | Dynamic and static E4M3 | W8A16, W8A8 GEMV/GEMM, block-scale GEMV/GEMM | All projection and serving fusions | E4M3 cache insert/gather and direct attention |
| FP8 E5M2 | Same lifecycle as E4M3 with E5M2 encoding | Dynamic and static E5M2 | W8A16, W8A8, and mixed E4M3/E5M2 pairs | All projection and serving fusions | E5M2 cache insert/gather and direct attention |
| FP8 W8A16 | Reuse FP8 weight formats | F16/BF16/F32 direct loads | Direct packed GEMV and blocked GEMM | Gate/up, QKV, LM head, MoE | Not implied |
| INT4 W4A16 | Signed row/group formats and affine groupwise U4 | F16/BF16/F32 direct loads | Decode GEMV, blocked GEMM, selected rows, row AXPY | Epilogue, gate/up, QKV, embedding, LM head, MoE | Not implied |
| INT4 W4A8 | Same INT4 weights, including group scales and zero points | Symmetric/asymmetric A8 per row or group | Integer-dot GEMV/GEMM with combined scales and zero-point correction | Norm-add-A8, SwiGLU-A8, projection epilogues, MoE | Not implied |
| INT4 W4A4 | Same INT4 weights | Symmetric/asymmetric INT4 and FP4 activation packing | W4A4 GEMV/GEMM with integer or table-lookup accumulation | Norm-add-A4, gate/up-A4, MoE | Not implied |
| INT8 W8A16 | Per-row/per-channel INT8 and scales | F16/BF16/F32 direct loads | Existing W8A32 route generalized to direct A16 | Projection epilogues, gate/up, QKV, LM head, MoE | Not implied |
| INT8 W8A8 | Per-channel weights, scales, row sums | Symmetric/asymmetric per-token/group A8 | Existing native GEMM plus native GEMV and fused quantize routes | Norm-add-A8, SwiGLU-A8, QKV, MoE | Not implied |
| AWQ | Import `qweight`, `qzeros`, scales, group size to affine U4 | A16 initially; A8/A4 reuse shared activation paths | Shared W4A16/W4A8/W4A4 | Shared fusions and serving | Not implied |
| GPTQ | Import packed INT fields, scales, qzeros, `g_idx`, `desc_act` | A16/A8/A4 shared | Shared grouped low-bit plus direct act-order | Shared fusions; permutation consumed in hot loop | Not implied |
| AutoRound | Dispatch exporter metadata to U4, FP8, MXFP4, MXFP8, or NVFP4 importer | Determined by emitted scheme | No AutoRound-only dot product | Reuse target-format fusions | Reuse target-format cache behavior only |
| SmoothQuant | Import smoothed per-channel INT8 weights and scale metadata | Static or dynamic A8, symmetric/AZP | Shared W8A8 GEMV/GEMM | Fused norm/add/quant, QKV, gate/up, MoE | Not implied |
| TurboQuant | Existing packed key/value codec plus signs, centroids, scales, zero points | Query sign/FWHT transform | No weight projection kernel | Cache management only | Packed-K score, rotated-V accumulation, online softmax, deferred inverse FWHT |
| BitNet b1.58 | Canonical ternary blocks plus I2_S/TL import/preparation | Per-token A8 | Ternary x A8 GEMV/GEMM using integer dot or LUT | ReLU2/FFN, embedding, LM head, MoE as model requires | Existing model cache precision unless separately requested |
| BitNet a4.8 | Extend BitNet metadata for activation policy and sparsity | A4/FP4 layer inputs; sparse A8 intermediates | Ternary x A4 and ternary x sparse A8 GEMV/GEMM | Sparse activation selection/compaction and FFN fusion | 3-bit KV codec and direct attention |

## Kernel work packages

Each work package produces a reference implementation, tests, benchmarks, and
dispatch plumbing unless it explicitly says it is an importer-only package.

### C: contract, metadata, and infrastructure

| ID | Work | Dependencies | Deliverables |
|---|---|---|---|
| C0 | Expand umbrella format specifications | None | Exact FP8/FP4/MX/BitNet encodings, scale rules, rounding, saturation, accumulation, and cache contracts |
| C1 | Add an operation-level quant coverage manifest | C0 | Machine-checkable format x activation x operation x dtype matrix; checker rejects unsupported claims |
| C2 | Define canonical metadata descriptors | C0 | Group size, scale mode, zero-point mode, act-order, global scales, sparse masks, centroid tables |
| C3 | Generalize `CpuPackedWeights` prepared metadata | C2 | Versioned CPU-private panels with ISA, tile, side-table, and memory diagnostics |
| C4 | Add quantized microkernel interfaces | C2 | Block-dot, MxN tile, paired projection, selected-row, and epilogue interfaces |
| C5 | Add benchmark case families | C1 | `quant_formats`, `quant_activation`, `quant_gemv_matrix`, `quant_gemm_matrix`, `quant_fusions`, `quant_serving`, `quant_kv`, `bitnet_matrix` |
| C6 | Add golden-vector infrastructure | C0 | Checked-in small canonical byte fixtures and independent decoded/oracle outputs |

### L: lifecycle, packing, and checkpoint ingestion

| ID | Kernel/importer | Required modes | Primary references |
|---|---|---|---|
| L1 | Grouped U4 pack/unpack | Symmetric and affine, group 32/64/128/256 and rowwise, fractional or integer zero points as the canonical spec decides | LLM-AWQ quantizer, GPTQ quantizer, Metal `kU4`/`kU4B8` decode |
| L2 | AWQ importer | `qweight`, `qzeros`, scales, group size, transposition and nibble order | LLM-AWQ `qmodule.py`, vLLM `awq_marlin_repack.cu` |
| L3 | GPTQ importer | GPTQ and GPTQ-v2 zero convention, symmetric/asymmetric, `g_idx`, act-order | GPTQ, T-MAC `model_utils.py`, vLLM `gptq_marlin_repack.cu` |
| L4 | AutoRound importer | INT, FP8, FP8 block, MXFP4, MXFP8, NVFP4, per-layer overrides | AutoRound formats and exporters |
| L5 | FP8 pack/unpack | E4M3/E5M2, raw, tensor, row, group, block | TorchAO FP8, CUTLASS `float8.h`, Metal dequant formats |
| L6 | MXFP8 pack/unpack | E8M0 plus 32 E4M3 values | TorchAO `mx_formats`, CUTLASS block-scaled reference |
| L7 | MXFP4 pack/unpack | E8M0 plus 32 packed E2M1 values | TorchAO `mx_formats`, Metal `mxfp4` |
| L8 | NVFP4 pack/unpack | Global scale, per-16 E4M3 scale, packed E2M1, 1-D and 2-D scale modes | TorchAO `nvfp4_tensor.py`, CUTLASS NVFP4 examples, Metal `nvfp4` |
| L9 | INT8 weight pack | Per-row/per-channel scales, row sums, symmetric/asymmetric metadata | Existing `int8_gemm`, SmoothQuant exporter |
| L10 | BitNet import and prepare | Canonical ternary, I2_S compatibility, TL1/TL2 private LUT preparation | BitNet and T-MAC |
| L11 | Lifecycle conversion benchmarks | All above | Cold pack time, prepared-panel time, canonical/prepared bytes, round-trip checks |

### Q: activation quantization

| ID | Kernel | Required inputs and modes | Output consumer |
|---|---|---|---|
| Q1 | Dynamic INT8 quantize/dequantize | F32/F16/BF16; symmetric per-token/per-group | W8A8, W4A8, BitNet A8 |
| Q2 | Asymmetric INT8 quantize/dequantize | F32/F16/BF16; zero point plus row sum correction | SmoothQuant/AZP |
| Q3 | INT4 quantize/dequantize | F32/F16/BF16; symmetric and affine per-token/per-group | W4A4 |
| Q4 | FP4 E2M1 quantize/dequantize | F32/F16/BF16; packed pairs | W4A4, BitNet a4.8 |
| Q5 | FP8 E4M3/E5M2 quantize/dequantize | Static/dynamic, tensor/token/group/block, power-of-two option | FP8 GEMV/GEMM and caches |
| Q6 | MXFP8 activation quantize | E8M0 per 32 | Dual MXFP8 |
| Q7 | MXFP4 activation quantize | E8M0 per 32 | Dual MXFP4 |
| Q8 | NVFP4 activation quantize | Dynamic/static global scale and local E4M3 scale | Dual NVFP4 |
| Q9 | BitNet a4.8 sparse activation prepare | A4/FP4 layer input; threshold/mask, compact indices, A8 intermediate | BitNet a4.8 projection |

Activation quantizers must support a caller-provided workspace or reusable
thread-local storage. Hot fused paths must not allocate a full temporary with
`std::vector` on each invocation.

### K: core GEMV and GEMM

| ID | Kernel | Format set | Shape focus |
|---|---|---|---|
| K1 | Direct WNA16 GEMV block-dot framework | U4/U4B8, INT4, INT8, FP8 E4M3/E5M2, MXFP8, MXFP4, NVFP4, BitNet | M=1, N/K 2K-16K |
| K2 | Prepared WNA16 GEMV panels | Same | Repeated decode, cache-resident metadata, selected rows |
| K3 | Blocked WNA16 GEMM | Same | M=16 and 128, N/K 4K-16K |
| K4 | W4A8 GEMV | Symmetric row/group and affine U4 | M=1 |
| K5 | W4A8 GEMM | Same | M=16/128; A8 quantization amortized over N |
| K6 | W4A4 GEMV | Symmetric/asymmetric INT4 and FP4 activation | M=1 |
| K7 | W4A4 GEMM | Same | M=16/128; compare integer dot and LUT approaches |
| K8 | W8A8 GEMV | Symmetric and AZP | M=1 |
| K9 | W8A8 GEMM completion | Existing native routes plus prepared packing and fused activation input | M=16/128 |
| K10 | FP8 scaled GEMV | E4M3/E5M2 and mixed pairs | M=1 |
| K11 | FP8 scaled/block GEMM | Tensor/row/group/block scales | M=16/128 |
| K12 | Dual MXFP8 GEMV/GEMM | E8M0 block scales on both operands | M=1/16/128 |
| K13 | Dual MXFP4 GEMV/GEMM | E8M0 block scales on both operands | M=1/16/128 |
| K14 | Dual NVFP4 GEMV/GEMM | Local and global scales on both operands | M=1/16/128 |
| K15 | Direct act-order GEMV/GEMM | U4B8/U4 and supported GPTQ variants | M=1/16/128 without materialized permutation |
| K16 | Selected-row/AXPY native blocks | All requested weight-only formats | Embedding, candidates, adapters, backward-input |
| K17 | Quantized backward-input | All requested weight-only formats where contract requires it | Frozen-weight training/inference adapters |

K1 and K3 are frameworks with format-specialized block implementations, not one
giant switch inside an element loop. K3 must replace repeated row-wise GEMV with
M/K/N tiling and reuse each prepared weight tile across multiple activation rows.

### F: fused transformer projections

| ID | Kernel | Format coverage | Required fusion |
|---|---|---|---|
| F1 | Quantized epilogue | All core projection formats | Bias plus none/GELU/SiLU/ReLU2; output F32/F16/BF16 |
| F2 | Gate/up paired projection | All weight-only A16 formats | One weight traversal/schedule for two projections |
| F3 | Gate/up plus SwiGLU quantize | W4A8/W4A4/W8A8/FP8/MX/NV | Paired projection, SwiGLU, next-layer activation quantization |
| F4 | QKV projection | All weight-only A16 formats | Shared input traversal and output placement |
| F5 | QKV plus RoPE plus KV write | Projection formats plus supported cache formats | No materialized Q/K/V staging buffers |
| F6 | RMSNorm-add-quant | INT8, INT4/FP4, FP8, MXFP8, MXFP4, NVFP4 | Row-local two-pass statistics and direct packing |
| F7 | LayerNorm-add-quant | Same | Mean/variance, residual output, direct packing |
| F8 | QFlux/linear activation epilogue | Requested weight-only formats | Fuse bias and activation into store |

### S: embeddings, LM head, and MoE

| ID | Kernel | Required formats | Completion condition |
|---|---|---|---|
| S1 | Quantized embedding/dequant gather | Every requested stored-weight format | Vector/block decode, scale/add, F32/F16/BF16 output |
| S2 | Quantized embedding bag | Same | Sum/mean/weighted modes without scalar element decoder |
| S3 | Streaming LM-head argmax/top-k/top-p | Same | Vocabulary tiles, multi-row reuse, no full logits allocation when selection permits |
| S4 | Masked/candidate/beam LM head | Same | Selected-row primitives and deterministic tie rules |
| S5 | Grouped MoE quantized GEMM | Same plus dual FP8/MX/NV and W4A8 | Prepared expert tiles, sorted expert runs, padded rows |
| S6 | Grouped MoE fused SwiGLU | Same | No rows x 2I intermediate; optional next-layer quantization |
| S7 | Named sibling compatibility routes | FP8, WNA16, NVFP4 | Adapt metadata once, then call shared CPU microkernels |

### A: quantized caches and attention

| ID | Kernel | Cache format | Completion condition |
|---|---|---|---|
| A1 | FP8 cache insert/gather | E4M3 and E5M2 | Direct F32/F16/BF16 conversion, static/dynamic scales |
| A2 | FP8 online paged attention | E4M3 and E5M2 | Decode K/V in registers during online softmax; no full cache decode |
| A3 | MXFP8 cache insert/gather | E4M3 plus E8M0 per group | Canonical scale layout and D64/D128 paths |
| A4 | MXFP8 online paged attention | Same | Direct block-scaled score and V accumulation |
| A5 | TurboQuant query transform | Signs plus normalized FWHT | Standalone typed transform seam; canonical cache keys remain untransformed |
| A6 | TurboQuant packed-K score | 2-8 bit keys | Dot from packed codes, scale, and zero point without materialized K |
| A7 | TurboQuant rotated-V accumulation | 2/3/4/8-bit centroid values | Accumulate in rotated domain through online softmax |
| A8 | TurboQuant fused paged attention | Packed K plus rotated V | One deferred inverse FWHT after final normalized V state |
| A9 | BitNet a4.8 KV3 codec | 3-bit cache | Pack/unpack/scatter/gather with specified scale policy |
| A10 | BitNet a4.8 KV3 attention | 3-bit cache | Direct packed score and V accumulation |

TurboQuant encode/decode remains the bit-exact oracle. A5-A8 are new serving
kernels; merely decoding the complete cache before existing attention does not
meet the completion definition.

### B: BitNet compute and model fusions

| ID | Kernel | Required behavior | References |
|---|---|---|---|
| B1 | Ternary x A8 GEMV | M=1 integer-dot and LUT candidates | BitNet MAD/LUT, T-MAC |
| B2 | Ternary x A8 GEMM | M=16/128 prepared LUT/tile reuse | BitNet generated kernels, T-MAC QGEMM |
| B3 | Ternary x A4/FP4 GEMV | a4.8 layer inputs | BitNet a4.8 numerical spec plus T-MAC low-bit LUT method |
| B4 | Ternary x A4/FP4 GEMM | a4.8 prefill | Same |
| B5 | Ternary x sparse A8 GEMV/GEMM | Sparse intermediate states, mask/indices, exact zero handling | a4.8 design; CPU sparse scheduling must be measured |
| B6 | BitNet projection epilogues | Bias rules, ReLU2 or model-selected activation, output quantization | BitNet model code |
| B7 | BitNet embedding and LM head | Canonical ternary and any model-required embedding representation | BitNet embedding guide |
| B8 | BitNet grouped MoE | Only for BitNet model contracts that contain experts | Shared MoE scheduler plus B1-B5 |

BitNet LUTs are CPU-private prepared data. Canonical ternary weights remain the
source of truth, and the runtime must retain a non-LUT portable fallback.

## ISA implementation matrix

Every row begins with a portable implementation. ISA cells are implementation
targets, not support claims until their evidence is recorded.

| Kernel class | AArch64 baseline | AArch64 advanced | x86-64 baseline | x86-64 advanced |
|---|---|---|---|---|
| WNA16 integer/low-bit GEMV | NEON table/unpack and FMA | DotProd, I8MM, SVE2; evaluate SME2 for tiles | AVX2/FMA/F16C | AVX-512F/BW/VNNI; evaluate AMX for multi-row |
| W4A8/W8A8 | NEON | DotProd, I8MM, SVE2, SME2 | AVX2 | AVX-512 VNNI, AMX INT8 |
| W4A4 | NEON lookup/unpack | SVE2/SME2; compare LUT versus widened integer path | AVX2 `pshufb`/unpack | AVX-512/VNNI where profitable |
| FP8 E4M3/E5M2 | NEON table/bit decode | SVE2 widening/conversion | AVX2 table/bit decode | AVX-512 conversion/table path |
| MXFP8/MXFP4/NVFP4 | NEON decode with hoisted scales | SVE2/SME2 tiles | AVX2 decode with hoisted scales | AVX-512 tiles |
| Blocked GEMM | NEON row panels | I8MM/SME/SME2 tiles | AVX2 row panels | AVX-512/VNNI and AMX tiles |
| BitNet | NEON table lookup | DotProd/SVE2/SME2 candidates | AVX2 shuffle LUT | AVX-512/VNNI candidates |
| Activation quantizers | NEON reductions/pack | SVE2 reductions | AVX2 reductions/pack | AVX-512 reductions/pack |
| Quantized attention | NEON online recurrence | SVE2 score/value vectors | AVX2 online recurrence | AVX-512 score/value vectors |

Do not make AMX, SVE, or SME a prerequisite for the public operation. Matrix
engines are optional prepared routes behind the same API and must include OS
state checks already modeled by `CpuFeatures`.

## Phased execution order

```text
Phase 0: C0-C6 contracts, manifests, goldens, and benchmark scaffolding
    -> Phase 1: L1-L11 canonical lifecycle and checkpoint import
        -> Phase 2: Q1-Q8 activation quantization
        -> Phase 3: K1-K3 direct WNA16 GEMV and blocked GEMM
            -> Phase 4: K4-K15 W4A8/W4A4/W8A8/FP8/MX/NV/act-order
                -> Phase 5: F1-F8 projection and norm fusions
                    -> Phase 6: S1-S7 embedding, LM head, and MoE
                    -> Phase 7: A1-A8 FP8/MXFP8/TurboQuant attention
        -> Phase 8: L10, Q9, B1-B8, A9-A10 BitNet b1.58/a4.8
            -> Phase 9: K16-K17 selected-row and backward closure
                -> Phase 10: ISA breadth, full evidence, and parity publication
```

Some work can proceed concurrently after Phase 0, but a format may not enter an
optimized kernel until canonical golden vectors and reference decode exist.

### Phase 0: contracts and executable matrix

- Expand the umbrella format specs through a separate umbrella change.
- Add the machine-readable matrix and a checker under `parity/` or `scripts/`.
- Define exact golden fixtures for every encoding and external importer.
- Add benchmark registrations before optimization so pass 0 is reproducible.
- Record current scalar/reference baselines with a dirty-tree label if needed.

Exit gate: every requested name maps unambiguously to canonical metadata and
one or more kernel work packages.

### Phase 1: lifecycle closure

- Implement L1-L10 using portable code first.
- Test external checkpoint fragments generated by each cloned exporter.
- Verify canonical pack/unpack byte order and scale/zero-point conventions.
- Extend `CpuPackedWeights::prepare` without changing canonical bytes.
- Measure conversion cost and prepared memory amplification separately from
  steady-state projection cost.

Exit gate: all requested weight formats can be produced or imported and decoded
without undefined or implicit metadata.

### Phase 2: activation closure

- Implement Q1-Q8 references for F32/F16/BF16 inputs.
- Add portable exact-code tests and dequantized error tests.
- Add NEON/AVX2 reducers and packers, followed by wider ISA routes.
- Eliminate per-call heap allocation from fused activation consumers.

Exit gate: every A4/A8/MX/NV activation representation required by the matrix
has a reusable packet and a checked producer.

### Phase 3: direct WNA16 projection

- Implement K1 with format-specialized block dots.
- Extend prepared row panels for K2.
- Implement K3 with separate M=16 and M=128 tiling decisions.
- Route F16/BF16 directly through register conversion or native loads rather
  than full-tensor staging.

Exit gate: every requested stored-weight format has M=1 and multi-row packed
projection that does not allocate a dequantized weight matrix.

### Phase 4: quantized-activation projection

- Extend W8A8 and W4A8 first because native integer dot instructions apply.
- Implement W4A4 with both widened integer and LUT candidates; keep the winner
  per ISA and shape.
- Replace scalar FP8/MX/NV dual-operand references with blocked kernels.
- Consume GPTQ act-order in the K loop.

Exit gate: W4A4, W4A8, W8A8, FP8, MXFP8, MXFP4, and NVFP4 pass the full
M=1/16/128 correctness and benchmark matrix.

### Phase 5: fused projections and norms

- Implement F1-F8 on the shared block/tile interfaces.
- Preserve independent unfused oracles for each fused result.
- Ensure a fusion removes a real pass, allocation, or packed-weight traversal.
- Reject fusions that only rename a composition without improving data flow.

Exit gate: decode QKV and gate/up paths, plus next-layer quantization, avoid
full intermediate tensors where the contract permits it.

### Phase 6: serving closure

- Implement S1-S7 using selected-row and multi-row block dots.
- Stream LM-head selection without allocating rows x vocabulary logits when the
  selection mode permits it.
- Sort/group MoE rows once and reuse prepared expert tiles.
- Retain deterministic tie breaking and expert padding semantics.

Exit gate: each requested stored-weight format works through embedding, LM head,
and applicable MoE operations with focused evidence.

### Phase 7: cache and attention closure

- Complete A1-A4 for FP8 and MXFP8.
- Implement A5-A8 as direct compressed TurboQuant attention.
- Use online softmax and keep V accumulation in its compressed/rotated domain
  as long as possible.
- Benchmark cache bandwidth, query transform cost, and end-to-end decode.

Exit gate: no format advertised for KV-cache use requires materializing the
complete cache in FP32 before attention.

### Phase 8: BitNet closure

- Implement L10 and B1-B2 for b1.58 A8 first.
- Compare direct integer dot with BitNet/T-MAC LUT preparation and execution.
- Add Q9 and B3-B6 for a4.8 hybrid activation behavior.
- Add B7-B8 only where required by executable model contracts.
- Implement A9-A10 for the a4.8 3-bit cache contract.

Exit gate: b1.58 and a4.8 are separate executable configurations with explicit
activation and cache policies, not aliases for generic W2A8 or W4A4.

### Phases 9-10: completeness and publication

- Complete selected-row, AXPY, and packed backward-input routes.
- Fill ISA gaps on at least one AArch64 and one x86-64 host.
- Run sanitizers and cross-architecture builds.
- Update `docs/sibling-port-matrix.md`, parity manifests, and umbrella matrices
  only for evidenced cells.
- Remove temporary experimental variants or mark them internal and disabled.

## Correctness plan

### Oracle hierarchy

Use the strongest applicable oracle in this order:

1. Checked-in byte-exact golden vectors from a reference exporter or format
   specification.
2. Independent scalar decode of the exact canonical codes and metadata.
3. Composition of existing public CPU operations for fused kernels.
4. Cross-backend sibling output for shared semantics.

Do not use the candidate SIMD helper to generate its own expected output.

### Required tests

Add focused executables or sections for:

- `test_quant_import.cpp`: AWQ, GPTQ, AutoRound, SmoothQuant, BitNet fragments;
- `test_quant_float_formats.cpp`: E4M3/E5M2, MXFP8, MXFP4, NVFP4;
- `test_quant_activation_matrix.cpp`: A4/A8/FP8/MX/NV from F32/F16/BF16;
- `test_quant_projection_matrix.cpp`: all W/A combinations at M=1/16/128;
- `test_quant_fusions.cpp`: epilogue, gate/up, QKV, norm-add-quant;
- `test_quant_serving_matrix.cpp`: embedding, LM head, MoE;
- `test_quant_kv_matrix.cpp`: FP8, MXFP8, TurboQuant, BitNet KV3; and
- `test_bitnet_matrix.cpp`: b1.58 and a4.8 activation policies.

Each suite must cover:

- all-zero blocks and rows;
- maximum finite and saturation-boundary values;
- positive and negative zero where the encoding distinguishes them;
- tie-to-even and adjacent representable values;
- non-finite input rejection or specified encoding behavior;
- ragged N/M tails and legal K blocks;
- illegal K/group sizes;
- null pointers and overflow-sized shapes;
- non-identity GPTQ permutations and repeated/invalid indices;
- asymmetric zero-point corrections;
- mixed zero and nonzero experts/cache slots; and
- exact in-place or alias behavior where allowed.

### Tolerances

Use umbrella tolerances as upper bounds:

| Class | rtol | atol |
|---|---:|---:|
| FP32 | 1e-5 | 1e-6 |
| FP16 | 1e-3 | 1e-3 |
| BF16 | 2e-3 | 2e-3 |
| FP8 | 2e-2 | 2e-2 |
| Quantized | 3e-2 | 3e-2 |

Codec tests should normally be byte exact. Integer-dot kernels should be exact
relative to the specified integer accumulation before scale conversion. Use the
looser quantized tolerance only for comparison with the original unquantized
tensor, not to hide a disagreement with the exact-code oracle.

### Validation commands

Every work package ends with at least:

```sh
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUIXICORE_CPU_BUILD_TESTS=ON \
  -DQUIXICORE_CPU_BUILD_BENCHMARKS=ON
cmake --build build-release --config Release
ctest --test-dir build-release --build-config Release --output-on-failure
```

Final phase validation additionally includes ASan/UBSan with float-cast overflow
checking and an x86-64 build/run where available.

## Required three-pass optimization protocol

Every kernel work package in Q, K, F, S, A, or B receives three measured
optimization passes after its portable reference is correct. A pass is an
experiment; regressions are rejected and documented rather than forced into the
final implementation.

| Pass | Primary question | Typical experiments | Required record |
|---|---|---|---|
| 1: algorithm and dataflow | Are unnecessary decodes, allocations, passes, or cache reads present? | Online recurrence, row/block decode, activation reuse, prepared panels, loop interchange, selected rows | Baseline/candidate medians, correctness, keep/reject rationale |
| 2: SIMD and microkernel | Is the hot loop mapped efficiently to the target ISA? | Multiple accumulators, DotProd/I8MM/VNNI, table lookup, vector conversion, scale hoisting, M/N/K tile | Variant name, compiler flags, CPU features, code-path proof, measurements |
| 3: scheduling and fusion | Does the kernel scale and compose well at real model shapes? | Grain size, thread partition, cache tiles, paired projections, fused quantization/epilogue, tiny-M bypass | 1-thread and multi-thread results, regressions, final route decision |

Use artifact names of the form:

```text
perf/results/YYYY-MM-DD/full-matrix-<package>-pass0-t<threads>/
perf/results/YYYY-MM-DD/full-matrix-<package>-pass1-t<threads>/
perf/results/YYYY-MM-DD/full-matrix-<package>-pass2-t<threads>/
perf/results/YYYY-MM-DD/full-matrix-<package>-pass3-t<threads>/
perf/results/YYYY-MM-DD/full-matrix-<package>-final-t<threads>/
```

Each pass must use the same checked shape subset and same build class. Record
hardware, compiler, source revision or dirty label, selected ISA, warmups,
iterations, median, dispersion, correctness error, and decision in
`perf/optimization_status.md`.

## Benchmark matrix

The umbrella `quant_matmul` shapes are the authoritative comprehensive sweep:

- M: 1, 16, 128;
- N: 4096, 8192, 16384; and
- K: 4096, 8192, 16384.

Use three benchmark levels:

| Level | Purpose | Shape policy |
|---|---|---|
| Smoke | Build/dispatch/correctness | One small aligned case plus one tail case per kernel |
| Quick | Each optimization pass | M=1/16/128 with representative N/K around 4096 and model-specific tails |
| Comprehensive | Completion gate | Full umbrella matrix plus format-specific groups and serving shapes |

Additional required shapes:

- decode projections at N/K 2048, 4096, and 8192;
- gate/up with K=4096 and intermediate sizes 11008, 13824, and 14336;
- QKV with GQA head layouts and head dimensions 64/128;
- LM head with vocabulary 32K, 128K, and a tail not divisible by the row tile;
- MoE tokens 128/1024/4096, experts 8/16/64, top-k 1/2/4;
- embeddings with repeated and random ids;
- FP8/MXFP8 attention at D64/D128 and context 512/2048/8192;
- TurboQuant and BitNet KV3 at head sizes 64/128/256; and
- pack/prepare cold cost plus steady-state reuse counts 1/16/128/1024.

Measure one thread and a representative physical-core count. Do not claim an
architecture tier from Rosetta, emulation, or cross-compilation alone.

## File layout plan

Prefer small format-independent interfaces and format/ISA implementation files:

```text
include/quixicore_cpu/
  quantization.h                 public codecs and activation quantization
  qgemv.h                        weight-only and selected-row semantics
  qgemm.h                        matrix and quantized-activation semantics
  packed_weights.h               prepared-panel ownership and diagnostics

kernels/quantization/
  quant_layout_ref.cpp           canonical metadata and validation
  checkpoint_import_ref.cpp      AWQ/GPTQ/AutoRound/SmoothQuant adapters
  activation_quant_ref.cpp       portable A4/A8/FP8/MX/NV producers
  activation_quant_neon.cpp
  activation_quant_avx2.cpp
  activation_quant_avx512.cpp
  qgemv_wna16_ref.cpp
  qgemv_wna16_neon.cpp
  qgemv_wna16_avx2.cpp
  qgemv_wna16_avx512.cpp
  qgemm_wna16_ref.cpp
  qgemm_wna16_<isa>.cpp
  qgemm_lowbit_activation_ref.cpp
  qgemm_lowbit_activation_<isa>.cpp
  microscale_ref.cpp             codecs and portable oracles
  microscale_<isa>.cpp           FP8/MX/NV compute
  bitnet_ref.cpp
  bitnet_<isa>.cpp
  qgemv_fused_ref.cpp
  qgemv_fused_<isa>.cpp

src/dispatch/
  quant_layout.cpp
  activation_quant.cpp
  qgemv_wna16.cpp
  qgemm_wna16.cpp
  lowbit_activation.cpp
  microscale.cpp
  bitnet.cpp

kernels/attention/
  attention_quant_ref.cpp
  attention_quant_<isa>.cpp
  turboquant_attention_ref.cpp
  turboquant_attention_<isa>.cpp

benchmarks/cases/
  quant_formats.cpp
  quant_projection_matrix.cpp
  quant_fusions.cpp
  quant_serving.cpp
  quant_kv.cpp
  bitnet_matrix.cpp
```

These names are proposed organization, not a requirement to break compatible
existing files merely for aesthetics. Reuse current dispatch files when doing so
keeps the implementation clearer.

## Milestones and completion checklist

### M0: executable contract

- [x] C0-C6 complete.
- [x] Requested matrix is machine-readable.
- [x] Golden vectors exist for every canonical layout.
- [x] Benchmark pass-0 cases are registered.

M0 evidence: the umbrella format and quantization specifications define the
wire semantics; `parity/full_quant_matrix.tsv` contains 85 checked requirement
cells; `tests/testdata/quant_contract_golden.tsv` covers all 13 canonical
layouts; `tests/testdata/quant_import_golden.tsv` covers every planned external
importer; and the eight pass-0 benchmark families are recorded in
`perf/results/2026-07-22/m0-pass0-t1/`. All matrix cells remain `planned` until
the later milestone's correctness and performance evidence exists.

### M1: model ingestion

- [x] AWQ import is byte- and output-checked.
- [x] GPTQ symmetric/asymmetric and act-order import is checked.
- [x] AutoRound target formats import correctly.
- [x] SmoothQuant metadata imports to W8A8/AZP.
- [x] FP8/MX/NV packers are complete.
- [x] BitNet canonical and prepared formats are complete.

M1 evidence: `CanonicalQuantTensor` owns portable bytes and explicit logical
side tables; AWQ, GPTQ v1/v2, AutoRound, SmoothQuant/AZP, and checked BitNet
I2_S importers normalize into that representation. FP8 covers tensor, row,
channel, group, and declared row-by-K block scales; MXFP8/MXFP4 use embedded
E8M0 blocks; NVFP4 covers 1-D and 2-D E4M3 local scales; all accept FP32,
FP16, and BF16 storage. `CpuPackedWeights::prepare` retains canonical bytes
unchanged and appends aligned, rebuildable CPU side tables. Exact byte/output,
invalid-metadata, reserved-code, scale-topology, round-trip, and panel tests
live in `tests/correctness/test_quant_import.cpp` and
`tests/correctness/test_quantization.cpp`.

Three measured lifecycle passes are recorded in
`perf/results/2026-07-22/m1-pass{1,2,3}-t1/` (pass 1's corrected artifact is
`m1-pass1-t1-fixed`; the stable final is `m1-pass3-final2-t1`). The retained
direct FP32-bit FP8 encoder is exact against the exhaustive oracle and reduces
E4M3, E5M2, MXFP8, and NVFP4 conversion by 96.39x, 68.89x, 87.10x, and 5.32x
respectively versus pass 1. These are lifecycle results, not projection support
claims; matrix compute cells remain planned for M2 and later milestones.

### M2: universal projection

- [x] Every requested weight format has direct F32/F16/BF16 GEMV.
- [x] Every requested weight format has blocked M=16/128 GEMM.
- [x] W4A4, W4A8, W8A8, FP8, MXFP8, MXFP4, and NVFP4 dual modes pass.
- [x] GPTQ act-order performs no full activation permutation allocation.
- [x] Each kernel has three optimization-pass records.

M2 evidence: importer-normalized INT4/U4, INT8, FP8 E4M3/E5M2,
FP4, MXFP8, MXFP4, NVFP4, and BitNet tensors now execute directly from
version-2 `CpuPackedWeights` row panels. `qgemm_prepacked_storage` consumes
FP32/FP16/BF16 activations in the K loop, tiles M=16/128 without a dequantized
weight matrix, and supports typed output conversion. The quantized-activation
route covers the requested A4/A8/FP8/MX/NV combinations at M=1/16/128.
GPTQ act-order is indexed directly and the focused test verifies no workspace
allocation. `tests/correctness/test_quant_projection_matrix.cpp` checks all 11
non-cache layouts; the current full Apple Release suite passes 55/55 tests.

The shared K1-K15 framework has three optimization passes recorded in
`perf/optimization_status.md`, with stable artifacts under
`perf/results/2026-07-23/full-matrix-k1-k3-final2-t{1,6}/`. M>=16 is retained
as a measured AArch64 NEON tier. K1 now also has three direct M=1 passes and
checked final results for all 11 layouts under
`perf/results/2026-07-23/canonical-gemv-final-all-t{1,6}/`; the comprehensive
N4096 K4096 run is `canonical-gemv-comprehensive-all-t1`. The retained
AArch64 FP32 block dots are 1.26x-6.30x over the same-run already-dequantized
scalar matrix at that shape. Ten quantized-activation M=1 pairs now also have
three direct packed optimization passes under
`canonical-dual-gemv-pass{0,1-portable,2-neon-integer,3-neon-fp8}-t1` and a
comprehensive N4096 K4096 final. Integer, FP4, NVFP4, and BitNet pairs beat the
predecoded scalar baseline. The subsequent `canonical-dual-fp8-pass4-*`
through `pass6-*` output-panel reuse sweep closes the M1 FP8 gap: E4M3/E4M3,
E5M2/E4M3, and MXFP8 are 1.09x-1.16x over the scalar comparator at quick and
1.27x-1.35x at comprehensive shapes. M16/128 dual-quant panels also have
three measured passes under the `canonical-dual-gemm-pass*` artifacts; the
retained activation/output-panel route reuses one
packed activation decode across 16 output lanes and reaches 0.3911-3.6073 ms
at M16. A second `canonical-dual-fp8-gemm-pass4-*` through `pass6-*` sweep
extends reuse to eight panels per activation row and closes the checked
multi-row FP8 gap: M16 FP8 pairs are 1.10x-1.25x over the scalar comparator
and M128 E4M3 is 1.12x. Format-specialized FP16/BF16 M1 routes have three
passes under the
`canonical-typed-gemv-*` artifacts and match the FP32 tier. Typed M16/128 GEMM
now also has three passes under `canonical-typed-gemm-pass*`; the retained
four-panel-group route reuses a converted activation tile across output panels
and is 1.16x-2.91x over its scalar-conversion pass 0. Native x86 M1 has three
independent passes under `x86-canonical-pass*`: AVX2/F16C typed panel groups
are 1.19x-2.34x over scalar-conversion pass 0, while the format-selected FP32
E4M3/E5M2 panel route is 1.31x-1.32x faster. Native x86 M16/128 now has three
further passes under `x86-m2-pass*` and final selected artifacts under
`x86-m2-final-selected-t{1,16}`. AVX2/FMA M tiles and four-panel activation
reuse are 1.58x-3.69x over the native pass-1 weight-only routes; direct
dual-code dots are 1.00x-4.49x, with measured-regressing W8A8 explicitly
retained on the portable panel path. This closes the per-kernel three-pass
item for M2. Applicable AVX-512/VNNI routes are measured separately in M5;
there is no AMX projection route or claim.

### M3: fused model path

- [x] Bias/activation epilogues cover the weight-only projection matrix.
- [x] Gate/up paired projection covers the weight-only matrix.
- [x] Gate/up/SwiGLU plus activation quantization covers the dual matrix.
- [x] QKV/RoPE/KV-write paths cover the matrix for floating cache storage.
- [x] Norm-add-quant covers A4/A8/FP8/MX/NV.
- [x] Embedding and embedding-bag paths cover the stored-weight matrix.
- [x] LM-head selection paths cover the stored-weight matrix.
- [x] Applicable MoE paths cover the matrix.

M3 F1 evidence: `qgemm_prepacked_epilogue_storage` now applies optional
channel bias and none/GELU-erf/GELU-tanh/SiLU/ReLU2 directly to each FP32
accumulator before FP32/FP16/BF16 output storage. All 11 non-cache canonical
weight layouts and FP32/FP16/BF16 activation storage share the M2 prepared
panel routes; no complete dequantized weight or FP32 output tensor is
materialized. `test_quant_projection_matrix` covers every layout, activation,
and output storage at M=1/16/128. Three measured passes plus retained Apple and
native x86 results are recorded under `f1-fused-epilogue-*`; the stable x86
route is throughput-neutral versus a stronger preallocated composition, so F1
is a materialization/semantic closure rather than a performance-tier claim.

M3 F2 evidence: `qgemm_prepacked_gate_up_storage` accepts independently
prepared gate/up planes for all 11 non-cache canonical layouts, shares one CPU
schedule and the multi-row activation traversal, and writes FP32/FP16/BF16
outputs without a combined-weight or intermediate allocation. Correctness
covers every layout, M=1/16/128, and all three activation/output storage types.
Three primary passes plus an x86 locality-selection follow-up are recorded
under `f2-gate-up-*`; stable one-thread EPYC M16/128 results are
1.04x-1.17x over two prepared projection calls. This completes F2.

M3 F3 evidence: `qgemm_prepacked_swiglu_storage` eliminates both gate/up
outputs and the post-projection activation pass while preserving direct
FP32/FP16/BF16 stores. `qgemm_prepacked_swiglu_quantized` goes on to emit
canonical symmetric/affine INT4 and INT8, FP4, FP8 E4M3/E5M2, MXFP4, MXFP8,
or NVFP4 bytes and scales from bounded per-worker groups; no complete M-by-N
floating-point SwiGLU tensor is allocated. Reused output vectors and the
caller/thread workspace make the warmed path allocation-free. Correctness
crosses every one of the 11 prepared weight layouts, M=1/16/128, all three
input storage types, row/group/tensor scale modes, and NVFP4 1-D/2-D scales.
Three primary passes for both the typed-store and activation-quantized kernels,
plus measured native-x86 locality selection, are recorded under `f3-swiglu-*`
and `f3-quant-*`. Apple group-A8 cases are 0.975x-2.188x versus fused typed
SwiGLU plus standalone quantization; the stable EPYC range is
0.892x-1.052x, so F3 completion is a semantic/materialization claim rather
than a blanket cross-architecture speedup.

M3 F4 evidence: `qgemm_prepacked_qkv_storage` accepts independently prepared
Q, K, and V matrices with unequal output rows, direct FP32/FP16/BF16 inputs
and mixed outputs, and one CPU scheduling region. Selected multi-row routes
share activation tiles across active Q/K/V panels; measured-regressing typed
or highly threaded shapes stay on their mature panel groups. Correctness
crosses all 11 weight layouts, M=1/16/128, every input storage type, exact
typed output rounding, unequal heads, wrapper behavior, and failure-atomic
validation against three independent projection oracles. Three primary passes
and native-x86 selector follow-ups are recorded under `f4-qkv-*`. Stable
one-thread M128 FP32 ratios are 1.04x-1.08x on Apple and 1.03x-1.11x on EPYC
versus three public prepared calls. This completed F4 QKV projection; the
combined QKV/RoPE/KV-write item was kept open until the F5 evidence below.

M3 F5 evidence: `qgemv_prepacked_qkv_rope_kv_storage` projects all 11
canonical weight layouts from FP32/FP16/BF16 activation storage, applies
split-half RoPE, writes direct FP32/FP16/BF16 Q output, and inserts K/V into
independently typed ordinary floating caches without separate raw Q/K/V
buffers. `slot=-1` skips cache projection and preserves cache bytes.
Correctness uses independent projection and RoPE/cache oracles across every
layout and exact mixed typed-store boundaries. Three primary passes and
architecture-locality follow-ups are recorded under `f5-qkv-rope-*`; rejected
generic panel experiments remain visible. Native x86 typed cases are
1.095x-1.148x over the composed one-thread comparator, while the full Apple
and x86 matrix is treated as materialization closure rather than a blanket
speedup. This closes the M3 QKV/RoPE/KV-write item for FP32/FP16/BF16 caches;
FP8, MXFP8, TurboQuant, and BitNet compressed caches are closed under M4 below.

M3 F6/F7 evidence: `rms_norm_add_quantized_storage` and
`layer_norm_add_quantized_storage` accept independent FP32/FP16/BF16 storage
for residual inputs, parameters, and residual output, then directly pack
canonical symmetric/affine INT4 or INT8, FP4, FP8 E4M3/E5M2, MXFP4, MXFP8,
or NVFP4 activations. Only per-row statistics and bounded worker scratch are
retained; no rows-by-hidden normalized tensor is allocated. Correctness
crosses both norm types, every output layout, 1-D/2-D and partial NV domains,
and homogeneous plus mixed storage. Three measured Apple passes retain direct
F32 traversal and full-row reuse. The final threaded matrix is 1.41x-5.71x
over preallocated composition on Apple and 1.77x-6.52x on native EPYC; the
one-thread MX/NV tails are reported as neutral/regressive rather than a
blanket speedup. This closes F6/F7 and the M3 norm-add-quant checklist item.

M3 S1/S2 evidence: `canonical_quantized_embedding_storage` and
`canonical_quantized_embedding_bag_storage` decode only selected rows from
all 11 canonical stored-weight layouts. Gather supports optional independently
typed add input and FP32/FP16/BF16 output; bag supports sum, mean, and weighted
reduction with bounded per-worker row scratch. Correctness compares against an
independently dequantized full-table oracle, checks exact typed storage
boundaries, and verifies invalid IDs are failure-atomic on Apple and native
x86-64. Three measured passes retain direct FP32 stores and item-major bag
accumulation. Final Apple one-thread ratios versus full-table dequantization
then selection are 2.57x-25.69x and native EPYC ratios are 1.84x-31.09x;
threaded results are also recorded. These ratios quantify the removed table
materialization and are not presented as equal-work projection speedups. This
closes S1/S2; quantized MoE is handled by S5-S7 below.

M3 S3/S4 evidence: `qgemm_prepacked_lm_head_sample_storage`,
`qgemm_prepacked_lm_head_masked_topk_storage`,
`qgemm_prepacked_lm_head_candidates_storage`, and
`qgemm_prepacked_lm_head_beam_advance_storage` operate on every prepared
canonical weight layout with FP32/FP16/BF16 hidden storage. Argmax, top-k,
packed-mask, candidate, and beam routes retain only deterministic selection
partials and online log-sum-exp state; exact categorical/top-p sampling uses
one vocabulary row because the final nucleus is data-dependent, never a
rows-by-vocabulary tensor. The algorithms adapt the two-stage partial/reduce
scheme in QuixiCore-Metal revision `bc968fc` to CPU row panels, cache tiles,
and the persistent thread pool. Correctness covers all 11 layouts, all input
storage types, exact seeded top-k/top-p selection, structured log
probabilities, beam parents/scores, strict candidate validation, and
failure-atomic outputs on Apple and native x86-64. Three primary passes retain
format-selected direct-row versus multi-row panel traversal; follow-ups add
threaded vocabulary partials and x86 multi-row panel reuse. Apple one-thread
top-k is 1.14x-2.96x and structured routes reach 3.37x-130.40x versus
materialized composition. Native EPYC one-thread multi-row paths range
0.95x-12.49x; 16-thread top-k reaches 2.04x-3.85x for three of four formats,
with all neutral/regressive cases preserved in the evidence. This closes
S3/S4 without making a blanket throughput claim.

M3 S5/S6/S7 evidence: `moe_grouped_prepacked_storage` and
`moe_grouped_prepacked_swiglu_storage` accept expert-indexed arrays of every
prepared canonical stored-weight layout with direct FP32/FP16/BF16 activation
and output storage. Sorted expert runs avoid permutation scratch; unsorted
runs use stable workspace grouping and typed gather/scatter; `-1` padded rows
store zero. The fused route directly stores `SiLU(gate)*up` without a
rows-by-2I tensor. Named FP8, WNA16, and NVFP4 wrappers validate metadata once
and reuse the same implementation. Correctness crosses all 11 layouts, all
three storage types, sorted/unsorted/all-padded routing, wrappers, and
failure-atomic invalid IDs on Apple and native x86-64. Three passes retain an
Apple one-thread compact-row selector, grouped FP8, and one shared threaded
dispatch across sorted experts. Final 16-thread ratios versus per-token
prepared GEMVs are 3.18x-8.72x on Apple and 3.56x-8.30x on EPYC. This closes
the stored-weight/A16 slices and their named adapters.

`moe_grouped_prepacked_quantized` adds canonical W4A8, W8A8, FP4/FP8,
MXFP4/MXFP8, NVFP4, and BitNet-A8 activation packets. Direct block-pair dots
are retained where measured; multi-row x86 routes decode the activation once
into reusable workspace before invoking the optimized grouped panels.
`moe_grouped_prepacked_swiglu_quantized` projects gate/up and directly packs
all ten supported non-BitNet activation layouts from bounded worker/domain
scratch, including tensor scales and NVFP4 1-D/2-D domains. Sorted FP8 expert
runs reuse decoded panels and share one dispatch. Correctness covers every
weight layout, every output packet, FP32/FP16/BF16 input, unsorted and padded
rows, and a sorted threaded panel route on Apple and native x86-64. Three
passes for each new kernel, retained and rejected selectors, and one-/16-thread
measurements are recorded under `s5-moe-dual-*`,
`s6-moe-swiglu-quant-*`, and `x86-s5-s6-moe-*`. Regressive cases remain in
the notebook, so this is matrix and materialization closure rather than a
blanket performance-tier claim. S5-S7 and the aggregate M3 MoE item are now
complete.

Metal BaseQN drift closure: `base_q_dequant`, `base_q_gemv`, `base_q_gemm`,
`base_q_embedding`, `base_q_gemv_qkv`, `base_q_gemv_swiglu`, and
`base_q_lm_head_argmax`, `base_q_moe_gemm`, and `base_q_moe_swiglu` cover the
canonical BaseQ2/3/4/5/6/8 sibling contract, including every legal group,
scale, symmetry, and FP32/FP16/BF16 storage combination. LM-head scores are
rounded to the activation storage type before selection and ties choose the
lower token, matching Metal. Grouped expert projection consumes the existing
32-row padded schedule, and the fused gate/up row axis is
`[gate(intermediate), up(intermediate)]`. Three direct projection, LM-head,
and grouped-MoE optimization passes, retained Apple and native x86
measurements, and the complete correctness matrix are recorded under
`base-q-*`. This adds the newly discovered sibling family without broadening
the F3 output-format claim to BitNet Q9 sparse activation preparation.

### M4: compressed-cache path

- [x] FP8 E4M3/E5M2 direct paged attention is complete.
- [x] MXFP8 direct paged attention is complete.
- [x] TurboQuant compressed attention is complete.
- [x] BitNet a4.8 KV3 codec and attention are complete.

M4 FP8 evidence: canonical E4M3FN/E5M2 cache insert and gather now accept
FP32, FP16, or BF16 storage, with per-head static scales or contract-correct
dynamic scales (`amax/448` and `amax/57344`). The direct paged kernel decodes
K/V bytes inside online softmax, hoists K/V scales out of the element loops,
and supports all three query/output storage types at D64/D128. The focused
correctness matrix crosses both formats, both dimensions, and all 3x3 typed
input/output boundaries. Three measured attention and cache-I/O passes plus
Apple AArch64 and native EPYC results are recorded under `m4-fp8-*` and
`x86-m4-fp8-*`; direct attention is 1.22x-1.78x over cache materialization on
Apple at one thread and 3.83x-5.09x on EPYC. Typed cache I/O has mixed
architecture results, which are retained without a blanket speedup claim.

M4 MXFP8 evidence: cache rows use the canonical interleaved 33-byte block
(`E8M0 scale + 32 E4M3FN codes`) for every D64/D128 head group. Scatter and
gather accept FP32/FP16/BF16 storage, and direct paged attention consumes block
scales and codes inside online softmax. Correctness crosses both dimensions and
all typed boundaries. Three optimization passes replace scalar bit decoding
with lookup tables, hoist block scales, and schedule collision-safe cache
blocks. Direct attention is 1.95x-2.10x over materialization on Apple at one
thread and 1.53x-1.57x on EPYC; threaded results and all artifacts are indexed
in `perf/optimization_status.md`. This closes A3/A4 only.

M4 TurboQuant evidence: the canonical 2-8-bit packed-key codec, centroid-coded
signed-FWHT value cache, FP16-rounded group metadata, and D64/D128/D256 heads
now have FP32/FP16/BF16 codec adapters. Direct online attention computes K
scores from packed codes and accumulates V in its rotated domain, performing
one inverse FWHT only after softmax normalization. The standalone normalized
signed-FWHT query transform is exposed for the A5 seam; it is intentionally
not applied to canonical cache attention because the contract stores K
untransformed. Correctness covers every 2-8-bit diagonal pair, signed K8,
D64/D128/D256, the complete typed codec and 3x3 attention boundaries, and an
explicit materialized-cache oracle on Apple and native x86-64. Three passes
replace generic bit addressing with 2/4/8-bit extractors, hoist 32-value group
metadata, remove per-head heap scratch, and add collision-safe codec
scheduling. Retained direct attention is 1.29x-1.54x / 3.00x-3.28x over
materialization on Apple and 1.36x-2.12x / 6.28x-7.99x on EPYC at one/16
threads. Typed codec staging ratios remain mixed and are not generalized.
Artifacts are indexed in `perf/optimization_status.md`. This closes A5-A8
only.

M4 BitNet a4.8 KV3 evidence: cache heads are a canonical low-bit-first 3-bit
stream with a declared group size, explicit signed/unsigned interpretation,
explicit no-zero/integer-zero-point mode, and FP16 or FP32 group scales.
Scatter, gather, and direct online paged attention support D64/D128/D256 and
FP32/FP16/BF16 boundaries. Correctness independently unpacks every code and
crosses signed symmetric, unsigned nonnegative, unsigned affine, signed
affine, both scale encodings, multiple group sizes, the complete typed codec
matrix, and all 3x3 query/output pairs on Apple and native x86-64. Three
primary passes establish scalar semantics, adopt 8-code/3-byte packet
addressing with group metadata hoisting, and add collision-safe cache/query
scheduling; a retained follow-up reuses each loaded packet across all eight
codes. Apple one-thread direct attention is 0.87x-0.97x materialization and is
therefore not claimed as faster; at 16 threads it is 2.14x-3.02x. EPYC is
0.46x-0.48x at one thread and 1.74x-2.08x at 16 threads. Threaded codec I/O is
1.14x-1.24x on Apple and 1.04x-1.33x on EPYC. Artifacts and regressions are
recorded in `perf/optimization_status.md`. This closes A9/A10 and M4 without a
blanket single-thread performance claim.

### M5: architecture and publication closure

- [x] Portable fallback passes all tests.
- [x] AArch64 NEON plus applicable DotProd/I8MM routes pass and are benchmarked.
- [x] x86-64 AVX2 plus applicable AVX-512/VNNI routes pass and are benchmarked.
- [x] No optional SVE/SME/AMX kernel tier is claimed without an implemented,
  correctness-tested, and benchmarked route.
- [x] Release, sanitizer, and cross-architecture suites pass.
- [x] `perf/optimization_status.md` contains three passes per kernel family.
- [x] `perf/baseline_status.md` indexes accepted finals.
- [x] `docs/sibling-port-matrix.md` and parity manifests match the evidence.
- [x] Umbrella CPU registry and matrices match the backend evidence.

M5 evidence: Apple AArch64 Release passes 55/55 tests, including the live
sibling drift gate and the enforced 110-operation sibling inventory. Intel
Sapphire Rapids Release passes all 57 locally executable tests; parity is
checked on the Apple workspace because the isolated x86 checkout does not
contain sibling repositories. Eight forced-route tests prove exact GGUF
AVX2/AVX-512, W8A32 AVX2/AVX-512, INT8 AVX2/AVX-512-VNNI, and W4A8
AVX2/AVX-512-VNNI execution. Three focused passes retain AVX-512 for W8A32,
VNNI for INT8/W4A8, and AVX2 for automatic GGUF dispatch after the generic
AVX-512 candidate trailed on 19 of 23 formats. The prior AMD EPYC 7702 native
and portable-only Release suites pass 46/46, and its
ASan/UBSan/float-cast-overflow suite passes 45/45 with leak detection.
Sapphire Rapids runtime detection exposes AMX tile/int8/bf16, but no AMX
source or dispatch route exists, so no AMX kernel or performance tier is
claimed. No SVE or SME tier is claimed either. Three-pass artifacts for
M0-M4, later sibling semantic ports, and the advanced x86 routes are indexed
in both performance status documents. The separate umbrella working tree now
registers CPU as active, declares portable scalar execution as its minimum,
and marks all 16 contract families complete with an explicit semantic-versus-
optimized-tier qualification.

## Per-package handoff template

Use this template for every work package so progress remains auditable:

```text
Package:
Public operation(s):
Canonical format and metadata:
Reference revision/path:
Portable oracle:
Target ISA variants:
Shapes and thread counts:
Correctness command and maximum error:
Pass 0 artifact:
Pass 1 experiment, artifact, decision:
Pass 2 experiment, artifact, decision:
Pass 3 experiment, artifact, decision:
Final artifact and selected variant:
Rejected alternatives:
Documentation/parity cells updated:
Remaining architecture gaps:
```

The implementation is complete only when the checklist is supported by files,
tests, and raw benchmark artifacts. A public symbol, scalar fallback, or sibling
implementation alone is not sufficient evidence for a completed matrix cell.
