# Remaining Cross-Backend Parity Dimensions

The pinned portable-f32 semantic inventories are closed and enforced by
`scripts/check_parity_manifest.py`. That is not the same claim as universal
backend parity. The remaining dimensions are explicit so a semantic mapping
cannot be mistaken for dtype, ISA, or performance evidence.

| Dimension | Current evidence | Remaining closure condition |
|---|---|---|
| llama CPU operation semantics | 105 top-level ops, 22 unary modes, and six GLU modes mapped at pinned revision `2beefef68` | Re-audit when the pinned source revision changes |
| llama quant lifecycle | 25 stored formats have pack, unpack, and f32 GEMV; Q8_1/Q8_K activation layouts are public | Per-format qgemm/activation-dot/ISA cells marked `reference` in `llama_quants.tsv` need optimized measured routes before any optimized-tier claim |
| sibling operation manifests | All 66 operation-level entries published by pinned Metal/XPU/ROCm manifests map to public CPU symbols | CUDA currently publishes family-level metadata only; exact CUDA operation drift cannot be proven until its manifest is expanded upstream |
| sibling quant manifests | All 29 published format-family entries map to CPU format/compute surfaces | Family mappings do not imply every storage-layout × operation × dtype product is optimized |
| dtype surface | Every public floating tensor can use FP16 or BF16 storage through the generic typed adapter, with named activation/norm/GEMM/attention/quantized-projection routes; FP32 is zero-copy | Native half accumulation is intentionally not claimed; FP64 remains absent, and operation-specific native half compute requires separate contract demand and evidence |
| aarch64 ISA performance | NEON/DotProd/I8MM routes have focused local evidence for the paths recorded in `perf/` | Reference-only cells remain reference-only until a correctness-gated benchmark is recorded |
| x86 ISA performance | AVX2/AVX-512/VNNI/AMX sources compile and the x86_64 Release suite passes under Rosetta | Run on native AVX2 and AVX-512 hosts before upgrading any unmeasured x86 tier |
| accelerator execution strategy | GPU tile/pipeline/staging variants collapse to equivalent CPU results | GPU occupancy, tensor-core throughput, asynchronous copies, and network overlap are not meaningful CPU parity dimensions |

The drift gate deliberately fails on source revision mismatch instead of
silently carrying these counts forward. `reference` is correctness coverage,
not an optimized-performance claim.
