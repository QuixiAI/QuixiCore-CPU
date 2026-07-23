# CPU Parity Ledger

This directory is the machine-readable source of truth for CPU parity work.
It deliberately separates semantic coverage from format authorship, optimized
ISA coverage, and measured performance.

`gaps.md` states the remaining dtype, ISA, performance, and source-metadata
limits after portable-f32 semantic closure.

- `sources.tsv` pins the audited upstream revisions.
- `llama_ops.tsv` classifies every `GGML_OP_*` symbol referenced by llama.cpp's
  CPU backend.
- `llama_unary_ops.tsv` and `llama_glu_ops.tsv` expand the selector enums
  nested under `GGML_OP_UNARY` and `GGML_OP_GLU`; top-level coverage cannot
  hide a missing activation mode.
- `llama_quants.tsv` records the lifecycle and ISA state of every live llama
  quantized type used for stored weights or activation dot partners.
- `sibling_semantics.tsv` records the top-level semantic inventory shared with
  the Metal, XPU, CUDA, and ROCm repositories.
- `sibling_operations.tsv` maps every operation-level entry currently exposed
  by the pinned Metal, XPU, and ROCm backend manifests. CUDA's pinned manifest
  is family-level only, so its families remain represented in
  `sibling_semantics.tsv` rather than inventing operation metadata upstream
  does not publish.
- `sibling_quant_families.tsv` maps every format-family entry in all four
  pinned sibling quant manifests.
- `dtype_coverage.tsv` records the universal FP16/BF16 storage adapter,
  accumulation precision, named hot-path routes, and architecture evidence.
- `full_quant_matrix.tsv` is the executable M0 matrix for every requested
  format/scheme, activation mode, operation, storage dtype, work package, and
  pass-0 benchmark family. `planned` rows are requirements, not support claims.

At the pinned revisions the llama operation ledger has 89 public numerical
mappings, seven validated view/layout adapters, and nine enum markers,
callbacks, or selector values that are not numerical kernels. Its nested
inventories add all 22 numerical unary modes and all six numerical GLU modes.
No row is `missing`. The quant ledger separately prevents semantic coverage
from being mistaken for an ISA performance tier.

Run `scripts/check_parity_manifest.py` from the repository root. The check
fails when a pinned checkout changes, llama adds or removes a CPU operation or
nested unary/GLU mode, a sibling adds or removes an operation-level manifest
entry or quant family, or a manifest row/status is malformed. A missing row is
never silently treated as unsupported.

Status vocabulary:

- `mapped`: the CPU public surface implements the source semantics.
- `adapter`: no numerical kernel is required, but a validated CPU adapter is.
- `reference`: a portable implementation exists without an optimized tier.
- `optimized`: a measured ISA-specific implementation exists.
- `missing`: implementation or evidence is still required.
- `internal`: an intermediate format rather than a stored-weight format.
- `non_kernel`: source metadata or an external callback, not numerical work.

The native numerical kernels remain FP32 contracts. Universal FP16/BF16
*storage* is provided by `dispatch_float_storage`/`with_float_storage`: decode
once, run the same FP32 kernel, round once on successful output commit. This is
not a claim of native half-precision accumulation, a separate half arithmetic
implementation for every operation, or FP64 coverage.
