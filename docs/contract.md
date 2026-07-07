# Contract Alignment

QuixiCore CPU implements the shared QuixiCore contract as a native CPU backend.
The umbrella repository owns the contract; this repository owns only CPU
implementation, test, and benchmark work.

## Current Declaration

- Backend: `cpu`
- Name: `QuixiCore CPU`
- Contract target: `v0.1`
- Status: `planned`
- Targets: `x86_64`, `aarch64`

The local library currently reports no supported kernel families. Planned
families are listed so tools and tests can verify naming against the umbrella
contract without claiming implementation support.

## Support Rule

A kernel family can move from planned to supported only after this repository has:

- a CPU implementation for the contract behavior,
- correctness tests for supported dtypes, layouts, and edge cases,
- benchmark results for the relevant umbrella shape family,
- an optimization entry in `perf/optimization_status.md`, and
- a baseline entry in `perf/baseline_status.md`.

CPU-specific implementation choices such as packing, ISA dispatch, thread
partitioning, NUMA policy, or fused epilogues must remain invisible at the
contract boundary unless documented as explicit CPU extensions.

## De-Facto Family Semantics

The umbrella kernel specs are stubs at contract v0.1, so the implemented
sibling backends (Metal, CUDA) define de-facto operation semantics. The CPU
backend follows them (verified 2026-07-07 against QuixiCore-Metal
`dequant.metal`/`tk/quant.py` and QuixiCore-CUDA `quant_formats.cuh`):

- `qgemv`: `out = dequantize(wq) @ x`, full-precision activations (f32 on
  CPU; Metal/CUDA use fp16), f32 accumulation. Quant block layouts are
  llama.cpp/GGUF byte-compatible (`q8_0` = 34 bytes: fp16 scale + 32 int8,
  round-to-nearest-even packing). Activation-quantized integer math is a
  separate op (`qgemv_w8a8` in the siblings; planned here) — never the
  default `qgemv` path.
- `rms_norm`: `y = x * rsqrt(mean(x^2) + eps) * weight`, eps inside the
  sqrt, fp32 (or better) accumulation, multiplicative weight, no bias,
  default `eps = 1e-5`.
- Op naming follows the siblings (`qgemv`, `qgemm`, `rms_norm`,
  `softmax`, fused `_add` suffixes); the umbrella registry family keys
  (`quant_gemv`, ...) are labels, not op names.

