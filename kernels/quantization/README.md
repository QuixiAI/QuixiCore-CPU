# Quantization

Contract families: `quant_gemv`, `quant_gemm`, `quantized_lm_head`. Spec:
umbrella `specs/kernels/quantization.md`, formats under `specs/formats/`.
This is the first kernel track for the CPU backend (see `docs/roadmap.md`
Phase 2): quantized GEMV/GEMM dominates low-batch CPU decode.

Status: `quant_gemv` family in progress, exposed as `qgemv` (family op
naming). Contract semantics per Metal/CUDA: `out = dequantize(wq) @ x`,
full-precision activations, f32 accumulation, GGUF-byte-compatible q8_0.
Variants: `ref` scalar, `neon` f32-activation (contract default on
aarch64), `dotprod_i8` int8 SDOT (activation-quantizing — contract-
divergent, env-forced only; previews a future `qgemv_w8a8` twin op).
Dispatch in `src/dispatch/qgemv.cpp`, correctness in
`tests/correctness/test_qgemv.cpp`, evidence in
`perf/optimization_status.md` (2026-07-07). Not claimed supported; next:
`qgemv_w8a8`, q4_0, i8mm qgemm. `quant_gemm`, `quantized_lm_head`:
planned.
