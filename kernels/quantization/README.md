# Quantization

Contract families: `quant_gemv`, `quant_gemm`, `quantized_lm_head`. Spec:
umbrella `specs/kernels/quantization.md`, formats under `specs/formats/`.
This is the first kernel track for the CPU backend (see `docs/roadmap.md`
Phase 2): quantized GEMV/GEMM dominates low-batch CPU decode.

Status: `quant_gemv` family in progress, exposed as `qgemv` (family op
naming). Contract semantics per Metal/CUDA: `out = dequantize(wq) @ x`,
full-precision activations, f32 accumulation, and GGUF-byte-compatible q8_0
and q4_0 weights. Public q8_0 variants are portable `ref` and aarch64 `neon`;
q4_0 currently uses its portable reference. The separate `qgemv_w8a8` route
quantizes activations per 32-element block and supports q4_0 (`ref`) and q8_0
(`ref` or aarch64 `dotprod`). Dispatch lives in `src/dispatch/qgemv*.cpp`,
correctness in `tests/correctness/test_qgemv*.cpp`, and evidence in
`perf/optimization_status.md`. q8_0 QGEMM and quantized LM-head composition are
implemented; further GGUF formats and ISA-tuned q4_0 remain planned.
