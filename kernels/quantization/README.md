# Quantization

Contract families: `quant_gemv`, `quant_gemm`, `quantized_lm_head`. Spec:
umbrella `specs/kernels/quantization.md`, formats under `specs/formats/`.
This is the first kernel track for the CPU backend (see `docs/roadmap.md`
Phase 2): quantized GEMV/GEMM dominates low-batch CPU decode.

Status: `quant_gemv` in progress — GGUF-compatible q8_0 scalar reference
(`qgemv_ref.cpp`) with pack/unpack, dispatch via
`src/dispatch/quant_gemv.cpp`, correctness in
`tests/correctness/test_quant_gemv.cpp`, evidence in
`perf/optimization_status.md` (2026-07-07). Not claimed supported; ISA
variants (NEON dotprod first) and further formats pending. `quant_gemm`,
`quantized_lm_head`: planned.
