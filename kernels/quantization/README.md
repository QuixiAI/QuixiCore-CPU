# Quantization

Contract families: `quant_gemv`, `quant_gemm`, `quantized_lm_head`. Spec:
umbrella `specs/kernels/quantization.md`, formats under `specs/formats/`.
This is the first kernel track for the CPU backend (see `docs/roadmap.md`
Phase 2): quantized GEMV/GEMM dominates low-batch CPU decode. Status:
planned; no implementation yet.
