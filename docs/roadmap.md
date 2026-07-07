# Roadmap

This roadmap turns the CPU backend vision in `plan.md` into staged engineering
work. It is intentionally evidence-gated.

## Phase 0: Repository Scaffold

- Add backend metadata.
- Add build, smoke test, and info tool.
- Add performance evidence templates.
- Keep all kernel support status planned.

## Phase 1: Reference Harness

- Add scalar reference implementations for selected kernels.
- Add deterministic correctness tests against simple host-side references.
- Add benchmark harness plumbing without performance claims.

## Phase 2: First CPU Kernel Track

Start with quantized GEMV or quantized GEMM because CPU decode workloads are
often memory-bandwidth and low-batch sensitive.

Required evidence:

- selected dtype or quant format,
- umbrella shape set,
- correctness run,
- baseline measurement,
- candidate measurement,
- hardware and compiler details,
- keep or reject decision.

## Phase 3: ISA Variants

Add optimized variants only when matching hardware is available:

- x86: AVX2, AVX-512, AVX-512 VNNI, AMX.
- Arm: NEON, DotProd, I8MM, SVE, SME.

Each variant needs separate correctness and performance evidence.

## Phase 4: Decode Utilities

After the first quantized matmul path has evidence, expand to decode-adjacent
operations such as RMSNorm, softmax, sampling, KV cache utilities, and
embedding lookup.

