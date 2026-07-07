# CPU Kernels

Microkernel implementations, organized by QuixiCore contract kernel family.
The umbrella registry (`registry/kernels.yaml`) owns family names and specs;
this tree owns only the CPU implementations.

## Family Map

| Directory | Umbrella kernel families |
|---|---|
| `common/` | Shared microkernel helpers (not a contract family) |
| `norms/` | `norms` |
| `softmax/` | `softmax` |
| `activations/` | `activations` |
| `attention/` | `causal_attention`, `paged_attention`, `mla_decode` |
| `quantization/` | `quant_gemv`, `quant_gemm`, `quantized_lm_head` |
| `sampling/` | `sampling`, `beam_search`, `speculative_decode` |
| `moe/` | `moe_routing`, `grouped_moe_gemm` |
| `serving/` | Embedding lookup, KV cache utilities |

Directories for the remaining contract families (`mamba_ssd` under `ssm/`,
`optimizers/`) are created when work on them starts.

## Source File Convention

Each operation follows a two-axis layout: the directory encodes the contract
family, the file suffix encodes the ISA variant.

```text
kernels/quantization/
  qgemv.h            Kernel-internal interface used by the dispatch layer.
  qgemv_ref.cpp      Portable scalar reference. Builds on every platform.
  qgemv_avx2.cpp     x86_64 AVX2 variant.
  qgemv_avx512.cpp   x86_64 AVX-512 variant.
  qgemv_neon.cpp     aarch64 NEON variant.
  qgemv_dotprod.cpp  aarch64 DotProd variant.
```

Rules:

- `_ref.cpp` comes first. Every operation needs a portable scalar reference
  before any ISA variant. It is the correctness oracle and the fallback the
  dispatcher always has.
- ISA variants are registered in the build with
  `quixicore_cpu_add_isa_sources(... ISA <isa> SOURCES <files>)` from
  `cmake/QuixiCoreCPUFeatures.cmake`. That compiles only those files with the
  ISA's flags. Never add `-march=native` or a global `/arch` flag.
- Variant selection happens at runtime in `src/dispatch/`, keyed off
  `quixicore_cpu::cpu_features()`. A variant that was compiled but is not
  supported by the executing CPU is never called.
- Files that only build on one architecture must also be preprocessor-guarded
  (`#if defined(__x86_64__) || defined(_M_X64)` etc.) so multi-arch builds
  stay safe.
- Per `AGENTS.md`, an ISA variant may only be committed with correctness
  evidence and a focused optimization run on matching hardware, recorded in
  `perf/optimization_status.md`. No hardware, no variant.
