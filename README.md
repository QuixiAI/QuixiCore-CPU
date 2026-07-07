# QuixiCore CPU

QuixiCore CPU is the planned host CPU backend for the QuixiCore kernel contract.
It is intended for CPU-resident AI inference paths built around SIMD, matrix
extensions, cache-aware packing, and thread scheduling.

Current status: initialized scaffold. No QuixiCore kernel family is claimed
supported yet.

## Scope

The CPU backend targets:

- `x86_64`: scalar baseline first, then AVX2, AVX-512, VNNI, and AMX variants
  when hardware and benchmark evidence are available.
- `aarch64`: scalar baseline first, then NEON, DotProd, I8MM, SVE, and SME
  variants when hardware and benchmark evidence are available.

The first useful kernel work should focus on LLM inference paths where CPUs have
a realistic role: quantized GEMV, quantized GEMM, RMSNorm, LayerNorm, softmax,
sampling, embedding lookup, KV cache utilities, and small-batch decode support.

## Repository Layout

```text
.quixicore/backend.yaml     Backend compatibility metadata.
cmake/                      Build modules, including per-ISA compile flags.
include/quixicore_cpu/      Public CPU backend headers.
src/
  backend.cpp               Contract metadata and status reporting.
  runtime/                  Runtime CPU feature detection per OS/arch.
  dispatch/                 Kernel variant selection (placeholder).
  threading/                Thread pool and partitioning (placeholder).
  memory/                   Aligned alloc and packing buffers (placeholder).
kernels/                    Microkernels by contract family: <op>_<isa>.cpp.
tools/                      Local command-line utilities.
tests/
  smoke/                    Build- and metadata-level checks.
  unit/                     Component tests (feature detection, utils).
  correctness/              Contract correctness vs specs and tolerances.
  testdata/                 Small deterministic fixtures.
benchmarks/                 Native benchmark harness (quixicore_cpu_bench).
scripts/                    Convenience wrappers (scripts/bench).
perf/                       Performance guide and evidence logs.
docs/                       CPU backend design notes (see architecture.md).
plan.md                     Original vision notes for the backend.
```

The library is built once per architecture with no global ISA flags; ISA
kernel variants are compiled per file and selected at runtime from detected
CPU features. `docs/architecture.md` explains the layout and the dispatch,
threading, and packing design.

## Build

```sh
cmake -S . -B build -DQUIXICORE_CPU_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Inspect the initialized backend metadata and compiler-target feature hints:

```sh
./build/quixicore_cpu_info
```

## Evidence Policy

Kernel support requires:

- a native CPU implementation,
- correctness coverage against the QuixiCore contract,
- benchmark coverage for the relevant registry shape set,
- a focused optimization run recorded in `perf/optimization_status.md`, and
- a baseline or status entry in `perf/baseline_status.md`.

Scaffold-only changes may skip kernel benchmarking, but they must not claim a
speedup or supported kernel.

