# QuixiCore CPU

QuixiCore CPU is the experimental host CPU backend for the QuixiCore kernel
contract. It provides portable f32 reference semantics for every active v0.1
family and for the top-level operation union inventoried from the Metal, XPU,
CUDA, and ROCm siblings, including their quantized, serving, training, vision,
state-space, and collective extensions.
CPU-resident implementations use runtime ISA dispatch and a persistent
fork-join thread pool behind the shared API semantics.

Current status: correctness-tested reference implementation. Family-level
support remains conservative until each operation, dtype, and registry shape
has its own benchmark evidence; see `docs/sibling-port-matrix.md`.
The Colibri CPU-algorithm excavation is tracked separately in
`docs/colibri-port-matrix.md`.

## Scope

The CPU backend targets:

- `x86_64`: scalar baseline first, then AVX2, AVX-512, VNNI, and AMX variants
  when hardware and benchmark evidence are available.
- `aarch64`: scalar baseline first, then NEON, DotProd, I8MM, SVE, and SME
  variants when hardware and benchmark evidence are available.

The current portable surface includes norms and backward paths, activations,
dense/quantized matmul, attention and decode, the sibling packed-format
decoders, sampling/serving, embeddings and KV caches, linear attention,
Mamba/SSD/FFT-convolution semantics, MoE, vision, training utilities, and
host-reference collectives. Runtime-selected SIMD routes include aarch64
RMSNorm/q8_0 GEMV plus Colibri-derived W8A32, W8A8, and low-bit matmul paths on
supported aarch64 and x86_64 ISAs; remaining portable kernels are still future
performance work.

## Repository Layout

```text
.quixicore/backend.yaml     Backend compatibility metadata.
cmake/                      Build modules, including per-ISA compile flags.
include/quixicore_cpu/      Public CPU backend headers.
src/
  backend.cpp               Contract metadata and status reporting.
  runtime/                  Runtime CPU feature detection per OS/arch.
  dispatch/                 Public op entry points and variant selection.
  threading/                Fork-join pool; set_num_threads() control.
  memory/                   Reusable workspaces and CPU packed-weight panels.
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
