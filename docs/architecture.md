# CPU Backend Architecture

How the repository is organized and why. The design target: **one library
build per architecture that runs correctly on any CPU of that architecture**
across macOS, Linux, and Windows on x86_64 and aarch64, and picks the fastest
compiled kernel variant at runtime.

## Two-Axis Kernel Organization

Kernel code is organized along two axes:

- **Contract family** (directory): `kernels/<family>/` mirrors the umbrella
  `registry/kernels.yaml` families, matching the layout convention of the
  sibling backends.
- **ISA variant** (file suffix): `<op>_ref.cpp` is the portable scalar
  reference; `<op>_<isa>.cpp` are optional optimized variants
  (`avx2`, `avx512`, `avx512_vnni`, `amx`, `neon`, `dotprod`, `i8mm`, `sve`,
  `sve2`, `sme2`).

See `kernels/README.md` for the full file convention and rules.

## Build Policy: Per-File ISA Flags

The library never compiles with `-march=native` or a global MSVC `/arch`
flag. Doing so would make the binary crash with illegal-instruction faults on
older CPUs of the same architecture.

Instead, `cmake/QuixiCoreCPUFeatures.cmake` provides
`quixicore_cpu_add_isa_sources()`, which compiles only the variant's own
source files with that ISA's flags (e.g. `-mavx512vnni`, `/arch:AVX2`,
`-march=armv8.2-a+dotprod`). If the host compiler cannot target an ISA, the
variant is skipped at configure time and the dispatcher simply never offers
it. This is the same strategy used by llama.cpp, oneDNN, and KleidiAI.

Two guard layers keep this safe:

1. **Configure-time**: variant sources are only added when the target
   architecture matches and the compiler accepts the flags.
2. **Run-time**: a compiled variant is only ever *called* after
   `cpu_features()` confirms the executing CPU supports it.

Architecture-specific files are additionally preprocessor-guarded
(`#if defined(__x86_64__) || defined(_M_X64)` etc.) so multi-arch
compilations never break. macOS universal binaries should be built as two
single-arch builds combined with `lipo`; per-file ISA flags cannot be applied
per-slice in a single fat build.

## Runtime Feature Detection

`src/runtime/` answers "what can this CPU execute?" once per process, cached:

| Platform | Mechanism |
|---|---|
| x86_64, all OSes | `CPUID` leaves 1 and 7 + `XGETBV` (verifies the OS enabled AVX/AVX-512/AMX register state, not just the silicon) |
| aarch64 macOS | `sysctlbyname("hw.optional.arm.FEAT_*")` |
| aarch64 Linux | `getauxval(AT_HWCAP / AT_HWCAP2)` |
| aarch64 Windows | `IsProcessorFeaturePresent(PF_ARM_*)` |

Notes:

- NEON is reported unconditionally on aarch64; it is architecturally
  baseline.
- AMX detection reports hardware + XSAVE support. Linux additionally requires
  a one-time `arch_prctl(ARCH_REQ_XCOMP_PERM, XTILE_DATA)` handshake before
  tile instructions run; that belongs to the dispatch layer, not detection.
- Windows/aarch64 has no query for NEON-only I8MM or SME yet, so those
  under-report there until the SDK adds constants. Under-reporting costs
  speed, never correctness.

## Dispatch, Threading, Memory

- `src/dispatch/` — per-operation variant tables keyed off `cpu_features()`,
  resolved once, scalar reference as the universal fallback, with a test/bench
  override hook. See `src/dispatch/README.md`.
- `src/threading/` — thread pool and partitioning helpers; kernels receive a
  thread context rather than spawning threads. See `src/threading/README.md`.
- `src/memory/` — aligned allocation, packed-weight buffers, workspace
  arenas. See `src/memory/README.md`.

## Directory Map

```text
.quixicore/backend.yaml    Backend compatibility metadata.
cmake/                     Build modules (per-ISA flag policy).
include/quixicore_cpu/     Public headers (backend metadata, cpu_features).
src/
  backend.cpp              Contract metadata and status reporting.
  runtime/                 Runtime CPU feature detection per OS/arch.
  dispatch/                Public op entry points and variant selection.
  threading/               Fork-join pool; set_num_threads() control.
  memory/                  Aligned alloc and packing buffers (placeholder).
kernels/                   Microkernels: <family>/<op>_<isa>.cpp.
tests/
  smoke/                   Build- and metadata-level checks.
  unit/                    Component tests (feature detection, utils).
  correctness/             Contract correctness vs specs and tolerances.
  testdata/                Small deterministic fixtures.
benchmarks/                Benchmark harness (consumes umbrella shape sets).
perf/                      Performance guide, evidence logs, raw results.
tools/                     Command-line utilities (quixicore_cpu_info).
docs/                      Design notes, contract alignment, roadmap.
```

## Platform Support Matrix

| OS | x86_64 | aarch64 |
|---|---|---|
| Linux | GCC/Clang | GCC/Clang |
| macOS | Clang | Clang (Apple silicon) |
| Windows | MSVC/clang-cl | MSVC (NEON/DotProd only until MSVC grows SVE/I8MM support) |

CI builds and tests all five OS/arch combinations
(`.github/workflows/ci.yml`). What CI cannot provide is *ISA coverage*:
runners do not expose AMX or SME, so per-variant correctness and performance
evidence still requires matching hardware, per `AGENTS.md`.

## Adding a Kernel (checklist)

1. Read the umbrella spec and tolerance entries for the family.
2. Implement `<op>_ref.cpp` (portable scalar) + correctness tests in
   `tests/correctness/`.
3. Wire the operation into the dispatch table with the reference as the only
   variant.
4. Add an ISA variant only with matching hardware available: implement
   `<op>_<isa>.cpp`, register via `quixicore_cpu_add_isa_sources()`, extend
   the dispatch table, compare against the reference for correctness.
5. Run the benchmark harness on the umbrella shape set; complete a focused
   optimization run and record it in `perf/optimization_status.md` and
   `perf/baseline_status.md` before claiming support.
