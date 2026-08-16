# Agent Instructions

This is the QuixiCore CPU backend repository. The umbrella contract repository is
`QuixiAI/QuixiCore`.

## Rules

- Preserve the public QuixiCore contract. CPU-specific packing, ISA dispatch, and
  thread scheduling must stay behind the shared API semantics.
- Before changing a contract-facing kernel, read the relevant umbrella registry
  and spec files:
  - `registry/kernels.yaml`
  - `registry/quant-formats.yaml`
  - `registry/benchmark-shapes.yaml`
  - `registry/tolerances.yaml`
  - `matrices/`
- Do not claim support for a kernel, dtype, quant format, architecture feature,
  or performance tier without correctness and benchmark evidence in this repo.
- Kernel implementation or routing work requires at least one focused
  performance optimization run on an affected CPU path. Read `perf/perf.md`
  first and record the result in `perf/optimization_status.md`.
- Pure documentation, metadata, and scaffolding changes may skip a kernel
  performance run, but they must not assert performance improvements.

## Build Rules

- Use the checked-in CMake presets or Makefile targets.
- Keep every generated configuration under `build/<profile>/`.
- Do not create top-level `build-*` or `cmake-build-*` directories. Add a
  reusable preset for a lasting configuration, or use `build/scratch/` for a
  temporary experiment and remove it afterward.
- Build trees are disposable. Record durable correctness and performance
  evidence under the documented test and `perf/` paths.

## Evidence Locations

- Operating guide: `perf/perf.md`
- Optimization notebook: `perf/optimization_status.md`
- Baseline index: `perf/baseline_status.md`
- Raw benchmark outputs: `perf/results/`
