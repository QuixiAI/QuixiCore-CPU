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

## Evidence Locations

- Operating guide: `perf/perf.md`
- Optimization notebook: `perf/optimization_status.md`
- Baseline index: `perf/baseline_status.md`
- Raw benchmark outputs: `perf/results/`

