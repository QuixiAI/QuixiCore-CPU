# Contract Alignment

QuixiCore CPU implements the shared QuixiCore contract as a native CPU backend.
The umbrella repository owns the contract; this repository owns only CPU
implementation, test, and benchmark work.

## Current Declaration

- Backend: `cpu`
- Name: `QuixiCore CPU`
- Contract target: `v0.1`
- Status: `planned`
- Targets: `x86_64`, `aarch64`

The local library currently reports no supported kernel families. Planned
families are listed so tools and tests can verify naming against the umbrella
contract without claiming implementation support.

## Support Rule

A kernel family can move from planned to supported only after this repository has:

- a CPU implementation for the contract behavior,
- correctness tests for supported dtypes, layouts, and edge cases,
- benchmark results for the relevant umbrella shape family,
- an optimization entry in `perf/optimization_status.md`, and
- a baseline entry in `perf/baseline_status.md`.

CPU-specific implementation choices such as packing, ISA dispatch, thread
partitioning, NUMA policy, or fused epilogues must remain invisible at the
contract boundary unless documented as explicit CPU extensions.

