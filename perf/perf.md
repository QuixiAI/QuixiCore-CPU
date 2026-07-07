# CPU Performance Guide

QuixiCore CPU benchmarks use host-side timing and must document enough system
state to make results reproducible.

## Required Report Fields

- Repository and git commit or working-tree label.
- QuixiCore contract version.
- Kernel family and operation.
- Input dtype, output dtype, and quant format when applicable.
- Shape family and concrete dimensions.
- CPU model, microarchitecture if known, core count, SMT state, and memory
  configuration.
- ISA target and compiler flags.
- Operating system and compiler version.
- Thread count, affinity policy, NUMA policy, and frequency governor or power
  mode when applicable.
- Warmup iterations, measurement iterations, median, and variance or min/max.
- Correctness command and result.

## Measurement Policy

- Use a monotonic clock for host-side timing.
- Separate warmup iterations from measured iterations.
- Pin or document thread placement for threaded kernels.
- Document frequency scaling state when it can affect measurements.
- Report scalar/reference baselines separately from ISA-tuned candidates.
- Do not compare results across machines unless hardware, memory, compiler, and
  thread policy are clearly identified.

## Minimum Optimization Run

A valid optimization run includes:

```text
kernel:
operation:
dtype_or_format:
shape_set:
correctness_command:
baseline_command:
candidate_command:
hardware:
os:
compiler:
thread_policy:
warmups:
iterations:
median:
variance_or_min_max:
decision:
artifact:
```

Record completed runs in `perf/optimization_status.md`.

