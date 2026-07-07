# Baseline Status

No CPU kernel baselines have been recorded yet.

This file tracks accepted baseline measurements for supported or in-progress CPU
kernel paths. A baseline is required before reporting a speedup or updating
backend coverage status.

## Current Harness Index

`quixicore_cpu_bench` cases available for baselining (see
`benchmarks/README.md`). Both are system probes — not kernel baselines and
not contract kernels:

| Case | What it measures | Role | Reproduce |
|---|---|---|---|
| `mem_triad` | Single-thread STREAM-triad bandwidth, 96 KiB - 768 MiB working sets | Machine memory roofline for judging memory-bound kernels | `scripts/bench --preset quick --kernel mem_triad` |
| `sgemv_naive` | Naive scalar f32 GEMV on `quant_matmul` m=1 shapes | Reference semantics future GEMV variants must beat | `scripts/bench --preset quick --kernel sgemv_naive` |

| Date | Kernel | Dtype / Format | Shape Set | Target | Command | Median | Min / Max or Variance | Artifact | Notes |
|---|---|---|---|---|---|---:|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | No baselines yet. |

