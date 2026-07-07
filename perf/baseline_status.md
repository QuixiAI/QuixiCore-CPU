# Baseline Status

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
| 2026-07-07 | rms_norm (`neon`) | f32 | decode_small R1 H4096 | Apple M4 Max, 1 thread, NEON | `scripts/bench --preset quick --kernel rms_norm` | 0.52 us | CV 0.056 | `perf/results/2026-07-07/024347-quick/` | 94.4 GB/s cache-resident; in-progress |
| 2026-07-07 | rms_norm (`neon`) | f32 | stress R512 H4096 | Apple M4 Max, 1 thread, NEON | `scripts/bench --preset quick --kernel rms_norm` | 263.33 us | CV 0.052 | `perf/results/2026-07-07/024347-quick/` | 63.8 GB/s; in-progress |
| 2026-07-07 | quant_gemv (`dotprod`) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, NEON DotProd | `scripts/bench --preset quick --kernel qgemv` | 0.3006 ms | CV 0.020 | `perf/results/2026-07-07/023619-quick/` | 59.3 W-GB/s = 51% of triad DRAM roofline; in-progress |
| 2026-07-07 | quant_gemv (`dotprod`) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, NEON DotProd | `scripts/bench --preset quick --kernel qgemv` | 1.2010 ms | CV 0.018 | `perf/results/2026-07-07/023619-quick/` | 59.4 W-GB/s; in-progress |
| 2026-07-07 | quant_gemv (`dotprod`) | q8_0 | quant_matmul m=1 N16384 K4096 | Apple M4 Max, 1 thread, NEON DotProd | `scripts/bench --preset quick --kernel qgemv` | 1.2012 ms | CV 0.025 | `perf/results/2026-07-07/023619-quick/` | 59.4 W-GB/s; in-progress |
| 2026-07-07 | quant_gemv (`ref`) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 4.3189 ms | CV 0.014 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
| 2026-07-07 | quant_gemv (`ref`) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 17.2801 ms | CV 0.022 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
| 2026-07-07 | quant_gemv (`ref`) | q8_0 | quant_matmul m=1 N16384 K4096 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 17.1908 ms | CV 0.011 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |

