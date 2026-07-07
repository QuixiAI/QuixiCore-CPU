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
| 2026-07-07 | qgemv (`neon`, contract path) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, NEON f32-act | `scripts/bench --preset quick --kernel qgemv` | 1.0344 ms | CV 0.027 | `perf/results/2026-07-07/033244-quick/` | family numerics (dequant W x f32 x); 17.2 W-GB/s; in-progress |
| 2026-07-07 | qgemv (`neon`, contract path) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, NEON f32-act | `scripts/bench --preset quick --kernel qgemv` | 4.1164 ms | CV 0.022 | `perf/results/2026-07-07/033244-quick/` | family numerics; 17.3 W-GB/s; in-progress |
| 2026-07-07 | qgemv (`dotprod_i8`, 8 threads) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 8 threads, NEON DotProd | `scripts/bench --preset quick --kernel qgemv --threads 8` | 0.0679 ms | CV 0.024 | `perf/results/2026-07-07/030745-quick/` | 263 W-GB/s ~= aggregate DRAM roofline; env-forced variant since the 2026-07-07 realignment (previews qgemv_w8a8) |
| 2026-07-07 | mem_triad (8 threads) | f32 | ws_192MiB (aggregate DRAM roofline) | Apple M4 Max, 8 threads | `scripts/bench --preset quick --kernel mem_triad --threads 8` | 0.8022 ms | CV 0.058 | `perf/results/2026-07-07/030745-quick/` | 251 GB/s (304 at 12 threads); system probe |
| 2026-07-07 | rms_norm (`neon`) | f32 | decode_small R1 H4096 | Apple M4 Max, 1 thread, NEON | `scripts/bench --preset quick --kernel rms_norm` | 0.52 us | CV 0.056 | `perf/results/2026-07-07/024347-quick/` | 94.4 GB/s cache-resident; in-progress |
| 2026-07-07 | rms_norm (`neon`) | f32 | stress R512 H4096 | Apple M4 Max, 1 thread, NEON | `scripts/bench --preset quick --kernel rms_norm` | 263.33 us | CV 0.052 | `perf/results/2026-07-07/024347-quick/` | 63.8 GB/s; in-progress |
| 2026-07-07 | qgemv (`dotprod_i8`) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, NEON DotProd | `QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8 scripts/bench --preset quick --kernel qgemv` | 0.3006 ms | CV 0.020 | `perf/results/2026-07-07/023619-quick/` | 59.3 W-GB/s = 51% of triad DRAM roofline; env-forced variant since realignment |
| 2026-07-07 | qgemv (`dotprod_i8`) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, NEON DotProd | `QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8 scripts/bench --preset quick --kernel qgemv` | 1.2010 ms | CV 0.018 | `perf/results/2026-07-07/023619-quick/` | 59.4 W-GB/s; env-forced variant since realignment |
| 2026-07-07 | qgemv (`ref`) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 4.3189 ms | CV 0.014 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
| 2026-07-07 | qgemv (`ref`) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 17.2801 ms | CV 0.022 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
| 2026-07-07 | qgemv (`ref`) | q8_0 | quant_matmul m=1 N16384 K4096 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 17.1908 ms | CV 0.011 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |

