# Baseline Status

This file tracks accepted baseline measurements for supported or in-progress CPU
kernel paths. A baseline is required before reporting a speedup or updating
backend coverage status.

## Current Harness Index

`quixicore_cpu_bench` cases available for baselining (see
`benchmarks/README.md`):

| Case | What it measures | Role | Reproduce |
|---|---|---|---|
| `mem_triad` | Single-thread STREAM-triad bandwidth, 96 KiB - 768 MiB working sets | Machine memory roofline for judging memory-bound kernels | `scripts/bench --preset quick --kernel mem_triad` |
| `sgemv_naive` | Naive scalar f32 GEMV on `quant_matmul` m=1 shapes | Reference semantics future GEMV variants must beat | `scripts/bench --preset quick --kernel sgemv_naive` |
| `qgemv` | Public q8_0 GEMV plus scalar/decomposed references | Quantized decode baseline | `scripts/bench --preset quick --kernel qgemv` |
| `qgemv_formats` | Public q4_0 weight-only and q4_0/q8_0 W8A8 GEMV | Continued quant-format baselines | `scripts/bench --preset quick --kernel qgemv_formats` |
| `rms_norm` | Public f32 RMSNorm plus scalar reference | Norm baseline | `scripts/bench --preset quick --kernel rms_norm` |
| `contract_ops` | Softmax, causal attention, MoE routing, Mamba scan, q8_0 QGEMM | Representative sibling-port baselines | `scripts/bench --preset quick --kernel contract_ops` |

| Date | Kernel | Dtype / Format | Shape Set | Target | Command | Median | Min / Max or Variance | Artifact | Notes |
|---|---|---|---|---|---|---:|---|---|---|
| 2026-07-22 | qgemv (`ref`) | q4_0 / f32 activation | quant_matmul m=1 N4096 K4096 | Apple M5 Max, 6 threads | `quixicore_cpu_bench --preset quick --kernel qgemv_formats --threads 6 --warmup 5 --iters 30 --min-sample-ms 5` | 1.048475 ms | CV 0.0358 | `perf/results/2026-07-22/qgemv-formats-final-t6-fixed/` | 9.0 W-GB/s; portable candidate, no ISA speedup claim |
| 2026-07-22 | qgemv_w8a8 (`ref`) | q4_0 / blockwise int8 activation | quant_matmul m=1 N4096 K4096 | Apple M5 Max, 6 threads | same | 0.858071 ms | CV 0.0200 | same | 11.0 W-GB/s; portable candidate, no ISA speedup claim |
| 2026-07-22 | qgemv_w8a8 (`dotprod`) | q8_0 / blockwise int8 activation | quant_matmul m=1 N4096 K4096 | Apple M5 Max, 6 threads, aarch64 DotProd | same | 0.138599 ms | CV 0.0852 | same | 5.33x over direct portable ref; 128.6 W-GB/s; candidate |
| 2026-07-22 | softmax | f32 | R512 H4096 | Apple M5 Max, 1 / 6 threads | `quixicore_cpu_bench --preset quick --kernel contract_ops --threads {1,6}` | 2.571875 / 0.549761 ms | CV 0.0450 / 0.0463 | `perf/results/2026-07-22/all-kernels-final-{t1,t6}/` | scalar baseline 3.487917 ms at t1; candidate |
| 2026-07-22 | causal attention | f32 | H8 S128 D64 | Apple M5 Max, 1 / 6 threads | same | 2.570625 / 0.555919 ms | CV 0.0156 / 0.0993 | same | materialized scalar baseline 2.677979 ms at t1; candidate |
| 2026-07-22 | MoE top-k routing | f32 | T1024 E64 K4 | Apple M5 Max, 1 / 6 threads | same | 0.156679 / 0.151962 ms | CV 0.0936 / 0.0506 | same | semantically equivalent full-sort baseline 0.827618 ms at t1; serial candidate |
| 2026-07-22 | Mamba selective scan | f32 | C256 S512 N16 | Apple M5 Max, 1 / 6 threads | same | 3.560417 / 0.773006 ms | CV 0.0172 / 0.0550 | same | serial baseline 3.586625 ms at t1; candidate |
| 2026-07-22 | q8_0 QGEMM | q8_0 weights / f32 activation | M16 N2048 K2048 | Apple M5 Max, 1 / 6 threads | same | 3.815875 / 1.377552 ms | CV 0.0161 / 0.0968 | same | dequantized scalar baseline 31.739709 ms at t1; candidate |
| 2026-07-07 | qgemv (`neon`, contract path) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, NEON f32-act | `scripts/bench --preset quick --kernel qgemv` | 1.0344 ms | CV 0.027 | `perf/results/2026-07-07/033244-quick/` | family numerics (dequant W x f32 x); 17.2 W-GB/s; in-progress |
| 2026-07-07 | qgemv (`neon`, contract path) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, NEON f32-act | `scripts/bench --preset quick --kernel qgemv` | 4.1164 ms | CV 0.022 | `perf/results/2026-07-07/033244-quick/` | family numerics; 17.3 W-GB/s; in-progress |
| 2026-07-07 | qgemv (`dotprod_i8`, 8 threads) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 8 threads, NEON DotProd | `scripts/bench --preset quick --kernel qgemv --threads 8` | 0.0679 ms | CV 0.024 | `perf/results/2026-07-07/030745-quick/` | Historical internal measurement; the variant now routes only through public qgemv_w8a8 |
| 2026-07-07 | mem_triad (8 threads) | f32 | ws_192MiB (aggregate DRAM roofline) | Apple M4 Max, 8 threads | `scripts/bench --preset quick --kernel mem_triad --threads 8` | 0.8022 ms | CV 0.058 | `perf/results/2026-07-07/030745-quick/` | 251 GB/s (304 at 12 threads); system probe |
| 2026-07-07 | rms_norm (`neon`) | f32 | decode_small R1 H4096 | Apple M4 Max, 1 thread, NEON | `scripts/bench --preset quick --kernel rms_norm` | 0.52 us | CV 0.056 | `perf/results/2026-07-07/024347-quick/` | 94.4 GB/s cache-resident; in-progress |
| 2026-07-07 | rms_norm (`neon`) | f32 | stress R512 H4096 | Apple M4 Max, 1 thread, NEON | `scripts/bench --preset quick --kernel rms_norm` | 263.33 us | CV 0.052 | `perf/results/2026-07-07/024347-quick/` | 63.8 GB/s; in-progress |
| 2026-07-07 | qgemv (`dotprod_i8`) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, NEON DotProd | `QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8 scripts/bench --preset quick --kernel qgemv` | 0.3006 ms | CV 0.020 | `perf/results/2026-07-07/023619-quick/` | 59.3 W-GB/s = 51% of triad DRAM roofline; env-forced variant since realignment |
| 2026-07-07 | qgemv (`dotprod_i8`) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, NEON DotProd | `QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8 scripts/bench --preset quick --kernel qgemv` | 1.2010 ms | CV 0.018 | `perf/results/2026-07-07/023619-quick/` | 59.4 W-GB/s; env-forced variant since realignment |
| 2026-07-07 | qgemv (`ref`) | q8_0 | quant_matmul m=1 N4096 K4096 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 4.3189 ms | CV 0.014 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
| 2026-07-07 | qgemv (`ref`) | q8_0 | quant_matmul m=1 N8192 K8192 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 17.2801 ms | CV 0.022 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
| 2026-07-07 | qgemv (`ref`) | q8_0 | quant_matmul m=1 N16384 K4096 | Apple M4 Max, 1 thread, baseline flags | `scripts/bench --preset quick --kernel qgemv` | 17.1908 ms | CV 0.011 | `perf/results/2026-07-07/022305-quick/` | scalar reference; in-progress, not claimed supported |
