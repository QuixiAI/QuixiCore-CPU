# Benchmarks

This directory is reserved for CPU-native benchmark harnesses.

No benchmark binary is present yet. The first benchmark harness should consume
the umbrella shape families from `registry/benchmark-shapes.yaml`, starting with:

- `quant_matmul` for quantized GEMV and GEMM work.
- `decode_small` for low-batch decode-adjacent kernels.
- `prefill` only after dense matmul or attention work exists.

CPU benchmark reports must include the fields listed in `perf/perf.md`.

