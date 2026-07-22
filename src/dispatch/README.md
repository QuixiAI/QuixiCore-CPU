# Dispatch

Runtime kernel-variant selection. For each contract operation this layer owns
a dispatch table mapping detected CPU features (`cpu_features()`) to the best
compiled variant, falling back to the scalar reference. Selection is resolved
once per process (or per explicit override), never per call.

Responsibilities:

- per-operation variant tables and registration from kernel families,
- feature-to-variant policy, including an environment override for forcing a
  contract-compatible variant during testing and benchmarking,
- one-time platform enablement handshakes (e.g. Linux AMX XTILE_DATA
  permission via `arch_prctl` before any tile instruction runs).

Current dispatchers cover qgemv, qgemv_w8a8, and RMSNorm. Weight-only and
activation-quantized GEMV remain separate public operations and dispatch
tables.
