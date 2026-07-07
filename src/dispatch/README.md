# Dispatch

Runtime kernel-variant selection. For each contract operation this layer owns
a dispatch table mapping detected CPU features (`cpu_features()`) to the best
compiled variant, falling back to the scalar reference. Selection is resolved
once per process (or per explicit override), never per call.

Planned responsibilities:

- per-operation variant tables and registration from kernel families,
- feature-to-variant policy, including an environment override for forcing a
  variant during testing and benchmarking,
- one-time platform enablement handshakes (e.g. Linux AMX XTILE_DATA
  permission via `arch_prctl` before any tile instruction runs).

Empty apart from this note until the first kernel family lands.
