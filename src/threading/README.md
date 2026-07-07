# Threading

CPU work partitioning. Planned responsibilities:

- a lightweight thread pool sized to physical topology,
- work partitioning helpers for row/tile blocking across threads,
- affinity and NUMA placement policy (documented per `perf/perf.md` whenever
  it affects a measurement).

Kernels take a thread-context argument rather than spawning threads
themselves, so single-threaded correctness testing stays trivial.

Empty apart from this note until the first threaded kernel lands.
