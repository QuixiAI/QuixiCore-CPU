# Threading

CPU work partitioning. Implemented:

- `thread_pool.h/.cpp` — fork-join pool with persistent workers and a
  deterministic contiguous-range partition (`parallel_ranges`). Public
  thread-count control is `quixicore_cpu::set_num_threads()` /
  `num_threads()` (`include/quixicore_cpu/threading.h`); default 1 keeps
  everything synchronous. Small counts and nested calls execute inline;
  ranges never split finer than the caller's `min_per_chunk`, so kernel
  outputs are bit-identical at any thread count (rows are never split
  across workers). No allocation per call (fn-pointer + context, not
  `std::function`). macOS workers request user-initiated QoS to stay on
  performance cores.
- Kernels take the pool via `parallel_ranges`; they never spawn threads.
  Hot loop bodies live in free functions with by-value arguments — see the
  codegen note in `thread_pool.h` (capture-frame aliasing measured 1.6-1.9x
  slowdowns before that rule).

Planned: affinity pinning and NUMA placement policy when server-class
multi-socket targets arrive (documented per `perf/perf.md` whenever it
affects a measurement).
