#pragma once

// Kernel-internal fork-join pool. Public thread-count control lives in
// include/quixicore_cpu/threading.h; kernels use parallel_ranges and never
// spawn threads themselves.
//
// Codegen note for callers: keep the hot loops out of the lambda's capture
// environment. Either call a free function with by-value arguments from the
// lambda, or copy captures into locals before looping — loop bounds and
// pointers read through a capture frame defeat register allocation because
// stores through output pointers may alias the frame (measured 1.6-1.9x
// slowdown on rms_norm before this rule).

namespace quixicore_cpu::threading {

using RangeFn = void (*)(void* ctx, long long begin, long long end,
                         int worker);

// Runs fn over disjoint contiguous ranges covering [0, count), on the
// calling thread plus the pool workers. The partition is a deterministic
// function of (count, parallelism, min_per_chunk); ranges never split finer
// than min_per_chunk, and small counts (or num_threads() == 1) execute
// inline on the caller. Blocking; fn must not throw and must not call
// parallel_ranges (nested calls execute inline). No allocation.
void parallel_ranges_impl(long long count, long long min_per_chunk,
                          RangeFn fn, void* ctx);

template <class F>
void parallel_ranges(long long count, long long min_per_chunk, F&& f) {
  parallel_ranges_impl(
      count, min_per_chunk,
      [](void* ctx, long long begin, long long end, int worker) {
        (*static_cast<F*>(ctx))(begin, end, worker);
      },
      &f);
}

}  // namespace quixicore_cpu::threading
