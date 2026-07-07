#pragma once

#include <atomic>
#include <memory>

namespace qcb {

// Out-of-line and opaque to the optimizer; defined in donotopt_sink.cpp.
// The harness must never be built with LTO or the sink stops being opaque.
void sink(const void* p);

// Compiler barriers so measured work cannot be deleted, hoisted, or folded.
// Thunk contract: every benchmark thunk routes its output through
// do_not_optimize() (or calls clobber()) once per invocation.
#if defined(__GNUC__) || defined(__clang__)

template <class T>
inline void do_not_optimize(T const& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

inline void clobber() { asm volatile("" : : : "memory"); }

#else  // MSVC

template <class T>
inline void do_not_optimize(T const& value) {
  sink(std::addressof(value));
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

inline void clobber() {
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

#endif

}  // namespace qcb
