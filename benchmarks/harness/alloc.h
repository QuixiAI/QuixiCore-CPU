#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace qcb {

struct AlignedDeleter {
  void operator()(void* p) const noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

// No destructors run on free; use for trivially destructible element types
// only (benchmark buffers are float/int arrays).
template <class T>
using AlignedBuffer = std::unique_ptr<T[], AlignedDeleter>;

// 64-byte-aligned (cache line / vector width) array; throws std::bad_alloc.
template <class T>
AlignedBuffer<T> aligned_alloc_array(long long count) {
  // std::aligned_alloc requires a size that is a multiple of the alignment.
  const size_t bytes =
      ((static_cast<size_t>(count) * sizeof(T) + 63) / 64) * 64;
  void* p =
#if defined(_MSC_VER)
      _aligned_malloc(bytes, 64);
#else
      std::aligned_alloc(64, bytes);
#endif
  if (p == nullptr) {
    throw std::bad_alloc{};
  }
  return AlignedBuffer<T>(static_cast<T*>(p));
}

}  // namespace qcb
