#pragma once

#include <cstddef>
#include <memory>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

namespace detail {
class WorkspaceFrame;
}

// Reusable, segmented scratch arena for CPU kernels. Every returned pointer is
// stable until reset() even when later allocations grow the arena. Storage is
// retained across reset calls, so warmed-up kernels do not allocate on their
// hot path. Workspace objects are not thread-safe; callers may own one per
// concurrent invocation. Kernel-internal worker slices use the same arena.
class Workspace {
 public:
  static constexpr std::size_t kAlignment = 64;

  Workspace();
  ~Workspace();
  Workspace(Workspace&& other) noexcept;
  Workspace& operator=(Workspace&& other) noexcept;

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  // Ensures a future contiguous allocation of at least bytes can be served.
  Status reserve(std::size_t bytes);

  // Returns aligned scratch or nullptr for zero bytes, unsupported alignment,
  // overflow, or allocation failure. alignment must be a power of two no
  // greater than kAlignment.
  void* allocate(std::size_t bytes,
                 std::size_t alignment = kAlignment) noexcept;

  // Releases all allocations logically while retaining their backing blocks.
  void reset() noexcept;
  std::size_t capacity() const noexcept;
  std::size_t used() const noexcept;

 private:
  struct Impl;
  struct Marker {
    std::size_t block = 0;
    std::size_t offset = 0;
    bool valid = false;
  };

  Marker mark() const noexcept;
  void rewind(Marker marker) noexcept;

  std::unique_ptr<Impl> impl_;
  friend class detail::WorkspaceFrame;
};

}  // namespace quixicore_cpu
