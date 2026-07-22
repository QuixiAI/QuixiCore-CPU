#pragma once

#include <cstddef>
#include <limits>

#include "quixicore_cpu/workspace.h"

namespace quixicore_cpu::detail {

// Persistent workspace owned by the current application/pool thread.
Workspace& thread_workspace();

// Nested-safe arena frame. Destruction rewinds logical usage while keeping all
// backing storage warm for the next invocation.
class WorkspaceFrame {
 public:
  explicit WorkspaceFrame(Workspace* workspace = nullptr)
      : workspace_(workspace != nullptr ? workspace : &thread_workspace()),
        marker_(workspace_->mark()) {}

  ~WorkspaceFrame() { workspace_->rewind(marker_); }
  WorkspaceFrame(const WorkspaceFrame&) = delete;
  WorkspaceFrame& operator=(const WorkspaceFrame&) = delete;

  void* allocate_bytes(std::size_t bytes,
                       std::size_t alignment = Workspace::kAlignment) {
    return workspace_->allocate(bytes, alignment);
  }

  template <typename T>
  T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return nullptr;
    }
    constexpr std::size_t alignment =
        alignof(T) > Workspace::kAlignment ? alignof(T) : Workspace::kAlignment;
    return static_cast<T*>(
        allocate_bytes(count * sizeof(T), alignment));
  }

 private:
  Workspace* workspace_;
  Workspace::Marker marker_;
};

}  // namespace quixicore_cpu::detail
