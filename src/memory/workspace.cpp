#include "quixicore_cpu/workspace.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quixicore_cpu {
namespace {

constexpr std::size_t kMinimumBlock = 64 * 1024;

bool power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool aligned_offset(std::size_t offset, std::size_t alignment,
                    std::size_t* result) {
  const std::size_t mask = alignment - 1;
  if (offset > std::numeric_limits<std::size_t>::max() - mask) return false;
  *result = (offset + mask) & ~mask;
  return true;
}

std::size_t growth_size(std::size_t required, std::size_t previous) {
  std::size_t candidate = std::max(required, kMinimumBlock);
  if (previous != 0 && previous <= std::numeric_limits<std::size_t>::max() / 2) {
    candidate = std::max(candidate, previous * 2);
  }
  return candidate;
}

struct Block {
  explicit Block(std::size_t requested)
      : data(static_cast<std::byte*>(::operator new(
            requested, std::align_val_t(Workspace::kAlignment)))),
        capacity(requested) {}

  ~Block() {
    if (data != nullptr) {
      ::operator delete(data, std::align_val_t(Workspace::kAlignment));
    }
  }
  Block(Block&& other) noexcept
      : data(std::exchange(other.data, nullptr)),
        capacity(std::exchange(other.capacity, 0)),
        used(std::exchange(other.used, 0)) {}
  Block& operator=(Block&& other) noexcept {
    if (this == &other) return *this;
    if (data != nullptr) {
      ::operator delete(data, std::align_val_t(Workspace::kAlignment));
    }
    data = std::exchange(other.data, nullptr);
    capacity = std::exchange(other.capacity, 0);
    used = std::exchange(other.used, 0);
    return *this;
  }
  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  std::byte* data = nullptr;
  std::size_t capacity = 0;
  std::size_t used = 0;
};

}  // namespace

struct Workspace::Impl {
  std::vector<Block> blocks;
  std::size_t current = 0;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&& other) noexcept = default;
Workspace& Workspace::operator=(Workspace&& other) noexcept = default;

Status Workspace::reserve(std::size_t bytes) {
  if (bytes == 0) return Status::kOk;
  try {
    if (impl_ == nullptr) impl_ = std::make_unique<Impl>();
    for (std::size_t index = impl_->current; index < impl_->blocks.size();
         ++index) {
      const Block& block = impl_->blocks[index];
      std::size_t offset = 0;
      if (aligned_offset(block.used, kAlignment, &offset) &&
          offset <= block.capacity && bytes <= block.capacity - offset) {
        return Status::kOk;
      }
    }
    const std::size_t previous =
        impl_->blocks.empty() ? 0 : impl_->blocks.back().capacity;
    impl_->blocks.emplace_back(growth_size(bytes, previous));
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

void* Workspace::allocate(std::size_t bytes, std::size_t alignment) noexcept {
  if (bytes == 0 || impl_ == nullptr || !power_of_two(alignment) ||
      alignment > kAlignment) {
    return nullptr;
  }
  for (std::size_t index = impl_->current; index < impl_->blocks.size(); ++index) {
    Block& block = impl_->blocks[index];
    std::size_t offset = 0;
    if (!aligned_offset(block.used, alignment, &offset) ||
        offset > block.capacity || bytes > block.capacity - offset) {
      continue;
    }
    block.used = offset + bytes;
    impl_->current = index;
    return block.data + offset;
  }
  try {
    const std::size_t padding = alignment - 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - padding) {
      return nullptr;
    }
    const std::size_t previous =
        impl_->blocks.empty() ? 0 : impl_->blocks.back().capacity;
    impl_->blocks.emplace_back(growth_size(bytes + padding, previous));
    impl_->current = impl_->blocks.size() - 1;
    Block& block = impl_->blocks.back();
    block.used = bytes;
    return block.data;
  } catch (const std::bad_alloc&) {
    return nullptr;
  } catch (const std::length_error&) {
    return nullptr;
  }
}

void Workspace::reset() noexcept {
  if (impl_ == nullptr) return;
  for (Block& block : impl_->blocks) block.used = 0;
  impl_->current = 0;
}

std::size_t Workspace::capacity() const noexcept {
  if (impl_ == nullptr) return 0;
  std::size_t total = 0;
  for (const Block& block : impl_->blocks) {
    if (block.capacity > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += block.capacity;
  }
  return total;
}

std::size_t Workspace::used() const noexcept {
  if (impl_ == nullptr) return 0;
  std::size_t total = 0;
  for (const Block& block : impl_->blocks) {
    if (block.used > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += block.used;
  }
  return total;
}

Workspace::Marker Workspace::mark() const noexcept {
  if (impl_ == nullptr || impl_->blocks.empty()) return {};
  return {impl_->current, impl_->blocks[impl_->current].used, true};
}

void Workspace::rewind(Marker marker) noexcept {
  if (impl_ == nullptr) return;
  if (!marker.valid || marker.block >= impl_->blocks.size()) {
    reset();
    return;
  }
  impl_->blocks[marker.block].used = marker.offset;
  for (std::size_t index = marker.block + 1; index < impl_->blocks.size(); ++index) {
    impl_->blocks[index].used = 0;
  }
  impl_->current = marker.block;
}

namespace detail {

Workspace& thread_workspace() {
  thread_local Workspace workspace;
  return workspace;
}

}  // namespace detail
}  // namespace quixicore_cpu
