#include "quixicore_cpu/workspace.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>

#define REQUIRE(condition)                                                  \
  do {                                                                      \
    if (!(condition)) {                                                     \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ':'         \
                << __LINE__ << '\n';                                        \
      return 1;                                                             \
    }                                                                       \
  } while (0)

int main() {
  using quixicore_cpu::Status;
  using quixicore_cpu::Workspace;

  Workspace workspace;
  REQUIRE(workspace.capacity() == 0);
  REQUIRE(workspace.used() == 0);
  REQUIRE(workspace.reserve(4096) == Status::kOk);
  REQUIRE(workspace.capacity() >= 4096);

  auto* first = static_cast<std::uint8_t*>(workspace.allocate(257));
  REQUIRE(first != nullptr);
  REQUIRE(reinterpret_cast<std::uintptr_t>(first) % Workspace::kAlignment == 0);
  for (int index = 0; index < 257; ++index) first[index] = index & 0xFF;

  // Growing into another segment must not invalidate earlier allocations.
  void* large = workspace.allocate(2 * 1024 * 1024);
  REQUIRE(large != nullptr);
  REQUIRE(reinterpret_cast<std::uintptr_t>(large) % Workspace::kAlignment == 0);
  for (int index = 0; index < 257; ++index) {
    REQUIRE(first[index] == static_cast<std::uint8_t>(index & 0xFF));
  }
  REQUIRE(workspace.used() >= 2 * 1024 * 1024 + 257);

  const std::size_t retained = workspace.capacity();
  workspace.reset();
  REQUIRE(workspace.used() == 0);
  REQUIRE(workspace.capacity() == retained);
  REQUIRE(workspace.allocate(257) == first);
  REQUIRE(workspace.allocate(1, 3) == nullptr);
  REQUIRE(workspace.allocate(1, 128) == nullptr);
  REQUIRE(workspace.allocate(0) == nullptr);

  Workspace partially_used;
  REQUIRE(partially_used.reserve(1024) == Status::kOk);
  const std::size_t first_capacity = partially_used.capacity();
  REQUIRE(first_capacity > 64);
  REQUIRE(partially_used.allocate(first_capacity - 32) != nullptr);
  REQUIRE(partially_used.reserve(64) == Status::kOk);
  REQUIRE(partially_used.capacity() > first_capacity);
  REQUIRE(partially_used.allocate(64) != nullptr);

  Workspace moved = std::move(workspace);
  REQUIRE(moved.capacity() == retained);
  moved.reset();
  REQUIRE(moved.used() == 0);
  std::cout << "workspace tests passed\n";
  return 0;
}
