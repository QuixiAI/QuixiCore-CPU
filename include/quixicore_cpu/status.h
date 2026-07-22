#pragma once

namespace quixicore_cpu {

enum class Status {
  kOk,
  kInvalidShape,       // non-positive dims, block-size violations, bad eps
  kUnsupportedFormat,
  kInvalidArgument,    // null output/input pointer or non-finite pack input
  kOutOfMemory,        // CPU-owned workspace or packed-panel allocation failed
};

}  // namespace quixicore_cpu
