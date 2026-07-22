#pragma once

namespace quixicore_cpu {

enum class Status {
  kOk,
  kInvalidShape,       // non-positive dims, block-size violations, bad eps
  kUnsupportedFormat,
  kInvalidArgument,    // null output/input pointer or non-finite pack input
};

}  // namespace quixicore_cpu
