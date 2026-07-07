#pragma once

namespace quixicore_cpu {

enum class Status {
  kOk,
  kInvalidShape,       // non-positive dims, block-size violations, bad eps
  kUnsupportedFormat,
};

}  // namespace quixicore_cpu
