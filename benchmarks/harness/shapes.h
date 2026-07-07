#pragma once

#include "harness/case.h"

namespace qcb {

// Source of truth: QuixiAI/QuixiCore registry/benchmark-shapes.yaml
// (contract v0.1). Values copied verbatim; update by hand when the registry
// changes. No YAML parsing here, matching the Metal harness precedent.
inline constexpr long long kQuantMatmulM[] = {1, 16, 128};
inline constexpr long long kQuantMatmulN[] = {4096, 8192, 16384};
inline constexpr long long kQuantMatmulK[] = {4096, 8192, 16384};

inline constexpr long long kDecodeSmallBatch[] = {1, 2, 4};
inline constexpr long long kDecodeSmallSequence[] = {1};
inline constexpr long long kDecodeSmallHidden[] = {2048, 4096};

template <class T>
const T& pick(Preset preset, const T& smoke, const T& quick,
              const T& comprehensive) {
  switch (preset) {
    case Preset::kSmoke:
      return smoke;
    case Preset::kQuick:
      return quick;
    default:
      return comprehensive;
  }
}

}  // namespace qcb
