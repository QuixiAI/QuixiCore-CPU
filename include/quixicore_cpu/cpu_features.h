#pragma once

#include <vector>

#include "quixicore_cpu/backend.h"

namespace quixicore_cpu {

// CPU features detected at runtime on the executing machine. Fields for the
// non-native architecture are always false. Detection reports hardware plus
// OS support for the instructions and their register state; it does not pick
// a kernel variant. Variant selection belongs to the dispatch layer.
struct CpuFeatures {
  // x86_64
  bool fma = false;
  bool f16c = false;
  bool avx2 = false;
  bool avx512f = false;
  bool avx512_vnni = false;
  bool amx_tile = false;
  bool amx_int8 = false;
  bool amx_bf16 = false;

  // aarch64
  bool neon = false;
  bool fp16 = false;
  bool dotprod = false;
  bool i8mm = false;
  bool sve = false;
  bool sve2 = false;
  bool sme = false;
  bool sme2 = false;
};

// Detects once on first use and caches the result. Thread-safe.
const CpuFeatures& cpu_features();

// Runtime analogue of compile_time_feature_hints().
std::vector<FeatureHint> runtime_feature_hints();

}  // namespace quixicore_cpu
