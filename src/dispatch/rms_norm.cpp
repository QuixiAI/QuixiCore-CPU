// Public rms_norm entry point and runtime variant dispatch. Same pattern
// as quant_gemv: variants slowest-first, last supported wins, env override
// QUIXICORE_CPU_RMS_NORM_VARIANT for A/B runs.

#include "quixicore_cpu/rms_norm.h"

#include <cstdlib>
#include <cstring>

#include "kernels/norms/rms_norm.h"
#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {
namespace {

struct Variant {
  const char* name;
  norms::RmsNormFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Variant kVariants[] = {
    {"ref", &norms::rms_norm_ref, [](const CpuFeatures&) { return true; }},
#if defined(__aarch64__) || defined(_M_ARM64)
    {"neon", &norms::rms_norm_neon,
     [](const CpuFeatures& f) { return f.neon; }},
#endif
};

const Variant& resolve() {
  static const Variant& chosen = []() -> const Variant& {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_RMS_NORM_VARIANT");
    if (forced != nullptr) {
      for (const auto& variant : kVariants) {
        if (std::strcmp(variant.name, forced) == 0 &&
            variant.supported(features)) {
          return variant;
        }
      }
    }
    const Variant* best = &kVariants[0];
    for (const auto& variant : kVariants) {
      if (variant.supported(features)) {
        best = &variant;
      }
    }
    return *best;
  }();
  return chosen;
}

}  // namespace

Status rms_norm(const float* x, const float* weight, float* y, long long rows,
                long long hidden, float eps) {
  if (rows <= 0 || hidden <= 0 || !(eps >= 0.0f)) {
    return Status::kInvalidShape;
  }
  resolve().fn(x, weight, y, rows, hidden, eps);
  return Status::kOk;
}

const char* rms_norm_variant() { return resolve().name; }

}  // namespace quixicore_cpu
