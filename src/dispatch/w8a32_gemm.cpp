#include "kernels/quantization/w8a32_gemm.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "kernels/common/validation.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/qgemm.h"

namespace quixicore_cpu {
namespace {

struct Variant {
  const char* name;
  quant::W8A32GemmFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Variant kVariants[] = {
    {"ref", &quant::w8a32_gemm_ref, [](const CpuFeatures&) { return true; }},
#if defined(__aarch64__) || defined(_M_ARM64)
    {"neon", &quant::w8a32_gemm_neon,
     [](const CpuFeatures& features) { return features.neon; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_W8A32_AVX2)
    {"avx2", &quant::w8a32_gemm_avx2,
     [](const CpuFeatures& features) { return features.avx2 && features.fma; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_W8A32_AVX512)
    {"avx512", &quant::w8a32_gemm_avx512,
     [](const CpuFeatures& features) { return features.avx512f; }},
#endif
};

const Variant& resolve() {
  static const Variant* selected = []() -> const Variant* {
    const CpuFeatures& features = cpu_features();
    if (const char* forced = std::getenv("QUIXICORE_CPU_W8A32_VARIANT")) {
      for (const Variant& variant : kVariants) {
        if (std::strcmp(forced, variant.name) == 0 &&
            variant.supported(features)) {
          return &variant;
        }
      }
    }
    const Variant* best = &kVariants[0];
    for (const Variant& variant : kVariants) {
      if (variant.supported(features)) best = &variant;
    }
    return best;
  }();
  return *selected;
}

}  // namespace

Status qgemm_w8a32(const std::int8_t* weights, const float* weight_scale,
                   const float* x, float* y, long long m, long long n,
                   long long k) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(weights, weight_scale, x, y)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < n; ++row) {
    if (!std::isfinite(weight_scale[row])) return Status::kInvalidArgument;
  }
  resolve().fn(weights, weight_scale, x, y, m, n, k);
  return Status::kOk;
}

const char* qgemm_w8a32_variant() { return resolve().name; }

}  // namespace quixicore_cpu
