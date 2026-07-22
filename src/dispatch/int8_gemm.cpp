#include "quixicore_cpu/qgemm.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "kernels/common/validation.h"
#include "kernels/quantization/int8_gemm.h"
#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {
namespace {

struct Variant {
  const char* name;
  quant::Int8GemmFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Variant kVariants[] = {
    {"ref", &quant::int8_gemm_ref_kernel,
     [](const CpuFeatures&) { return true; }},
#if defined(QUIXICORE_CPU_HAVE_INT8_GEMM_AVX2)
    {"avx2", &quant::int8_gemm_avx2_kernel,
     [](const CpuFeatures& features) { return features.avx2; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_INT8_GEMM_DOTPROD)
    {"dotprod", &quant::int8_gemm_dotprod_kernel,
     [](const CpuFeatures& features) { return features.dotprod; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_INT8_GEMM_AVX512_VNNI)
    {"avx512_vnni", &quant::int8_gemm_avx512_vnni_kernel,
     [](const CpuFeatures& features) { return features.avx512_vnni; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_INT8_GEMM_I8MM)
    {"i8mm", &quant::int8_gemm_i8mm_kernel,
     [](const CpuFeatures& features) { return features.i8mm; }},
#endif
};

const Variant& resolve() {
  static const Variant& selected = []() -> const Variant& {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_INT8_GEMM_VARIANT");
    if (forced != nullptr) {
      for (const Variant& variant : kVariants) {
        if (std::strcmp(forced, variant.name) == 0 &&
            variant.supported(features)) {
          return variant;
        }
      }
    }
    const Variant* best = &kVariants[0];
    for (const Variant& variant : kVariants) {
      if (variant.supported(features)) best = &variant;
    }
    return *best;
  }();
  return selected;
}

}  // namespace

Status int8_gemm(const std::int8_t* weights, const std::int8_t* x,
                 const float* weight_scale, const float* activation_scale,
                 const std::int32_t* weight_row_sum,
                 const int* activation_zero_point, float* y, long long m,
                 long long n, long long k, bool asymmetric) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(weights, x, weight_scale, activation_scale, y) ||
      (asymmetric &&
       !detail::all_nonnull(weight_row_sum, activation_zero_point))) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < n; ++row) {
    if (!std::isfinite(weight_scale[row])) return Status::kInvalidArgument;
  }
  for (long long row = 0; row < m; ++row) {
    if (!std::isfinite(activation_scale[row])) {
      return Status::kInvalidArgument;
    }
  }
  // SIMD accumulators are signed 32-bit. Model dimensions are far below this
  // conservative exactness limit; retain the int64 scalar anchor above it.
  const Variant& variant = resolve();
  const quant::Int8GemmFn kernel = k > 131000
                                       ? &quant::int8_gemm_ref_kernel
                                       : variant.fn;
  kernel(weights, x, weight_scale, activation_scale, weight_row_sum,
         activation_zero_point, y, m, n, k, asymmetric);
  return Status::kOk;
}

const char* int8_gemm_variant() { return resolve().name; }

}  // namespace quixicore_cpu
