#include "quixicore_cpu/lowbit.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernels/common/validation.h"
#include "kernels/quantization/lowbit.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/quantization.h"

namespace quixicore_cpu {
namespace {

struct Variant {
  const char* name;
  quant::LowBitW8A8Fn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Variant kVariants[] = {
    {"ref", &quant::lowbit_w8a8_ref,
     [](const CpuFeatures&) { return true; }},
#if defined(QUIXICORE_CPU_HAVE_LOWBIT_W8A8_AVX2)
    {"avx2", &quant::lowbit_w8a8_avx2,
     [](const CpuFeatures& features) { return features.avx2; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_LOWBIT_W8A8_DOTPROD)
    {"dotprod", &quant::lowbit_w8a8_dotprod,
     [](const CpuFeatures& features) { return features.dotprod; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_LOWBIT_W8A8_I8MM)
    {"i8mm", &quant::lowbit_w8a8_i8mm,
     [](const CpuFeatures& features) { return features.i8mm; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_LOWBIT_W8A8_AVX512_VNNI)
    {"avx512_vnni", &quant::lowbit_w8a8_avx512_vnni,
     [](const CpuFeatures& features) { return features.avx512_vnni; }},
#endif
};

const Variant& resolve() {
  static const Variant& selected = []() -> const Variant& {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_LOWBIT_W8A8_VARIANT");
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

Status lowbit_gemm_w8a8(const std::uint8_t* packed_weights,
                        const float* weight_scales, const float* x, float* y,
                        long long m, long long n, long long k) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_weights, weight_scales, x, y)) {
    return Status::kInvalidArgument;
  }
  std::size_t bytes = 0;
  std::size_t scales = 0;
  Status status = lowbit_packed_size(LowBitFormat::kInt4Row, n, k, 0,
                                     &bytes, &scales);
  if (status != Status::kOk) return status;
  (void)bytes;
  (void)scales;
  thread_local std::vector<std::int8_t> quantized;
  thread_local std::vector<float> activation_scales;
  quantized.resize(static_cast<std::size_t>(m * k));
  activation_scales.resize(static_cast<std::size_t>(m));
  status = quantize_int8(x, quantized.data(), activation_scales.data(), m, k,
                         k);
  if (status != Status::kOk) return status;
  const Variant& variant = resolve();
  const quant::LowBitW8A8Fn kernel =
      k > 131000 ? &quant::lowbit_w8a8_ref : variant.fn;
  kernel(packed_weights, weight_scales, quantized.data(),
         activation_scales.data(), y, m, n, k);
  return Status::kOk;
}

const char* lowbit_gemm_w8a8_variant() { return resolve().name; }

}  // namespace quixicore_cpu
