// Public qgemv_w8a8 entry points — the activation-quantizing quantized GEMV
// twin of qgemv (see the reservation in include/quixicore_cpu/qgemv.h).
//
// Weights stay quantized (q4_0 or q8_0); activations are
// quantized to per-32-block int8 inside the kernel and the dot is integer with
// f32-accumulated combined scales. A scalar reference is the correctness
// anchor; SIMD variants resolve last-wins by CPU feature (as in qgemv).

#include "quixicore_cpu/qgemv_w8a8.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "kernels/quantization/qgemv.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {
namespace {

struct Q8_0Variant {
  const char* name;
  quant::Q8_0GemvFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Q8_0Variant kQ8_0Variants[] = {
    {"ref", &quant::q8_0_gemv_w8a8_ref,
     [](const CpuFeatures&) { return true; }},
#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    {"dotprod", &quant::q8_0_gemv_dotprod,
     [](const CpuFeatures& features) { return features.dotprod; }},
#endif
};

const Q8_0Variant& resolve_q8_0() {
  static const Q8_0Variant* chosen = []() -> const Q8_0Variant* {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_QGEMV_W8A8_VARIANT");
    if (forced != nullptr) {
      for (const auto& variant : kQ8_0Variants) {
        if (std::strcmp(variant.name, forced) == 0 &&
            variant.supported(features)) {
          return &variant;
        }
      }
    }
    const Q8_0Variant* best = &kQ8_0Variants[0];
    for (const auto& variant : kQ8_0Variants) {
      if (variant.supported(features)) {
        best = &variant;
      }
    }
    return best;
  }();
  return *chosen;
}

struct Q4_0Variant {
  const char* name;
  quant::Q4_0W8A8GemvFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Q4_0Variant kQ4_0Variants[] = {
    {"ref", &quant::q4_0_gemv_w8a8_ref,
     [](const CpuFeatures&) { return true; }},
#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    {"dotprod", &quant::q4_0_gemv_w8a8_dotprod,
     [](const CpuFeatures& features) { return features.dotprod; }},
#endif
};

const Q4_0Variant& resolve_q4_0() {
  static const Q4_0Variant* chosen = []() -> const Q4_0Variant* {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_QGEMV_W8A8_VARIANT");
    if (forced != nullptr) {
      for (const auto& variant : kQ4_0Variants) {
        if (std::strcmp(variant.name, forced) == 0 &&
            variant.supported(features)) {
          return &variant;
        }
      }
    }
    const Q4_0Variant* best = &kQ4_0Variants[0];
    for (const auto& variant : kQ4_0Variants) {
      if (variant.supported(features)) {
        best = &variant;
      }
    }
    return best;
  }();
  return *chosen;
}

Status validate(QuantFormat format, long long n, long long k) {
  if (format != QuantFormat::kQ8_0 && format != QuantFormat::kQ4_0) {
    return Status::kUnsupportedFormat;
  }
  std::size_t ignored = 0;
  return qgemv_packed_size(format, n, k, &ignored);
}

}  // namespace

Status qgemv_w8a8_packed_size(QuantFormat format, long long n, long long k,
                              long long* bytes) {
  std::size_t packed_bytes = 0;
  const Status status = qgemv_packed_size(format, n, k, &packed_bytes);
  if (status != Status::kOk) {
    return status;
  }
  if (bytes == nullptr) {
    return Status::kInvalidArgument;
  }
  if (packed_bytes > static_cast<std::size_t>(LLONG_MAX)) {
    return Status::kInvalidShape;
  }
  *bytes = static_cast<long long>(packed_bytes);
  return Status::kOk;
}

Status qgemv_w8a8_pack(QuantFormat format, const float* weights, long long n,
                       long long k, void* packed) {
  return qgemv_pack(format, weights, n, k, packed);
}

Status qgemv_w8a8(QuantFormat format, const void* packed_weights,
                  const float* x, float* y, long long n, long long k) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  if (packed_weights == nullptr || x == nullptr || y == nullptr) {
    return Status::kInvalidArgument;
  }
  for (long long element = 0; element < k; ++element) {
    if (!std::isfinite(x[element])) {
      return Status::kInvalidArgument;
    }
  }
  if (format == QuantFormat::kQ8_0) {
    resolve_q8_0().fn(static_cast<const quant::BlockQ8_0*>(packed_weights), x,
                      y, n, k);
  } else {
    resolve_q4_0().fn(static_cast<const quant::BlockQ4_0*>(packed_weights), x,
                      y, n, k);
  }
  return Status::kOk;
}

const char* qgemv_w8a8_variant(QuantFormat format) {
  if (format == QuantFormat::kQ8_0) {
    return resolve_q8_0().name;
  }
  if (format == QuantFormat::kQ4_0) {
    return resolve_q4_0().name;
  }
  return "unsupported";
}

}  // namespace quixicore_cpu
