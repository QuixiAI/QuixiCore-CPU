// Public quant_gemv entry points and runtime variant dispatch.
//
// Variants are listed slowest-first; resolution picks the last variant whose
// feature predicate passes on the executing CPU, resolved once per process.
// QUIXICORE_CPU_QGEMV_VARIANT forces a named variant (if supported) for
// testing and benchmarking A/B runs.

#include "quixicore_cpu/quant_gemv.h"

#include <cstdlib>
#include <cstring>

#include "kernels/quantization/qgemv.h"
#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {
namespace {

struct Q8_0Variant {
  const char* name;
  qgemv::Q8_0GemvFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr Q8_0Variant kQ8_0Variants[] = {
    {"ref", &qgemv::q8_0_gemv_ref, [](const CpuFeatures&) { return true; }},
#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    {"dotprod", &qgemv::q8_0_gemv_dotprod,
     [](const CpuFeatures& f) { return f.dotprod; }},
#endif
};

const Q8_0Variant& resolve_q8_0() {
  static const Q8_0Variant& chosen = []() -> const Q8_0Variant& {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_QGEMV_VARIANT");
    if (forced != nullptr) {
      for (const auto& variant : kQ8_0Variants) {
        if (std::strcmp(variant.name, forced) == 0 &&
            variant.supported(features)) {
          return variant;
        }
      }
    }
    const Q8_0Variant* best = &kQ8_0Variants[0];
    for (const auto& variant : kQ8_0Variants) {
      if (variant.supported(features)) {
        best = &variant;
      }
    }
    return *best;
  }();
  return chosen;
}

Status validate(QuantFormat format, long long n, long long k) {
  if (format != QuantFormat::kQ8_0) {
    return Status::kUnsupportedFormat;
  }
  if (n <= 0 || k <= 0 || k % qgemv::kQ8_0BlockSize != 0) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

}  // namespace

Status quant_gemv_packed_size(QuantFormat format, long long n, long long k,
                              size_t* size) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  *size = static_cast<size_t>(n) *
          static_cast<size_t>(k / qgemv::kQ8_0BlockSize) *
          sizeof(qgemv::BlockQ8_0);
  return Status::kOk;
}

Status quant_gemv_pack(QuantFormat format, const float* weights, long long n,
                       long long k, void* packed) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  qgemv::q8_0_pack_ref(weights, n, k,
                       static_cast<qgemv::BlockQ8_0*>(packed));
  return Status::kOk;
}

Status quant_gemv_unpack(QuantFormat format, const void* packed, long long n,
                         long long k, float* weights) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  qgemv::q8_0_unpack_ref(static_cast<const qgemv::BlockQ8_0*>(packed), n, k,
                         weights);
  return Status::kOk;
}

Status quant_gemv(QuantFormat format, const void* packed, const float* x,
                  float* y, long long n, long long k) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  resolve_q8_0().fn(static_cast<const qgemv::BlockQ8_0*>(packed), x, y, n, k);
  return Status::kOk;
}

const char* quant_gemv_variant(QuantFormat format) {
  if (format != QuantFormat::kQ8_0) {
    return "unsupported";
  }
  return resolve_q8_0().name;
}

}  // namespace quixicore_cpu
