// Public qgemv entry points and runtime variant dispatch.
//
// Auto-selectable variants are listed slowest-first; resolution picks the
// last one whose feature predicate passes, resolved once per process.
// Variants with auto_select == false implement different numerics than the
// family qgemv contract (e.g. activation quantization) and run only when
// QUIXICORE_CPU_QGEMV_VARIANT names them explicitly.

#include "quixicore_cpu/qgemv.h"

#include <cstdlib>
#include <cstring>

#include "kernels/quantization/qgemv.h"
#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {
namespace {

struct Q8_0Variant {
  const char* name;
  quant::Q8_0GemvFn fn;
  bool (*supported)(const CpuFeatures&);
  bool auto_select;
};

constexpr Q8_0Variant kQ8_0Variants[] = {
    {"ref", &quant::q8_0_gemv_ref, [](const CpuFeatures&) { return true; },
     true},
#if defined(__aarch64__) || defined(_M_ARM64)
    {"neon", &quant::q8_0_gemv_neon,
     [](const CpuFeatures& f) { return f.neon; }, true},
#endif
#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    // Activation-quantizing int8 path; contract-divergent, env-forced only.
    {"dotprod_i8", &quant::q8_0_gemv_dotprod,
     [](const CpuFeatures& f) { return f.dotprod; }, false},
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
      if (variant.auto_select && variant.supported(features)) {
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
  if (n <= 0 || k <= 0 || k % quant::kQ8_0BlockSize != 0) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

}  // namespace

Status qgemv_packed_size(QuantFormat format, long long n, long long k,
                         size_t* size) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  *size = static_cast<size_t>(n) *
          static_cast<size_t>(k / quant::kQ8_0BlockSize) *
          sizeof(quant::BlockQ8_0);
  return Status::kOk;
}

Status qgemv_pack(QuantFormat format, const float* weights, long long n,
                  long long k, void* packed) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  quant::q8_0_pack_ref(weights, n, k,
                       static_cast<quant::BlockQ8_0*>(packed));
  return Status::kOk;
}

Status qgemv_unpack(QuantFormat format, const void* packed, long long n,
                    long long k, float* weights) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  quant::q8_0_unpack_ref(static_cast<const quant::BlockQ8_0*>(packed), n, k,
                         weights);
  return Status::kOk;
}

Status qgemv(QuantFormat format, const void* packed, const float* x, float* y,
             long long n, long long k) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  resolve_q8_0().fn(static_cast<const quant::BlockQ8_0*>(packed), x, y, n, k);
  return Status::kOk;
}

const char* qgemv_variant(QuantFormat format) {
  if (format != QuantFormat::kQ8_0) {
    return "unsupported";
  }
  return resolve_q8_0().name;
}

}  // namespace quixicore_cpu
