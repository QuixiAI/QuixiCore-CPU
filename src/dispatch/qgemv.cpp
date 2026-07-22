// Public qgemv entry points and runtime variant dispatch.
//
// Contract-compatible variants are listed slowest-first; resolution picks the
// last one whose feature predicate passes, resolved once per process. Internal
// activation-quantizing experiments are not members of this dispatch table.

#include "quixicore_cpu/qgemv.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>

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
    {"ref", &quant::q8_0_gemv_ref, [](const CpuFeatures&) { return true; }},
#if defined(__aarch64__) || defined(_M_ARM64)
    {"neon", &quant::q8_0_gemv_neon,
     [](const CpuFeatures& f) { return f.neon; }},
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
  if (format != QuantFormat::kQ8_0 && format != QuantFormat::kQ4_0) {
    return Status::kUnsupportedFormat;
  }
  const long long block_size = format == QuantFormat::kQ8_0
                                   ? quant::kQ8_0BlockSize
                                   : quant::kQ4_0BlockSize;
  if (n <= 0 || k <= 0 || k % block_size != 0) {
    return Status::kInvalidShape;
  }
  // All implementations use signed long-long pointer offsets. Reject shapes
  // whose logical matrix or packed block count cannot be represented before
  // those products reach a hot loop.
  if (n > LLONG_MAX / k || n > LLONG_MAX / (k / block_size)) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

bool checked_mul(size_t a, size_t b, size_t* out) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    return false;
  }
  *out = a * b;
  return true;
}

Status packed_size(QuantFormat format, long long n, long long k,
                   size_t* size) {
  const long long block_size = format == QuantFormat::kQ8_0
                                   ? quant::kQ8_0BlockSize
                                   : quant::kQ4_0BlockSize;
  const size_t block_bytes = format == QuantFormat::kQ8_0
                                 ? sizeof(quant::BlockQ8_0)
                                 : sizeof(quant::BlockQ4_0);
  size_t blocks = 0;
  if (!checked_mul(static_cast<size_t>(n),
                   static_cast<size_t>(k / block_size), &blocks) ||
      !checked_mul(blocks, block_bytes, size)) {
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
  if (size == nullptr) {
    return Status::kInvalidArgument;
  }
  return packed_size(format, n, k, size);
}

Status qgemv_pack(QuantFormat format, const float* weights, long long n,
                  long long k, void* packed) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  size_t ignored = 0;
  if (packed_size(format, n, k, &ignored) != Status::kOk) {
    return Status::kInvalidShape;
  }
  if (weights == nullptr || packed == nullptr) {
    return Status::kInvalidArgument;
  }
  const bool packed_ok =
      format == QuantFormat::kQ8_0
          ? quant::q8_0_pack_ref(weights, n, k,
                                 static_cast<quant::BlockQ8_0*>(packed))
          : quant::q4_0_pack_ref(weights, n, k,
                                 static_cast<quant::BlockQ4_0*>(packed));
  return packed_ok ? Status::kOk : Status::kInvalidArgument;
}

Status qgemv_unpack(QuantFormat format, const void* packed, long long n,
                    long long k, float* weights) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  size_t ignored = 0;
  if (packed_size(format, n, k, &ignored) != Status::kOk) {
    return Status::kInvalidShape;
  }
  if (packed == nullptr || weights == nullptr) {
    return Status::kInvalidArgument;
  }
  if (format == QuantFormat::kQ8_0) {
    quant::q8_0_unpack_ref(static_cast<const quant::BlockQ8_0*>(packed), n, k,
                           weights);
  } else {
    quant::q4_0_unpack_ref(static_cast<const quant::BlockQ4_0*>(packed), n, k,
                           weights);
  }
  return Status::kOk;
}

Status qgemv(QuantFormat format, const void* packed, const float* x, float* y,
             long long n, long long k) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  size_t ignored = 0;
  if (packed_size(format, n, k, &ignored) != Status::kOk) {
    return Status::kInvalidShape;
  }
  if (packed == nullptr || x == nullptr || y == nullptr) {
    return Status::kInvalidArgument;
  }
  if (format == QuantFormat::kQ8_0) {
    resolve_q8_0().fn(static_cast<const quant::BlockQ8_0*>(packed), x, y, n,
                      k);
  } else {
    quant::q4_0_gemv_ref(static_cast<const quant::BlockQ4_0*>(packed), x, y,
                         n, k);
  }
  return Status::kOk;
}

const char* qgemv_variant(QuantFormat format) {
  if (format == QuantFormat::kQ8_0) {
    return resolve_q8_0().name;
  }
  if (format == QuantFormat::kQ4_0) {
    return "ref";
  }
  return "unsupported";
}

}  // namespace quixicore_cpu
