// Public qgemv entry points and runtime variant dispatch.
//
// Contract-compatible variants are listed slowest-first; resolution picks the
// last one whose feature predicate passes, resolved once per process. Internal
// activation-quantizing experiments are not members of this dispatch table.

#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "kernels/quantization/activation_quant.h"
#include "kernels/quantization/qgemv.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "kernels/quantization/gguf_pack_ref.h"
#include "kernels/quantization/gguf_ref.h"
#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {
namespace {

struct Q8_0Variant {
  const char* name;
  quant::Q8_0GemvFn fn;
  bool (*supported)(const CpuFeatures&);
};

struct GgufVariant {
  const char* name;
  quant::GgufGemvFn fn;
  bool (*supported)(const CpuFeatures&);
};

constexpr GgufVariant kGgufVariants[] = {
    {"ref", &quant::gguf_gemv_ref,
     [](const CpuFeatures&) { return true; }},
    {"blocked_ref", &quant::gguf_gemv_blocked_ref,
     [](const CpuFeatures&) { return true; }},
#if defined(__aarch64__) || defined(_M_ARM64)
    {"neon", &quant::gguf_gemv_neon,
     [](const CpuFeatures& features) { return features.neon; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_GGUF_GEMV_AVX2)
    {"avx2", &quant::gguf_gemv_avx2,
     [](const CpuFeatures& features) { return features.avx2; }},
#endif
#if defined(QUIXICORE_CPU_HAVE_GGUF_GEMV_AVX512)
    {"avx512", &quant::gguf_gemv_avx512,
     [](const CpuFeatures& features) { return features.avx512f; }},
#endif
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

const GgufVariant& resolve_gguf() {
  static const GgufVariant& chosen = []() -> const GgufVariant& {
    const CpuFeatures& features = cpu_features();
    const char* forced = std::getenv("QUIXICORE_CPU_GGUF_GEMV_VARIANT");
    if (forced != nullptr) {
      for (const auto& variant : kGgufVariants) {
        if (std::strcmp(variant.name, forced) == 0 &&
            variant.supported(features)) {
          return variant;
        }
      }
    }
    const GgufVariant* best = &kGgufVariants[0];
    for (const auto& variant : kGgufVariants) {
      if (variant.supported(features)) best = &variant;
    }
    return *best;
  }();
  return chosen;
}

Status validate(QuantFormat format, long long n, long long k) {
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
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
  long long block_size = 0;
  size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  size_t blocks = 0;
  if (!checked_mul(static_cast<size_t>(n),
                   static_cast<size_t>(k / block_size), &blocks) ||
      !checked_mul(blocks, block_bytes, size)) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

Status validate_activation(QuantActivationFormat format, long long n,
                           long long k) {
  long long block_size = 0;
  std::size_t ignored = 0;
  if (!quant::activation_format_info(format, &block_size, &ignored)) {
    return Status::kUnsupportedFormat;
  }
  if (n <= 0 || k <= 0 || k % block_size != 0 || n > LLONG_MAX / k ||
      n > LLONG_MAX / (k / block_size)) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

Status activation_packed_size(QuantActivationFormat format, long long n,
                              long long k, size_t* size) {
  long long block_size = 0;
  size_t block_bytes = 0;
  if (!quant::activation_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
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
  return qgemv_pack_weighted(format, weights, nullptr, n, k, packed);
}

Status qgemv_pack_weighted(QuantFormat format, const float* weights,
                           const float* importance, long long n, long long k,
                           void* packed) {
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
  if (importance != nullptr) {
    const long long elements = n * k;
    for (long long i = 0; i < elements; ++i) {
      if (!std::isfinite(importance[i]) || importance[i] < 0.0f) {
        return Status::kInvalidArgument;
      }
    }
  }
  if (format != QuantFormat::kQ8_0 && format != QuantFormat::kQ4_0 &&
      format != QuantFormat::kTQ2_0 &&
      !quant::gguf_pack_supported(format)) {
    return Status::kUnsupportedFormat;
  }
  if (quant::gguf_pack_supported(format)) {
    return quant::gguf_pack_ref(format, weights, n, k, packed, importance)
               ? Status::kOk
               : Status::kInvalidArgument;
  }
  if (format == QuantFormat::kTQ2_0) {
    // The public pack operation does not expose the optional dequantized tensor.
    // Reuse the output matrix as temporary storage after validating finiteness;
    // tq2_0_pack reads a whole block before writing the corresponding values.
    // A dedicated scratch avoids surprising callers that overlap the buffers.
    size_t elements = 0;
    if (!checked_mul(static_cast<size_t>(n), static_cast<size_t>(k),
                     &elements) ||
        elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
      return Status::kInvalidShape;
    }
    auto scratch = std::make_unique<float[]>(elements);
    return tq2_0_pack(weights, static_cast<std::uint8_t*>(packed), scratch.get(),
                      n, k);
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
  } else if (format == QuantFormat::kQ4_0) {
    quant::q4_0_unpack_ref(static_cast<const quant::BlockQ4_0*>(packed), n, k,
                           weights);
  } else {
    quant::gguf_unpack_ref(format, packed, n, k, weights);
  }
  return Status::kOk;
}

Status quant_activation_packed_size(QuantActivationFormat format, long long n,
                                    long long k, size_t* size) {
  const Status status = validate_activation(format, n, k);
  if (status != Status::kOk) return status;
  if (size == nullptr) return Status::kInvalidArgument;
  return activation_packed_size(format, n, k, size);
}

Status quant_activation_pack(QuantActivationFormat format, const float* input,
                             long long n, long long k, void* packed) {
  const Status status = validate_activation(format, n, k);
  if (status != Status::kOk) return status;
  size_t ignored = 0;
  if (activation_packed_size(format, n, k, &ignored) != Status::kOk) {
    return Status::kInvalidShape;
  }
  if (input == nullptr || packed == nullptr) return Status::kInvalidArgument;
  return quant::activation_pack_ref(format, input, n, k, packed)
             ? Status::kOk
             : Status::kInvalidArgument;
}

Status quant_activation_unpack(QuantActivationFormat format,
                               const void* packed, long long n, long long k,
                               float* output) {
  const Status status = validate_activation(format, n, k);
  if (status != Status::kOk) return status;
  size_t ignored = 0;
  if (activation_packed_size(format, n, k, &ignored) != Status::kOk) {
    return Status::kInvalidShape;
  }
  if (packed == nullptr || output == nullptr) return Status::kInvalidArgument;
  quant::activation_unpack_ref(format, packed, n, k, output);
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
    resolve_gguf().fn(format, packed, x, y, n, k);
  }
  return Status::kOk;
}

Status qgemv_quantized_activation(QuantFormat weight_format,
                                  const void* packed_weights,
                                  QuantActivationFormat activation_format,
                                  const void* packed_activation, float* y,
                                  long long n, long long k) {
  const Status weight_status = validate(weight_format, n, k);
  if (weight_status != Status::kOk) return weight_status;
  const Status activation_status = validate_activation(activation_format, 1, k);
  if (activation_status != Status::kOk) return activation_status;
  if (packed_weights == nullptr || packed_activation == nullptr || y == nullptr) {
    return Status::kInvalidArgument;
  }
  try {
    thread_local std::vector<float> activation;
    activation.resize(static_cast<size_t>(k));
    quant::activation_unpack_ref(activation_format, packed_activation, 1, k,
                                 activation.data());
    return qgemv(weight_format, packed_weights, activation.data(), y, n, k);
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }
}

const char* qgemv_variant(QuantFormat format) {
  if (format == QuantFormat::kQ8_0) {
    return resolve_q8_0().name;
  }
  long long ignored_size = 0;
  std::size_t ignored_bytes = 0;
  return quant::gguf_format_info(format, &ignored_size, &ignored_bytes)
             ? resolve_gguf().name
             : "unsupported";
}

}  // namespace quixicore_cpu
