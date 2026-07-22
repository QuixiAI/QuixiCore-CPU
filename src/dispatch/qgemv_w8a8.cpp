// Public qgemv_w8a8 entry points — the activation-quantizing quantized GEMV
// twin of qgemv (see the reservation in include/quixicore_cpu/qgemv.h).
//
// Weights stay quantized (kQ4_0 for now; kQ8_0 to follow); activations are
// quantized to per-32-block int8 inside the kernel and the dot is integer with
// f32-accumulated combined scales. A scalar reference is the correctness
// anchor; SIMD variants resolve last-wins by CPU feature (as in qgemv).

#include "quixicore_cpu/qgemv_w8a8.h"

#include <climits>

#include "kernels/quantization/qgemv_w8a8.h"

namespace quixicore_cpu {
namespace {

Status validate(QuantFormat format, long long n, long long k) {
  if (format != QuantFormat::kQ4_0) {
    return Status::kUnsupportedFormat;
  }
  if (n <= 0 || k <= 0 || k % quant::kQ4_0BlockSize != 0) {
    return Status::kInvalidShape;
  }
  // Signed long-long offsets: reject shapes whose logical matrix or packed
  // block count cannot be represented before those products reach a hot loop.
  if (n > LLONG_MAX / k ||
      n > LLONG_MAX / (k / quant::kQ4_0BlockSize)) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

}  // namespace

Status qgemv_w8a8_packed_size(QuantFormat format, long long n, long long k,
                              long long* bytes) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  if (bytes == nullptr) {
    return Status::kInvalidArgument;
  }
  *bytes = n * (k / quant::kQ4_0BlockSize) *
           static_cast<long long>(sizeof(quant::BlockQ4_0));
  return Status::kOk;
}

Status qgemv_w8a8_pack(QuantFormat format, const float* weights, long long n,
                       long long k, void* packed) {
  const Status status = validate(format, n, k);
  if (status != Status::kOk) {
    return status;
  }
  if (weights == nullptr || packed == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!quant::q4_0_pack_ref(weights, n, k,
                            static_cast<quant::BlockQ4_0*>(packed))) {
    return Status::kInvalidArgument;
  }
  return Status::kOk;
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
  quant::q4_0_gemv_w8a8_ref(
      static_cast<const quant::BlockQ4_0*>(packed_weights), x, y, n, k);
  return Status::kOk;
}

const char* qgemv_w8a8_variant(QuantFormat format) {
  (void)format;
  return "ref";
}

}  // namespace quixicore_cpu
