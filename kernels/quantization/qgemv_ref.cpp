// Portable scalar reference for q8_0 GEMV. Builds with baseline arch flags
// on every platform; this is the correctness oracle and dispatch fallback
// that ISA variants are measured against.

#include "kernels/quantization/qgemv.h"

#include <cmath>

#include "kernels/common/fp16.h"

namespace quixicore_cpu::qgemv {

void q8_0_pack_ref(const float* weights, long long n, long long k,
                   BlockQ8_0* packed) {
  const long long blocks_per_row = k / kQ8_0BlockSize;
  for (long long i = 0; i < n; ++i) {
    const float* row = weights + i * k;
    BlockQ8_0* out = packed + i * blocks_per_row;
    for (long long b = 0; b < blocks_per_row; ++b) {
      const float* src = row + b * kQ8_0BlockSize;
      float amax = 0.0f;
      for (long long j = 0; j < kQ8_0BlockSize; ++j) {
        amax = std::fmax(amax, std::fabs(src[j]));
      }
      const float d = amax / 127.0f;
      const float id = d != 0.0f ? 1.0f / d : 0.0f;
      out[b].d = fp32_to_fp16(d);
      for (long long j = 0; j < kQ8_0BlockSize; ++j) {
        const float q = std::nearbyint(src[j] * id);
        out[b].qs[j] =
            static_cast<int8_t>(q < -127.0f ? -127.0f
                                            : (q > 127.0f ? 127.0f : q));
      }
    }
  }
}

void q8_0_unpack_ref(const BlockQ8_0* packed, long long n, long long k,
                     float* weights) {
  const long long blocks_per_row = k / kQ8_0BlockSize;
  for (long long i = 0; i < n; ++i) {
    const BlockQ8_0* row = packed + i * blocks_per_row;
    float* out = weights + i * k;
    for (long long b = 0; b < blocks_per_row; ++b) {
      const float d = fp16_to_fp32(row[b].d);
      for (long long j = 0; j < kQ8_0BlockSize; ++j) {
        out[b * kQ8_0BlockSize + j] = d * row[b].qs[j];
      }
    }
  }
}

void q8_0_gemv_ref(const BlockQ8_0* packed, const float* x, float* y,
                   long long n, long long k) {
  const long long blocks_per_row = k / kQ8_0BlockSize;
  for (long long i = 0; i < n; ++i) {
    const BlockQ8_0* row = packed + i * blocks_per_row;
    // Deliberately the plain loop: a manual 4-way multi-accumulator split
    // measured 2-3% slower because compilers auto-vectorize this form
    // already (rejected; see perf/optimization_status.md 2026-07-07).
    float acc = 0.0f;
    for (long long b = 0; b < blocks_per_row; ++b) {
      const float d = fp16_to_fp32(row[b].d);
      const int8_t* q = row[b].qs;
      const float* xb = x + b * kQ8_0BlockSize;
      float sum = 0.0f;
      for (long long j = 0; j < kQ8_0BlockSize; ++j) {
        sum += static_cast<float>(q[j]) * xb[j];
      }
      acc += d * sum;
    }
    y[i] = acc;
  }
}

}  // namespace quixicore_cpu::qgemv
