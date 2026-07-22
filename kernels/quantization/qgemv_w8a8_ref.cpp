// Portable reference for the activation-quantizing q4_0 GEMV (qgemv_w8a8).
//
// The integer dot is exact; quantizing the activations to int8 blocks adds
// bounded error well inside the umbrella `quantized` tolerance. This scalar
// path is the correctness anchor; SIMD variants (NEON SDOT, AVX2) are added
// separately and validated against it.

#include "kernels/quantization/qgemv_w8a8.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/common/fp16.h"

namespace quixicore_cpu::quant {
namespace {

inline float block_scale(uint16_t d) {
#if defined(_MSC_VER) && !defined(__clang__)
  return fp16_to_fp32(d);  // MSVC has no __fp16 type
#else
  __fp16 h;
  std::memcpy(&h, &d, sizeof h);
  return static_cast<float>(h);
#endif
}

inline uint16_t fp16_of(float f) {
#if defined(_MSC_VER) && !defined(__clang__)
  return fp32_to_fp16(f);
#else
  __fp16 h = static_cast<__fp16>(f);
  uint16_t d;
  std::memcpy(&d, &h, sizeof d);
  return d;
#endif
}

}  // namespace

bool q4_0_pack_ref(const float* weights, long long n, long long k,
                   BlockQ4_0* packed) {
  const long long blocks = k / kQ4_0BlockSize;
  for (long long r = 0; r < n; ++r) {
    for (long long b = 0; b < blocks; ++b) {
      const float* wb = weights + r * k + b * kQ4_0BlockSize;
      float amax = 0.0f;
      float vmax = 0.0f;  // signed value with the largest magnitude
      for (long long j = 0; j < kQ4_0BlockSize; ++j) {
        const float v = wb[j];
        if (!std::isfinite(v)) {
          return false;  // no undefined float->int conversion on NaN/inf
        }
        if (std::fabs(v) > amax) {
          amax = std::fabs(v);
          vmax = v;
        }
      }
      const float d = vmax / -8.0f;
      const float id = d != 0.0f ? 1.0f / d : 0.0f;
      BlockQ4_0& blk = packed[r * blocks + b];
      blk.d = fp16_of(d);
      for (int j = 0; j < 16; ++j) {
        int q0 = static_cast<int>(std::lround(wb[j] * id)) + 8;
        int q1 = static_cast<int>(std::lround(wb[j + 16] * id)) + 8;
        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
        blk.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
      }
    }
  }
  return true;
}

void q4_0_unpack_ref(const BlockQ4_0* packed, long long n, long long k,
                     float* weights) {
  const long long blocks = k / kQ4_0BlockSize;
  for (long long r = 0; r < n; ++r) {
    for (long long b = 0; b < blocks; ++b) {
      const BlockQ4_0& blk = packed[r * blocks + b];
      const float d = block_scale(blk.d);
      float* wb = weights + r * k + b * kQ4_0BlockSize;
      for (int j = 0; j < 16; ++j) {
        wb[j] = static_cast<float>((blk.qs[j] & 0x0F) - 8) * d;
        wb[j + 16] = static_cast<float>((blk.qs[j] >> 4) - 8) * d;
      }
    }
  }
}

void q4_0_gemv_w8a8_ref(const BlockQ4_0* packed, const float* x, float* y,
                        long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;

  // Quantize the activations to per-block int8 once (amortized over all rows).
  std::vector<int8_t> qx(static_cast<size_t>(k));
  std::vector<float> dx(static_cast<size_t>(blocks));
  for (long long b = 0; b < blocks; ++b) {
    const float* xb = x + b * kQ4_0BlockSize;
    float amax = 0.0f;
    for (long long j = 0; j < kQ4_0BlockSize; ++j) {
      amax = std::fmax(amax, std::fabs(xb[j]));
    }
    const float scale = amax / 127.0f;
    const float id = scale != 0.0f ? 1.0f / scale : 0.0f;
    dx[static_cast<size_t>(b)] = scale;
    for (long long j = 0; j < kQ4_0BlockSize; ++j) {
      const float v = std::nearbyint(xb[j] * id);
      qx[static_cast<size_t>(b * kQ4_0BlockSize + j)] =
          static_cast<int8_t>(v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v));
    }
  }

  for (long long i = 0; i < n; ++i) {
    const BlockQ4_0* row = packed + i * blocks;
    float sumf = 0.0f;
    for (long long b = 0; b < blocks; ++b) {
      const uint8_t* qs = row[b].qs;
      const int8_t* qxb = qx.data() + b * kQ4_0BlockSize;
      int sumi = 0;
      for (int j = 0; j < 16; ++j) {
        const int v0 = static_cast<int>(qs[j] & 0x0F) - 8;
        const int v1 = static_cast<int>(qs[j] >> 4) - 8;
        sumi += v0 * static_cast<int>(qxb[j]);
        sumi += v1 * static_cast<int>(qxb[j + 16]);
      }
      sumf += static_cast<float>(sumi) * block_scale(row[b].d) *
              dx[static_cast<size_t>(b)];
    }
    y[i] = sumf;
  }
}

}  // namespace quixicore_cpu::quant
