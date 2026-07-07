// AArch64 DotProd (SDOT) variant of q8_0 GEMV.
//
// Activations are quantized to int8 blocks once per call (d = amax/127,
// round-to-nearest), then each 32-element block is two vdotq_s32 int8x16
// dot products accumulated as float32x4 lanes with the combined
// weight*activation scale (the llama.cpp q8_0 x q8_0 structure). The int
// dot is exact; activation quantization adds bounded error well inside the
// umbrella quantized tolerance (see tests/correctness/test_quant_gemv.cpp).
//
// This file is added to the build only when the toolchain can target
// dotprod (cmake/QuixiCoreCPUFeatures.cmake) and is called only after
// runtime detection confirms the feature (src/dispatch/quant_gemv.cpp).

#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    defined(QUIXICORE_CPU_ISA_DOTPROD)

#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv.h"

namespace quixicore_cpu::qgemv {
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

// Scalar activation quantization; one pass over x per call, amortized over
// all n rows. Matches the packing semantics: d = amax/127, nearbyint.
void quantize_activations(const float* x, long long k, std::vector<int8_t>& q,
                          std::vector<float>& d) {
  const long long blocks = k / kQ8_0BlockSize;
  q.resize(static_cast<size_t>(k));
  d.resize(static_cast<size_t>(blocks));
  for (long long b = 0; b < blocks; ++b) {
    const float* xb = x + b * kQ8_0BlockSize;
    float amax = 0.0f;
    for (long long j = 0; j < kQ8_0BlockSize; ++j) {
      amax = std::fmax(amax, std::fabs(xb[j]));
    }
    const float scale = amax / 127.0f;
    const float id = scale != 0.0f ? 1.0f / scale : 0.0f;
    d[static_cast<size_t>(b)] = scale;
    for (long long j = 0; j < kQ8_0BlockSize; ++j) {
      const float v = std::nearbyint(xb[j] * id);
      q[static_cast<size_t>(b * kQ8_0BlockSize + j)] =
          static_cast<int8_t>(v < -127.0f ? -127.0f
                                          : (v > 127.0f ? 127.0f : v));
    }
  }
}

}  // namespace

void q8_0_gemv_dotprod(const BlockQ8_0* packed, const float* x, float* y,
                       long long n, long long k) {
  const long long blocks = k / kQ8_0BlockSize;

  // Grow-only per-thread scratch: no allocation on repeat calls.
  thread_local std::vector<int8_t> qx;
  thread_local std::vector<float> dx;
  quantize_activations(x, k, qx, dx);
  const int8_t* qx_data = qx.data();
  const float* dx_data = dx.data();

  for (long long i = 0; i < n; ++i) {
    const BlockQ8_0* row = packed + i * blocks;
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    long long b = 0;
    for (; b + 1 < blocks; b += 2) {
      const BlockQ8_0& blk0 = row[b];
      const BlockQ8_0& blk1 = row[b + 1];
      const int8_t* qxb = qx_data + b * kQ8_0BlockSize;

      int32x4_t p0 = vdupq_n_s32(0);
      p0 = vdotq_s32(p0, vld1q_s8(blk0.qs), vld1q_s8(qxb));
      p0 = vdotq_s32(p0, vld1q_s8(blk0.qs + 16), vld1q_s8(qxb + 16));
      int32x4_t p1 = vdupq_n_s32(0);
      p1 = vdotq_s32(p1, vld1q_s8(blk1.qs), vld1q_s8(qxb + 32));
      p1 = vdotq_s32(p1, vld1q_s8(blk1.qs + 16), vld1q_s8(qxb + 48));

      sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0),
                          block_scale(blk0.d) * dx_data[b]);
      sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1),
                          block_scale(blk1.d) * dx_data[b + 1]);
    }
    float acc = vaddvq_f32(vaddq_f32(sumv0, sumv1));
    for (; b < blocks; ++b) {  // odd trailing block
      const int8_t* qw = row[b].qs;
      const int8_t* qxb = qx_data + b * kQ8_0BlockSize;
      int32x4_t p = vdupq_n_s32(0);
      p = vdotq_s32(p, vld1q_s8(qw), vld1q_s8(qxb));
      p = vdotq_s32(p, vld1q_s8(qw + 16), vld1q_s8(qxb + 16));
      acc += block_scale(row[b].d) * dx_data[b] *
             static_cast<float>(vaddvq_s32(p));
    }
    y[i] = acc;
  }
}

}  // namespace quixicore_cpu::qgemv

#endif  // aarch64 && QUIXICORE_CPU_ISA_DOTPROD
