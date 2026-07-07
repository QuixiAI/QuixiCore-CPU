// NEON RMSNorm variant. Advanced SIMD is architecturally baseline on
// aarch64, so this file needs no special build flags — only the
// architecture guard for multi-arch compilations.
//
// Sum of squares uses four f32 vector accumulators (pairwise-style
// summation, tighter error than a sequential scalar f32 chain); the scale
// pass is vectorized multiply. Deterministic.

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>

#include "kernels/norms/rms_norm.h"

namespace quixicore_cpu::norms {

void rms_norm_neon(const float* x, const float* weight, float* y,
                   long long rows, long long hidden, float eps) {
  for (long long r = 0; r < rows; ++r) {
    const float* xr = x + r * hidden;
    float* yr = y + r * hidden;

    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    long long j = 0;
    for (; j + 15 < hidden; j += 16) {
      const float32x4_t v0 = vld1q_f32(xr + j);
      const float32x4_t v1 = vld1q_f32(xr + j + 4);
      const float32x4_t v2 = vld1q_f32(xr + j + 8);
      const float32x4_t v3 = vld1q_f32(xr + j + 12);
      s0 = vfmaq_f32(s0, v0, v0);
      s1 = vfmaq_f32(s1, v1, v1);
      s2 = vfmaq_f32(s2, v2, v2);
      s3 = vfmaq_f32(s3, v3, v3);
    }
    float sumsq = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; j < hidden; ++j) {
      sumsq += xr[j] * xr[j];
    }

    const float scale =
        1.0f / std::sqrt(sumsq / static_cast<float>(hidden) + eps);
    const float32x4_t vscale = vdupq_n_f32(scale);
    j = 0;
    for (; j + 3 < hidden; j += 4) {
      const float32x4_t vx = vld1q_f32(xr + j);
      const float32x4_t vw = vld1q_f32(weight + j);
      vst1q_f32(yr + j, vmulq_f32(vmulq_f32(vx, vw), vscale));
    }
    for (; j < hidden; ++j) {
      yr[j] = xr[j] * weight[j] * scale;
    }
  }
}

}  // namespace quixicore_cpu::norms

#endif  // aarch64
