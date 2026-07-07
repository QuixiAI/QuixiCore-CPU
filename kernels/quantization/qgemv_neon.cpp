// NEON f32-activation variant of q8_0 GEMV — the QuixiCore contract path:
// out = dequantize(wq) @ x, activations stay full precision, accumulation
// in f32. Weights are widened int8 -> f32 in registers and FMA'd against
// the f32 activations; the per-block scale is applied once per block.
//
// Advanced SIMD is architecturally baseline on aarch64, so this file needs
// no special build flags — only the architecture guard.

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

// Free function with by-value arguments so the hot loops run on true
// locals; see the codegen note in src/threading/thread_pool.h.
void gemv_rows_neon(const BlockQ8_0* packed, const float* x, float* y,
                    long long blocks, long long i0, long long i1) {
  for (long long i = i0; i < i1; ++i) {
    const BlockQ8_0* row = packed + i * blocks;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (long long b = 0; b < blocks; ++b) {
      const BlockQ8_0& blk = row[b];
      const float* xb = x + b * kQ8_0BlockSize;
      const float d = fp16_to_fp32(blk.d);

      const int8x16_t q0 = vld1q_s8(blk.qs);
      const int8x16_t q1 = vld1q_s8(blk.qs + 16);
      const int16x8_t w0 = vmovl_s8(vget_low_s8(q0));
      const int16x8_t w1 = vmovl_s8(vget_high_s8(q0));
      const int16x8_t w2 = vmovl_s8(vget_low_s8(q1));
      const int16x8_t w3 = vmovl_s8(vget_high_s8(q1));

      // s = sum(q_j * x_j) over the block, two independent f32 chains.
      float32x4_t s0 = vdupq_n_f32(0.0f);
      float32x4_t s1 = vdupq_n_f32(0.0f);
      s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))),
                     vld1q_f32(xb));
      s1 = vfmaq_f32(s1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))),
                     vld1q_f32(xb + 4));
      s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))),
                     vld1q_f32(xb + 8));
      s1 = vfmaq_f32(s1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))),
                     vld1q_f32(xb + 12));
      s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2))),
                     vld1q_f32(xb + 16));
      s1 = vfmaq_f32(s1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2))),
                     vld1q_f32(xb + 20));
      s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3))),
                     vld1q_f32(xb + 24));
      s1 = vfmaq_f32(s1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3))),
                     vld1q_f32(xb + 28));

      // acc += d * s, alternating accumulators to break the chain.
      acc0 = vfmaq_n_f32(acc0, s0, d);
      acc1 = vfmaq_n_f32(acc1, s1, d);
    }
    y[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
  }
}

}  // namespace

void q8_0_gemv_neon(const BlockQ8_0* packed, const float* x, float* y,
                    long long n, long long k) {
  const long long blocks = k / kQ8_0BlockSize;
  threading::parallel_ranges(n, 32, [&](long long i0, long long i1, int) {
    gemv_rows_neon(packed, x, y, blocks, i0, i1);
  });
}

}  // namespace quixicore_cpu::quant

#endif  // aarch64
