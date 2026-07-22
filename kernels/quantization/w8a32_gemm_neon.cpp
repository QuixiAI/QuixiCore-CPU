#if defined(__aarch64__) || defined(_M_ARM64)

#include "kernels/quantization/w8a32_gemm.h"

#include <arm_neon.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void w8a32_gemm_neon(const std::int8_t* weights, const float* weight_scales,
                     const float* x, float* y, long long m, long long n,
                     long long k) {
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::int8_t* weight_row = weights + output * k;
      const float scale = weight_scales[output];
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        long long column = 0;
        for (; column + 8 <= k; column += 8) {
          const int16x8_t w16 = vmovl_s8(vld1_s8(weight_row + column));
          acc0 = vfmaq_f32(
              acc0, vld1q_f32(input + column),
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
          acc1 = vfmaq_f32(
              acc1, vld1q_f32(input + column + 4),
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; column < k; ++column) {
          sum += input[column] * static_cast<float>(weight_row[column]);
        }
        y[row * n + output] = sum * scale;
      }
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
