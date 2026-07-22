#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    defined(QUIXICORE_CPU_ISA_DOTPROD)

#include "kernels/quantization/int8_gemm.h"

#include <arm_neon.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

std::int32_t dotprod(const std::int8_t* weights, const std::int8_t* x,
                     long long k) {
  long long input = 0;
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  for (; input + 64 <= k; input += 64) {
    acc0 = vdotq_s32(acc0, vld1q_s8(weights + input),
                     vld1q_s8(x + input));
    acc1 = vdotq_s32(acc1, vld1q_s8(weights + input + 16),
                     vld1q_s8(x + input + 16));
    acc2 = vdotq_s32(acc2, vld1q_s8(weights + input + 32),
                     vld1q_s8(x + input + 32));
    acc3 = vdotq_s32(acc3, vld1q_s8(weights + input + 48),
                     vld1q_s8(x + input + 48));
  }
  int32x4_t acc =
      vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
  for (; input + 16 <= k; input += 16) {
    acc = vdotq_s32(acc, vld1q_s8(weights + input),
                    vld1q_s8(x + input));
  }
  std::int32_t sum = vaddvq_s32(acc);
  for (; input < k; ++input) {
    sum += static_cast<int>(weights[input]) * static_cast<int>(x[input]);
  }
  return sum;
}

}  // namespace

void int8_gemm_dotprod_kernel(
    const std::int8_t* weights, const std::int8_t* x,
    const float* weight_scale, const float* activation_scale,
    const std::int32_t* weight_row_sum, const int* activation_zero_point,
    float* y, long long m, long long n, long long k, bool asymmetric) {
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      std::int64_t dot = dotprod(weights + output * k, x + row * k, k);
      if (asymmetric) {
        dot -= static_cast<std::int64_t>(activation_zero_point[row]) *
               weight_row_sum[output];
      }
      y[item] = static_cast<float>(dot) * weight_scale[output] *
                activation_scale[row];
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
