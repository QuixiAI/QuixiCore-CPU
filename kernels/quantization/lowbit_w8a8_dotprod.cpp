#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    defined(QUIXICORE_CPU_ISA_DOTPROD)

#include "kernels/quantization/lowbit.h"

#include <arm_neon.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

std::int32_t dot_int4_int8(const std::uint8_t* weights,
                           const std::int8_t* x, long long k) {
  const uint8x16_t mask = vdupq_n_u8(15);
  const int8x16_t offset = vdupq_n_s8(8);
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  long long input = 0;
  for (; input + 64 <= k; input += 64) {
    const uint8x16_t bytes0 = vld1q_u8(weights + (input >> 1));
    const uint8x16_t bytes1 = vld1q_u8(weights + (input >> 1) + 16);
    const uint8x16x2_t codes0 =
        vzipq_u8(vandq_u8(bytes0, mask), vshrq_n_u8(bytes0, 4));
    const uint8x16x2_t codes1 =
        vzipq_u8(vandq_u8(bytes1, mask), vshrq_n_u8(bytes1, 4));
    acc0 = vdotq_s32(acc0,
                     vsubq_s8(vreinterpretq_s8_u8(codes0.val[0]), offset),
                     vld1q_s8(x + input));
    acc1 = vdotq_s32(acc1,
                     vsubq_s8(vreinterpretq_s8_u8(codes0.val[1]), offset),
                     vld1q_s8(x + input + 16));
    acc2 = vdotq_s32(acc2,
                     vsubq_s8(vreinterpretq_s8_u8(codes1.val[0]), offset),
                     vld1q_s8(x + input + 32));
    acc3 = vdotq_s32(acc3,
                     vsubq_s8(vreinterpretq_s8_u8(codes1.val[1]), offset),
                     vld1q_s8(x + input + 48));
  }
  int32x4_t acc =
      vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
  for (; input + 32 <= k; input += 32) {
    const uint8x16_t bytes = vld1q_u8(weights + (input >> 1));
    const uint8x16x2_t codes =
        vzipq_u8(vandq_u8(bytes, mask), vshrq_n_u8(bytes, 4));
    acc = vdotq_s32(acc,
                    vsubq_s8(vreinterpretq_s8_u8(codes.val[0]), offset),
                    vld1q_s8(x + input));
    acc = vdotq_s32(acc,
                    vsubq_s8(vreinterpretq_s8_u8(codes.val[1]), offset),
                    vld1q_s8(x + input + 16));
  }
  std::int32_t sum = vaddvq_s32(acc);
  for (; input + 1 < k; input += 2) {
    const std::uint8_t byte = weights[input >> 1];
    sum += (int(byte & 15) - 8) * int(x[input]);
    sum += (int(byte >> 4) - 8) * int(x[input + 1]);
  }
  if (input < k) {
    sum += (int(weights[input >> 1] & 15) - 8) * int(x[input]);
  }
  return sum;
}

}  // namespace

void lowbit_w8a8_dotprod(const std::uint8_t* packed,
                         const float* weight_scales, const std::int8_t* x,
                         const float* activation_scales, float* y,
                         long long m, long long n, long long k) {
  const long long row_bytes = (k + 1) / 2;
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      const std::int32_t dot = dot_int4_int8(
          packed + output * row_bytes, x + row * k, k);
      y[item] = static_cast<float>(dot) * weight_scales[output] *
                activation_scales[row];
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
