#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    defined(QUIXICORE_CPU_ISA_I8MM)

#include "kernels/quantization/lowbit.h"

#include <arm_neon.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

int32x4_t tile16(int32x4_t acc, int8x16_t weight0, int8x16_t weight1,
                 int8x16_t input0, int8x16_t input1) {
  acc = vmmlaq_s32(acc,
                   vcombine_s8(vget_low_s8(weight0), vget_low_s8(weight1)),
                   vcombine_s8(vget_low_s8(input0), vget_low_s8(input1)));
  return vmmlaq_s32(
      acc, vcombine_s8(vget_high_s8(weight0), vget_high_s8(weight1)),
      vcombine_s8(vget_high_s8(input0), vget_high_s8(input1)));
}

std::int32_t scalar_dot(const std::uint8_t* weights, const std::int8_t* x,
                        long long k) {
  std::int32_t dot = 0;
  for (long long input = 0; input < k; ++input) {
    dot += (int((weights[input >> 1] >> (4 * (input & 1))) & 15) - 8) *
           int(x[input]);
  }
  return dot;
}

}  // namespace

void lowbit_w8a8_i8mm(const std::uint8_t* packed,
                      const float* weight_scales, const std::int8_t* x,
                      const float* activation_scales, float* y,
                      long long m, long long n, long long k) {
  const long long row_bytes = (k + 1) / 2;
  const uint8x16_t mask = vdupq_n_u8(15);
  const int8x16_t offset = vdupq_n_s8(8);
  threading::parallel_ranges(n / 2, 1,
                             [&](long long begin, long long end, int) {
    for (long long pair = begin; pair < end; ++pair) {
      const long long output = 2 * pair;
      const std::uint8_t* weight0 = packed + output * row_bytes;
      const std::uint8_t* weight1 = weight0 + row_bytes;
      for (long long row = 0; row + 1 < m; row += 2) {
        const std::int8_t* input0 = x + row * k;
        const std::int8_t* input1 = input0 + k;
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        long long inner = 0;
        for (; inner + 64 <= k; inner += 64) {
          const uint8x16_t bytes0a =
              vld1q_u8(weight0 + (inner >> 1));
          const uint8x16_t bytes1a =
              vld1q_u8(weight1 + (inner >> 1));
          const uint8x16_t bytes0b =
              vld1q_u8(weight0 + (inner >> 1) + 16);
          const uint8x16_t bytes1b =
              vld1q_u8(weight1 + (inner >> 1) + 16);
          const uint8x16x2_t codes0a = vzipq_u8(
              vandq_u8(bytes0a, mask), vshrq_n_u8(bytes0a, 4));
          const uint8x16x2_t codes1a = vzipq_u8(
              vandq_u8(bytes1a, mask), vshrq_n_u8(bytes1a, 4));
          const uint8x16x2_t codes0b = vzipq_u8(
              vandq_u8(bytes0b, mask), vshrq_n_u8(bytes0b, 4));
          const uint8x16x2_t codes1b = vzipq_u8(
              vandq_u8(bytes1b, mask), vshrq_n_u8(bytes1b, 4));
          acc0 = tile16(
              acc0,
              vsubq_s8(vreinterpretq_s8_u8(codes0a.val[0]), offset),
              vsubq_s8(vreinterpretq_s8_u8(codes1a.val[0]), offset),
              vld1q_s8(input0 + inner), vld1q_s8(input1 + inner));
          acc1 = tile16(
              acc1,
              vsubq_s8(vreinterpretq_s8_u8(codes0a.val[1]), offset),
              vsubq_s8(vreinterpretq_s8_u8(codes1a.val[1]), offset),
              vld1q_s8(input0 + inner + 16),
              vld1q_s8(input1 + inner + 16));
          acc2 = tile16(
              acc2,
              vsubq_s8(vreinterpretq_s8_u8(codes0b.val[0]), offset),
              vsubq_s8(vreinterpretq_s8_u8(codes1b.val[0]), offset),
              vld1q_s8(input0 + inner + 32),
              vld1q_s8(input1 + inner + 32));
          acc3 = tile16(
              acc3,
              vsubq_s8(vreinterpretq_s8_u8(codes0b.val[1]), offset),
              vsubq_s8(vreinterpretq_s8_u8(codes1b.val[1]), offset),
              vld1q_s8(input0 + inner + 48),
              vld1q_s8(input1 + inner + 48));
        }
        for (; inner + 32 <= k; inner += 32) {
          const uint8x16_t bytes0 =
              vld1q_u8(weight0 + (inner >> 1));
          const uint8x16_t bytes1 =
              vld1q_u8(weight1 + (inner >> 1));
          const uint8x16x2_t codes0 =
              vzipq_u8(vandq_u8(bytes0, mask), vshrq_n_u8(bytes0, 4));
          const uint8x16x2_t codes1 =
              vzipq_u8(vandq_u8(bytes1, mask), vshrq_n_u8(bytes1, 4));
          acc0 = tile16(
              acc0, vsubq_s8(vreinterpretq_s8_u8(codes0.val[0]), offset),
              vsubq_s8(vreinterpretq_s8_u8(codes1.val[0]), offset),
              vld1q_s8(input0 + inner), vld1q_s8(input1 + inner));
          acc1 = tile16(
              acc1, vsubq_s8(vreinterpretq_s8_u8(codes0.val[1]), offset),
              vsubq_s8(vreinterpretq_s8_u8(codes1.val[1]), offset),
              vld1q_s8(input0 + inner + 16),
              vld1q_s8(input1 + inner + 16));
        }
        const int32x4_t acc =
            vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        std::int32_t dot00 = vgetq_lane_s32(acc, 0);
        std::int32_t dot01 = vgetq_lane_s32(acc, 1);
        std::int32_t dot10 = vgetq_lane_s32(acc, 2);
        std::int32_t dot11 = vgetq_lane_s32(acc, 3);
        for (; inner < k; ++inner) {
          const int a =
              int((weight0[inner >> 1] >> (4 * (inner & 1))) & 15) - 8;
          const int b =
              int((weight1[inner >> 1] >> (4 * (inner & 1))) & 15) - 8;
          dot00 += a * int(input0[inner]);
          dot01 += a * int(input1[inner]);
          dot10 += b * int(input0[inner]);
          dot11 += b * int(input1[inner]);
        }
        y[row * n + output] = float(dot00) * weight_scales[output] *
                              activation_scales[row];
        y[(row + 1) * n + output] =
            float(dot01) * weight_scales[output] *
            activation_scales[row + 1];
        y[row * n + output + 1] =
            float(dot10) * weight_scales[output + 1] *
            activation_scales[row];
        y[(row + 1) * n + output + 1] =
            float(dot11) * weight_scales[output + 1] *
            activation_scales[row + 1];
      }
      if (m & 1) {
        const long long row = m - 1;
        y[row * n + output] =
            float(scalar_dot(weight0, x + row * k, k)) *
            weight_scales[output] * activation_scales[row];
        y[row * n + output + 1] =
            float(scalar_dot(weight1, x + row * k, k)) *
            weight_scales[output + 1] * activation_scales[row];
      }
    }
  });
  if (n & 1) {
    const long long output = n - 1;
    const std::uint8_t* weights = packed + output * row_bytes;
    for (long long row = 0; row < m; ++row) {
      y[row * n + output] =
          float(scalar_dot(weights, x + row * k, k)) *
          weight_scales[output] * activation_scales[row];
    }
  }
}

}  // namespace quixicore_cpu::quant

#endif
