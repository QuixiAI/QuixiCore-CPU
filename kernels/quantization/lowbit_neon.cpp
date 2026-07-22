#if defined(__aarch64__) || defined(_M_ARM64)

#include "kernels/quantization/lowbit.h"

#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include <cstring>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float dot_int4(const std::uint8_t* weights, const float* x, long long first,
               long long last) {
  long long input = first;
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  const uint8x8_t mask = vdup_n_u8(15);
  const int8x8_t offset = vdup_n_s8(8);
  if ((input & 1) == 0) {
    for (; input + 16 <= last; input += 16) {
      const uint8x8_t bytes = vld1_u8(weights + (input >> 1));
      const uint8x8x2_t zipped =
          vzip_u8(vand_u8(bytes, mask), vshr_n_u8(bytes, 4));
      const int16x8_t low =
          vmovl_s8(vsub_s8(vreinterpret_s8_u8(zipped.val[0]), offset));
      const int16x8_t high =
          vmovl_s8(vsub_s8(vreinterpret_s8_u8(zipped.val[1]), offset));
      acc0 = vfmaq_f32(
          acc0, vld1q_f32(x + input),
          vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))));
      acc1 = vfmaq_f32(
          acc1, vld1q_f32(x + input + 4),
          vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))));
      acc0 = vfmaq_f32(
          acc0, vld1q_f32(x + input + 8),
          vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))));
      acc1 = vfmaq_f32(
          acc1, vld1q_f32(x + input + 12),
          vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))));
    }
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; input < last; ++input) {
    const int code =
        int((weights[input >> 1] >> (4 * (input & 1))) & 15) - 8;
    sum += x[input] * static_cast<float>(code);
  }
  return sum;
}

float dot_int2(const std::uint8_t* weights, const float* x, long long first,
               long long last) {
  long long input = first;
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  const uint8x8_t mask = vdup_n_u8(3);
  const int8x8_t offset = vdup_n_s8(2);
  if ((input & 3) == 0) {
    for (; input + 16 <= last; input += 16) {
      std::uint32_t word = 0;
      std::memcpy(&word, weights + (input >> 2), sizeof(word));
      const uint8x8_t bytes = vreinterpret_u8_u32(vdup_n_u32(word));
      const uint8x8x2_t p01 =
          vzip_u8(vand_u8(bytes, mask),
                  vand_u8(vshr_n_u8(bytes, 2), mask));
      const uint8x8x2_t p23 =
          vzip_u8(vand_u8(vshr_n_u8(bytes, 4), mask),
                  vshr_n_u8(bytes, 6));
      const uint16x4x2_t joined = vzip_u16(
          vreinterpret_u16_u8(p01.val[0]), vreinterpret_u16_u8(p23.val[0]));
      const int16x8_t low = vmovl_s8(vsub_s8(
          vreinterpret_s8_u16(joined.val[0]), offset));
      const int16x8_t high = vmovl_s8(vsub_s8(
          vreinterpret_s8_u16(joined.val[1]), offset));
      acc0 = vfmaq_f32(
          acc0, vld1q_f32(x + input),
          vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))));
      acc1 = vfmaq_f32(
          acc1, vld1q_f32(x + input + 4),
          vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))));
      acc0 = vfmaq_f32(
          acc0, vld1q_f32(x + input + 8),
          vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))));
      acc1 = vfmaq_f32(
          acc1, vld1q_f32(x + input + 12),
          vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))));
    }
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; input < last; ++input) {
    const int code =
        int((weights[input >> 2] >> (2 * (input & 3))) & 3) - 2;
    sum += x[input] * static_cast<float>(code);
  }
  return sum;
}

float dot_int3_group(const std::uint8_t* group, const float* x,
                     long long count) {
  const std::uint8_t* low_plane = group;
  const std::uint8_t* high_plane = group + 16;
  int input = 0;
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  const uint8x8_t mask = vdup_n_u8(3);
  const int8x16_t offset = vdupq_n_s8(4);
  const uint8x16_t bit_mask = {1,   2,  4,  8,  16, 32, 64, 128,
                               1,   2,  4,  8,  16, 32, 64, 128};
  const uint8x16_t four = vdupq_n_u8(4);
  for (; input + 16 <= count; input += 16) {
    std::uint32_t word = 0;
    std::memcpy(&word, low_plane + (input >> 2), sizeof(word));
    const uint8x8_t bytes = vreinterpret_u8_u32(vdup_n_u32(word));
    const uint8x8x2_t p01 =
        vzip_u8(vand_u8(bytes, mask),
                vand_u8(vshr_n_u8(bytes, 2), mask));
    const uint8x8x2_t p23 =
        vzip_u8(vand_u8(vshr_n_u8(bytes, 4), mask),
                vshr_n_u8(bytes, 6));
    const uint16x4x2_t joined = vzip_u16(
        vreinterpret_u16_u8(p01.val[0]), vreinterpret_u16_u8(p23.val[0]));
    const uint8x16_t low = vcombine_u8(
        vreinterpret_u8_u16(joined.val[0]),
        vreinterpret_u8_u16(joined.val[1]));
    const uint8x16_t high_bytes =
        vcombine_u8(vdup_n_u8(high_plane[input >> 3]),
                    vdup_n_u8(high_plane[(input >> 3) + 1]));
    const uint8x16_t high =
        vandq_u8(vtstq_u8(high_bytes, bit_mask), four);
    const int8x16_t codes =
        vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(low, high)), offset);
    const int16x8_t codes0 = vmovl_s8(vget_low_s8(codes));
    const int16x8_t codes1 = vmovl_s8(vget_high_s8(codes));
    acc0 = vfmaq_f32(
        acc0, vld1q_f32(x + input),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(codes0))));
    acc1 = vfmaq_f32(
        acc1, vld1q_f32(x + input + 4),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(codes0))));
    acc0 = vfmaq_f32(
        acc0, vld1q_f32(x + input + 8),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(codes1))));
    acc1 = vfmaq_f32(
        acc1, vld1q_f32(x + input + 12),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(codes1))));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; input < count; ++input) {
    const unsigned code =
        ((low_plane[input >> 2] >> (2 * (input & 3))) & 3u) |
        (((high_plane[input >> 3] >> (input & 7)) & 1u) << 2);
    sum += x[input] * static_cast<float>(static_cast<int>(code) - 4);
  }
  return sum;
}

}  // namespace

void lowbit_gemm_neon(LowBitFormat format, const std::uint8_t* packed,
                      const float* scales, const float* x, float* y,
                      long long m, long long n, long long k,
                      long long group_size) {
  const long long groups =
      format == LowBitFormat::kInt3Group64
          ? (k + kInt3Group - 1) / kInt3Group
          : (format == LowBitFormat::kInt4Group
                 ? (k + group_size - 1) / group_size
                 : 1);
  const long long scale_group =
      format == LowBitFormat::kInt3Group64
          ? kInt3Group
          : (format == LowBitFormat::kInt4Group ? group_size : k);
  const long long row_bytes =
      format == LowBitFormat::kInt2Row
          ? (k + 3) / 4
          : (format == LowBitFormat::kInt3Group64
                 ? groups * kInt3GroupBytes
                 : (k + 1) / 2);
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::uint8_t* weights = packed + output * row_bytes;
      const float* row_scales = scales + output * groups;
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        float sum = 0.0f;
        for (long long group = 0; group < groups; ++group) {
          const long long first = group * scale_group;
          const long long last = std::min(k, first + scale_group);
          float partial = 0.0f;
          if (format == LowBitFormat::kInt2Row) {
            partial = dot_int2(weights, input, first, last);
          } else if (format == LowBitFormat::kInt3Group64) {
            partial = dot_int3_group(
                weights + group * kInt3GroupBytes, input + first,
                last - first);
          } else {
            partial = dot_int4(weights, input, first, last);
          }
          sum += partial * row_scales[group];
        }
        y[row * n + output] = sum;
      }
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
