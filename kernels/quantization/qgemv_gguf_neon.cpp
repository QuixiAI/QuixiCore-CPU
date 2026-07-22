#include "kernels/quantization/gguf_ref.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float dot_i8x16_f32(int8x16_t weights, const float* input) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(weights));
  const int16x8_t hi = vmovl_s8(vget_high_s8(weights));
  float32x4_t sum = vmulq_f32(
      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(input));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                  vld1q_f32(input + 4));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                  vld1q_f32(input + 8));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                  vld1q_f32(input + 12));
  return vaddvq_f32(sum);
}

void q4_rows(const BlockQ4_0* packed, const float* x, float* y,
             long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    const uint8x16_t mask = vdupq_n_u8(15);
    const uint8x16_t offset = vdupq_n_u8(8);
    for (long long row = begin; row < end; ++row) {
      float total = 0.0f;
      const BlockQ4_0* row_weights = packed + row * blocks;
      for (long long block = 0; block < blocks; ++block) {
        const uint8x16_t codes = vld1q_u8(row_weights[block].qs);
        const int8x16_t low = vreinterpretq_s8_u8(
            vsubq_u8(vandq_u8(codes, mask), offset));
        const int8x16_t high = vreinterpretq_s8_u8(
            vsubq_u8(vshrq_n_u8(codes, 4), offset));
        const float* input = x + block * kQ4_0BlockSize;
        const float dot = dot_i8x16_f32(low, input) +
                          dot_i8x16_f32(high, input + 16);
        total += fp16_to_fp32(row_weights[block].d) * dot;
      }
      y[row] = total;
    }
  });
}

}  // namespace

void gguf_gemv_neon(QuantFormat format, const void* packed, const float* x,
                    float* y, long long n, long long k) {
  if (format == QuantFormat::kQ4_0) {
    q4_rows(static_cast<const BlockQ4_0*>(packed), x, y, n, k);
    return;
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  (void)gguf_format_info(format, &block_size, &block_bytes);
  const long long blocks_per_row = k / block_size;
  const auto* bytes = static_cast<const std::uint8_t*>(packed);
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    alignas(64) float decoded[256];
    for (long long row = begin; row < end; ++row) {
      double total = 0.0;
      for (long long block = 0; block < blocks_per_row; ++block) {
        gguf_dequant_block_ref(
            format, bytes + (row * blocks_per_row + block) * block_bytes,
            decoded);
        const float* input = x + block * block_size;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int column = 0;
        for (; column + 7 < block_size; column += 8) {
          acc0 = vfmaq_f32(acc0, vld1q_f32(decoded + column),
                           vld1q_f32(input + column));
          acc1 = vfmaq_f32(acc1, vld1q_f32(decoded + column + 4),
                           vld1q_f32(input + column + 4));
        }
        float block_total = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; column < block_size; ++column) {
          block_total += decoded[column] * input[column];
        }
        total += block_total;
      }
      y[row] = static_cast<float>(total);
    }
  });
}

}  // namespace quixicore_cpu::quant

#else

namespace quixicore_cpu::quant {
void gguf_gemv_neon(QuantFormat, const void*, const float*, float*, long long,
                    long long) {}
}  // namespace quixicore_cpu::quant

#endif
