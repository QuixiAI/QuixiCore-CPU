#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    defined(QUIXICORE_CPU_ISA_I8MM)

#include "kernels/quantization/int8_gemm.h"

#include <arm_neon.h>
#include <vector>

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

void store_tile(const std::int8_t* weights, const std::int8_t* x,
                const float* weight_scale, const float* activation_scale,
                const std::int32_t* weight_row_sum,
                const int* activation_zero_point, float* y, long long row,
                long long output, long long m, long long n, long long k,
                bool asymmetric) {
  const std::int8_t* weight0 = weights + output * k;
  const std::int8_t* weight1 = weights + (output + 1) * k;
  const std::int8_t* input0 = x + row * k;
  const std::int8_t* input1 = x + (row + 1) * k;
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  long long inner = 0;
  for (; inner + 64 <= k; inner += 64) {
    acc0 = tile16(acc0, vld1q_s8(weight0 + inner),
                  vld1q_s8(weight1 + inner), vld1q_s8(input0 + inner),
                  vld1q_s8(input1 + inner));
    acc1 = tile16(acc1, vld1q_s8(weight0 + inner + 16),
                  vld1q_s8(weight1 + inner + 16),
                  vld1q_s8(input0 + inner + 16),
                  vld1q_s8(input1 + inner + 16));
    acc2 = tile16(acc2, vld1q_s8(weight0 + inner + 32),
                  vld1q_s8(weight1 + inner + 32),
                  vld1q_s8(input0 + inner + 32),
                  vld1q_s8(input1 + inner + 32));
    acc3 = tile16(acc3, vld1q_s8(weight0 + inner + 48),
                  vld1q_s8(weight1 + inner + 48),
                  vld1q_s8(input0 + inner + 48),
                  vld1q_s8(input1 + inner + 48));
  }
  for (; inner + 16 <= k; inner += 16) {
    acc0 = tile16(acc0, vld1q_s8(weight0 + inner),
                  vld1q_s8(weight1 + inner), vld1q_s8(input0 + inner),
                  vld1q_s8(input1 + inner));
  }
  const int32x4_t acc =
      vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
  std::int64_t dot00 = vgetq_lane_s32(acc, 0);
  std::int64_t dot01 = vgetq_lane_s32(acc, 1);
  std::int64_t dot10 = vgetq_lane_s32(acc, 2);
  std::int64_t dot11 = vgetq_lane_s32(acc, 3);
  for (; inner < k; ++inner) {
    const int a = weight0[inner];
    const int b = weight1[inner];
    const int u = input0[inner];
    const int v = input1[inner];
    dot00 += a * u;
    dot01 += a * v;
    dot10 += b * u;
    dot11 += b * v;
  }
  if (asymmetric) {
    dot00 -= static_cast<std::int64_t>(activation_zero_point[row]) *
             weight_row_sum[output];
    dot01 -= static_cast<std::int64_t>(activation_zero_point[row + 1]) *
             weight_row_sum[output];
    dot10 -= static_cast<std::int64_t>(activation_zero_point[row]) *
             weight_row_sum[output + 1];
    dot11 -= static_cast<std::int64_t>(activation_zero_point[row + 1]) *
             weight_row_sum[output + 1];
  }
  y[row * n + output] = static_cast<float>(dot00) * weight_scale[output] *
                        activation_scale[row];
  y[(row + 1) * n + output] =
      static_cast<float>(dot01) * weight_scale[output] *
      activation_scale[row + 1];
  y[row * n + output + 1] =
      static_cast<float>(dot10) * weight_scale[output + 1] *
      activation_scale[row];
  y[(row + 1) * n + output + 1] =
      static_cast<float>(dot11) * weight_scale[output + 1] *
      activation_scale[row + 1];
  (void)m;
}

}  // namespace

void int8_gemm_i8mm_kernel(
    const std::int8_t* weights, const std::int8_t* x,
    const float* weight_scale, const float* activation_scale,
    const std::int32_t* weight_row_sum, const int* activation_zero_point,
    float* y, long long m, long long n, long long k, bool asymmetric) {
  const long long output_pairs = n / 2;
  threading::parallel_ranges(output_pairs, 1,
                             [&](long long begin, long long end, int) {
    for (long long pair = begin; pair < end; ++pair) {
      const long long output = 2 * pair;
      for (long long row = 0; row + 1 < m; row += 2) {
        store_tile(weights, x, weight_scale, activation_scale, weight_row_sum,
                   activation_zero_point, y, row, output, m, n, k,
                   asymmetric);
      }
      if (m & 1) {
        const long long row = m - 1;
        for (int lane = 0; lane < 2; ++lane) {
          std::int64_t dot = 0;
          const std::int8_t* weight = weights + (output + lane) * k;
          const std::int8_t* input = x + row * k;
          long long inner = 0;
          int32x4_t acc = vdupq_n_s32(0);
          for (; inner + 16 <= k; inner += 16) {
            acc = vdotq_s32(acc, vld1q_s8(weight + inner),
                            vld1q_s8(input + inner));
          }
          dot = vaddvq_s32(acc);
          for (; inner < k; ++inner) dot += weight[inner] * input[inner];
          if (asymmetric) {
            dot -= static_cast<std::int64_t>(activation_zero_point[row]) *
                   weight_row_sum[output + lane];
          }
          y[row * n + output + lane] =
              static_cast<float>(dot) * weight_scale[output + lane] *
              activation_scale[row];
        }
      }
    }
  });
  if (n & 1) {
    // Reuse the DotProd driver for the odd output column.
    std::vector<float> column(static_cast<std::size_t>(m));
    int8_gemm_dotprod_kernel(
        weights + (n - 1) * k, x, weight_scale + n - 1, activation_scale,
        asymmetric ? weight_row_sum + n - 1 : nullptr,
        activation_zero_point, column.data(), m, 1, k, asymmetric);
    for (long long row = 0; row < m; ++row) {
      y[row * n + n - 1] = column[static_cast<std::size_t>(row)];
    }
  }
}

}  // namespace quixicore_cpu::quant

#endif
