#include "kernels/quantization/w8a32_gemm.h"

#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

}  // namespace

void w8a32_gemm_avx2(const std::int8_t* weights, const float* weight_scales,
                     const float* x, float* y, long long m, long long n,
                     long long k) {
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::int8_t* weight_row = weights + output * k;
      const float scale = weight_scales[output];
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        __m256 accumulator = _mm256_setzero_ps();
        long long column = 0;
        for (; column + 8 <= k; column += 8) {
          const __m128i bytes =
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weight_row + column));
          const __m256 weights_f32 =
              _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
          accumulator = _mm256_fmadd_ps(
              _mm256_loadu_ps(input + column), weights_f32, accumulator);
        }
        float sum = horizontal_sum(accumulator);
        for (; column < k; ++column) {
          sum += input[column] * static_cast<float>(weight_row[column]);
        }
        y[row * n + output] = sum * scale;
      }
    }
  });
}

}  // namespace quixicore_cpu::quant
