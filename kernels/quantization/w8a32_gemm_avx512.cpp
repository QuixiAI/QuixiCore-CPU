#include "kernels/quantization/w8a32_gemm.h"

#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void w8a32_gemm_avx512(const std::int8_t* weights,
                       const float* weight_scales, const float* x, float* y,
                       long long m, long long n, long long k) {
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::int8_t* weight_row = weights + output * k;
      const float scale = weight_scales[output];
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        __m512 accumulator = _mm512_setzero_ps();
        long long column = 0;
        for (; column + 16 <= k; column += 16) {
          const __m128i bytes = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(weight_row + column));
          const __m512 weights_f32 =
              _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes));
          accumulator = _mm512_fmadd_ps(
              _mm512_loadu_ps(input + column), weights_f32, accumulator);
        }
        float sum = _mm512_reduce_add_ps(accumulator);
        for (; column < k; ++column) {
          sum += input[column] * static_cast<float>(weight_row[column]);
        }
        y[row * n + output] = sum * scale;
      }
    }
  });
}

}  // namespace quixicore_cpu::quant
