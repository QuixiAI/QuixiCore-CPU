#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_AVX2)

#include "kernels/quantization/int8_gemm.h"

#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

int horizontal_sum(__m256i value) {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

std::int32_t dot_avx2(const std::int8_t* weights, const std::int8_t* x,
                      long long k) {
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i acc = _mm256_setzero_si256();
  long long input = 0;
  for (; input + 32 <= k; input += 32) {
    const __m256i w =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + input));
    const __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + input));
    const __m256i pairs = _mm256_maddubs_epi16(
        _mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones));
  }
  std::int32_t sum = horizontal_sum(acc);
  for (; input < k; ++input) sum += weights[input] * x[input];
  return sum;
}

}  // namespace

void int8_gemm_avx2_kernel(
    const std::int8_t* weights, const std::int8_t* x,
    const float* weight_scale, const float* activation_scale,
    const std::int32_t* weight_row_sum, const int* activation_zero_point,
    float* y, long long m, long long n, long long k, bool asymmetric) {
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      std::int64_t dot =
          dot_avx2(weights + output * k, x + row * k, k);
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
