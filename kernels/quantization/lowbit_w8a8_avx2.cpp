#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_AVX2)

#include "kernels/quantization/lowbit.h"

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

std::int32_t dot_int4_avx2(const std::uint8_t* weights,
                           const std::int8_t* x, long long k) {
  const __m128i mask = _mm_set1_epi8(15);
  const __m256i offset = _mm256_set1_epi8(8);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i acc = _mm256_setzero_si256();
  long long input = 0;
  for (; input + 32 <= k; input += 32) {
    const __m128i bytes = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weights + (input >> 1)));
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    const __m128i codes0 = _mm_unpacklo_epi8(low, high);
    const __m128i codes1 = _mm_unpackhi_epi8(low, high);
    const __m256i codes =
        _mm256_sub_epi8(_mm256_set_m128i(codes1, codes0), offset);
    const __m256i input_codes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x + input));
    const __m256i pairs = _mm256_maddubs_epi16(
        _mm256_sign_epi8(codes, codes),
        _mm256_sign_epi8(input_codes, codes));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones));
  }
  std::int32_t sum = horizontal_sum(acc);
  for (; input < k; ++input) {
    sum += (int((weights[input >> 1] >> (4 * (input & 1))) & 15) - 8) *
           int(x[input]);
  }
  return sum;
}

}  // namespace

void lowbit_w8a8_avx2(const std::uint8_t* packed,
                      const float* weight_scales, const std::int8_t* x,
                      const float* activation_scales, float* y,
                      long long m, long long n, long long k) {
  const long long row_bytes = (k + 1) / 2;
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      const std::int32_t dot = dot_int4_avx2(
          packed + output * row_bytes, x + row * k, k);
      y[item] = float(dot) * weight_scales[output] * activation_scales[row];
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
