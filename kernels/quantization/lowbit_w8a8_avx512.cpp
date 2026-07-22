#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_AVX512_VNNI)

#include "kernels/quantization/lowbit.h"

#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

std::int32_t dot_int4_vnni(const std::uint8_t* weights,
                           const std::int8_t* x, long long k) {
  const __m256i mask = _mm256_set1_epi8(15);
  const __m512i offset = _mm512_set1_epi8(8);
  const __m512i input_order = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
  __m512i acc = _mm512_setzero_si512();
  long long input = 0;
  for (; input + 64 <= k; input += 64) {
    const __m256i bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(weights + (input >> 1)));
    const __m256i low = _mm256_and_si256(bytes, mask);
    const __m256i high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
    const __m256i codes0 = _mm256_unpacklo_epi8(low, high);
    const __m256i codes1 = _mm256_unpackhi_epi8(low, high);
    const __m512i codes = _mm512_sub_epi8(
        _mm512_inserti64x4(_mm512_castsi256_si512(codes0), codes1, 1),
        offset);
    const __m512i input_codes = _mm512_permutexvar_epi64(
        input_order, _mm512_loadu_si512(x + input));
    const __mmask64 negative = _mm512_movepi8_mask(codes);
    const __m512i signed_input = _mm512_mask_sub_epi8(
        input_codes, negative, _mm512_setzero_si512(), input_codes);
    acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(codes), signed_input);
  }
  std::int32_t sum = _mm512_reduce_add_epi32(acc);
  for (; input < k; ++input) {
    sum += (int((weights[input >> 1] >> (4 * (input & 1))) & 15) - 8) *
           int(x[input]);
  }
  return sum;
}

}  // namespace

void lowbit_w8a8_avx512_vnni(
    const std::uint8_t* packed, const float* weight_scales,
    const std::int8_t* x, const float* activation_scales, float* y,
    long long m, long long n, long long k) {
  const long long row_bytes = (k + 1) / 2;
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      const std::int32_t dot = dot_int4_vnni(
          packed + output * row_bytes, x + row * k, k);
      y[item] = float(dot) * weight_scales[output] * activation_scales[row];
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
