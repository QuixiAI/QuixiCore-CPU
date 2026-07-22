#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_AVX512_VNNI)

#include "kernels/quantization/int8_gemm.h"

#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

std::int32_t dot_vnni(const std::int8_t* weights, const std::int8_t* x,
                      long long k) {
  __m512i acc = _mm512_setzero_si512();
  long long input = 0;
  for (; input + 64 <= k; input += 64) {
    const __m512i w = _mm512_loadu_si512(weights + input);
    const __m512i a = _mm512_loadu_si512(x + input);
    const __mmask64 negative = _mm512_movepi8_mask(w);
    const __m512i signed_a = _mm512_mask_sub_epi8(
        a, negative, _mm512_setzero_si512(), a);
    acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(w), signed_a);
  }
  std::int32_t sum = _mm512_reduce_add_epi32(acc);
  for (; input < k; ++input) sum += weights[input] * x[input];
  return sum;
}

}  // namespace

void int8_gemm_avx512_vnni_kernel(
    const std::int8_t* weights, const std::int8_t* x,
    const float* weight_scale, const float* activation_scale,
    const std::int32_t* weight_row_sum, const int* activation_zero_point,
    float* y, long long m, long long n, long long k, bool asymmetric) {
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      std::int64_t dot = dot_vnni(weights + output * k, x + row * k, k);
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
