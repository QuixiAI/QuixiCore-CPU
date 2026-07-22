#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_AVX512)

#include "kernels/quantization/lowbit.h"

#include <algorithm>
#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void lowbit_gemm_avx512(LowBitFormat format, const std::uint8_t* packed,
                        const float* scales, const float* x, float* y,
                        long long m, long long n, long long k,
                        long long group_size) {
  if (format != LowBitFormat::kInt4Row &&
      format != LowBitFormat::kInt4Group) {
    lowbit_gemm_ref(format, packed, scales, x, y, m, n, k, group_size);
    return;
  }
  const long long groups = format == LowBitFormat::kInt4Group
                               ? (k + group_size - 1) / group_size
                               : 1;
  const long long scale_group =
      format == LowBitFormat::kInt4Group ? group_size : k;
  const long long row_bytes = (k + 1) / 2;
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
          long long inner = first;
          __m512 acc0 = _mm512_setzero_ps();
          __m512 acc1 = _mm512_setzero_ps();
          const __m128i mask = _mm_set1_epi8(15);
          const __m512i offset = _mm512_set1_epi32(8);
          if ((inner & 1) == 0) {
            for (; inner + 32 <= last; inner += 32) {
              const __m128i bytes = _mm_loadu_si128(
                  reinterpret_cast<const __m128i*>(weights + (inner >> 1)));
              const __m128i low = _mm_and_si128(bytes, mask);
              const __m128i high =
                  _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
              const __m128i codes0 = _mm_unpacklo_epi8(low, high);
              const __m128i codes1 = _mm_unpackhi_epi8(low, high);
              const __m512 weight0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                  _mm512_cvtepu8_epi32(codes0), offset));
              const __m512 weight1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                  _mm512_cvtepu8_epi32(codes1), offset));
              acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(input + inner),
                                     weight0, acc0);
              acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(input + inner + 16),
                                     weight1, acc1);
            }
          }
          float partial = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
          for (; inner < last; ++inner) {
            partial += input[inner] *
                       float(int((weights[inner >> 1] >>
                                 (4 * (inner & 1))) & 15) - 8);
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
