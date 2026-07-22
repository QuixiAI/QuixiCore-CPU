#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_AVX2)

#include "kernels/quantization/lowbit.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float horizontal_sum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                          _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  const __m128 shuffled = _mm_shuffle_ps(sum, sum, 1);
  return _mm_cvtss_f32(_mm_add_ss(sum, shuffled));
}

float dot_int4(const std::uint8_t* weights, const float* x, long long first,
               long long last) {
  const __m128i mask = _mm_set1_epi8(15);
  const __m256i offset = _mm256_set1_epi32(8);
  __m256 acc = _mm256_setzero_ps();
  long long input = first;
  if ((input & 1) == 0) {
    for (; input + 16 <= last; input += 16) {
      const __m128i bytes = _mm_loadl_epi64(
          reinterpret_cast<const __m128i*>(weights + (input >> 1)));
      const __m128i low = _mm_and_si128(bytes, mask);
      const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
      const __m128i codes = _mm_unpacklo_epi8(low, high);
      const __m256 weight0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
          _mm256_cvtepu8_epi32(codes), offset));
      const __m256 weight1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
          _mm256_cvtepu8_epi32(_mm_srli_si128(codes, 8)), offset));
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + input), weight0, acc);
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + input + 8), weight1, acc);
    }
  }
  float sum = horizontal_sum(acc);
  for (; input < last; ++input) {
    sum += x[input] *
           float(int((weights[input >> 1] >> (4 * (input & 1))) & 15) - 8);
  }
  return sum;
}

float dot_int2(const std::uint8_t* weights, const float* x, long long first,
               long long last) {
  const __m128i mask = _mm_set1_epi8(3);
  const __m256i offset = _mm256_set1_epi32(2);
  __m256 acc = _mm256_setzero_ps();
  long long input = first;
  if ((input & 3) == 0) {
    for (; input + 16 <= last; input += 16) {
      std::uint32_t word = 0;
      std::memcpy(&word, weights + (input >> 2), sizeof(word));
      const __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(word));
      const __m128i p0 = _mm_and_si128(bytes, mask);
      const __m128i p1 = _mm_and_si128(_mm_srli_epi16(bytes, 2), mask);
      const __m128i p2 = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
      const __m128i p3 = _mm_and_si128(_mm_srli_epi16(bytes, 6), mask);
      const __m128i low = _mm_unpacklo_epi8(p0, p1);
      const __m128i high = _mm_unpacklo_epi8(p2, p3);
      const __m128i codes = _mm_unpacklo_epi16(low, high);
      const __m256 weight0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
          _mm256_cvtepu8_epi32(codes), offset));
      const __m256 weight1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
          _mm256_cvtepu8_epi32(_mm_srli_si128(codes, 8)), offset));
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + input), weight0, acc);
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + input + 8), weight1, acc);
    }
  }
  float sum = horizontal_sum(acc);
  for (; input < last; ++input) {
    sum += x[input] *
           float(int((weights[input >> 2] >> (2 * (input & 3))) & 3) - 2);
  }
  return sum;
}

float dot_int3(const std::uint8_t* group, const float* x, long long count) {
  const std::uint8_t* low = group;
  const std::uint8_t* high = group + 16;
  float sum = 0.0f;
  for (long long input = 0; input < count; ++input) {
    const unsigned code =
        ((low[input >> 2] >> (2 * (input & 3))) & 3u) |
        (((high[input >> 3] >> (input & 7)) & 1u) << 2);
    sum += x[input] * float(static_cast<int>(code) - 4);
  }
  return sum;
}

}  // namespace

void lowbit_gemm_avx2(LowBitFormat format, const std::uint8_t* packed,
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
            partial = dot_int3(weights + group * kInt3GroupBytes,
                               input + first, last - first);
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
