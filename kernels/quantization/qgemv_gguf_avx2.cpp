#include "kernels/quantization/gguf_ref.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float horizontal(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

float dot_i8x16_f32(__m128i weights, const float* input) {
  const __m256i low = _mm256_cvtepi8_epi32(weights);
  const __m256i high = _mm256_cvtepi8_epi32(_mm_srli_si128(weights, 8));
  const __m256 sum0 = _mm256_mul_ps(_mm256_cvtepi32_ps(low),
                                    _mm256_loadu_ps(input));
  const __m256 sum1 = _mm256_mul_ps(_mm256_cvtepi32_ps(high),
                                    _mm256_loadu_ps(input + 8));
  return horizontal(_mm256_add_ps(sum0, sum1));
}

void q4_rows(const BlockQ4_0* packed, const float* x, float* y,
             long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i offset = _mm_set1_epi8(8);
    for (long long row = begin; row < end; ++row) {
      float total = 0.0f;
      const BlockQ4_0* row_weights = packed + row * blocks;
      for (long long block = 0; block < blocks; ++block) {
        const __m128i codes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(row_weights[block].qs));
        const __m128i low = _mm_sub_epi8(_mm_and_si128(codes, mask), offset);
        const __m128i high =
            _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(codes, 4), mask), offset);
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

void gguf_gemv_avx2(QuantFormat format, const void* packed, const float* x,
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
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int column = 0;
        for (; column + 15 < block_size; column += 16) {
          acc0 = _mm256_fmadd_ps(_mm256_load_ps(decoded + column),
                                 _mm256_loadu_ps(input + column), acc0);
          acc1 = _mm256_fmadd_ps(_mm256_load_ps(decoded + column + 8),
                                 _mm256_loadu_ps(input + column + 8), acc1);
        }
        float block_total = horizontal(_mm256_add_ps(acc0, acc1));
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

#endif
