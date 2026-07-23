#include "kernels/quantization/gguf_ref.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float horizontal(__m512 value) {
  return _mm512_reduce_add_ps(value);
}

}  // namespace

void gguf_gemv_avx512(QuantFormat format, const void* packed, const float* x,
                      float* y, long long n, long long k) {
  // The direct AVX2 packed-block path avoids a 256-float scratch decode and is
  // faster for these formats on current x86 CPUs. AVX-512 retains its wider
  // generic decoded fallback until a dedicated direct kernel is measured.
  switch (format) {
    case QuantFormat::kQ4_0:
    case QuantFormat::kQ4_1:
    case QuantFormat::kQ5_0:
    case QuantFormat::kQ5_1:
    case QuantFormat::kQ2_K:
    case QuantFormat::kQ3_K:
    case QuantFormat::kQ4_K:
    case QuantFormat::kQ5_K:
    case QuantFormat::kQ6_K:
    case QuantFormat::kIQ4_NL:
    case QuantFormat::kIQ4_XS:
      gguf_gemv_avx2(format, packed, x, y, n, k);
      return;
    default:
      break;
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
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int column = 0;
        for (; column + 31 < block_size; column += 32) {
          acc0 = _mm512_fmadd_ps(_mm512_load_ps(decoded + column),
                                 _mm512_loadu_ps(input + column), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_load_ps(decoded + column + 16),
                                 _mm512_loadu_ps(input + column + 16), acc1);
        }
        float block_total = horizontal(_mm512_add_ps(acc0, acc1));
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
