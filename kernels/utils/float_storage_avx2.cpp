#if (defined(__x86_64__) || defined(_M_X64)) && defined(QUIXICORE_CPU_ISA_AVX2)

#include <immintrin.h>

#include "kernels/utils/float_storage_isa.h"
#include "quixicore_cpu/float_storage.h"

namespace quixicore_cpu::float_storage_detail {

void bf16_to_f32_avx2(const std::uint16_t* input, float* output,
                      long long begin, long long end) {
  const __m128i zero = _mm_setzero_si128();
  long long i = begin;
  for (; i + 7 < end; i += 8) {
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
    const __m128i low = _mm_unpacklo_epi16(zero, packed);
    const __m128i high = _mm_unpackhi_epi16(zero, packed);
    _mm_storeu_ps(output + i, _mm_castsi128_ps(low));
    _mm_storeu_ps(output + i + 4, _mm_castsi128_ps(high));
  }
  for (; i < end; ++i) output[i] = bf16_to_float(input[i]);
}

}  // namespace quixicore_cpu::float_storage_detail

#endif
