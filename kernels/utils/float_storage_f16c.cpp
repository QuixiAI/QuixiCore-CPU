#if (defined(__x86_64__) || defined(_M_X64)) && \
    defined(QUIXICORE_CPU_ISA_F16C)

#include <immintrin.h>

#include "kernels/common/fp16.h"
#include "kernels/utils/float_storage_isa.h"

namespace quixicore_cpu::float_storage_detail {

void f16_to_f32_f16c(const std::uint16_t* input, float* output,
                     long long begin, long long end) {
  long long i = begin;
  for (; i + 7 < end; i += 8) {
    const __m128i half =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
    _mm256_storeu_ps(output + i, _mm256_cvtph_ps(half));
  }
  for (; i < end; ++i) output[i] = fp16_to_fp32(input[i]);
}

void f32_to_f16_f16c(const float* input, std::uint16_t* output,
                     long long begin, long long end) {
  long long i = begin;
  for (; i + 7 < end; i += 8) {
    const __m128i half = _mm256_cvtps_ph(
        _mm256_loadu_ps(input + i), _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), half);
  }
  for (; i < end; ++i) output[i] = fp32_to_fp16(input[i]);
}

}  // namespace quixicore_cpu::float_storage_detail

#endif
