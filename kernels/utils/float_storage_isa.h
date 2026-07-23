#pragma once

#include <cstdint>

namespace quixicore_cpu::float_storage_detail {

#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_NEON_FP16)
void f16_to_f32_neon(const std::uint16_t* input, float* output,
                     long long begin, long long end);
void f32_to_f16_neon(const float* input, std::uint16_t* output,
                     long long begin, long long end);
#endif

#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_F16C)
void f16_to_f32_f16c(const std::uint16_t* input, float* output,
                     long long begin, long long end);
void f32_to_f16_f16c(const float* input, std::uint16_t* output,
                     long long begin, long long end);
#endif

}  // namespace quixicore_cpu::float_storage_detail
