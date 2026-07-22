#pragma once

#include <cstdint>

namespace quixicore_cpu::quant {

using W8A32GemmFn = void (*)(const std::int8_t* weights,
                             const float* weight_scales, const float* x,
                             float* y, long long m, long long n, long long k);

void w8a32_gemm_ref(const std::int8_t* weights, const float* weight_scales,
                    const float* x, float* y, long long m, long long n,
                    long long k);

#if defined(__aarch64__) || defined(_M_ARM64)
void w8a32_gemm_neon(const std::int8_t* weights, const float* weight_scales,
                     const float* x, float* y, long long m, long long n,
                     long long k);
#endif

void w8a32_gemm_avx2(const std::int8_t* weights, const float* weight_scales,
                     const float* x, float* y, long long m, long long n,
                     long long k);
void w8a32_gemm_avx512(const std::int8_t* weights,
                       const float* weight_scales, const float* x, float* y,
                       long long m, long long n, long long k);

}  // namespace quixicore_cpu::quant
