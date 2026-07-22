#pragma once

#include <cstdint>

#include "quixicore_cpu/lowbit.h"

namespace quixicore_cpu::quant {

inline constexpr long long kInt3Group = 64;
inline constexpr long long kInt3GroupBytes = 24;
inline constexpr long long kE8Block = 256;
inline constexpr long long kE8BlockBytes = 98;

using LowBitGemmFn = void (*)(LowBitFormat format,
                              const std::uint8_t* packed,
                              const float* scales, const float* x, float* y,
                              long long m, long long n, long long k,
                              long long group_size);
using LowBitW8A8Fn = void (*)(const std::uint8_t* packed,
                              const float* weight_scales,
                              const std::int8_t* x,
                              const float* activation_scales, float* y,
                              long long m, long long n, long long k);

void lowbit_gemm_ref(LowBitFormat format, const std::uint8_t* packed,
                     const float* scales, const float* x, float* y,
                     long long m, long long n, long long k,
                     long long group_size);

#if defined(__aarch64__) || defined(_M_ARM64)
void lowbit_gemm_neon(LowBitFormat format, const std::uint8_t* packed,
                      const float* scales, const float* x, float* y,
                      long long m, long long n, long long k,
                      long long group_size);
#endif
void lowbit_gemm_avx2(LowBitFormat format, const std::uint8_t* packed,
                      const float* scales, const float* x, float* y,
                      long long m, long long n, long long k,
                      long long group_size);
void lowbit_gemm_avx512(LowBitFormat format, const std::uint8_t* packed,
                        const float* scales, const float* x, float* y,
                        long long m, long long n, long long k,
                        long long group_size);

void lowbit_w8a8_ref(const std::uint8_t* packed, const float* weight_scales,
                     const std::int8_t* x, const float* activation_scales,
                     float* y, long long m, long long n, long long k);
void lowbit_w8a8_dotprod(const std::uint8_t* packed,
                         const float* weight_scales, const std::int8_t* x,
                         const float* activation_scales, float* y,
                         long long m, long long n, long long k);
void lowbit_w8a8_i8mm(const std::uint8_t* packed,
                      const float* weight_scales, const std::int8_t* x,
                      const float* activation_scales, float* y,
                      long long m, long long n, long long k);
void lowbit_w8a8_avx2(const std::uint8_t* packed,
                      const float* weight_scales, const std::int8_t* x,
                      const float* activation_scales, float* y,
                      long long m, long long n, long long k);
void lowbit_w8a8_avx512_vnni(
    const std::uint8_t* packed, const float* weight_scales,
    const std::int8_t* x, const float* activation_scales, float* y,
    long long m, long long n, long long k);

}  // namespace quixicore_cpu::quant
