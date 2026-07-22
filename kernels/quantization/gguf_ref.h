#pragma once

#include <cstddef>
#include <cstdint>

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu::quant {

bool gguf_format_info(QuantFormat format, long long* block_size,
                      std::size_t* block_bytes);
void gguf_unpack_ref(QuantFormat format, const void* packed, long long n,
                     long long k, float* weights);
void gguf_gemv_ref(QuantFormat format, const void* packed, const float* x,
                   float* y, long long n, long long k);

// Decode one complete format block into contiguous f32 values. This keeps
// format semantics centralized while ISA GEMV kernels own only the dot path.
void gguf_dequant_block_ref(QuantFormat format, const std::uint8_t* block,
                            float* values);

using GgufGemvFn = void (*)(QuantFormat format, const void* packed,
                            const float* x, float* y, long long n,
                            long long k);
void gguf_gemv_blocked_ref(QuantFormat format, const void* packed,
                           const float* x, float* y, long long n,
                           long long k);
void gguf_gemv_neon(QuantFormat format, const void* packed, const float* x,
                    float* y, long long n, long long k);
void gguf_gemv_avx2(QuantFormat format, const void* packed, const float* x,
                    float* y, long long n, long long k);
void gguf_gemv_avx512(QuantFormat format, const void* packed, const float* x,
                      float* y, long long n, long long k);

// Internal element decoder used by packed embedding and LM-head composites.
float gguf_dequant_element(QuantFormat format, const std::uint8_t* block,
                           int column);

}  // namespace quixicore_cpu::quant
