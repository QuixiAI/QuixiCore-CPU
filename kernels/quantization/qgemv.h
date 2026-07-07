#pragma once

// Kernel-internal interface for quantized GEMV. The public API and variant
// dispatch live in src/dispatch/quant_gemv.cpp; this header is not installed.

#include <cstdint>

namespace quixicore_cpu::qgemv {

// GGUF-compatible q8_0: 32-element blocks, fp16 scale + 32 int8 values
// (34 bytes per 32 weights). Layout matches llama.cpp block_q8_0 so packed
// weights are interoperable.
inline constexpr long long kQ8_0BlockSize = 32;

struct BlockQ8_0 {
  uint16_t d;      // fp16 scale
  int8_t qs[32];   // quantized values; weight = d * qs[j]
};
static_assert(sizeof(BlockQ8_0) == 34, "q8_0 block layout must be packed");

// Quantize one row-major f32 matrix (n x k, k % 32 == 0) into blocks.
void q8_0_pack_ref(const float* weights, long long n, long long k,
                   BlockQ8_0* packed);

// Dequantize back to row-major f32.
void q8_0_unpack_ref(const BlockQ8_0* packed, long long n, long long k,
                     float* weights);

// y = W x, f32 accumulation. Deterministic.
using Q8_0GemvFn = void (*)(const BlockQ8_0* packed, const float* x, float* y,
                            long long n, long long k);

void q8_0_gemv_ref(const BlockQ8_0* packed, const float* x, float* y,
                   long long n, long long k);

}  // namespace quixicore_cpu::qgemv
