#pragma once

// Kernel-internal interface for the activation-quantizing quantized GEMV
// (qgemv_w8a8). Public API + variant dispatch live in
// src/dispatch/qgemv_w8a8.cpp; this header is not installed.

#include <cstdint>

#include "kernels/quantization/qgemv.h"

namespace quixicore_cpu::quant {

inline constexpr long long kQ4_0BlockSize = 32;

// GGUF-compatible q4_0: fp16 scale + 16 bytes of packed 4-bit weights. Two
// nibbles per byte — low nibble is element j, high nibble is element j+16 —
// each a value in [0,15] with an implicit -8 offset, so weight = d*(nibble-8).
// 18 bytes per 32 weights, matching llama.cpp block_q4_0 so packed weights are
// interoperable.
struct BlockQ4_0 {
  uint16_t d;       // fp16 scale
  uint8_t qs[16];   // 32 packed nibbles
};
static_assert(sizeof(BlockQ4_0) == 18, "q4_0 block layout must be packed");

// Pack a row-major f32 matrix (n x k, k % 32 == 0) into q4_0 blocks. Returns
// false (without any undefined float->int conversion) on NaN/inf; the public
// entry reports kInvalidArgument. llama.cpp q4_0 rounding: with the largest
// magnitude signed value vmax, d = vmax / -8 and q = clamp(round(x/d)+8, 0, 15).
bool q4_0_pack_ref(const float* weights, long long n, long long k,
                   BlockQ4_0* packed);
void q4_0_unpack_ref(const BlockQ4_0* packed, long long n, long long k,
                     float* weights);

// Weight-only q4_0 GEMV: dequantized weights times full-precision f32
// activations, matching the public qgemv contract.
void q4_0_gemv_ref(const BlockQ4_0* packed, const float* x, float* y,
                   long long n, long long k);

// y = dequant(W) . quant(x): quantize x [k] to per-32-block int8 once
// (d = amax/127, round-to-nearest), then per row a 4-bit x int8 integer dot
// per block scaled by the combined fp16 scales, f32 accumulation. Deterministic.
using Q4_0W8A8GemvFn = void (*)(const BlockQ4_0* packed, const float* x,
                                float* y, long long n, long long k);
void q4_0_gemv_w8a8_ref(const BlockQ4_0* packed, const float* x, float* y,
                        long long n, long long k);

// AArch64 DotProd (SDOT) variant of q4_0_gemv_w8a8_ref: identical activation
// quantization and integer-dot semantics, with each 32-weight block nibble-
// unpacked (-8 offset) and dotted via vdotq_s32. Built only under
// QUIXICORE_CPU_ISA_DOTPROD; selected at runtime when features.dotprod
// (src/dispatch/qgemv_w8a8.cpp).
void q4_0_gemv_w8a8_dotprod(const BlockQ4_0* packed, const float* x, float* y,
                            long long n, long long k);

// Portable q8_0 x per-block-int8-activation anchor for qgemv_w8a8. The
// aarch64 DotProd implementation is an ISA variant of these semantics.
void q8_0_gemv_w8a8_ref(const BlockQ8_0* packed, const float* x, float* y,
                        long long n, long long k);

}  // namespace quixicore_cpu::quant
