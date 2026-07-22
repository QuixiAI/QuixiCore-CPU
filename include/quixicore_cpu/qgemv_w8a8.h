#pragma once

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu {

// Activation-quantizing quantized GEMV — the "w8a8" twin of qgemv named in
// qgemv.h. Packed quantized weights W [N,K] and f32 activations x [K]; the
// activations are quantized on the fly to per-32-element int8 blocks, then an
// integer dot-product (int8 x int8, or 4-bit x int8) accumulates per block with
// the combined weight*activation fp16 scales, in f32. Exact semantics are
// y = dequant(W) @ dequant(blockwise_int8(x)).
//
// Unlike qgemv, this op quantizes activations and therefore is not
// bitwise-identical to the f32-activation path. Supported weight formats are
// kQ8_0 and kQ4_0; both use per-block int8 activations.
Status qgemv_w8a8(QuantFormat format, const void* packed_weights,
                  const float* x, float* y, long long n, long long k);

// Packed size / pack for the w8a8 weight formats (kQ8_0 shares qgemv's q8_0
// layout; kQ4_0 is the 18-byte GGUF q4_0 block). k must be a multiple of 32.
Status qgemv_w8a8_packed_size(QuantFormat format, long long n, long long k,
                              long long* bytes);
Status qgemv_w8a8_pack(QuantFormat format, const float* weights, long long n,
                       long long k, void* packed);

// Name of the resolved variant for the given weight format ("ref", "dotprod",
// ...).
const char* qgemv_w8a8_variant(QuantFormat format);

}  // namespace quixicore_cpu
