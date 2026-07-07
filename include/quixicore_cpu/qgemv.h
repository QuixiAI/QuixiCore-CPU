#pragma once

#include <cstddef>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

enum class QuantFormat {
  kQ8_0,  // GGUF q8_0: 32-element blocks, fp16 scale + 32 int8 values
};

// Quantized GEMV, QuixiCore family contract: out = dequantize(wq) @ x with
// full-precision activations (f32 on CPU) and f32 accumulation. Matches the
// Metal/CUDA `qgemv` semantics and oracle; block layouts are llama.cpp/GGUF
// byte-compatible. Activation-quantizing integer paths are NOT this op —
// they belong to a future `qgemv_w8a8` twin, per the sibling backends.

// Packed size in bytes of an n x k row-major weight matrix
// (layout (n, k/block_k, block_bytes), blocks along the contraction axis).
Status qgemv_packed_size(QuantFormat format, long long n, long long k,
                         size_t* size);

// Quantize row-major f32 weights (n x k) into the packed format.
Status qgemv_pack(QuantFormat format, const float* weights, long long n,
                  long long k, void* packed);

// Dequantize packed weights back to row-major f32 (n x k).
Status qgemv_unpack(QuantFormat format, const void* packed, long long n,
                    long long k, float* weights);

// y = dequantize(wq) @ x with packed W (n x k), f32 x (k), f32 y (n).
// Deterministic. The kernel variant is resolved once per process from
// runtime CPU features; QUIXICORE_CPU_QGEMV_VARIANT forces a named variant
// for testing and benchmarking.
Status qgemv(QuantFormat format, const void* packed, const float* x, float* y,
             long long n, long long k);

// Name of the variant qgemv resolves to for this format ("ref", "neon", ...).
const char* qgemv_variant(QuantFormat format);

}  // namespace quixicore_cpu
