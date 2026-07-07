#pragma once

#include <cstddef>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

enum class QuantFormat {
  kQ8_0,  // GGUF q8_0: 32-element blocks, fp16 scale + 32 int8 values
};

// Packed size in bytes of an n x k row-major weight matrix.
Status quant_gemv_packed_size(QuantFormat format, long long n, long long k,
                              size_t* size);

// Quantize row-major f32 weights (n x k) into the packed format.
Status quant_gemv_pack(QuantFormat format, const float* weights, long long n,
                       long long k, void* packed);

// Dequantize packed weights back to row-major f32 (n x k).
Status quant_gemv_unpack(QuantFormat format, const void* packed, long long n,
                         long long k, float* weights);

// y = W x with packed quantized W (n x k) and f32 activations x (k), y (n).
// Deterministic, f32 accumulation. The kernel variant is resolved once per
// process from runtime CPU features; set QUIXICORE_CPU_QGEMV_VARIANT to
// force a named variant for testing and benchmarking.
Status quant_gemv(QuantFormat format, const void* packed, const float* x,
                  float* y, long long n, long long k);

// Name of the variant quant_gemv resolves to for this format ("ref", ...).
const char* quant_gemv_variant(QuantFormat format);

}  // namespace quixicore_cpu
