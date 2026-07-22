#pragma once

#include <cstddef>
#include <cstdint>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

enum class QuantFormat {
  kQ8_0,  // GGUF q8_0: 32-element blocks, fp16 scale + 32 int8 values
  kQ4_0,  // GGUF q4_0: 32-element blocks, fp16 scale + 16 packed nibbles
          // (-8 offset)
  kQ4_1,
  kQ5_0,
  kQ5_1,
  kU4B8,
  kU4,
  kHQQ,
  kFP8E4M3,
  kFP8E5M2,
  kFP8Block,
  kFP8Raw,
  kFP4E2M1,
  kMXFP8,
  kNVFP4,
  kMXFP4,
  kMXFP6E3M2,
  kMXFP6E2M3,
  kBitnet,
  kQ2_K,
  kQ3_K,
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kIQ4_NL,
  kIQ4_XS,
  kIQ2_XXS,
  kIQ2_XS,
  kIQ3_XXS,
  kIQ1_S,
  kTQ2_0,
};

// Quantized GEMV, QuixiCore family contract: out = dequantize(wq) @ x with
// full-precision activations (f32 on CPU) and f32 accumulation. Matches the
// sibling `qgemv` semantics and oracle; q8_0 and q4_0 block layouts are
// llama.cpp/GGUF byte-compatible. Activation-quantizing integer paths are a
// separate `qgemv_w8a8` operation.

// Packed size in bytes of an n x k row-major weight matrix
// (layout (n, k/block_k, block_bytes), blocks along the contraction axis).
Status qgemv_packed_size(QuantFormat format, long long n, long long k,
                         size_t* size);

// Quantize row-major f32 weights (n x k) into the packed format. Packing is
// currently defined for q8_0/q4_0/tq2_0; the other formats are decode-compatible
// with GGUF inputs and return kUnsupportedFormat from this entry point.
Status qgemv_pack(QuantFormat format, const float* weights, long long n,
                  long long k, void* packed);

// Dequantize packed weights back to row-major f32 (n x k).
Status qgemv_unpack(QuantFormat format, const void* packed, long long n,
                    long long k, float* weights);

// y = dequantize(wq) @ x with packed W (n x k), f32 x (k), f32 y (n).
// Deterministic. The kernel variant is resolved once per process from
// runtime CPU features; QUIXICORE_CPU_QGEMV_VARIANT forces a named
// contract-compatible variant for testing and benchmarking. Internal
// activation-quantizing experiments are deliberately not selectable here.
Status qgemv(QuantFormat format, const void* packed, const float* x, float* y,
             long long n, long long k);

// XPU-style logical split layouts: adjacent E2M1 nibbles in packed_weights,
// row-major scale bytes, and f32 activations/output. MXFP4 uses one E8M0 scale
// per 32 values; NVFP4 uses one E4M3 scale per 16 plus a tensor scale.
Status mxfp4_gemv(const std::uint8_t* packed_weights,
                  const std::uint8_t* scale_codes, const float* x, float* y,
                  long long n, long long k);
Status nvfp4_gemv(const std::uint8_t* packed_weights,
                  const std::uint8_t* scale_codes, float global_scale,
                  const float* x, float* y, long long n, long long k);

// Fused projection compositions matching the Metal decode seams.
Status qgemv_up_gate(QuantFormat format, const void* packed_up,
                     const void* packed_gate, const float* x, float* up,
                     float* gate, long long n, long long k);
Status qgemv_up_gate_activation(QuantFormat format, const void* packed_up,
                                const void* packed_gate, const float* x,
                                float* out, long long n, long long k,
                                bool gelu_tanh = true);
Status qgemv_qkv(QuantFormat format, const void* packed_q,
                 const void* packed_k, const void* packed_v, const float* x,
                 float* q, float* k_out, float* v_out, long long query_dim,
                 long long kv_dim, long long input_dim);

// Name of the variant qgemv resolves to for this format ("ref", "neon", ...).
const char* qgemv_variant(QuantFormat format);

}  // namespace quixicore_cpu
