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
  // Additional canonical llama.cpp/GGUF weight formats. Keep these appended so
  // the numeric values of the existing QuixiCore formats remain stable.
  kQ1_0,
  kQ2_0,
  kIQ3_S,
  kIQ2_S,
  kIQ1_M,
  kTQ1_0,
};

// llama.cpp-compatible activation/intermediate formats. These are kept
// separate from QuantFormat because Q8_1 and Q8_K are dot-product partners,
// not GGUF model-storage formats.
enum class QuantActivationFormat {
  kQ8_0,
  kQ8_1,
  kQ8_K,
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
// currently defined for q1_0/q2_0/q4_0/q4_1/q5_0/q5_1/q8_0, Q2_K-Q6_K,
// IQ formats, MXFP4, NVFP4, tq1_0, and tq2_0. Importance-matrix IQ formats
// use uniform importance here; qgemv_pack_weighted accepts calibration data.
Status qgemv_pack(QuantFormat format, const float* weights, long long n,
                  long long k, void* packed);

// Importance-aware authoring for llama.cpp IQ formats. importance is an n x k
// row-major non-negative finite matrix. For formats that do not use an
// importance matrix this is equivalent to qgemv_pack. Passing nullptr selects
// uniform importance and is also what qgemv_pack uses.
Status qgemv_pack_weighted(QuantFormat format, const float* weights,
                           const float* importance, long long n, long long k,
                           void* packed);

// Dequantize packed weights back to row-major f32 (n x k).
Status qgemv_unpack(QuantFormat format, const void* packed, long long n,
                    long long k, float* weights);

// Packed size, quantization, and dequantization for row-major activation
// matrices. The byte layouts are canonical llama.cpp block_q8_0, block_q8_1,
// and block_q8_K layouts. k must be divisible by the selected block size.
Status quant_activation_packed_size(QuantActivationFormat format, long long n,
                                    long long k, size_t* size);
Status quant_activation_pack(QuantActivationFormat format, const float* input,
                             long long n, long long k, void* packed);
Status quant_activation_unpack(QuantActivationFormat format,
                               const void* packed, long long n, long long k,
                               float* output);

// y = dequantize(wq) @ x with packed W (n x k), f32 x (k), f32 y (n).
// Deterministic. The kernel variant is resolved once per process from
// runtime CPU features; QUIXICORE_CPU_QGEMV_VARIANT forces a named
// contract-compatible variant for testing and benchmarking. Internal
// activation-quantizing experiments are deliberately not selectable here.
Status qgemv(QuantFormat format, const void* packed, const float* x, float* y,
             long long n, long long k);

// y = dequantize(wq) @ dequantize(xq). This portable contract route accepts
// all supported stored-weight formats and all activation formats above. ISA
// implementations may fuse the two block decoders behind the same semantics.
Status qgemv_quantized_activation(QuantFormat weight_format,
                                  const void* packed_weights,
                                  QuantActivationFormat activation_format,
                                  const void* packed_activation, float* y,
                                  long long n, long long k);

// Compute only selected packed rows. row_ids has row_count entries in [0,n);
// y is compact [row_count] in the same order. This is the packed projection
// seam used by constrained vocabularies and MLA absorption.
Status qgemv_rows(QuantFormat format, const void* packed, const float* x,
                  const int* row_ids, float* y, long long row_count,
                  long long n, long long k);

// In-place out[K] += coefficient * dequantize(W[row,:]) without materializing
// the complete packed matrix.
Status qgemv_axpy_row(QuantFormat format, const void* packed, long long row,
                      float coefficient, float* out, long long n,
                      long long k);

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
// Single-token decode projection with split-half RoPE on Q/K and direct KV
// cache insertion. Caches are flattened [slots,kv_heads,head_dim].
Status qgemv_qkv_rope_kv(
    QuantFormat format, const void* packed_q, const void* packed_k,
    const void* packed_v, const float* x, const float* cosine,
    const float* sine, float* q, float* key_cache, float* value_cache,
    long long query_heads, long long kv_heads, long long head_dim,
    long long input_dim, long long slots, long long max_position, int position,
    int slot);

// Name of the variant qgemv resolves to for this format ("ref", "neon", ...).
const char* qgemv_variant(QuantFormat format);

}  // namespace quixicore_cpu
