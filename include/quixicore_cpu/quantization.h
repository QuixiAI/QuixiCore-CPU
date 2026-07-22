#pragma once

#include <cstdint>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

enum class Float8Format { kE4M3FN, kE5M2 };

// Portable float8 conversion. Encoding uses round-to-nearest-even and saturates
// finite overflow to the largest finite value. NaNs retain NaN semantics.
std::uint8_t float8_encode(float value, Float8Format format);
float float8_decode(std::uint8_t code, Float8Format format);

// Dynamic symmetric quantization over contiguous groups in each row. A
// group_size of zero means one group per row. scales is [rows,dim/group_size].
Status quantize_int8(const float* x, std::int8_t* codes, float* scales,
                     long long rows, long long dim,
                     long long group_size = 0);
Status dequantize_int8(const std::int8_t* codes, const float* scales,
                       float* out, long long rows, long long dim,
                       long long group_size = 0);

// Per-row asymmetric INT8. Reconstruct with scale[row]*(code-zero_point[row]).
Status quantize_int8_asymmetric(const float* x, std::int8_t* codes,
                                float* scales, int* zero_points,
                                long long rows, long long dim);
Status dequantize_int8_asymmetric(const std::int8_t* codes,
                                  const float* scales,
                                  const int* zero_points, float* out,
                                  long long rows, long long dim);

// Signed symmetric INT4, two two's-complement nibbles per byte. Groups run
// along the last dimension and must divide dim.
Status quantize_int4_group(const float* x, std::uint8_t* packed,
                           float* scales, long long rows, long long dim,
                           long long group_size);
Status dequantize_int4_group(const std::uint8_t* packed,
                             const float* scales, float* out,
                             long long rows, long long dim,
                             long long group_size);

// Dynamic float8 quantization. E4M3FN matches the runtime activation quantizer
// used by sibling backends. power_of_two_scale implements UE8M0/MX scaling.
Status quantize_float8(const float* x, std::uint8_t* codes, float* scales,
                       long long rows, long long dim,
                       long long group_size = 0,
                       Float8Format format = Float8Format::kE4M3FN,
                       bool power_of_two_scale = false);
Status dequantize_float8(const std::uint8_t* codes, const float* scales,
                         float* out, long long rows, long long dim,
                         long long group_size = 0,
                         Float8Format format = Float8Format::kE4M3FN);

// Fake-quantized results are reconstructed in f32 from the exact emitted codes.
Status fake_quant_int8(const float* x, float* out, std::int8_t* codes,
                       float* scales, long long rows, long long dim);
Status silu_mul_fake_quant_int8(const float* x, const float* gate, float* out,
                                std::int8_t* codes, float* scales,
                                long long rows, long long dim,
                                bool oai_mode = false,
                                float alpha = 1.0f,
                                float limit = 0.0f);
Status silu_mul_quant_int8(const float* x, const float* gate,
                           std::int8_t* codes, float* scales, long long rows,
                           long long dim, bool oai_mode = false,
                           float alpha = 1.702f, float limit = 7.0f);
Status silu_mul_quant_float8(
    const float* x, const float* gate, std::uint8_t* codes, float* scales,
    long long rows, long long dim, long long group_size = 0,
    bool power_of_two_scale = false, bool oai_mode = false,
    float alpha = 1.702f, float limit = 7.0f,
    Float8Format format = Float8Format::kE4M3FN);
Status fake_quant_float8(const float* x, float* out, std::uint8_t* codes,
                         float* scale, long long count,
                         Float8Format format = Float8Format::kE4M3FN);

// Residual-add + normalization + dynamic activation quantization. Scales are
// per row when group_size==0 and per group otherwise.
Status rms_norm_add_quant_int8(
    const float* x, const float* residual, const float* weight,
    std::int8_t* codes, float* residual_out, float* scales, long long rows,
    long long hidden, float eps = 1e-5f, long long group_size = 0);
Status layer_norm_add_quant_int8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::int8_t* codes, float* residual_out,
    float* scales, long long rows, long long hidden, float eps = 1e-5f,
    long long group_size = 0);
Status rms_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    std::uint8_t* codes, float* residual_out, float* scales, long long rows,
    long long hidden, float eps = 1e-5f, long long group_size = 0,
    bool power_of_two_scale = false,
    Float8Format format = Float8Format::kE4M3FN);
Status layer_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out,
    float* scales, long long rows, long long hidden, float eps = 1e-5f,
    long long group_size = 0, bool power_of_two_scale = false,
    Float8Format format = Float8Format::kE4M3FN);

// BitNet blocks are byte-compatible {fp16 scale, 8 packed 2-bit codes}; K and
// group_k must be divisible by 32. Packed storage is rows*(K/32)*10 bytes.
Status ternary_pack(const float* weights, std::uint8_t* packed,
                    float* dequantized, long long rows, long long k,
                    long long group_k);
Status ternary_unpack(const std::uint8_t* packed, float* weights,
                      long long rows, long long k);
Status ternary_stats(const std::uint8_t* packed, std::uint32_t* counts,
                     long long rows, long long k);
Status ternary_code_flip_count(const std::uint8_t* a, const std::uint8_t* b,
                               std::uint32_t* flips, long long rows,
                               long long k);

// llama.cpp TQ2_0: one 66-byte block per 256 values ({qs[64], fp16 d}).
Status tq2_0_pack(const float* weights, std::uint8_t* packed,
                  float* dequantized, long long rows, long long k);
Status tq2_0_unpack(const std::uint8_t* packed, float* weights,
                    long long rows, long long k);

// TurboQuant KV codec. Caches are flattened [slots,heads,packed_bytes] and
// scales/zero-points [slots,heads,head_size/32]. Scale outputs are half-rounded
// but exposed as f32 on CPU. signs is [head_size] and centroids [2^value_bits].
Status turboquant_packed_bytes(long long head_size, int bits,
                               long long* packed_bytes);
Status turboquant_encode(
    const float* key, const float* value, const int* slot_mapping,
    const float* value_centroids, const float* signs,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    float* key_scale_cache, float* value_scale_cache,
    float* key_zero_cache, long long tokens, long long slots,
    long long heads, long long head_size, int key_bits,
    bool key_signed, int value_bits);
Status turboquant_decode(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const float* key_scale_cache, const float* value_scale_cache,
    const float* key_zero_cache, const int* slots_to_gather,
    const float* value_centroids, const float* signs, float* key_out,
    float* value_out, long long cache_slots, long long rows,
    long long heads, long long head_size, int key_bits,
    bool key_signed, int value_bits);

}  // namespace quixicore_cpu
