#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Universal floating-point storage accepted by the CPU fallback layer.
// FP16/BF16 tensors are decoded once, accumulated by the existing FP32
// kernels, and rounded once when an output is committed. kF32 is zero-copy.
// The 16-bit representations are the raw IEEE binary16/bfloat16 bits.
enum class FloatStorageType { kF32, kF16, kBF16 };

struct FloatStorageInput {
  const void* data = nullptr;
  FloatStorageType type = FloatStorageType::kF32;
  long long count = 0;
};

struct FloatStorageOutput {
  void* data = nullptr;
  FloatStorageType type = FloatStorageType::kF32;
  long long count = 0;
};

// Reusable conversion arena. One workspace may be used by one invocation at
// a time. Keeping it at the model/executor level removes allocation after the
// largest observed operation has warmed the arena.
class FloatStorageWorkspace {
 public:
  Status reserve(std::size_t float_elements);
  std::size_t capacity() const noexcept { return scratch_.capacity(); }

 private:
  std::vector<float> scratch_;
  friend Status dispatch_float_storage(const FloatStorageInput*, long long,
                                       const FloatStorageOutput*, long long,
                                       Status (*)(const float* const*,
                                                  float* const*, void*),
                                       void*, FloatStorageWorkspace*);
};

using Float32StorageKernel = Status (*)(const float* const* inputs,
                                        float* const* outputs, void* context);

// Adapt arbitrary floating storage to any FP32 kernel. All inputs are decoded
// before the callback and outputs are encoded only when it returns kOk.
// Exact input/output aliases with identical type and count are in-place. Any
// partial overlap, or an exact alias described with different metadata, is
// rejected because it cannot preserve the FP32 kernel's alias contract. FP16
// and BF16 outputs commit only after success; zero-copy FP32 outputs retain the
// wrapped kernel's normal failure/alias behavior.
Status dispatch_float_storage(const FloatStorageInput* inputs,
                              long long input_count,
                              const FloatStorageOutput* outputs,
                              long long output_count,
                              Float32StorageKernel kernel, void* context,
                              FloatStorageWorkspace* workspace = nullptr);

// C++ convenience for wrapping any existing FP32 operation without declaring
// a separate context struct. The callable receives (inputs, outputs) and must
// return Status. Dispatch is synchronous, so temporary lambdas are safe.
template <class Kernel>
Status with_float_storage(const FloatStorageInput* inputs,
                          long long input_count,
                          const FloatStorageOutput* outputs,
                          long long output_count, Kernel&& kernel,
                          FloatStorageWorkspace* workspace = nullptr) {
  using Callable = std::remove_reference_t<Kernel>;
  return dispatch_float_storage(
      inputs, input_count, outputs, output_count,
      [](const float* const* f32_inputs, float* const* f32_outputs,
         void* opaque) -> Status {
        return (*static_cast<Callable*>(opaque))(f32_inputs, f32_outputs);
      },
      &kernel, workspace);
}

std::uint16_t float_to_f16(float value) noexcept;
float f16_to_float(std::uint16_t bits) noexcept;
std::uint16_t float_to_bf16(float value) noexcept;
float bf16_to_float(std::uint16_t bits) noexcept;

Status float_storage_to_f32(FloatStorageType type, const void* input,
                            float* output, long long count);
Status float_storage_from_f32(FloatStorageType type, const float* input,
                              void* output, long long count);
const char* float_storage_variant(FloatStorageType type) noexcept;

// Typed convenience routes for the dominant model kernels. Individual inputs
// and outputs may use different storage types; all reductions remain FP32.
Status unary_storage(FloatStorageInput x, FloatStorageOutput y, UnaryOp op,
                     XiEluParams xielu = {});
Status calibration_absmax_storage(
    FloatStorageInput x, const float* running, float* out, long long tokens,
    long long channels, FloatStorageWorkspace* workspace = nullptr);
Status logits_softcap_storage(FloatStorageInput logits, FloatStorageOutput out,
                              float cap,
                              FloatStorageWorkspace* workspace = nullptr);
Status value_clip_storage(FloatStorageInput x, FloatStorageOutput out,
                          float minimum, float maximum,
                          FloatStorageWorkspace* workspace = nullptr);
Status softmax_storage(FloatStorageInput x, FloatStorageOutput y,
                       long long rows, long long dim);
Status rms_norm_storage(FloatStorageInput x, FloatStorageInput weight,
                        FloatStorageOutput y, long long rows, long long hidden,
                        float eps = 1e-5f);
Status dense_gemm_storage(FloatStorageInput a, FloatStorageInput b,
                          FloatStorageOutput c, long long m, long long n,
                          long long k);
Status embedding_lookup_types_storage(
    const int* token_ids, const int* type_ids, FloatStorageInput token_table,
    FloatStorageInput type_table, FloatStorageOutput out,
    long long token_vocab, long long type_vocab, long long count,
    long long dim, float token_scale = 1.0f,
    FloatStorageWorkspace* workspace = nullptr);
Status masked_mean_pool_rms_l2_storage(
    FloatStorageInput x, const int* mask, FloatStorageInput weight,
    FloatStorageOutput out, long long batch, long long sequence,
    long long hidden, float eps = 1e-5f,
    FloatStorageWorkspace* workspace = nullptr);
Status extract_patches_2d_storage(
    FloatStorageInput input, FloatStorageOutput output, long long batch,
    long long input_height, long long input_width, long long channels,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, long long pad_height = 0, long long pad_width = 0,
    FloatStorageWorkspace* workspace = nullptr);
Status extract_patches_3d_storage(
    FloatStorageInput input, FloatStorageOutput output, long long batch,
    long long input_frames, long long input_height, long long input_width,
    long long channels, long long kernel_frames, long long kernel_height,
    long long kernel_width, long long stride_frames, long long stride_height,
    long long stride_width, long long pad_frames = 0,
    long long pad_height = 0, long long pad_width = 0,
    FloatStorageWorkspace* workspace = nullptr);
Status interpolate_position_2d_storage(
    FloatStorageInput table, FloatStorageOutput output, long long input_height,
    long long input_width, long long output_height, long long output_width,
    long long channels, bool align_corners = false,
    FloatStorageWorkspace* workspace = nullptr);
Status avg_pool2d_tokens_storage(
    FloatStorageInput input, FloatStorageOutput output, long long batch,
    long long input_height, long long input_width, long long channels,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, bool ceil_mode = false,
    FloatStorageWorkspace* workspace = nullptr);
Status factorized_position_2d_storage(
    const int* position_ids, FloatStorageInput table, const int* valid_mask,
    FloatStorageOutput output, long long batch, long long tokens,
    long long max_position, long long channels,
    FloatStorageWorkspace* workspace = nullptr);
Status pool_tokens_by_position_storage(
    FloatStorageInput input, const int* position_ids, const int* valid_mask,
    float* output, int* output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width, FloatStorageWorkspace* workspace = nullptr);
Status vision_patch_projection_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_height,
    long long input_width, long long input_channels,
    long long output_channels, long long kernel_height,
    long long kernel_width, long long stride_height, long long stride_width,
    long long pad_height = 0, long long pad_width = 0,
    FloatStorageWorkspace* workspace = nullptr);
Status vision_patch_projection_3d_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_frames,
    long long input_height, long long input_width, long long input_channels,
    long long output_channels, long long kernel_frames,
    long long kernel_height, long long kernel_width, long long stride_frames,
    long long stride_height, long long stride_width, long long pad_frames = 0,
    long long pad_height = 0, long long pad_width = 0,
    FloatStorageWorkspace* workspace = nullptr);
Status audio_conv1d_direct_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_length,
    long long input_channels, long long output_channels, long long kernel,
    long long stride = 1, long long padding = 0, long long dilation = 1,
    FloatStorageWorkspace* workspace = nullptr);
Status audio_depthwise_conv1d_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_length,
    long long channels, long long kernel, long long stride = 1,
    long long padding = 0, long long dilation = 1, bool apply_silu = false,
    FloatStorageWorkspace* workspace = nullptr);
Status audio_causal_depthwise_conv1d_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_length,
    long long channels, long long kernel, long long dilation = 1,
    FloatStorageWorkspace* workspace = nullptr);
Status lora_apply_direct_f16_storage(
    FloatStorageInput x, const std::uint16_t* adapter_a,
    const std::uint16_t* adapter_b, FloatStorageInput base,
    FloatStorageOutput out, long long rows, long long input_dim,
    long long output_dim, long long rank, float scale,
    FloatStorageWorkspace* workspace = nullptr);
Status attention_storage(FloatStorageInput q, FloatStorageInput k,
                         FloatStorageInput v, FloatStorageOutput out,
                         long long query_heads, long long kv_heads,
                         long long query_length, long long kv_length,
                         long long head_dim, bool causal);
Status cross_attention_storage(
    FloatStorageInput q, FloatStorageInput k, FloatStorageInput v,
    const int* key_lengths, const float* bias, FloatStorageOutput out,
    long long batch, long long query_heads, long long kv_heads,
    long long query_length, long long key_length, long long head_dim,
    float scale = 0.0f, float softcap = 0.0f,
    FloatStorageWorkspace* workspace = nullptr);
Status audio_relative_attention_storage(
    FloatStorageInput q, FloatStorageInput k, FloatStorageInput v,
    FloatStorageInput relative_k, const float* per_dim_scale,
    const int* lengths, FloatStorageOutput out, long long batch,
    long long length, long long heads, long long head_dim,
    long long relative_positions, long long chunk_size,
    long long left_context, long long right_context, float q_scale = 0.0f,
    float k_scale = 0.0f, float softcap = 0.0f,
    FloatStorageWorkspace* workspace = nullptr);
Status vision_rope_2d_storage(
    FloatStorageInput x, FloatStorageInput cosine, FloatStorageInput sine,
    const int* positions, FloatStorageOutput out, long long batch,
    long long heads, long long tokens, long long head_dim,
    long long max_position, FloatStorageWorkspace* workspace = nullptr);
Status qwen_vision_rope_2d_storage(
    FloatStorageInput x, FloatStorageInput cosine, FloatStorageInput sine,
    const int* positions, FloatStorageOutput out, long long batch,
    long long heads, long long tokens, long long head_dim,
    long long max_position, FloatStorageWorkspace* workspace = nullptr);
Status sigmoid_mul_storage(FloatStorageInput gate_logits,
                           FloatStorageInput values, FloatStorageOutput out,
                           FloatStorageWorkspace* workspace = nullptr);
Status sigmoid_mul_backward_storage(FloatStorageInput grad_out,
                                    FloatStorageInput gate_logits,
                                    FloatStorageInput values,
                                    FloatStorageOutput grad_gate,
                                    FloatStorageOutput grad_values,
                                    FloatStorageWorkspace* workspace = nullptr);

Status gdn_recur_storage(FloatStorageInput q, FloatStorageInput k,
                         FloatStorageInput v, FloatStorageInput decay,
                         FloatStorageInput beta, const float* state_pool,
                         const int* cumulative_lengths, const int* slot_mapping,
                         FloatStorageOutput out, float* state_pool_out,
                         long long requests, long long slots,
                         long long key_heads, long long value_heads,
                         long long key_dim, long long value_dim,
                         bool load_initial = true,
                         FloatStorageWorkspace* workspace = nullptr);
Status gdn_short_conv_storage(FloatStorageInput x, FloatStorageInput weight,
                              const float* state_pool,
                              const int* cumulative_lengths,
                              const int* slot_mapping, FloatStorageOutput out,
                              float* state_pool_out, long long requests,
                              long long slots, long long channels,
                              long long kernel_size, bool load_initial = true,
                              bool apply_silu = true,
                              FloatStorageWorkspace* workspace = nullptr);
Status gdn_qkv_prepare_storage(
    FloatStorageInput mixed, FloatStorageOutput q, FloatStorageOutput k,
    FloatStorageOutput v, long long tokens, long long key_heads,
    long long value_heads, long long key_dim, long long value_dim,
    float eps = 1e-6f, float q_scale = std::numeric_limits<float>::quiet_NaN(),
    float k_scale = std::numeric_limits<float>::quiet_NaN(),
    FloatStorageWorkspace* workspace = nullptr);
Status gdn_gate_beta_storage(FloatStorageInput a, FloatStorageInput b,
                             const float* a_log, const float* dt_bias,
                             float* decay, float* beta, long long tokens,
                             long long value_heads,
                             FloatStorageWorkspace* workspace = nullptr);
Status gdn_gated_rmsnorm_storage(FloatStorageInput y, FloatStorageInput z,
                                 FloatStorageInput weight,
                                 FloatStorageOutput out, long long tokens,
                                 long long value_heads, long long value_dim,
                                 float eps = 1e-6f,
                                 FloatStorageWorkspace* workspace = nullptr);

// Typed FP8 KV-cache routes. Cache bytes remain canonical E4M3FN/E5M2 and
// reductions remain FP32; only the query/source/output storage varies.
Status kv_cache_scales_fp8_storage(FloatStorageInput key,
                                   FloatStorageInput value, float* key_scale,
                                   float* value_scale, long long count,
                                   long long heads, long long head_dim,
                                   Float8Format format,
                                   FloatStorageWorkspace* workspace = nullptr);
Status kv_cache_scatter_fp8_storage(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    const float* key_scale, const float* value_scale, std::uint8_t* key_cache,
    std::uint8_t* value_cache, long long max_slots, long long count,
    long long heads, long long head_dim, Float8Format format);
Status kv_cache_gather_fp8_storage(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const float* key_scale, const float* value_scale,
    FloatStorageOutput key_out, FloatStorageOutput value_out,
    long long max_slots, long long count, long long heads, long long head_dim,
    Float8Format format);
Status paged_attention_fp8_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const int* block_table,
    const int* context_lens, const float* key_scale, const float* value_scale,
    FloatStorageOutput out, long long cache_blocks, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, Float8Format format,
    float scale = 0.0f, long long window = 0, float softcap = 0.0f,
    FloatStorageWorkspace* workspace = nullptr);
Status kv_cache_scatter_q8_0_storage(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    std::int8_t* key_codes, std::uint16_t* key_scales, std::int8_t* value_codes,
    std::uint16_t* value_scales, long long cache_blocks, long long count,
    long long heads, long long head_dim, long long page_size);
Status kv_cache_gather_q8_0_storage(
    const std::int8_t* key_codes, const std::uint16_t* key_scales,
    const std::int8_t* value_codes, const std::uint16_t* value_scales,
    const int* block_table, const int* cumulative_lengths,
    FloatStorageOutput key_out, FloatStorageOutput value_out,
    long long cache_blocks, long long num_tokens, long long sequences,
    long long heads, long long head_dim, long long page_size,
    long long max_blocks);
Status paged_attention_q8_0_storage(
    FloatStorageInput q, const std::int8_t* key_codes,
    const std::uint16_t* key_scales, const std::int8_t* value_codes,
    const std::uint16_t* value_scales, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    float scale = 0.0f, long long window = 0,
    FloatStorageWorkspace* workspace = nullptr);
Status kv_cache_scatter_mxfp8_storage(FloatStorageInput key,
                                      FloatStorageInput value, const int* slots,
                                      std::uint8_t* key_cache,
                                      std::uint8_t* value_cache,
                                      long long max_slots, long long count,
                                      long long heads, long long head_dim);
Status kv_cache_gather_mxfp8_storage(const std::uint8_t* key_cache,
                                     const std::uint8_t* value_cache,
                                     const int* indices,
                                     FloatStorageOutput key_out,
                                     FloatStorageOutput value_out,
                                     long long max_slots, long long count,
                                     long long heads, long long head_dim);
Status paged_attention_mxfp8_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    float scale = 0.0f, long long window = 0,
    FloatStorageWorkspace* workspace = nullptr);
Status turboquant_query_transform_storage(
    FloatStorageInput q, const float* signs, FloatStorageOutput transformed,
    long long rows, long long heads, long long head_size,
    FloatStorageWorkspace* workspace = nullptr);
Status paged_attention_turboquant_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const float* key_scale_cache,
    const float* value_scale_cache, const float* key_zero_cache,
    const float* value_centroids, const float* signs, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_size, long long page_size, long long max_blocks,
    int key_bits, bool key_signed, int value_bits, float scale = 0.0f,
    long long window = 0, FloatStorageWorkspace* workspace = nullptr);
Status kv_cache_scatter_bitnet_kv3_storage(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    std::uint8_t* key_cache, std::uint8_t* value_cache, void* key_scale_cache,
    void* value_scale_cache, int* key_zero_cache, int* value_zero_cache,
    long long max_slots, long long count, long long heads, long long head_dim,
    const BitNetKv3Config& config);
Status kv_cache_gather_bitnet_kv3_storage(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, FloatStorageOutput key_out,
    FloatStorageOutput value_out, long long max_slots, long long count,
    long long heads, long long head_dim, const BitNetKv3Config& config);
Status paged_attention_bitnet_kv3_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    const BitNetKv3Config& config, float scale = 0.0f, long long window = 0,
    FloatStorageWorkspace* workspace = nullptr);

enum class QuantFormat;
Status qgemv_storage(QuantFormat format, const void* packed,
                     FloatStorageInput x, FloatStorageOutput y, long long n,
                     long long k);
Status qgemm_storage(QuantFormat format, const void* packed,
                     FloatStorageInput x, FloatStorageOutput y, long long m,
                     long long n, long long k);

}  // namespace quixicore_cpu
