#pragma once

#include <cstdint>
#include <limits>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

enum class QuantFormat;
enum class Float8Format;
struct BitNetKv3Config;

// Portable f32 operation surface shared by the sibling backends. Layouts are
// row-major and all reductions accumulate in f32 or better.

enum class GeluApprox { kErf, kTanh };
enum class UnaryOp {
  kAbs,
  kSign,
  kNegate,
  kStep,
  kTanh,
  kElu,
  kRelu,
  kSigmoid,
  kGelu,
  kGeluQuick,
  kSilu,
  kHardSwish,
  kHardSigmoid,
  kExp,
  kExpm1,
  kSoftplus,
  kGeluErf,
  kXiElu,
  kFloor,
  kCeil,
  kRound,
  kTrunc,
};
struct XiEluParams {
  float alpha_n = 0.0f;
  float alpha_p = 0.0f;
  float beta = 0.0f;
  float eps = 0.0f;
};
enum class GluMode {
  kSwiGlu,
  kGeGlu,
  kReGlu,
  kGlu,
  kGeGluErf,
  kGeGluQuick,
};
enum class LinearActivation { kNone, kGeluErf, kGeluTanh, kSilu, kRelu2 };
enum class MoeScoring { kSoftmax, kSigmoid, kSqrtSoftplus };
enum class PoolMode { kAverage, kMaximum };

// Per-channel absolute maximum over a [tokens,channels] calibration batch.
// running may be null; NaNs propagate and infinities remain infinities.
Status calibration_absmax(const float* x, const float* running, float* out,
                          long long tokens, long long channels);

Status gelu(const float* x, float* y, long long count,
            GeluApprox approx = GeluApprox::kErf);
Status gelu_backward(const float* grad_out, const float* x, float* grad_in,
                     long long count, GeluApprox approx = GeluApprox::kErf);
Status silu(const float* x, float* y, long long count);
Status silu_backward(const float* grad_out, const float* x, float* grad_in,
                     long long count);

// All GGML_UNARY_OP_* f32 semantics. XiEluParams are the unconstrained public
// ggml_xielu parameters; the kernel applies llama's softplus constraints.
Status unary(const float* x, float* y, long long count, UnaryOp op,
             XiEluParams xielu = {});

// Gemma-style final-logit softcap, separate from attention score capping.
Status logits_softcap(const float* logits, float* out, long long count,
                      float cap);

// x is [rows, 2*dim] (gate half followed by value half); y is [rows, dim].
Status glu(const float* x, float* y, long long rows, long long dim,
           GluMode mode = GluMode::kSwiGlu);
Status glu_backward(const float* grad_out, const float* x, float* grad_in,
                    long long rows, long long dim,
                    GluMode mode = GluMode::kSwiGlu);
// Exact GGML_GLU_OP_SWIGLU_OAI split-input rule:
// silu(min(gate,limit), alpha) * (clamp(value,-limit,limit) + 1).
Status swiglu_oai(const float* gate, const float* value, float* y,
                  long long rows, long long dim, float alpha, float limit);
// Split-input sigmoid GLU used by full-attention output gates.
Status sigmoid_mul(const float* gate_logits, const float* values, float* out,
                   long long count);
Status sigmoid_mul_backward(const float* grad_out, const float* gate_logits,
                            const float* values, float* grad_gate,
                            float* grad_values, long long count);

// Numerically stable row-wise softmax over [rows, dim]. x and y may alias.
Status softmax(const float* x, float* y, long long rows, long long dim);

// Row-wise LayerNorm over [rows, hidden]. bias may be null.
Status layer_norm(const float* x, const float* weight, const float* bias,
                  float* y, long long rows, long long hidden,
                  float eps = 1e-5f);

// Fully fused normalization backward passes. Gradient accumulators are
// overwritten, not added to. layer_norm_backward permits a null grad_bias;
// rms_norm_backward returns dweight only.
Status layer_norm_backward(const float* x, const float* weight,
                           const float* grad_out, float* grad_x,
                           float* grad_weight, float* grad_bias, long long rows,
                           long long hidden, float eps = 1e-5f);
Status rms_norm_backward(const float* x, const float* weight,
                         const float* grad_out, float* grad_x,
                         float* grad_weight, long long rows, long long hidden,
                         float eps = 1e-5f);

// Fused residual seams. residual_out may alias x or residual but y must not
// overlap either input.
Status rms_norm_add(const float* x, const float* residual, const float* weight,
                    float* y, float* residual_out, long long rows,
                    long long hidden, float eps = 1e-5f);
Status layer_norm_add(const float* x, const float* residual,
                      const float* weight, const float* bias, float* y,
                      float* residual_out, long long rows, long long hidden,
                      float eps = 1e-5f);
Status rms_norm_residual_next(const float* x, const float* post_weight,
                              const float* residual, const float* next_weight,
                              float* residual_out, float* next_out,
                              long long rows, long long hidden,
                              float eps = 1e-5f);

// C[M,N] = A[M,K] @ B[K,N]. Inputs and output are row-major.
Status dense_gemm(const float* a, const float* b, float* c, long long m,
                  long long n, long long k);
Status dense_gemm_ex(const float* a, const float* b, const float* addend,
                     float* c, long long m, long long n, long long k,
                     bool transpose_a, bool transpose_b, float alpha = 1.0f,
                     float beta = 0.0f);
// Direct F16-adapter LoRA projection. adapter_a is [rank,input_dim] and
// adapter_b is [output_dim,rank], both stored as raw IEEE-FP16 bits. Each
// projection result is rounded to FP16 before the next stage. base may be
// null.
Status lora_apply_direct_f16(const float* x,
                             const std::uint16_t* adapter_a,
                             const std::uint16_t* adapter_b,
                             const float* base, float* out, long long rows,
                             long long input_dim, long long output_dim,
                             long long rank, float scale);
// y [rows,out] = activation(x[rows,in] @ weight[out,in]^T + bias + residual).
// bias/residual may be null.
Status linear_epilogue(const float* x, const float* weight, const float* bias,
                       const float* residual, float* y, long long rows,
                       long long input_dim, long long output_dim,
                       LinearActivation activation = LinearActivation::kNone);
Status decode_swiglu(const float* x, const float* gate_weight,
                     const float* up_weight, const float* gate_bias,
                     const float* up_bias, float* y, long long rows,
                     long long input_dim, long long output_dim);
// Fused Flux-style y=(x@w+bias)*gate+residual, w [K,N]. Optional pointers
// contribute their identity values (zero bias/residual, unit gate).
Status gemm_gate_residual(const float* x, const float* weight,
                          const float* bias, const float* gate,
                          const float* residual, float* y, long long rows,
                          long long output_dim, long long input_dim);
Status flux_gelu(const float* x, const float* weight, const float* bias,
                 float* y, long long rows, long long output_dim,
                 long long input_dim, GeluApprox approx = GeluApprox::kTanh);
Status flux_gate(const float* x, const float* weight, const float* bias,
                 const float* gate, const float* residual, float* y,
                 long long rows, long long output_dim, long long input_dim);
Status decode_linear(const float* x, const float* weight, const float* bias,
                     float* y, long long rows, long long input_dim,
                     long long output_dim, bool gelu = false);
Status decode_linear_residual(const float* x, const float* weight,
                              const float* bias, const float* residual,
                              float* y, long long rows, long long input_dim,
                              long long output_dim);
// Complex GEMM with planar real/imaginary inputs and outputs.
Status complex_gemm(const float* a_real, const float* a_imag,
                    const float* b_real, const float* b_imag, float* c_real,
                    float* c_imag, long long m, long long n, long long k);

// NeoX half-split RoPE over [tokens, heads, head_dim].
Status rope(const float* x, float* y, long long tokens, long long heads,
            long long head_dim, float base, long long pos0);
// Gemma vision RoPE over x [B,H,N,D]. positions [B,N,2] select independent
// x/y cos/sin rows [P,D/4] for the two D/2 channel blocks.
Status vision_rope_2d(const float* x, const float* cosine, const float* sine,
                      const int* positions, float* out, long long batch,
                      long long heads, long long tokens, long long head_dim,
                      long long max_position);
// Qwen global split-half pairing with x/y frequency sections in each half.
Status qwen_vision_rope_2d(
    const float* x, const float* cosine, const float* sine,
    const int* positions, float* out, long long batch, long long heads,
    long long tokens, long long head_dim, long long max_position);

// Scaled dot-product GQA. Q [Hq,Sq,D], K/V [Hkv,Sk,D], O [Hq,Sq,D].
Status attention(const float* q, const float* k, const float* v, float* out,
                 long long query_heads, long long kv_heads,
                 long long query_length, long long kv_length,
                 long long head_dim, bool causal);
// Independent-length batched GQA cross-attention. Q [B,Hq,Tq,D], K/V
// [B,Hkv,Tk,D], lengths [B], optional FP32 bias [B,Hq,Tq,Tk]. scale<=0
// selects 1/sqrt(D); softcap==0 disables tanh score capping.
Status cross_attention(const float* q, const float* k, const float* v,
                       const int* key_lengths, const float* bias, float* out,
                       long long batch, long long query_heads,
                       long long kv_heads, long long query_length,
                       long long key_length, long long head_dim,
                       float scale = 0.0f, float softcap = 0.0f);
// Blocked Conformer relative-position attention. Q/K/V [B,T,H,D], relative K
// [P,H,D], learned per-dimension scale [D], and valid lengths [B].
Status audio_relative_attention(
    const float* q, const float* k, const float* v,
    const float* relative_k, const float* per_dim_scale, const int* lengths,
    float* out, long long batch, long long length, long long heads,
    long long head_dim, long long relative_positions, long long chunk_size,
    long long left_context, long long right_context, float q_scale = 0.0f,
    float k_scale = 0.0f, float softcap = 0.0f);

// Attention forward with row log-sum-exp retained for staged backward or
// state merging. softcap<=0 disables tanh score capping; sinks contributes a
// denominator-only logit per query head when non-null.
Status attention_with_lse(const float* q, const float* k, const float* v,
                          const float* sinks, float* out, float* logsumexp,
                          long long query_heads, long long kv_heads,
                          long long query_length, long long kv_length,
                          long long head_dim, bool causal, float scale = 0.0f,
                          float softcap = 0.0f);
Status attention_backward_prep(const float* out, const float* grad_out,
                               float* delta, long long query_heads,
                               long long query_length, long long head_dim);
Status attention_backward_dq(const float* q, const float* k, const float* v,
                             const float* grad_out, const float* logsumexp,
                             const float* delta, float* grad_q,
                             long long query_heads, long long kv_heads,
                             long long query_length, long long kv_length,
                             long long head_dim, bool causal);
Status attention_backward_dkv(const float* q, const float* k, const float* v,
                              const float* grad_out, const float* logsumexp,
                              const float* delta, float* grad_k, float* grad_v,
                              long long query_heads, long long kv_heads,
                              long long query_length, long long kv_length,
                              long long head_dim, bool causal);
Status merge_attention_states(const float* prefix_out,
                              const float* prefix_logsumexp,
                              const float* suffix_out,
                              const float* suffix_logsumexp, float* out,
                              float* output_logsumexp, long long tokens,
                              long long heads, long long head_dim,
                              long long prefix_tokens);
Status merge_attention_states_fp8(const float* prefix_out,
                                  const float* prefix_logsumexp,
                                  const float* suffix_out,
                                  const float* suffix_logsumexp,
                                  std::uint8_t* out_codes,
                                  float* output_logsumexp, long long tokens,
                                  long long heads, long long head_dim,
                                  long long prefix_tokens, float output_scale);

// Backward for the dense/GQA attention above. All gradients are overwritten.
Status attention_backward(const float* q, const float* k, const float* v,
                          const float* grad_out, float* grad_q, float* grad_k,
                          float* grad_v, long long query_heads,
                          long long kv_heads, long long query_length,
                          long long kv_length, long long head_dim, bool causal);

// Local attention. left_window/right_window are inclusive distances; a
// negative value means unbounded on that side.
Status window_attention(const float* q, const float* k, const float* v,
                        float* out, long long query_heads, long long kv_heads,
                        long long query_length, long long kv_length,
                        long long head_dim, long long left_window,
                        long long right_window);

// Packed variable-length GQA. q [total_q,Hq,D], k/v [total_k,Hkv,D].
Status varlen_attention(const float* q, const float* k, const float* v,
                        const int* cumulative_q, const int* cumulative_k,
                        float* out, long long batch, long long query_heads,
                        long long kv_heads, long long head_dim, bool causal);
Status varlen_build_worklist(const int* cumulative_lengths, int* padded_offsets,
                             int* sequence_lengths, int* tile_sequence,
                             int* tile_local_start, int* tile_count,
                             long long batch, long long max_tiles,
                             int tile_rows = 8);
Status varlen_pad_q(const float* packed, const int* cumulative_lengths,
                    const int* padded_offsets, float* padded_head_major,
                    long long batch, long long heads, long long head_dim,
                    long long total_tokens, long long total_padded);
Status varlen_regather_o(const float* padded_head_major,
                         const int* cumulative_lengths,
                         const int* padded_offsets, float* packed,
                         long long batch, long long heads, long long head_dim,
                         long long total_tokens, long long total_padded);

// Dense attention with optional additive bias [Hq,Sq,Sk] and optional mask
// [Sq,Sk] (zero means masked). This is the semantic core of Swin attention.
Status biased_attention(const float* q, const float* k, const float* v,
                        const float* bias, const std::uint8_t* mask, float* out,
                        long long query_heads, long long kv_heads,
                        long long query_length, long long kv_length,
                        long long head_dim, float scale = 0.0f);
// Swin packed layout qkv [windows,tokens,3,heads,32], output
// [windows,tokens,heads,32]. mask is additive
// [windows_per_image,tokens,tokens] and may be null.
Status swin_attention_d32(const float* qkv, const float* relative_bias,
                          const float* mask, float* out, long long windows,
                          long long tokens, long long heads,
                          long long windows_per_image = 0);

// RoPE using caller-provided tables. positions [tokens], cos/sin
// [max_position,head_dim/2]. interleaved selects adjacent-pair layout.
Status rope_table(const float* x, const float* cosine, const float* sine,
                  const int* positions, float* y, long long tokens,
                  long long heads, long long head_dim, long long max_position,
                  bool interleaved = false);
// Positioned/partial RoPE over [batch,heads,tokens,head_dim]. positions is
// [tokens] when positions_per_batch is false and [batch,tokens] otherwise.
// rotary_dim=0 rotates the complete head; the unrotated tail is copied.
Status rotary_positioned(const float* x, const float* cosine, const float* sine,
                         const int* positions, float* y, long long batch,
                         long long heads, long long tokens, long long head_dim,
                         long long rotary_dim, long long max_position,
                         bool interleaved = false,
                         bool positions_per_batch = false);
// Three-axis temporal/height/width RoPE over [batch,heads,tokens,head_dim].
// positions is [3,tokens] or [batch,3,tokens]. sections contains three
// nonnegative rotary-pair counts. M-RoPE always uses split-half pairing.
Status mrope(const float* x, const float* cosine, const float* sine,
             const int* positions, const int* sections, float* y,
             long long batch, long long heads, long long tokens,
             long long head_dim, long long rotary_dim, long long max_position,
             bool section_interleaved = false,
             bool positions_per_batch = false);
// RoPE seam used by latent/indexer projections that arrive as adjacent pairs
// [a0,b0,a1,b1,...] but are consumed as split halves [real...,imag...]. This
// operation is in-place compatible.
Status rope_interleaved_to_split(const float* x, float* y, long long tokens,
                                 long long heads, long long head_dim,
                                 float base, long long pos0);

// Normalize packed Q and K heads then apply table RoPE; V heads pass through.
// qkv/y [tokens,(query_heads+key_heads+value_heads)*head_dim].
Status qk_norm_rope(const float* qkv, const float* q_weight,
                    const float* k_weight, const float* cosine,
                    const float* sine, const int* positions, float* y,
                    long long tokens, long long query_heads,
                    long long key_heads, long long value_heads,
                    long long head_dim, long long max_position,
                    float eps = 1e-5f, bool interleaved = false);
// Generic packed QKV preparation with partial or multimodal RoPE. A null
// mrope_sections selects one-dimensional positions [tokens]; otherwise it
// selects positions [3,tokens] and split-half M-RoPE. norm_weight_offset
// expresses weight versus (offset + weight) without model-specific flags.
Status qk_norm_rope_positioned(
    const float* qkv, const float* q_weight, const float* k_weight,
    const float* cosine, const float* sine, const int* positions, float* y,
    long long tokens, long long query_heads, long long key_heads,
    long long value_heads, long long head_dim, long long rotary_dim,
    long long max_position, float eps = 1e-5f, bool interleaved = false,
    float norm_weight_offset = 0.0f, const int* mrope_sections = nullptr,
    bool section_interleaved = false);
// Split-output CPU counterpart of the sibling KV-f16 seam. CPU outputs remain
// f32 and are shaped [T,Hq,D], [T,Hk,D], [T,Hv,D].
Status qk_norm_rope_split(const float* qkv, const float* q_weight,
                          const float* k_weight, const float* cosine,
                          const float* sine, const int* positions, float* q_out,
                          float* k_out, float* v_out, long long tokens,
                          long long query_heads, long long key_heads,
                          long long value_heads, long long head_dim,
                          long long max_position, float eps = 1e-5f,
                          bool interleaved = false, bool gemma_weight = false);

// Quantized dense attention: K/V vectors use qgemv packed blocks independently
// at [B,H,S,blocks].
Status quantized_attention(const float* q, const void* packed_k,
                           const void* packed_v, float* out, long long batch,
                           long long heads, long long sequence,
                           long long head_dim, QuantFormat format,
                           bool causal = false);

// Serving RoPE seams. Tables use split-half/GPT-NeoX layout. slot_mapping < 0
// skips a cache write; caches are flattened [slots,heads,head_dim].
Status rope_q_norm(const float* q, const float* cosine, const float* sine,
                   const int* positions, const float* norm_weight, float* out,
                   long long tokens, long long heads, long long head_dim,
                   long long max_position, bool do_norm = false,
                   bool gemma_weight = false, float eps = 1e-6f);
Status rope_kv_insert(const float* k, const float* v, const float* cosine,
                      const float* sine, const int* positions,
                      const int* slot_mapping, const float* norm_weight,
                      float* key_cache, float* value_cache, long long tokens,
                      long long slots, long long kv_heads, long long head_dim,
                      long long max_position, bool do_norm = false,
                      bool gemma_weight = false, float eps = 1e-6f);

// One-token decode fusion over contiguous [B,Hkv,cache_length,D] caches.
// context_lengths is the append index; the call writes K/V then attends over
// tokens [0,context].
Status decode_cache_attention(
    const float* q, const float* new_k, const float* new_v, const float* cosine,
    const float* sine, const int* positions, const int* context_lengths,
    const float* q_weight, const float* k_weight, float* key_cache,
    float* value_cache, float* out, long long batch, long long query_heads,
    long long kv_heads, long long cache_length, long long head_dim,
    long long max_position, float eps = 1e-6f, bool do_q_norm = false,
    bool do_k_norm = false, bool gemma_weight = false, float scale = 0.0f);

// Paged decode attention. q/out [B,Hq,D], cache [blocks,page,Hkv,D],
// block_table [B,max_blocks], context_lens [B]. window<=0 means full context.
Status paged_attention(const float* q, const float* key_cache,
                       const float* value_cache, const int* block_table,
                       const int* context_lens, float* out,
                       long long cache_blocks, long long batch,
                       long long query_heads, long long kv_heads,
                       long long head_dim, long long page_size,
                       long long max_blocks, float scale = 0.0f,
                       long long window = 0);

// Extended paged decode: optional per-query-head block mask, ALiBi slope, and
// attention-sink logit. softcap==0 disables score capping.
Status paged_attention_advanced(
    const float* q, const float* key_cache, const float* value_cache,
    const int* block_table, const int* context_lens, const int* block_mask,
    const float* alibi_slopes, const float* sinks, float* out,
    long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, float scale = 0.0f, long long window = 0,
    float softcap = 0.0f);
Status paged_attention_fp8(const float* q, const std::uint8_t* key_cache,
                           const std::uint8_t* value_cache,
                           const int* block_table, const int* context_lens,
                           const float* key_scale, const float* value_scale,
                           float* out, long long cache_blocks, long long batch,
                           long long query_heads, long long kv_heads,
                           long long head_dim, long long page_size,
                           long long max_blocks, Float8Format format,
                           float scale = 0.0f, long long window = 0,
                           float softcap = 0.0f);
// QuixiCore Q8_0 KV-cache ABI: signed int8 codes and one raw IEEE FP16 scale
// per 32 consecutive head-dimension values. Cache planes are laid out
// [blocks,page,kv_heads,head_dim] and [blocks,page,kv_heads,head_dim/32].
Status paged_attention_q8_0(const float* q, const std::int8_t* key_codes,
                            const std::uint16_t* key_scales,
                            const std::int8_t* value_codes,
                            const std::uint16_t* value_scales,
                            const int* block_table, const int* context_lens,
                            float* out, long long cache_blocks, long long batch,
                            long long query_heads, long long kv_heads,
                            long long head_dim, long long page_size,
                            long long max_blocks, float scale = 0.0f,
                            long long window = 0);
// Canonical MXFP8 cache blocks are [E8M0 scale, 32 E4M3FN codes], repeated
// over the head dimension. Supported cache head dimensions are 64 and 128.
Status paged_attention_mxfp8(const float* q, const std::uint8_t* key_cache,
                             const std::uint8_t* value_cache,
                             const int* block_table, const int* context_lens,
                             float* out, long long cache_blocks,
                             long long batch, long long query_heads,
                             long long kv_heads, long long head_dim,
                             long long page_size, long long max_blocks,
                             float scale = 0.0f, long long window = 0);
// TurboQuant direct decode. Packed cache rows and scale/zero metadata use the
// canonical layout documented by turboquant_encode.
Status turboquant_query_transform(const float* q, const float* signs,
                                  float* transformed, long long rows,
                                  long long heads, long long head_size);
Status paged_attention_turboquant(
    const float* q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const float* key_scale_cache,
    const float* value_scale_cache, const float* key_zero_cache,
    const float* value_centroids, const float* signs, const int* block_table,
    const int* context_lens, float* out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_size, long long page_size, long long max_blocks,
    int key_bits, bool key_signed, int value_bits, float scale = 0.0f,
    long long window = 0);
// BitNet a4.8 KV3 cache rows are one low-bit-first 3-bit stream per head.
// Scale storage and signed/zero-point interpretation are explicit in config;
// scales and integer zero points are [slots,heads,D/group_size].
Status paged_attention_bitnet_kv3(
    const float* q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, const int* block_table,
    const int* context_lens, float* out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    const BitNetKv3Config& config, float scale = 0.0f, long long window = 0);
// Layout adapter for the vLLM/Metal xcache representation. key_cache is
// [blocks,Hkv,D/x,page,x], value_cache is [blocks,Hkv,D,page].
Status paged_attention_xcache(const float* q, const float* key_cache,
                              const float* value_cache, const int* block_table,
                              const int* context_lens, float* out,
                              long long cache_blocks, long long batch,
                              long long query_heads, long long kv_heads,
                              long long head_dim, long long page_size,
                              long long max_blocks, long long vector_width,
                              float scale = 0.0f);
// Shared contiguous prefix plus request-specific paged suffix, merged as one
// exact softmax. prefix K/V are [prefix_length,kv_heads,head_dim].
Status cascade_attention(const float* q, const float* prefix_k,
                         const float* prefix_v, const float* key_cache,
                         const float* value_cache, const int* block_table,
                         const int* context_lens, float* out,
                         long long cache_blocks, long long batch,
                         long long query_heads, long long kv_heads,
                         long long head_dim, long long prefix_length,
                         long long page_size, long long max_blocks,
                         float scale = 0.0f);
// FP8 shared-prefix variant; the request-specific paged suffix remains f32.
Status cascade_attention_fp8(
    const float* q, const std::uint8_t* prefix_k, const std::uint8_t* prefix_v,
    const float* key_cache, const float* value_cache, const int* block_table,
    const int* context_lens, const float* key_scale, const float* value_scale,
    float* out, long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long prefix_length,
    long long page_size, long long max_blocks, Float8Format format,
    float scale = 0.0f);
Status cascade_attention_multi(
    const float* q, const float* const* prefix_k, const float* const* prefix_v,
    const long long* prefix_lengths, long long levels, const float* key_cache,
    const float* value_cache, const int* block_table, const int* context_lens,
    float* out, long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, float scale = 0.0f);

// MLA latent decode. q [B,H,W], cache [blocks,page,W], out [B,H,value_dim].
// Scores use all W=latent_dim+rope_dim values; values use the latent prefix.
Status mla_decode(const float* q, const float* kv_cache, const int* block_table,
                  const int* context_lens, float* out, long long cache_blocks,
                  long long batch, long long heads, long long latent_dim,
                  long long rope_dim, long long page_size, long long max_blocks,
                  float scale = 0.0f);
// Quantized MLA weight absorption. packed_kv_b is
// [heads*(nope_dim+value_dim),latent_dim]. Instead of reconstructing K/V for
// every cached token, the kernel projects each non-RoPE query into latent
// space, attends over the latent/RoPE caches, then projects the mixed latent
// vector into value space. Cache vectors are paged through block_table.
Status quantized_mla_decode_absorbed(
    QuantFormat format, const void* packed_kv_b, const float* q,
    const float* latent_cache, const float* rope_cache, const int* block_table,
    const int* context_lengths, float* out, long long cache_blocks,
    long long batch, long long heads, long long latent_dim, long long nope_dim,
    long long rope_dim, long long value_dim, long long page_size,
    long long max_blocks, float scale = 0.0f);
// Sparse DSA counterpart. token_indices is [batch,max_topk] and entries are
// logical positions resolved through the same page table.
Status quantized_mla_decode_absorbed_sparse(
    QuantFormat format, const void* packed_kv_b, const float* q,
    const float* latent_cache, const float* rope_cache, const int* block_table,
    const int* token_indices, const int* topk_lengths, float* out,
    long long cache_blocks, long long batch, long long heads,
    long long latent_dim, long long nope_dim, long long rope_dim,
    long long value_dim, long long page_size, long long max_blocks,
    long long max_topk, float scale = 0.0f);
Status mla_q_norm_rope(const float* q, const float* cosine, const float* sine,
                       const int* positions, const float* norm_weight,
                       float* out, long long tokens, long long heads,
                       long long nope_dim, long long rope_dim,
                       long long max_position, int norm_mode,
                       float eps = 1e-6f);
Status mla_kv_insert(const float* latent, const float* key_rope,
                     const float* cosine, const float* sine,
                     const int* positions, const int* slot_mapping,
                     const float* norm_weight, float* cache, long long tokens,
                     long long slots, long long latent_dim, long long rope_dim,
                     long long max_position, int norm_mode, float eps = 1e-6f);
Status mla_kv_insert_fp8(const float* kv, const int* slot_mapping,
                         std::uint8_t* code_cache, float* scale_cache,
                         long long tokens, long long slots, long long width,
                         long long group_size, Float8Format format,
                         bool power_of_two_scale = true);
Status mla_decode_fp8(const float* q, const std::uint8_t* code_cache,
                      const float* scale_cache, const int* block_table,
                      const int* context_lengths, float* out,
                      long long cache_blocks, long long batch, long long heads,
                      long long width, long long page_size,
                      long long max_blocks, long long group_size,
                      Float8Format format, float scale = 0.0f);
Status mla_decode_fp8_sparse(const float* q, const std::uint8_t* code_cache,
                             const float* scale_cache, const int* block_table,
                             const int* token_indices, const int* topk_lengths,
                             float* out, long long cache_blocks,
                             long long batch, long long heads, long long width,
                             long long page_size, long long max_blocks,
                             long long max_topk, long long group_size,
                             Float8Format format, float scale = 0.0f);

// Sampling is exactly reproducible for a fixed seed and row.
Status argmax_sample(const float* logits, int* out, long long rows,
                     long long vocab);
Status sample_categorical(const float* logits, int* out, long long rows,
                          long long vocab, float temperature = 1.0f,
                          std::uint32_t seed = 0);
Status top_k_sample(const float* logits, int* out, long long rows,
                    long long vocab, int k, float temperature = 1.0f,
                    std::uint32_t seed = 0);
Status top_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float p, float temperature = 1.0f,
                    std::uint32_t seed = 0);
// Select the K greatest values per row using a linear-time threshold partition,
// then emit their indices in original column order. Equal-threshold entries use
// the lowest columns first. This is the stable selection seam used by DSA.
Status threshold_topk_indices(const float* scores, int* indices, long long rows,
                              long long width, int k);
Status min_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float min_p, float temperature = 1.0f,
                    std::uint32_t seed = 0);
Status typical_p_sample(const float* logits, int* out, long long rows,
                        long long vocab, float p, float temperature = 1.0f,
                        std::uint32_t seed = 0);

// In-place-compatible logit processors. A masked entry is set to -infinity.
Status apply_token_bitmask(const float* logits, const std::uint8_t* bitmask,
                           float* out, long long rows, long long vocab);
Status apply_bad_words(const float* logits, const int* bad_ids, float* out,
                       long long rows, long long vocab, long long bad_count);
Status apply_repetition_penalty(const float* logits, const int* previous,
                                const int* lengths, float* out, long long rows,
                                long long vocab, long long max_previous,
                                float repetition_penalty,
                                float presence_penalty = 0.0f,
                                float frequency_penalty = 0.0f);
Status top_k_renorm(const float* probabilities, float* out, long long rows,
                    long long vocab, int k);
Status top_p_renorm(const float* probabilities, float* out, long long rows,
                    long long vocab, float p);
// Metal sampler-zoo transforms. Logit transforms emit logits/temperature and
// use -infinity for masked entries. Probability transforms preserve row sums.
Status quadratic_transform(const float* logits, float* out, long long rows,
                           long long vocab, float factor, float curve = 1.0f,
                           float temperature = 1.0f);
Status top_nsigma_mask(const float* logits, float* out, long long rows,
                       long long vocab, float nsigma, float temperature = 1.0f);
Status top_a_mask(const float* logits, float* out, long long rows,
                  long long vocab, float top_a, float temperature = 1.0f);
Status epsilon_cutoff_mask(const float* logits, float* out, long long rows,
                           long long vocab, float epsilon,
                           float temperature = 1.0f);
Status eta_cutoff_mask(const float* logits, float* out, long long rows,
                       long long vocab, float eta, float temperature = 1.0f);
Status xtc_mask(const float* logits, float* out, long long rows,
                long long vocab, float threshold, float probability,
                std::uint32_t seed = 0, float temperature = 1.0f);
Status skew_transform(const float* probabilities, float* out, long long rows,
                      long long vocab, float skew);
Status no_repeat_ngram_mask(const float* logits, const int* previous,
                            const int* lengths, float* out, long long rows,
                            long long vocab, long long history, int ngram_size,
                            float temperature = 1.0f);
Status dry_penalty(const float* logits, const int* previous, const int* lengths,
                   const int* breakers, float* out, long long rows,
                   long long vocab, long long history, long long breaker_count,
                   float multiplier, float base = 1.75f, int allowed_length = 2,
                   int range = 0, int max_ngram = 64, int max_occurrences = 64,
                   int early_exit_match_length = 64, float temperature = 1.0f);

// One beam-search step. logits [B*beam,V], cumulative [B,beam].
Status beam_search_step(const float* logits, const float* cumulative,
                        int* next_token, int* parent_beam,
                        float* next_cumulative, long long batch, long long beam,
                        long long vocab);

// vLLM-style linear speculative verification. out_tokens [B,S+1] is -1
// padded; accepted_count [B] excludes the recovered/bonus token.
Status speculative_verify(const int* draft_tokens, const float* draft_probs,
                          const float* target_probs, const int* bonus_tokens,
                          const float* accept_uniforms, int* out_tokens,
                          int* accepted_count, long long batch,
                          long long draft_length, long long vocab,
                          std::uint32_t seed = 0);
Status build_dynamic_tree(const int* parents, int* first_child,
                          int* next_sibling, int* positions, long long batch,
                          long long nodes);
Status speculative_verify_tree(const int* draft_tokens,
                               const float* target_probs,
                               const int* first_child, const int* next_sibling,
                               const int* tree_valid, int* accepted_indices,
                               int* accepted_tokens, int* accepted_count,
                               long long batch, long long nodes,
                               long long vocab, std::uint32_t seed = 0);
Status speculative_compact(const int* out_tokens, const int* accepted_count,
                           const int* sequence_lengths, int* packed_tokens,
                           int* packed_positions, int* cumulative_accepted,
                           long long batch, long long draft_length);
Status speculative_update_kv_meta(const int* sequence_lengths,
                                  const int* accepted_count,
                                  int* new_sequence_lengths, long long batch);
Status rejection_greedy_sample(
    const int* cumulative_drafts, const int* draft_tokens,
    const int* target_argmax, const int* bonus_tokens,
    const std::uint8_t* greedy_mask, int* out_tokens, long long batch,
    long long total_drafts, long long max_draft, bool use_greedy_mask = false);
Status rejection_random_sample(
    const int* cumulative_drafts, const int* draft_tokens,
    const float* draft_probs, const float* target_probs,
    const int* bonus_tokens, const int* recovered_tokens,
    const float* uniform_probs, const std::uint8_t* greedy_mask,
    int* out_tokens, long long batch, long long total_drafts,
    long long max_draft, long long vocab, bool no_draft_probs = false,
    bool use_greedy_mask = false);
Status sample_recovered_tokens(
    const int* cumulative_drafts, const int* draft_tokens,
    const float* draft_probs, const float* target_probs,
    const float* inverse_noise, int* recovered_tokens, long long batch,
    long long total_drafts, long long vocab, bool no_draft_probs = false);
Status eagle_prepare_inputs_padded(const int* cumulative_drafts,
                                   const int* valid_sampled_count,
                                   const int* query_start_locations,
                                   int* token_indices_to_sample,
                                   int* rejected_count, long long batch);
Status eagle_prepare_next_token_padded(
    const int* sampled_tokens, const std::uint8_t* discard_mask,
    const int* backup_tokens, int* next_tokens, int* valid_sampled_count,
    long long batch, long long sampled_per_request, int vocab);
Status eagle_step_slot_mapping_metadata(
    const int* positions, const int* block_table, const int* sequence_lengths,
    int* clamped_positions, int* slot_mapping, int* new_sequence_lengths,
    long long batch, long long input_batch, long long max_blocks,
    int block_size, int max_model_length, int pad_id);
Status eagle_expand_int32(const int* input, const int* cumulative_tokens,
                          int* output, long long batch, long long total,
                          int replace_from, int replace_to);
Status eagle_prepare_next_token_int64(
    const long long* sampled_tokens, const std::uint8_t* discard_mask,
    const long long* backup_tokens, long long* next_tokens,
    long long* valid_sampled_count, long long batch,
    long long sampled_per_request, long long vocab);
Status eagle_expand_int64(const long long* input,
                          const long long* cumulative_tokens, long long* output,
                          long long batch, long long total,
                          long long replace_from, long long replace_to);
Status eagle_step_slot_mapping_int64(
    long long* sequence_lengths, const long long* positions,
    const int* block_table, long long* clamped_positions,
    long long* slot_mapping, long long batch, long long input_batch,
    long long max_blocks, long long block_size, long long max_model_length,
    long long pad_id);
Status copy_and_expand_eagle(
    const long long* target_token_ids, const long long* target_positions,
    const long long* next_token_ids, const int* query_start_locations,
    const int* query_end_locations, long long* output_input_ids,
    long long* output_positions, std::uint8_t* rejected_mask,
    std::uint8_t* masked_token_mask, int* new_token_indices,
    int* hidden_state_mapping, long long batch, long long input_tokens,
    long long output_tokens, long long padding_token_id,
    long long parallel_drafting_token_id, long long padding_slots_per_request,
    bool shift_input_ids);

Status embedding_lookup(const float* table, const int* ids, float* out,
                        long long vocab, long long count, long long dim);
// BERT-family token + type preparation. Each invalid id contributes zero.
Status embedding_lookup_types(const int* token_ids, const int* type_ids,
                              const float* token_table,
                              const float* type_table, float* out,
                              long long token_vocab, long long type_vocab,
                              long long count, long long dim,
                              float token_scale = 1.0f);
Status embedding_backward(const int* ids, const float* grad_out,
                          float* grad_table, long long vocab, long long count,
                          long long dim);
Status embedding_backward_sorted(const int* sorted_ids, const int* permutation,
                                 const float* grad_out, float* grad_table,
                                 long long vocab, long long count,
                                 long long dim, float scale = 1.0f);
// Builds the source map consumed by merge_multimodal_spans. Positions outside
// every span receive -1; overlapping spans use the first span.
Status build_multimodal_source_map(const int* span_offsets,
                                   const int* span_lengths,
                                   const int* modal_starts, int* source_map,
                                   long long spans, long long text_tokens);
Status merge_multimodal_spans(const float* text, const float* modal,
                              const int* source_map, float* out,
                              long long text_tokens, long long modal_tokens,
                              long long dim);
Status mean_pool_rms_l2(const float* x, const float* weight, const int* lengths,
                        float* out, long long batch, long long sequence,
                        long long hidden, float eps = 1e-5f);
// Mask-aware terminal embedding pool. Any nonzero mask entry participates;
// an empty batch row emits exactly zero.
Status masked_mean_pool_rms_l2(const float* x, const int* mask,
                               const float* weight, float* out,
                               long long batch, long long sequence,
                               long long hidden, float eps = 1e-5f);
// Packed-table gather operations. table uses the row-major packed layout of
// qgemv for the selected format; add may be null when use_add is false.
Status quantized_embedding(const void* table, const int* ids, const float* add,
                           float* out, long long vocab, long long count,
                           long long dim, QuantFormat quant_format,
                           float scale = 1.0f, bool use_add = false);
Status dequant_gather(const void* table, const int* ids, float* out,
                      long long vocab, long long count, long long dim,
                      QuantFormat quant_format, float scale = 1.0f);
Status quantized_embedding_bag(
    const void* table, const int* ids, const long long* offsets,
    const float* sample_weights, float* out, long long vocab,
    long long id_count, long long bags, long long dim, QuantFormat quant_format,
    float scale = 1.0f, bool use_weights = false, bool mean_mode = false);
Status kv_cache_scatter(float* cache, const float* src, const int* slots,
                        long long max_slots, long long count,
                        long long row_width);
Status kv_cache_gather(const float* cache, const int* indices, float* out,
                       long long max_slots, long long count,
                       long long row_width);
Status kv_cache_copy_blocks(float* key_cache, float* value_cache,
                            const long long* block_pairs, long long pair_count,
                            long long block_count,
                            long long elements_per_block);
Status beam_build_copy_pairs(const int* parent_beam, const int* block_table,
                             const int* sequence_lengths,
                             long long* block_pairs, long long batch,
                             long long beam_width, long long max_blocks,
                             int block_size);
Status beam_remap_block_table(const int* block_table, const int* parent_beam,
                              int* remapped_block_table, long long batch,
                              long long beam_width, long long max_blocks);
Status kv_cache_scales(const float* key, const float* value, float* key_scale,
                       float* value_scale, long long count);
Status kv_cache_scale_update(const float* key, const float* value,
                             float old_key_scale, float old_value_scale,
                             float* new_key_scale, float* new_value_scale,
                             long long count);
Status kv_cache_scatter_fp8(const float* key, const float* value,
                            const int* slots, const float* key_scale,
                            const float* value_scale, std::uint8_t* key_cache,
                            std::uint8_t* value_cache, long long max_slots,
                            long long count, long long heads,
                            long long head_dim, Float8Format format);
Status kv_cache_gather_fp8(const std::uint8_t* key_cache,
                           const std::uint8_t* value_cache, const int* indices,
                           const float* key_scale, const float* value_scale,
                           float* key_out, float* value_out,
                           long long max_slots, long long count,
                           long long heads, long long head_dim,
                           Float8Format format);
// Q8_0 scatter creates zero-filled code/scale planes before writing active
// logical slots. Negative slots are inactive. Gather packs sequences according
// to cumulative_lengths and returns zero rows for negative block-table entries.
Status kv_cache_scatter_q8_0(
    const float* key, const float* value, const int* slots,
    std::int8_t* key_codes, std::uint16_t* key_scales, std::int8_t* value_codes,
    std::uint16_t* value_scales, long long cache_blocks, long long count,
    long long heads, long long head_dim, long long page_size);
Status kv_cache_gather_q8_0(
    const std::int8_t* key_codes, const std::uint16_t* key_scales,
    const std::int8_t* value_codes, const std::uint16_t* value_scales,
    const int* block_table, const int* cumulative_lengths, float* key_out,
    float* value_out, long long cache_blocks, long long num_tokens,
    long long sequences, long long heads, long long head_dim,
    long long page_size, long long max_blocks);
// Functional block remap: clone all four input planes, then copy each
// source/destination pair from the immutable inputs into the outputs.
Status kv_cache_copy_blocks_q8_0(
    const std::int8_t* key_codes, const std::uint16_t* key_scales,
    const std::int8_t* value_codes, const std::uint16_t* value_scales,
    std::int8_t* key_codes_out, std::uint16_t* key_scales_out,
    std::int8_t* value_codes_out, std::uint16_t* value_scales_out,
    const long long* block_pairs, long long pair_count, long long cache_blocks,
    long long page_size, long long heads, long long head_dim);
Status kv_cache_scatter_mxfp8(const float* key, const float* value,
                              const int* slots, std::uint8_t* key_cache,
                              std::uint8_t* value_cache, long long max_slots,
                              long long count, long long heads,
                              long long head_dim);
Status kv_cache_gather_mxfp8(const std::uint8_t* key_cache,
                             const std::uint8_t* value_cache,
                             const int* indices, float* key_out,
                             float* value_out, long long max_slots,
                             long long count, long long heads,
                             long long head_dim);
Status kv_cache_scatter_bitnet_kv3(
    const float* key, const float* value, const int* slots,
    std::uint8_t* key_cache, std::uint8_t* value_cache, void* key_scale_cache,
    void* value_scale_cache, int* key_zero_cache, int* value_zero_cache,
    long long max_slots, long long count, long long heads, long long head_dim,
    const BitNetKv3Config& config);
Status kv_cache_gather_bitnet_kv3(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, float* key_out, float* value_out,
    long long max_slots, long long count, long long heads, long long head_dim,
    const BitNetKv3Config& config);

Status dropout(const float* x, float* y, long long count, float probability,
               std::uint32_t seed);
Status dropout_backward(const float* grad_out, float* grad_in, long long count,
                        float probability, std::uint32_t seed);
Status add(const float* x, const float* y, float* out, long long count);
Status add_scalar(const float* x, float value, float* out, long long count);
Status subtract(const float* x, const float* y, float* out, long long count);
Status multiply(const float* x, const float* y, float* out, long long count);
Status divide(const float* x, const float* y, float* out, long long count);
Status scale(const float* x, float value, float* out, long long count);
Status clamp(const float* x, float minimum, float maximum, float* out,
             long long count);
// Scalar-bounds clamp with framework semantics: infinite bounds are allowed,
// NaN bounds and reversed intervals are rejected.
Status value_clip(const float* x, float minimum, float maximum, float* out,
                  long long count);
Status square(const float* x, float* out, long long count);
Status square_root(const float* x, float* out, long long count);
Status logarithm(const float* x, float* out, long long count);
Status sine(const float* x, float* out, long long count);
Status cosine(const float* x, float* out, long long count);
Status leaky_relu(const float* x, float negative_slope, float* out,
                  long long count);
Status fill(float* out, long long count, float value);
Status arange(float start, float step, float* out, long long count);
Status cumulative_sum(const float* x, float* out, long long rows,
                      long long dim);
Status reduce_sum_all(const float* x, float* out, long long count);
Status reduce_mean(const float* x, float* out, long long rows, long long dim);
Status count_equal(const std::int32_t* x, const std::int32_t* y,
                   long long count, long long* out);
Status argsort(const float* x, int* indices, long long rows, long long dim,
               bool descending = false);
// Inputs are [outer,a_axis,inner] and [outer,b_axis,inner].
Status concat(const float* a, const float* b, float* out, long long outer,
              long long a_axis, long long b_axis, long long inner);
Status repeat_2d(const float* x, float* out, long long source_rows,
                 long long source_cols, long long output_rows,
                 long long output_cols);
Status repeat_backward_2d(const float* grad_out, float* grad_in,
                          long long source_rows, long long source_cols,
                          long long output_rows, long long output_cols);
Status diag_embed(const float* diagonal, float* out, long long batch,
                  long long dim);
Status diag_mask(const float* x, float* out, long long rows, long long cols,
                 long long past, bool use_negative_infinity);
Status triangular_fill(const float* x, float* out, long long rows,
                       long long cols, long long diagonal, bool upper,
                       float fill_value);
Status roll_2d(const float* x, float* out, long long rows, long long cols,
               long long row_shift, long long col_shift);
Status pad_2d(const float* x, float* out, long long rows, long long cols,
              long long top, long long bottom, long long left, long long right,
              float value = 0.0f);
Status pad_reflect_1d(const float* x, float* out, long long rows,
                      long long length, long long left, long long right);
Status upscale_nearest_2d(const float* x, float* out, long long channels,
                          long long input_height, long long input_width,
                          long long scale_height, long long scale_width);
Status group_norm(const float* x, const float* weight, const float* bias,
                  float* out, long long batch, long long channels,
                  long long spatial, long long groups, float eps = 1e-5f);
Status l2_normalize(const float* x, float* out, long long rows, long long dim,
                    float eps = 1e-12f);
Status softmax_backward(const float* grad_out, const float* softmax_output,
                        float* grad_in, long long rows, long long dim);
Status rope_backward(const float* grad_out, float* grad_in, long long tokens,
                     long long heads, long long head_dim, float base,
                     long long pos0);
Status outer_product(const float* x, const float* y, float* out, long long rows,
                     long long cols);
Status set_rows(const float* source, const int* row_ids, float* destination,
                long long source_rows, long long destination_rows,
                long long row_width);
Status accumulate(float* destination, const float* source, long long count,
                  float alpha = 1.0f);
Status sgd(float* parameters, const float* gradients, long long count,
           float learning_rate, float weight_decay = 0.0f);
Status add_id(const float* x, const float* rows, const int* ids, float* out,
              long long count, long long row_count, long long width);
Status tensor_copy(const float* source, float* destination, long long count);
// Copy base to output (unless aliased), then set a strided 4-D view from a
// contiguous update. Destination strides and offset are measured in elements;
// stride 0 is implicitly one.
Status tensor_set_4d(const float* base, const float* update, float* output,
                     long long output_count, long long n0, long long n1,
                     long long n2, long long n3, long long stride1,
                     long long stride2, long long stride3, long long offset);
// NCHW image -> [N,OH,OW,C*KH*KW]. OH/OW follow the standard convolution
// formula from the supplied stride, padding, and dilation.
Status im2col_2d(const float* image, float* columns, long long batch,
                 long long channels, long long input_height,
                 long long input_width, long long kernel_height,
                 long long kernel_width, long long stride_height,
                 long long stride_width, long long pad_height,
                 long long pad_width, long long dilation_height = 1,
                 long long dilation_width = 1);
Status col2im_2d(const float* columns, float* image, long long batch,
                 long long channels, long long input_height,
                 long long input_width, long long kernel_height,
                 long long kernel_width, long long stride_height,
                 long long stride_width, long long pad_height,
                 long long pad_width, long long dilation_height = 1,
                 long long dilation_width = 1);
// Scatter-add [time_in,channels,kernel] columns to [channels,time_out].
Status col2im_1d(const float* columns, float* signal, long long time_in,
                 long long channels, long long kernel, long long stride,
                 long long padding);
Status im2col_3d(const float* volume, float* columns, long long batch,
                 long long channels, long long input_depth,
                 long long input_height, long long input_width,
                 long long kernel_depth, long long kernel_height,
                 long long kernel_width, long long stride_depth,
                 long long stride_height, long long stride_width,
                 long long pad_depth, long long pad_height, long long pad_width,
                 long long dilation_depth = 1, long long dilation_height = 1,
                 long long dilation_width = 1);
// weights are [out_channels,in_channels,KH,KW], input/output are NCHW.
Status conv2d(const float* input, const float* weights, const float* bias,
              float* output, long long batch, long long input_channels,
              long long output_channels, long long input_height,
              long long input_width, long long kernel_height,
              long long kernel_width, long long stride_height = 1,
              long long stride_width = 1, long long pad_height = 0,
              long long pad_width = 0, long long dilation_height = 1,
              long long dilation_width = 1);
// Depthwise weights are [channels,multiplier,KH,KW].
Status depthwise_conv2d(const float* input, const float* weights,
                        const float* bias, float* output, long long batch,
                        long long channels, long long multiplier,
                        long long input_height, long long input_width,
                        long long kernel_height, long long kernel_width,
                        long long stride_height = 1, long long stride_width = 1,
                        long long pad_height = 0, long long pad_width = 0,
                 long long dilation_height = 1,
                 long long dilation_width = 1);
// NHWC image -> [B,OH*OW,KH*KW*C], matching transformer patch-token order.
Status extract_patches_2d(const float* input, float* output, long long batch,
                          long long input_height, long long input_width,
                          long long channels, long long kernel_height,
                          long long kernel_width, long long stride_height,
                          long long stride_width, long long pad_height = 0,
                          long long pad_width = 0);
// NTHWC volume -> [B,OT*OH*OW,KT*KH*KW*C].
Status extract_patches_3d(
    const float* input, float* output, long long batch, long long input_frames,
    long long input_height, long long input_width, long long channels,
    long long kernel_frames, long long kernel_height, long long kernel_width,
    long long stride_frames, long long stride_height, long long stride_width,
    long long pad_frames = 0, long long pad_height = 0,
    long long pad_width = 0);
// Resize a learned [IH,IW,C] position table with bilinear interpolation.
Status interpolate_position_2d(const float* table, float* output,
                               long long input_height, long long input_width,
                               long long output_height, long long output_width,
                               long long channels,
                               bool align_corners = false);
// NHWC average pooling. ceil_mode retains the final partial spatial window.
Status avg_pool2d_tokens(const float* input, float* output, long long batch,
                         long long input_height, long long input_width,
                         long long channels, long long kernel_height,
                         long long kernel_width, long long stride_height,
                         long long stride_width, bool ceil_mode = false);
// Sum independent x/y learned tables [2,P,D] for ids [B,N,2]. Invalid or
// masked tokens emit zero.
Status factorized_position_2d(const int* position_ids, const float* table,
                              const int* valid_mask, float* output,
                              long long batch, long long tokens,
                              long long max_position, long long channels);
// Coordinate-bucketed vision pooling. Input [B,N,D], ids [B,N,2], output
// [B,O,D] in FP32 with a matching int32 validity mask.
Status pool_tokens_by_position(
    const float* input, const int* position_ids, const int* valid_mask,
    float* output, int* output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width);
// Patch extraction followed by FP32 projection. weights [O,KH,KW,C], output
// [B,OH*OW,O], bias optional.
Status vision_patch_projection(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_height, long long input_width,
    long long input_channels, long long output_channels,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, long long pad_height = 0,
    long long pad_width = 0);
// Temporal/spatial patch extraction followed by FP32 projection. weights are
// [O,KT,KH,KW,C], output is [B,OT*OH*OW,O], and bias is optional.
Status vision_patch_projection_3d(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_frames, long long input_height,
    long long input_width, long long input_channels,
    long long output_channels, long long kernel_frames,
    long long kernel_height, long long kernel_width, long long stride_frames,
    long long stride_height, long long stride_width, long long pad_frames = 0,
    long long pad_height = 0, long long pad_width = 0);
Status conv3d(const float* input, const float* weights, const float* bias,
              float* output, long long batch, long long input_channels,
              long long output_channels, long long input_depth,
              long long input_height, long long input_width,
              long long kernel_depth, long long kernel_height,
              long long kernel_width, long long stride_depth = 1,
              long long stride_height = 1, long long stride_width = 1,
              long long pad_depth = 0, long long pad_height = 0,
              long long pad_width = 0, long long dilation_depth = 1,
              long long dilation_height = 1, long long dilation_width = 1);
// Transposed-convolution weights are [in_channels,out_channels,...].
Status conv_transpose_1d(const float* input, const float* weights,
                         const float* bias, float* output, long long batch,
                         long long input_channels, long long output_channels,
                         long long input_length, long long kernel,
                         long long stride = 1, long long padding = 0);
// Audio front-end kernels use NWC activations. General weights are [O,K,C];
// depthwise weights are [C,K]. Bias may be null.
Status audio_conv1d_direct(const float* input, const float* weights,
                           const float* bias, float* output, long long batch,
                           long long input_length, long long input_channels,
                           long long output_channels, long long kernel,
                           long long stride = 1, long long padding = 0,
                           long long dilation = 1);
Status audio_depthwise_conv1d(const float* input, const float* weights,
                              const float* bias, float* output,
                              long long batch, long long input_length,
                              long long channels, long long kernel,
                              long long stride = 1, long long padding = 0,
                              long long dilation = 1,
                              bool apply_silu = false);
// Causal NWC depthwise convolution with left padding dilation*(K-1), no right
// padding, and an output length equal to the input length.
Status audio_causal_depthwise_conv1d(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_length, long long channels,
    long long kernel, long long dilation = 1);
Status conv_transpose_2d(const float* input, const float* weights,
                         const float* bias, float* output, long long batch,
                         long long input_channels, long long output_channels,
                         long long input_height, long long input_width,
                         long long kernel_height, long long kernel_width,
                         long long stride_height = 1,
                         long long stride_width = 1, long long pad_height = 0,
                         long long pad_width = 0);
Status pool1d(const float* input, float* output, long long batch,
              long long channels, long long input_length, long long kernel,
              long long stride, long long padding,
              PoolMode mode = PoolMode::kAverage);
Status pool2d(const float* input, float* output, long long batch,
              long long channels, long long input_height, long long input_width,
              long long kernel_height, long long kernel_width,
              long long stride_height, long long stride_width,
              long long pad_height, long long pad_width,
              PoolMode mode = PoolMode::kAverage);
Status pool2d_backward(const float* input, const float* grad_out,
                       float* grad_in, long long batch, long long channels,
                       long long input_height, long long input_width,
                       long long kernel_height, long long kernel_width,
                       long long stride_height, long long stride_width,
                       long long pad_height, long long pad_width,
                       PoolMode mode = PoolMode::kAverage);
Status timestep_embedding(const float* timesteps, float* output,
                          long long count, long long dim,
                          float max_period = 10000.0f);
Status solve_lower_triangular(const float* a, const float* b, float* x,
                              long long batch, long long n,
                              long long right_hand_sides);
Status get_relative_position(const float* table, float* output, long long width,
                             long long dim);
Status add_relative_position_2d(const float* attention,
                                const float* relative_height,
                                const float* relative_width, float* output,
                                long long batches, long long query_height,
                                long long query_width, long long key_height,
                                long long key_width);
// NHWC image/window layouts, with zero padding at the lower/right edges.
Status window_partition(const float* image, float* windows, long long height,
                        long long width, long long channels,
                        long long window_size);
Status window_unpartition(const float* windows, float* image, long long height,
                          long long width, long long channels,
                          long long window_size);
// llama.cpp-compatible recurrent attention layouts. Token tensors are
// [sequences,tokens_per_sequence,heads,head_dim], while state tensors are
// [sequences,heads,head_dim,head_dim]. Initial and final state may alias.
Status gated_linear_attention(const float* key, const float* value,
                              const float* query, const float* gate,
                              const float* initial_state, float* output,
                              float* final_state, long long sequences,
                              long long tokens_per_sequence, long long heads,
                              long long head_dim, float scale = 1.0f);
Status rwkv_wkv6(const float* key, const float* value, const float* receptance,
                 const float* time_first, const float* time_decay,
                 const float* initial_state, float* output, float* final_state,
                 long long sequences, long long tokens_per_sequence,
                 long long heads, long long head_dim);
Status rwkv_wkv7(const float* receptance, const float* decay, const float* key,
                 const float* value, const float* a, const float* b,
                 const float* initial_state, float* output, float* final_state,
                 long long sequences, long long tokens_per_sequence,
                 long long heads, long long head_dim);
// DeepSeek-V4 hyper-connection stages. The fixed hyper-connection width is 4.
// mixes/base are [tokens,24]/[24], comb is [tokens,source,destination].
Status dsv4_hc_comb(const float* mixes, const float* scale, const float* base,
                    float* comb, long long tokens, float eps, int iterations);
Status dsv4_hc_pre(const float* x, const float* weights, float* output,
                   long long tokens, long long embedding);
Status dsv4_hc_post(const float* x, const float* residual, const float* post,
                    const float* comb, float* output, long long tokens,
                    long long embedding);
Status cross_entropy(const float* logits, const int* target, float* loss,
                     long long rows, long long vocab);
Status cross_entropy_forward(const float* logits, const int* target,
                             float* loss, float* logsumexp, long long rows,
                             long long vocab, int ignore_index = -100,
                             float label_smoothing = 0.0f, float z_loss = 0.0f,
                             float softcap = 0.0f);
Status cross_entropy_backward(const float* logits, const int* target,
                              const float* grad_out, float* grad_logits,
                              long long rows, long long vocab,
                              int ignore_index = -100,
                              float label_smoothing = 0.0f, float z_loss = 0.0f,
                              float softcap = 0.0f);
Status hadamard(const float* x, float* y, long long rows, long long dim);
Status fwht_rotate(const float* x, const float* sign, float* y, long long rows,
                   long long dim, bool inverse = false);
Status packbits(const std::uint8_t* x, std::uint8_t* out, long long count,
                bool bit_order_big = true);
Status segment_packbits(const std::uint8_t* x, const long long* input_offsets,
                        const long long* output_offsets, std::uint8_t* out,
                        long long segments, long long input_count,
                        long long output_bytes, bool bit_order_big = true);
Status permute_cols(const float* x, const int* permutation, float* out,
                    long long rows, long long cols);
Status tau_tail(const float* qkv, const float* token_qv_linear,
                const float* position_table, const int* positions, float* out,
                long long tokens, long long heads, long long head_dim,
                long long max_position);

// Knowledge-distillation losses. All losses and saved logsumexp arrays are
// row-shaped; backward overwrites the student gradient.
Status kd_kl_dense_forward(const float* teacher, const float* student,
                           float* loss, float* teacher_lse, float* student_lse,
                           long long rows, long long vocab,
                           float inverse_temperature = 1.0f);
Status kd_kl_dense_backward(const float* teacher, const float* student,
                            const float* teacher_lse, const float* student_lse,
                            const float* grad_out, float* grad_student,
                            long long rows, long long vocab,
                            float inverse_temperature = 1.0f);
Status kd_kl_topk_forward(const float* student, const int* teacher_indices,
                          const float* teacher_probabilities, float* loss,
                          float* student_lse, long long rows, long long vocab,
                          long long top_k, float inverse_temperature = 1.0f,
                          bool include_tail = false);
Status kd_kl_topk_backward(const float* student, const int* teacher_indices,
                           const float* teacher_probabilities,
                           const float* student_lse, const float* grad_out,
                           float* grad_student, long long rows, long long vocab,
                           long long top_k, float inverse_temperature = 1.0f,
                           bool include_tail = false);
Status kd_ce_fused_forward(const float* teacher, const float* student,
                           const int* targets, float* cross_entropy_loss,
                           float* kd_loss, float* raw_student_lse,
                           float* tempered_student_lse, float* teacher_lse,
                           long long rows, long long vocab,
                           float inverse_temperature = 1.0f,
                           int ignore_index = -100);
Status kd_ce_fused_backward(
    const float* teacher, const float* student, const int* targets,
    const float* raw_student_lse, const float* tempered_student_lse,
    const float* teacher_lse, const float* grad_cross_entropy,
    const float* grad_kd, float* grad_student, long long rows, long long vocab,
    float inverse_temperature = 1.0f, int ignore_index = -100);

// Vision composites use NHWC input. Patch merge emits
// [batch,height/2,width/2,4*channels]. Space-to-depth then normalizes and
// projects from block_size^2*channels to out_channels.
Status patch_merge_layer_norm(const float* input, const float* weight,
                              const float* bias, float* out, long long batch,
                              long long height, long long width,
                              long long channels, float eps = 1e-5f);
Status space_to_depth_norm_linear(
    const float* input, const float* norm_weight, const float* norm_bias,
    const float* projection_weight, const float* projection_bias, float* out,
    long long batch, long long height, long long width, long long channels,
    long long out_channels, long long block_size = 2, float eps = 1e-5f);
// Pairwise edge MLP: hidden [B,L,256], first_weight [256,512], output
// [B,7,L,L].
Status edge_mlp_256x7(const float* hidden, const float* first_weight,
                      const float* first_bias, const float* second_weight,
                      const float* second_bias, float* out, long long batch,
                      long long length);

// MoE top-k uses lowest expert id to break score ties and normalizes selected
// logits with softmax. Outputs are [tokens, top_k].
Status moe_route_topk(const float* router_logits, int* expert_ids,
                      float* expert_weights, long long tokens,
                      long long experts, int top_k);
Status moe_route_grouped(const float* router_logits, const float* bias,
                         int* expert_ids, float* expert_weights,
                         long long tokens, long long experts, int top_k,
                         int groups, int top_groups, bool renormalize,
                         float routed_scale = 1.0f,
                         MoeScoring scoring = MoeScoring::kSoftmax);
Status moe_route_scored(const float* router_logits, int* expert_ids,
                        float* expert_weights, long long tokens,
                        long long experts, int top_k, int mode = 0,
                        bool renormalize = true, float scaling = 1.0f);
// Stable expert-major permutation of flattened [tokens,top_k] routing rows.
Status moe_permute(const int* expert_ids, int* sorted_rows, int* offsets,
                   int* inverse, long long tokens, long long experts,
                   int top_k);
Status moe_pad_schedule(const int* sorted_rows, const int* offsets,
                        const int* inverse, int* expert_of_tile,
                        int* gather_rows, int* inverse_padded,
                        int* padded_offsets, long long tokens,
                        long long experts, int top_k, int tile_rows = 32);
Status moe_lora_align(const int* topk_ids, const int* token_lora_mapping,
                      const int* lora_ids, const std::uint8_t* adapter_enabled,
                      int* sorted_token_ids, int* expert_ids,
                      int* tokens_post_pad, long long tokens,
                      long long max_loras, long long num_experts, int top_k,
                      int block_size, long long sorted_capacity_per_lora,
                      long long expert_capacity_per_lora);
Status moe_gather(const float* x, const int* gather_rows, float* out,
                  long long tokens, long long gathered_rows, long long dim);
Status moe_finalize(const float* expert_out, const int* inverse,
                    const float* expert_weights, float* out, long long tokens,
                    int top_k, long long dim);
Status moe_finalize_backward(const float* grad_out, const float* expert_out,
                             const int* inverse, const float* expert_weights,
                             float* grad_expert_out, float* grad_expert_weights,
                             long long tokens, int top_k, long long dim);
Status moe_gather_backward(const float* grad_gathered, const int* gather_rows,
                           float* grad_input, long long tokens,
                           long long gathered_rows, long long dim);
// Row-indexed expert GEMM: expert_ids[rows] chooses weight [E,K,N].
Status moe_grouped_gemm(const float* x, const float* weights,
                        const int* expert_ids, float* out, long long rows,
                        long long experts, long long input_dim,
                        long long output_dim);
Status moe_grouped_swiglu(const float* x, const float* weights,
                          const int* expert_ids, float* out, long long rows,
                          long long experts, long long input_dim,
                          long long intermediate_dim);
Status moe_grouped_gemm_backward_input(const float* grad_out,
                                       const float* weights,
                                       const int* expert_ids, float* grad_input,
                                       long long rows, long long experts,
                                       long long input_dim,
                                       long long output_dim);
Status moe_grouped_gemm_backward_weight(const float* x, const float* grad_out,
                                        const int* expert_ids,
                                        float* grad_weights, long long rows,
                                        long long experts, long long input_dim,
                                        long long output_dim);
Status moe_grouped_qgemm(const float* x, const void* packed_weights,
                         const int* expert_ids, const float* bias, float* out,
                         long long rows, long long experts, long long input_dim,
                         long long output_dim, QuantFormat format,
                         bool use_bias = false);
Status moe_grouped_qswiglu(const float* x, const void* packed_weights,
                           const int* expert_ids, const float* bias, float* out,
                           long long rows, long long experts,
                           long long input_dim, long long intermediate_dim,
                           QuantFormat format, bool use_bias = false,
                           bool oai_mode = false, float alpha = 1.702f,
                           float limit = 7.0f);
// Named CUDA/ROCm quantized-MoE seams with CPU-logical metadata layouts.
// expert_ids is row-shaped; -1 denotes a padded row and produces zeros.
Status moe_gemm_fp8(const float* x, const std::uint8_t* weight_codes,
                    const float* weight_scales, const int* expert_ids,
                    float* out, long long rows, long long experts,
                    long long input_dim, long long output_dim);
// WNA16 weights are uint32-packed in the sibling de-interleave order. Scales
// are logical f32 [E,N,K/group_size]; zero_points may be null for 8/128.
Status moe_gemm_wna16(const float* x, const std::uint32_t* packed_weights,
                      const float* scales, const std::uint8_t* zero_points,
                      const int* expert_ids, float* out, long long rows,
                      long long experts, long long input_dim,
                      long long output_dim, long long group_size, int bits);
// Dual-operand NVFP4: activation scales [rows,K/16], weight scales
// [E,N,K/16], and expert_scale is the post-accumulation multiplier.
Status moe_gemm_nvfp4(const std::uint8_t* activation_codes,
                      const std::uint8_t* weight_codes,
                      const std::uint8_t* activation_scale_codes,
                      const std::uint8_t* weight_scale_codes,
                      const float* expert_scale, const int* expert_ids,
                      float* out, long long rows, long long experts,
                      long long input_dim, long long output_dim);

// A [groups,M,K], B [groups,K,N], C [groups,M,N], all row-major.
Status grouped_gemm(const float* a, const float* b, float* c, long long groups,
                    long long m, long long n, long long k);

// Non-causal linear attention, tensors [heads,seq,dim].
Status linear_attention(const float* q, const float* k, const float* v,
                        float* out, long long heads, long long sequence,
                        long long dim, float eps = 1e-6f);

// Metal-family identity feature-map variants are intentionally separate from
// the normalized XPU-style linear_attention entry point above.
Status linear_attention_unnormalized(const float* q, const float* k,
                                     const float* v, float* out,
                                     long long batch, long long heads,
                                     long long sequence, long long dim);
Status causal_linear_attention(const float* q, const float* k, const float* v,
                               float* out, long long batch, long long heads,
                               long long sequence, long long dim);
Status decayed_linear_attention(const float* q, const float* k, const float* v,
                                const float* cumulative_log, float* out,
                                long long batch, long long heads,
                                long long sequence, long long dim);
Status based_attention(const float* q, const float* k, const float* v,
                       float* out, long long batch, long long heads,
                       long long sequence, long long qk_dim,
                       long long value_dim);
Status hedgehog_attention(const float* q, const float* k, const float* v,
                          float* out, long long batch, long long heads,
                          long long sequence, long long dim);

// Packed variable-length GatedDeltaNet recurrence. state_pool is updated in
// place at [slots,value_heads,value_dim,key_dim].
Status gdn_recurrence(const float* q, const float* k, const float* v,
                      const float* gate, const float* beta, float* state_pool,
                      const int* cumulative_lengths, const int* slot_mapping,
                      float* out, long long requests, long long slots,
                      long long key_heads, long long value_heads,
                      long long key_dim, long long value_dim,
                      bool load_initial = true);

// Live Metal GatedDeltaNet surface. State-bearing operations are functional:
// input pools remain unchanged and output pools clone/update request slots.
Status gdn_recur(const float* q, const float* k, const float* v,
                 const float* decay, const float* beta, const float* state_pool,
                 const int* cumulative_lengths, const int* slot_mapping,
                 float* out, float* state_pool_out, long long requests,
                 long long slots, long long key_heads, long long value_heads,
                 long long key_dim, long long value_dim,
                 bool load_initial = true);
Status gdn_short_conv(const float* x, const float* weight,
                      const float* state_pool, const int* cumulative_lengths,
                      const int* slot_mapping, float* out,
                      float* state_pool_out, long long requests,
                      long long slots, long long channels,
                      long long kernel_size, bool load_initial = true,
                      bool apply_silu = true);
Status gdn_qkv_prepare(const float* mixed, float* q, float* k, float* v,
                       long long tokens, long long key_heads,
                       long long value_heads, long long key_dim,
                       long long value_dim, float eps = 1e-6f,
                       float q_scale = std::numeric_limits<float>::quiet_NaN(),
                       float k_scale = std::numeric_limits<float>::quiet_NaN());
Status gdn_gate_beta(const float* a, const float* b, const float* a_log,
                     const float* dt_bias, float* decay, float* beta,
                     long long tokens, long long value_heads);
Status gdn_gated_rmsnorm(const float* y, const float* z, const float* weight,
                         float* out, long long tokens, long long value_heads,
                         long long value_dim, float eps = 1e-6f);

// Mamba S6 selective scan. u/delta/y [channels,seq], A [channels,state],
// B/C [seq,state], D [channels].
Status selective_scan(const float* u, const float* delta, const float* a,
                      const float* b, const float* c, const float* d, float* y,
                      long long channels, long long sequence,
                      long long state_size);
// Stateful packed varlen S6. B/C are [groups,state,total_tokens], channel-
// major u/delta/z/out are [channels,total_tokens]. checkpoint_slots may be
// null; otherwise every nonnegative token entry checkpoints the post-token
// state into that state-pool slot (the APC seam).
Status selective_scan_varlen(
    const float* u, const float* delta, const float* a, const float* b,
    const float* c, const float* d, const float* delta_bias, const float* z,
    const int* cumulative_lengths, const int* cache_indices,
    const std::uint8_t* has_initial_state, const int* checkpoint_slots,
    float* state_pool, float* out, long long batch, long long slots,
    long long channels, long long total_tokens, long long state_size,
    long long groups, bool delta_softplus = true, int null_slot = -1);

// Mamba-2/SSD materialized reference, tensors [B,H,S,D], cumulative_log
// [B,H,S]. The backward outputs match their corresponding inputs.
Status mamba2(const float* c, const float* b, const float* x,
              const float* cumulative_log, float* y, long long batch,
              long long heads, long long sequence, long long dim);
Status mamba2_backward(const float* c, const float* b, const float* x,
                       const float* cumulative_log, const float* grad_y,
                       float* grad_c, float* grad_b, float* grad_x,
                       float* grad_cumulative_log, long long batch,
                       long long heads, long long sequence, long long dim);
Status ssd_chunked(const float* c, const float* b, const float* x,
                   const float* cumulative_log, float* y, long long batch,
                   long long heads, long long sequence, long long dim);
Status ssd_chunked_backward(const float* c, const float* b, const float* x,
                            const float* cumulative_log, const float* grad_y,
                            float* grad_c, float* grad_b, float* grad_x,
                            float* grad_cumulative_log, long long batch,
                            long long heads, long long sequence, long long dim);
Status ssd_decode(const float* state, const float* alpha, const float* x,
                  const float* k, const float* q, float* y, float* next_state,
                  long long batch, long long heads, long long dim);

// Direct circular convolution, the portable oracle for sibling FFT kernels.
// x [B,H,N], kernel [H,N], out [B,H,N].
Status fft_convolution(const float* x, const float* kernel, float* out,
                       long long batch, long long heads, long long length);

// Fused AdamW, in-place, with a 1-based step and bias correction.
Status adamw(float* parameters, const float* gradients, float* first_moment,
             float* second_moment, long long count, float learning_rate,
             float beta1, float beta2, float eps, float weight_decay,
             long long step);
// mask indexes contiguous segments. mode 0 freezes inactive segments; mode 1
// updates their moments/parameters while suppressing weight decay.
Status adamw_masked(float* parameters, const float* gradients,
                    float* first_moment, float* second_moment,
                    const std::uint8_t* mask, long long count,
                    long long segment_size, int mask_mode, float learning_rate,
                    float beta1, float beta2, float eps, float weight_decay,
                    long long step);

// Sparse-serving metadata and FP8 index cache operations.
Status indexer_k_quant_and_cache(const float* keys, const int* slot_mapping,
                                 std::uint8_t* code_cache, float* scale_cache,
                                 long long tokens, long long slots,
                                 long long head_dim,
                                 long long quant_block_size = 128,
                                 bool power_of_two_scale = false);
Status indexer_k_gather(const std::uint8_t* code_cache,
                        const float* scale_cache, const int* slots_to_gather,
                        float* out, long long cache_slots, long long count,
                        long long head_dim, long long quant_block_size = 128);
Status indexer_k_gather_paged(const std::uint8_t* code_cache,
                              const float* scale_cache, const int* block_table,
                              const int* cumulative_lengths,
                              std::uint8_t* output_codes, float* output_scales,
                              long long cache_blocks, long long batch,
                              long long max_blocks, long long cache_block_size,
                              long long total_tokens, long long head_dim,
                              long long quant_block_size = 128);
Status convert_vertical_slash_indexes(
    const int* query_lengths, const int* kv_lengths,
    const int* vertical_indexes, const int* slash_indexes, int* block_count,
    int* block_offset, int* column_count, int* column_index, long long batch,
    long long heads, long long rows, long long vertical_width,
    long long slash_width, int block_size_m = 64, int block_size_n = 64,
    bool causal = true);
Status minference_block_mask(const int* vertical_indices,
                             const int* slash_offsets,
                             const int* context_lengths, int* block_mask,
                             long long batch, long long heads,
                             long long vertical_width, long long slash_width,
                             long long max_blocks, long long block_size,
                             long long vertical_top_k, long long slash_top_k,
                             long long last_n_blocks = 1);

}  // namespace quixicore_cpu
