#pragma once

#include <cstdint>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

enum class QuantFormat;
enum class Float8Format;

// Portable f32 operation surface shared by the sibling backends. Layouts are
// row-major and all reductions accumulate in f32 or better.

enum class GeluApprox { kErf, kTanh };
enum class GluMode { kSwiGlu, kGeGlu, kReGlu, kGlu };
enum class LinearActivation { kNone, kGeluErf, kGeluTanh, kSilu };
enum class MoeScoring { kSoftmax, kSigmoid, kSqrtSoftplus };

Status gelu(const float* x, float* y, long long count,
            GeluApprox approx = GeluApprox::kErf);
Status gelu_backward(const float* grad_out, const float* x, float* grad_in,
                     long long count,
                     GeluApprox approx = GeluApprox::kErf);
Status silu(const float* x, float* y, long long count);

// x is [rows, 2*dim] (gate half followed by value half); y is [rows, dim].
Status glu(const float* x, float* y, long long rows, long long dim,
           GluMode mode = GluMode::kSwiGlu);
Status glu_backward(const float* grad_out, const float* x, float* grad_in,
                    long long rows, long long dim,
                    GluMode mode = GluMode::kSwiGlu);

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
                           float* grad_weight, float* grad_bias,
                           long long rows, long long hidden,
                           float eps = 1e-5f);
Status rms_norm_backward(const float* x, const float* weight,
                         const float* grad_out, float* grad_x,
                         float* grad_weight, long long rows, long long hidden,
                         float eps = 1e-5f);

// Fused residual seams. residual_out may alias x or residual but y must not
// overlap either input.
Status rms_norm_add(const float* x, const float* residual,
                    const float* weight, float* y, float* residual_out,
                    long long rows, long long hidden, float eps = 1e-5f);
Status layer_norm_add(const float* x, const float* residual,
                      const float* weight, const float* bias, float* y,
                      float* residual_out, long long rows, long long hidden,
                      float eps = 1e-5f);
Status rms_norm_residual_next(const float* x, const float* post_weight,
                              const float* residual,
                              const float* next_weight, float* residual_out,
                              float* next_out, long long rows,
                              long long hidden, float eps = 1e-5f);

// C[M,N] = A[M,K] @ B[K,N]. Inputs and output are row-major.
Status dense_gemm(const float* a, const float* b, float* c, long long m,
                  long long n, long long k);
Status dense_gemm_ex(const float* a, const float* b, const float* addend,
                     float* c, long long m, long long n, long long k,
                     bool transpose_a, bool transpose_b, float alpha = 1.0f,
                     float beta = 0.0f);
// y [rows,out] = activation(x[rows,in] @ weight[out,in]^T + bias + residual).
// bias/residual may be null.
Status linear_epilogue(const float* x, const float* weight,
                       const float* bias, const float* residual, float* y,
                       long long rows, long long input_dim,
                       long long output_dim,
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
// Complex GEMM with planar real/imaginary inputs and outputs.
Status complex_gemm(const float* a_real, const float* a_imag,
                    const float* b_real, const float* b_imag,
                    float* c_real, float* c_imag, long long m,
                    long long n, long long k);

// NeoX half-split RoPE over [tokens, heads, head_dim].
Status rope(const float* x, float* y, long long tokens, long long heads,
            long long head_dim, float base, long long pos0);

// Scaled dot-product GQA. Q [Hq,Sq,D], K/V [Hkv,Sk,D], O [Hq,Sq,D].
Status attention(const float* q, const float* k, const float* v, float* out,
                 long long query_heads, long long kv_heads,
                 long long query_length, long long kv_length,
                 long long head_dim, bool causal);

// Backward for the dense/GQA attention above. All gradients are overwritten.
Status attention_backward(const float* q, const float* k, const float* v,
                          const float* grad_out, float* grad_q, float* grad_k,
                          float* grad_v, long long query_heads,
                          long long kv_heads, long long query_length,
                          long long kv_length, long long head_dim,
                          bool causal);

// Local attention. left_window/right_window are inclusive distances; a
// negative value means unbounded on that side.
Status window_attention(const float* q, const float* k, const float* v,
                        float* out, long long query_heads,
                        long long kv_heads, long long query_length,
                        long long kv_length, long long head_dim,
                        long long left_window, long long right_window);

// Packed variable-length GQA. q [total_q,Hq,D], k/v [total_k,Hkv,D].
Status varlen_attention(const float* q, const float* k, const float* v,
                        const int* cumulative_q, const int* cumulative_k,
                        float* out, long long batch, long long query_heads,
                        long long kv_heads, long long head_dim, bool causal);
Status varlen_build_worklist(const int* cumulative_lengths,
                             int* padded_offsets, int* sequence_lengths,
                             int* tile_sequence, int* tile_local_start,
                             int* tile_count, long long batch,
                             long long max_tiles, int tile_rows = 8);
Status varlen_pad_q(const float* packed, const int* cumulative_lengths,
                    const int* padded_offsets, float* padded_head_major,
                    long long batch, long long heads, long long head_dim,
                    long long total_tokens, long long total_padded);
Status varlen_regather_o(const float* padded_head_major,
                         const int* cumulative_lengths,
                         const int* padded_offsets, float* packed,
                         long long batch, long long heads,
                         long long head_dim, long long total_tokens,
                         long long total_padded);

// Dense attention with optional additive bias [Hq,Sq,Sk] and optional mask
// [Sq,Sk] (zero means masked). This is the semantic core of Swin attention.
Status biased_attention(const float* q, const float* k, const float* v,
                        const float* bias, const std::uint8_t* mask,
                        float* out, long long query_heads,
                        long long kv_heads, long long query_length,
                        long long kv_length, long long head_dim,
                        float scale = 0.0f);

// RoPE using caller-provided tables. positions [tokens], cos/sin
// [max_position,head_dim/2]. interleaved selects adjacent-pair layout.
Status rope_table(const float* x, const float* cosine, const float* sine,
                  const int* positions, float* y, long long tokens,
                  long long heads, long long head_dim,
                  long long max_position, bool interleaved = false);

// Normalize packed Q and K heads then apply table RoPE; V heads pass through.
// qkv/y [tokens,(query_heads+key_heads+value_heads)*head_dim].
Status qk_norm_rope(const float* qkv, const float* q_weight,
                    const float* k_weight, const float* cosine,
                    const float* sine, const int* positions, float* y,
                    long long tokens, long long query_heads,
                    long long key_heads, long long value_heads,
                    long long head_dim, long long max_position,
                    float eps = 1e-5f, bool interleaved = false);

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
    const int* block_table, const int* context_lens,
    const int* block_mask, const float* alibi_slopes, const float* sinks,
    float* out, long long cache_blocks, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, float scale = 0.0f,
    long long window = 0, float softcap = 0.0f);
Status paged_attention_fp8(
    const float* q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const int* block_table,
    const int* context_lens, const float* key_scale,
    const float* value_scale, float* out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    Float8Format format, float scale = 0.0f, long long window = 0,
    float softcap = 0.0f);
// Layout adapter for the vLLM/Metal xcache representation. key_cache is
// [blocks,Hkv,D/x,page,x], value_cache is [blocks,Hkv,D,page].
Status paged_attention_xcache(
    const float* q, const float* key_cache, const float* value_cache,
    const int* block_table, const int* context_lens, float* out,
    long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, long long vector_width, float scale = 0.0f);
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
    const float* q, const std::uint8_t* prefix_k,
    const std::uint8_t* prefix_v, const float* key_cache,
    const float* value_cache, const int* block_table,
    const int* context_lens, const float* key_scale,
    const float* value_scale, float* out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long prefix_length, long long page_size,
    long long max_blocks, Float8Format format, float scale = 0.0f);

// MLA latent decode. q [B,H,W], cache [blocks,page,W], out [B,H,value_dim].
// Scores use all W=latent_dim+rope_dim values; values use the latent prefix.
Status mla_decode(const float* q, const float* kv_cache,
                  const int* block_table, const int* context_lens, float* out,
                  long long cache_blocks, long long batch, long long heads,
                  long long latent_dim, long long rope_dim, long long page_size,
                  long long max_blocks, float scale = 0.0f);
Status mla_kv_insert_fp8(
    const float* kv, const int* slot_mapping, std::uint8_t* code_cache,
    float* scale_cache, long long tokens, long long slots, long long width,
    long long group_size, Float8Format format,
    bool power_of_two_scale = true);
Status mla_decode_fp8(
    const float* q, const std::uint8_t* code_cache,
    const float* scale_cache, const int* block_table,
    const int* context_lengths, float* out, long long cache_blocks,
    long long batch, long long heads, long long width, long long page_size,
    long long max_blocks, long long group_size, Float8Format format,
    float scale = 0.0f);
Status mla_decode_fp8_sparse(
    const float* q, const std::uint8_t* code_cache,
    const float* scale_cache, const int* block_table,
    const int* token_indices, const int* topk_lengths, float* out,
    long long cache_blocks, long long batch, long long heads,
    long long width, long long page_size, long long max_blocks,
    long long max_topk, long long group_size, Float8Format format,
    float scale = 0.0f);

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
Status min_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float min_p, float temperature = 1.0f,
                    std::uint32_t seed = 0);
Status typical_p_sample(const float* logits, int* out, long long rows,
                        long long vocab, float p,
                        float temperature = 1.0f,
                        std::uint32_t seed = 0);

// In-place-compatible logit processors. A masked entry is set to -infinity.
Status apply_token_bitmask(const float* logits, const std::uint8_t* bitmask,
                           float* out, long long rows, long long vocab);
Status apply_bad_words(const float* logits, const int* bad_ids, float* out,
                       long long rows, long long vocab,
                       long long bad_count);
Status apply_repetition_penalty(const float* logits, const int* previous,
                                const int* lengths, float* out,
                                long long rows, long long vocab,
                                long long max_previous,
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
                       long long vocab, float nsigma,
                       float temperature = 1.0f);
Status top_a_mask(const float* logits, float* out, long long rows,
                  long long vocab, float top_a,
                  float temperature = 1.0f);
Status epsilon_cutoff_mask(const float* logits, float* out, long long rows,
                           long long vocab, float epsilon,
                           float temperature = 1.0f);
Status eta_cutoff_mask(const float* logits, float* out, long long rows,
                       long long vocab, float eta,
                       float temperature = 1.0f);
Status xtc_mask(const float* logits, float* out, long long rows,
                long long vocab, float threshold, float probability,
                std::uint32_t seed = 0, float temperature = 1.0f);
Status skew_transform(const float* probabilities, float* out, long long rows,
                      long long vocab, float skew);
Status no_repeat_ngram_mask(const float* logits, const int* previous,
                            const int* lengths, float* out, long long rows,
                            long long vocab, long long history,
                            int ngram_size, float temperature = 1.0f);
Status dry_penalty(const float* logits, const int* previous,
                   const int* lengths, const int* breakers, float* out,
                   long long rows, long long vocab, long long history,
                   long long breaker_count, float multiplier,
                   float base = 1.75f, int allowed_length = 2,
                   int range = 0, int max_ngram = 64,
                   int max_occurrences = 64,
                   int early_exit_match_length = 64,
                   float temperature = 1.0f);

// One beam-search step. logits [B*beam,V], cumulative [B,beam].
Status beam_search_step(const float* logits, const float* cumulative,
                        int* next_token, int* parent_beam,
                        float* next_cumulative, long long batch,
                        long long beam, long long vocab);

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
Status speculative_verify_tree(
    const int* draft_tokens, const float* target_probs,
    const int* first_child, const int* next_sibling, const int* tree_valid,
    int* accepted_indices, int* accepted_tokens, int* accepted_count,
    long long batch, long long nodes, long long vocab,
    std::uint32_t seed = 0);
Status speculative_compact(const int* out_tokens, const int* accepted_count,
                           const int* sequence_lengths, int* packed_tokens,
                           int* packed_positions, int* cumulative_accepted,
                           long long batch, long long draft_length);
Status speculative_update_kv_meta(const int* sequence_lengths,
                                  const int* accepted_count,
                                  int* new_sequence_lengths,
                                  long long batch);
Status rejection_greedy_sample(
    const int* cumulative_drafts, const int* draft_tokens,
    const int* target_argmax, const int* bonus_tokens,
    const std::uint8_t* greedy_mask, int* out_tokens, long long batch,
    long long total_drafts, long long max_draft,
    bool use_greedy_mask = false);
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
    long long total_drafts, long long vocab,
    bool no_draft_probs = false);
Status eagle_prepare_inputs_padded(
    const int* cumulative_drafts, const int* valid_sampled_count,
    const int* query_start_locations, int* token_indices_to_sample,
    int* rejected_count, long long batch);
Status eagle_prepare_next_token_padded(
    const int* sampled_tokens, const std::uint8_t* discard_mask,
    const int* backup_tokens, int* next_tokens, int* valid_sampled_count,
    long long batch, long long sampled_per_request, int vocab);
Status eagle_step_slot_mapping_metadata(
    const int* positions, const int* block_table,
    const int* sequence_lengths, int* clamped_positions,
    int* slot_mapping, int* new_sequence_lengths, long long batch,
    long long input_batch, long long max_blocks, int block_size,
    int max_model_length, int pad_id);
Status eagle_expand_int32(const int* input, const int* cumulative_tokens,
                          int* output, long long batch, long long total,
                          int replace_from, int replace_to);

Status embedding_lookup(const float* table, const int* ids, float* out,
                        long long vocab, long long count, long long dim);
Status embedding_backward(const int* ids, const float* grad_out,
                          float* grad_table, long long vocab,
                          long long count, long long dim);
// Builds the source map consumed by merge_multimodal_spans. Positions outside
// every span receive -1; overlapping spans use the first span.
Status build_multimodal_source_map(
    const int* span_offsets, const int* span_lengths,
    const int* modal_starts, int* source_map, long long spans,
    long long text_tokens);
Status merge_multimodal_spans(const float* text, const float* modal,
                              const int* source_map, float* out,
                              long long text_tokens,
                              long long modal_tokens, long long dim);
Status mean_pool_rms_l2(const float* x, const float* weight,
                        const int* lengths, float* out, long long batch,
                        long long sequence, long long hidden,
                        float eps = 1e-5f);
// Packed-table gather operations. table uses the row-major packed layout of
// qgemv for the selected format; add may be null when use_add is false.
Status quantized_embedding(const void* table, const int* ids,
                           const float* add, float* out, long long vocab,
                           long long count, long long dim,
                           QuantFormat quant_format,
                           float scale = 1.0f, bool use_add = false);
Status quantized_embedding_bag(const void* table, const int* ids,
                               const long long* offsets,
                               const float* sample_weights, float* out,
                               long long vocab, long long id_count,
                               long long bags, long long dim,
                               QuantFormat quant_format, float scale = 1.0f,
                               bool use_weights = false,
                               bool mean_mode = false);
Status kv_cache_scatter(float* cache, const float* src, const int* slots,
                        long long max_slots, long long count,
                        long long row_width);
Status kv_cache_gather(const float* cache, const int* indices, float* out,
                       long long max_slots, long long count,
                       long long row_width);
Status kv_cache_copy_blocks(float* key_cache, float* value_cache,
                            const long long* block_pairs,
                            long long pair_count, long long block_count,
                            long long elements_per_block);
Status beam_build_copy_pairs(const int* parent_beam, const int* block_table,
                             const int* sequence_lengths,
                             long long* block_pairs, long long batch,
                             long long beam_width, long long max_blocks,
                             int block_size);
Status beam_remap_block_table(const int* block_table,
                              const int* parent_beam,
                              int* remapped_block_table, long long batch,
                              long long beam_width, long long max_blocks);
Status kv_cache_scales(const float* key, const float* value,
                       float* key_scale, float* value_scale,
                       long long count);
Status kv_cache_scale_update(const float* key, const float* value,
                             float old_key_scale, float old_value_scale,
                             float* new_key_scale, float* new_value_scale,
                             long long count);
Status kv_cache_scatter_fp8(
    const float* key, const float* value, const int* slots,
    const float* key_scale, const float* value_scale,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    long long max_slots, long long count, long long heads,
    long long head_dim, Float8Format format);
Status kv_cache_gather_fp8(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const float* key_scale, const float* value_scale,
    float* key_out, float* value_out, long long max_slots,
    long long count, long long heads, long long head_dim,
    Float8Format format);

Status dropout(const float* x, float* y, long long count, float probability,
               std::uint32_t seed);
Status add(const float* x, const float* y, float* out, long long count);
Status cross_entropy(const float* logits, const int* target, float* loss,
                     long long rows, long long vocab);
Status cross_entropy_forward(const float* logits, const int* target,
                             float* loss, float* logsumexp, long long rows,
                             long long vocab, int ignore_index = -100,
                             float label_smoothing = 0.0f,
                             float z_loss = 0.0f, float softcap = 0.0f);
Status cross_entropy_backward(const float* logits, const int* target,
                              const float* grad_out, float* grad_logits,
                              long long rows, long long vocab,
                              int ignore_index = -100,
                              float label_smoothing = 0.0f,
                              float z_loss = 0.0f,
                              float softcap = 0.0f);
Status hadamard(const float* x, float* y, long long rows, long long dim);
Status packbits(const std::uint8_t* x, std::uint8_t* out, long long count,
                bool bit_order_big = true);
Status segment_packbits(const std::uint8_t* x, const long long* input_offsets,
                        const long long* output_offsets, std::uint8_t* out,
                        long long segments, long long input_count,
                        long long output_bytes,
                        bool bit_order_big = true);
Status permute_cols(const float* x, const int* permutation, float* out,
                    long long rows, long long cols);
Status tau_tail(const float* qkv, const float* token_qv_linear,
                const float* position_table, const int* positions, float* out,
                long long tokens, long long heads, long long head_dim,
                long long max_position);

// Knowledge-distillation losses. All losses and saved logsumexp arrays are
// row-shaped; backward overwrites the student gradient.
Status kd_kl_dense_forward(const float* teacher, const float* student,
                           float* loss, float* teacher_lse,
                           float* student_lse, long long rows,
                           long long vocab, float inverse_temperature = 1.0f);
Status kd_kl_dense_backward(const float* teacher, const float* student,
                            const float* teacher_lse,
                            const float* student_lse,
                            const float* grad_out, float* grad_student,
                            long long rows, long long vocab,
                            float inverse_temperature = 1.0f);
Status kd_kl_topk_forward(const float* student, const int* teacher_indices,
                          const float* teacher_probabilities, float* loss,
                          float* student_lse, long long rows,
                          long long vocab, long long top_k,
                          float inverse_temperature = 1.0f,
                          bool include_tail = false);
Status kd_kl_topk_backward(const float* student,
                           const int* teacher_indices,
                           const float* teacher_probabilities,
                           const float* student_lse,
                           const float* grad_out, float* grad_student,
                           long long rows, long long vocab, long long top_k,
                           float inverse_temperature = 1.0f,
                           bool include_tail = false);
Status kd_ce_fused_forward(
    const float* teacher, const float* student, const int* targets,
    float* cross_entropy_loss, float* kd_loss, float* raw_student_lse,
    float* tempered_student_lse, float* teacher_lse, long long rows,
    long long vocab, float inverse_temperature = 1.0f,
    int ignore_index = -100);
Status kd_ce_fused_backward(
    const float* teacher, const float* student, const int* targets,
    const float* raw_student_lse, const float* tempered_student_lse,
    const float* teacher_lse, const float* grad_cross_entropy,
    const float* grad_kd, float* grad_student, long long rows,
    long long vocab, float inverse_temperature = 1.0f,
    int ignore_index = -100);

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
    long long out_channels, long long block_size = 2,
    float eps = 1e-5f);
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
// Stable expert-major permutation of flattened [tokens,top_k] routing rows.
Status moe_permute(const int* expert_ids, int* sorted_rows, int* offsets,
                   int* inverse, long long tokens, long long experts,
                   int top_k);
Status moe_pad_schedule(const int* sorted_rows, const int* offsets,
                        const int* inverse,
                        int* expert_of_tile, int* gather_rows,
                        int* inverse_padded, int* padded_offsets,
                        long long tokens, long long experts, int top_k,
                        int tile_rows = 32);
Status moe_gather(const float* x, const int* gather_rows, float* out,
                  long long tokens, long long gathered_rows, long long dim);
Status moe_finalize(const float* expert_out, const int* inverse,
                    const float* expert_weights, float* out,
                    long long tokens, int top_k, long long dim);
Status moe_finalize_backward(
    const float* grad_out, const float* expert_out, const int* inverse,
    const float* expert_weights, float* grad_expert_out,
    float* grad_expert_weights, long long tokens, int top_k, long long dim);
Status moe_gather_backward(const float* grad_gathered,
                           const int* gather_rows, float* grad_input,
                           long long tokens, long long gathered_rows,
                           long long dim);
// Row-indexed expert GEMM: expert_ids[rows] chooses weight [E,K,N].
Status moe_grouped_gemm(const float* x, const float* weights,
                        const int* expert_ids, float* out, long long rows,
                        long long experts, long long input_dim,
                        long long output_dim);
Status moe_grouped_swiglu(const float* x, const float* weights,
                          const int* expert_ids, float* out, long long rows,
                          long long experts, long long input_dim,
                          long long intermediate_dim);
Status moe_grouped_gemm_backward_input(
    const float* grad_out, const float* weights, const int* expert_ids,
    float* grad_input, long long rows, long long experts,
    long long input_dim, long long output_dim);
Status moe_grouped_gemm_backward_weight(
    const float* x, const float* grad_out, const int* expert_ids,
    float* grad_weights, long long rows, long long experts,
    long long input_dim, long long output_dim);
Status moe_grouped_qgemm(
    const float* x, const void* packed_weights, const int* expert_ids,
    const float* bias, float* out, long long rows, long long experts,
    long long input_dim, long long output_dim, QuantFormat format,
    bool use_bias = false);
Status moe_grouped_qswiglu(
    const float* x, const void* packed_weights, const int* expert_ids,
    const float* bias, float* out, long long rows, long long experts,
    long long input_dim, long long intermediate_dim, QuantFormat format,
    bool use_bias = false, bool oai_mode = false, float alpha = 1.702f,
    float limit = 7.0f);

// A [groups,M,K], B [groups,K,N], C [groups,M,N], all row-major.
Status grouped_gemm(const float* a, const float* b, float* c,
                    long long groups, long long m, long long n, long long k);

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
Status decayed_linear_attention(const float* q, const float* k,
                                const float* v, const float* cumulative_log,
                                float* out, long long batch, long long heads,
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
    const float* c, const float* d, const float* delta_bias,
    const float* z, const int* cumulative_lengths,
    const int* cache_indices, const std::uint8_t* has_initial_state,
    const int* checkpoint_slots, float* state_pool, float* out,
    long long batch, long long slots, long long channels,
    long long total_tokens, long long state_size, long long groups,
    bool delta_softplus = true, int null_slot = -1);

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
                    long long segment_size, int mask_mode,
                    float learning_rate, float beta1, float beta2, float eps,
                    float weight_decay, long long step);

// Sparse-serving metadata and FP8 index cache operations.
Status indexer_k_quant_and_cache(const float* keys, const int* slot_mapping,
                                 std::uint8_t* code_cache,
                                 float* scale_cache, long long tokens,
                                 long long slots, long long head_dim,
                                 long long quant_block_size = 128,
                                 bool power_of_two_scale = false);
Status indexer_k_gather(const std::uint8_t* code_cache,
                        const float* scale_cache, const int* slots_to_gather,
                        float* out, long long cache_slots, long long count,
                        long long head_dim,
                        long long quant_block_size = 128);
Status minference_block_mask(const int* vertical_indices,
                             const int* slash_offsets,
                             const int* context_lengths, int* block_mask,
                             long long batch, long long heads,
                             long long vertical_width,
                             long long slash_width, long long max_blocks,
                             long long block_size,
                             long long vertical_top_k,
                             long long slash_top_k,
                             long long last_n_blocks = 1);

}  // namespace quixicore_cpu
