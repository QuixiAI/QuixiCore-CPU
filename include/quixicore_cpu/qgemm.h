#pragma once

#include <cstdint>

#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/workspace.h"

namespace quixicore_cpu {

enum class Float8Format;
enum class LinearActivation;
enum class LmHeadSampling { kArgmax, kCategorical, kTopK, kTopP };
struct CanonicalQuantTensor;

// Quantized matrix multiply with packed W [N,K], activations X [M,K], and
// output Y [M,N]: Y = X @ dequantize(W)^T. Activations and output are f32.
Status qgemm(QuantFormat format, const void* packed_weights, const float* x,
             float* y, long long m, long long n, long long k);

// CPU-prepared equivalent of qgemm. The canonical QuantFormat bytes remain
// owned by weights; its private row panel amortizes format decode across input
// rows. A supplied workspace is retained by the caller. nullptr selects a
// persistent workspace local to the invoking application/pool thread.
Status qgemm_prepacked(const CpuPackedWeights& weights, const float* x,
                       float* y, long long m, Workspace* workspace = nullptr);

// Canonical prepared-panel projection. Input and output may be f32, f16, or
// bf16; conversion is performed directly at the kernel boundary without a
// full-size temporary tensor.
Status qgemm_prepacked_storage(const CpuPackedWeights& weights,
                               FloatStorageInput x, FloatStorageOutput y,
                               long long m, Workspace* workspace = nullptr);
// Canonical prepared projection with an output-channel bias and activation
// applied before the selected f32/f16/bf16 store. bias may be nullptr.
Status qgemm_prepacked_epilogue_storage(const CpuPackedWeights& weights,
                                        FloatStorageInput x, const float* bias,
                                        FloatStorageOutput y, long long m,
                                        LinearActivation activation,
                                        Workspace* workspace = nullptr);
Status qgemv_prepacked(const CpuPackedWeights& weights, const float* x,
                       float* y, Workspace* workspace = nullptr);
Status qgemv_prepacked_storage(const CpuPackedWeights& weights,
                               FloatStorageInput x, FloatStorageOutput y,
                               Workspace* workspace = nullptr);

// Paired gate/up projection. Both prepared matrices have shape [N,K]; input
// is [M,K] and outputs are [M,N]. A single CPU schedule owns both weight
// traversals and shares activation work without materializing a combined
// weight tensor.
Status qgemm_prepacked_gate_up_storage(
    const CpuPackedWeights& gate_weights, const CpuPackedWeights& up_weights,
    FloatStorageInput x, FloatStorageOutput gate_output,
    FloatStorageOutput up_output, long long m, Workspace* workspace = nullptr);
Status qgemv_prepacked_gate_up_storage(const CpuPackedWeights& gate_weights,
                                       const CpuPackedWeights& up_weights,
                                       FloatStorageInput x,
                                       FloatStorageOutput gate_output,
                                       FloatStorageOutput up_output,
                                       Workspace* workspace = nullptr);

// Q/K/V projection with independently prepared [Nq,K], [Nk,K], and [Nv,K]
// matrices. All inputs and outputs share one CPU scheduling region; row counts
// and output storage types may differ.
Status qgemm_prepacked_qkv_storage(
    const CpuPackedWeights& q_weights, const CpuPackedWeights& k_weights,
    const CpuPackedWeights& v_weights, FloatStorageInput x,
    FloatStorageOutput q_output, FloatStorageOutput k_output,
    FloatStorageOutput v_output, long long m, Workspace* workspace = nullptr);
Status qgemv_prepacked_qkv_storage(
    const CpuPackedWeights& q_weights, const CpuPackedWeights& k_weights,
    const CpuPackedWeights& v_weights, FloatStorageInput x,
    FloatStorageOutput q_output, FloatStorageOutput k_output,
    FloatStorageOutput v_output, Workspace* workspace = nullptr);

// Single-token canonical decode fusion. Q and K receive split-half/GPT-NeoX
// RoPE for position; V is stored directly. key_cache/value_cache are flattened
// [slots,kv_heads,head_dim]. slot=-1 skips K/V projection and cache writes.
Status qgemv_prepacked_qkv_rope_kv_storage(
    const CpuPackedWeights& q_weights, const CpuPackedWeights& k_weights,
    const CpuPackedWeights& v_weights, FloatStorageInput x, const float* cosine,
    const float* sine, FloatStorageOutput q_output,
    FloatStorageOutput key_cache, FloatStorageOutput value_cache,
    long long query_heads, long long kv_heads, long long head_dim,
    long long slots, long long max_position, int position, int slot,
    Workspace* workspace = nullptr);

// Paired projection with direct SiLU(gate) * up storage. Gate and up tensors
// are never materialized.
Status qgemm_prepacked_swiglu_storage(const CpuPackedWeights& gate_weights,
                                      const CpuPackedWeights& up_weights,
                                      FloatStorageInput x,
                                      FloatStorageOutput output, long long m,
                                      Workspace* workspace = nullptr);
Status qgemv_prepacked_swiglu_storage(const CpuPackedWeights& gate_weights,
                                      const CpuPackedWeights& up_weights,
                                      FloatStorageInput x,
                                      FloatStorageOutput output,
                                      Workspace* workspace = nullptr);

// Paired projection with direct SiLU(gate) * up activation quantization. The
// result uses the same canonical bytes and metadata as quantize_canonical;
// reusable output vector capacity and worker-local workspace bound hot-path
// allocation without materializing a complete floating-point output tensor.
Status qgemm_prepacked_swiglu_quantized(
    const CpuPackedWeights& gate_weights, const CpuPackedWeights& up_weights,
    FloatStorageInput x, CanonicalQuantLayout output_layout,
    long long output_group_size, CanonicalQuantTensor* output, long long m,
    bool scale_2d = false, Workspace* workspace = nullptr);
Status qgemv_prepacked_swiglu_quantized(
    const CpuPackedWeights& gate_weights, const CpuPackedWeights& up_weights,
    FloatStorageInput x, CanonicalQuantLayout output_layout,
    long long output_group_size, CanonicalQuantTensor* output,
    bool scale_2d = false, Workspace* workspace = nullptr);

// Canonical activation formats may be projected directly against canonical
// prepared weights. The activation shape is [M,K] and the f32 output is
// [M,N].
Status qgemm_prepacked_quantized(const CpuPackedWeights& weights,
                                 const CanonicalQuantTensor& activation,
                                 float* y, Workspace* workspace = nullptr);
Status qgemv_prepacked_quantized(const CpuPackedWeights& weights,
                                 const CanonicalQuantTensor& activation,
                                 float* y, Workspace* workspace = nullptr);

// Direct canonical-table serving. Rows selected by ids are decoded without
// materializing the complete table. add is optional (null data/count zero),
// and add/output may independently use f32/f16/bf16 storage.
Status canonical_quantized_embedding_storage(
    const CanonicalQuantTensor& table, const int* ids, FloatStorageInput add,
    FloatStorageOutput output, long long count, float scale = 1.0f);
Status canonical_quantized_embedding_bag_storage(
    const CanonicalQuantTensor& table, const int* ids, const long long* offsets,
    const float* sample_weights, FloatStorageOutput output, long long id_count,
    long long bags, float scale = 1.0f, bool use_weights = false,
    bool mean_mode = false, Workspace* workspace = nullptr);

// Streaming canonical LM-head selection over prepared [vocab,hidden] weights.
// Argmax/top-k keep only the retained candidates. Categorical/top-p use one
// vocabulary row of scratch, never a rows-by-vocabulary logits tensor.
Status qgemm_prepacked_lm_head_sample_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, int* token_ids, long long rows, LmHeadSampling mode,
    int k, float top_p, float temperature, std::uint32_t seed,
    Workspace* workspace = nullptr);
// allow_mask is packed MSB-first with ceil(vocab/8) bytes per input row.
Status qgemm_prepacked_lm_head_masked_topk_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, const std::uint8_t* allow_mask, int* token_ids,
    float* log_probabilities, long long rows, int top_k,
    bool normalize_allowed = true, Workspace* workspace = nullptr);
// candidate_ids is CSR-like through offsets[rows+1]. Candidate ids must be
// valid and unique within each row, matching the existing CPU LM-head API.
Status qgemm_prepacked_lm_head_candidates_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, const int* candidate_ids, const long long* offsets,
    int* token_ids, float* log_probabilities, long long rows,
    long long candidates, int top_k, Workspace* workspace = nullptr);
Status qgemm_prepacked_lm_head_beam_advance_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, const float* cumulative_log_probabilities,
    int* next_token, int* parent_beam, float* next_cumulative, long long batch,
    long long beam_width, Workspace* workspace = nullptr);

// Expert-indexed prepared projection. expert_ids is [rows]; -1 denotes a
// padded row and stores zero. Expert weights share K/N but may use any
// canonical prepared layout. bias, when present, is [experts,N].
Status moe_grouped_prepacked_storage(const CpuPackedWeights* expert_weights,
                                     long long experts, FloatStorageInput x,
                                     const int* expert_ids, const float* bias,
                                     FloatStorageOutput output, long long rows,
                                     LinearActivation activation,
                                     Workspace* workspace = nullptr);
// Gate and up expert arrays contain [I,K] matrices. The result is direct
// SiLU(gate)*up storage; no rows-by-2I projection tensor is materialized.
Status moe_grouped_prepacked_swiglu_storage(
    const CpuPackedWeights* gate_weights, const CpuPackedWeights* up_weights,
    long long experts, FloatStorageInput x, const int* expert_ids,
    FloatStorageOutput output, long long rows, Workspace* workspace = nullptr);

// Direct quantized-activation grouped projection. activation is [rows,K] in
// any canonical projection layout; output is [rows,N] FP32. This includes the
// W4A8, FP8/FP8, MX/MX, NV/NV, and BitNet activation pairs. Architectures
// without a competitive direct packet dot may decode once into caller/thread
// workspace before entering the optimized grouped projection.
Status moe_grouped_prepacked_quantized(const CpuPackedWeights* expert_weights,
                                       long long experts,
                                       const CanonicalQuantTensor& activation,
                                       const int* expert_ids, float* output,
                                       Workspace* workspace = nullptr);

// Expert-indexed fused SwiGLU with direct canonical activation output. The
// output is [rows,I] and uses the same packet contract as quantize_canonical;
// only bounded worker/domain scratch is retained.
Status moe_grouped_prepacked_swiglu_quantized(
    const CpuPackedWeights* gate_weights, const CpuPackedWeights* up_weights,
    long long experts, FloatStorageInput x, const int* expert_ids,
    CanonicalQuantLayout output_layout, long long output_group_size,
    CanonicalQuantTensor* output, long long rows, bool scale_2d = false,
    Workspace* workspace = nullptr);

// Named sibling compatibility routes. Metadata is validated once, then the
// common prepared CPU kernel is used.
Status moe_grouped_fp8_prepacked_storage(const CpuPackedWeights* expert_weights,
                                         long long experts, FloatStorageInput x,
                                         const int* expert_ids,
                                         FloatStorageOutput output,
                                         long long rows,
                                         Workspace* workspace = nullptr);
Status moe_grouped_wna16_prepacked_storage(
    const CpuPackedWeights* expert_weights, long long experts,
    FloatStorageInput x, const int* expert_ids, FloatStorageOutput output,
    long long rows, Workspace* workspace = nullptr);
Status moe_grouped_nvfp4_prepacked_storage(
    const CpuPackedWeights* expert_weights, long long experts,
    FloatStorageInput x, const int* expert_ids, FloatStorageOutput output,
    long long rows, Workspace* workspace = nullptr);

// Input-gradient seam for frozen packed weights: grad_x[M,K] =
// grad_y[M,N] @ dequantize(W[N,K]).
Status qgemm_backward_input(QuantFormat format, const void* packed_weights,
                            const float* grad_y, float* grad_x, long long m,
                            long long n, long long k);

// Quantized projection with optional output-channel bias and activation.
Status qgemm_epilogue(QuantFormat format, const void* packed_weights,
                      const float* x, const float* bias, float* y, long long m,
                      long long n, long long k, LinearActivation activation);
Status qflux_gelu(QuantFormat format, const void* packed_weights,
                  const float* x, const float* bias, float* y, long long m,
                  long long n, long long k);

// Prequantized integer GEMMs. X is token-major [M,K], W is [N,K].
Status int8_gemm(const std::int8_t* weights, const std::int8_t* x,
                 const float* weight_scale, const float* activation_scale,
                 const std::int32_t* weight_row_sum,
                 const int* activation_zero_point, float* y, long long m,
                 long long n, long long k, bool asymmetric = false);
const char* int8_gemm_variant();
// Weight-only row-scaled int8 GEMM. W is [N,K] int8 with one f32 scale per
// output row, X is [M,K] f32, and Y is [M,N] f32.
Status qgemm_w8a32(const std::int8_t* weights, const float* weight_scale,
                   const float* x, float* y, long long m, long long n,
                   long long k);
const char* qgemm_w8a32_variant();
Status bitnet_int8_gemm(const void* packed_weights, const std::int8_t* x,
                        const float* activation_scale, float* y, long long m,
                        long long n, long long k);
Status qgemm_w8a8(const std::int8_t* weights, const std::int8_t* x,
                  const float* weight_scale, const float* activation_scale,
                  float* y, long long m, long long n, long long k);
Status qgemm_w8a8_azp(const std::int8_t* weights, const std::int8_t* x,
                      const float* weight_scale, const float* activation_scale,
                      const std::int32_t* weight_row_sum,
                      const int* activation_zero_point, float* y, long long m,
                      long long n, long long k);
Status qgemm_w2a8(const void* packed_weights, const std::int8_t* x,
                  const float* activation_scale, float* y, long long m,
                  long long n, long long k);
Status qgemv_w2a8(const void* packed_weights, const std::int8_t* x,
                  const float* activation_scale, float* y, long long n,
                  long long k);
Status qgemv_w2a8_v2(const void* packed_weights, const std::int8_t* x,
                     const float* activation_scale, float* y, long long n,
                     long long k);

// Codes-only FP8 GEMM with rank-1 row scales.
Status fp8_scaled_gemm(const std::uint8_t* weights, const std::uint8_t* x,
                       const float* weight_scale, const float* activation_scale,
                       float* y, long long m, long long n, long long k,
                       Float8Format format);
// Microscaled GEMMs use logical row-major scale tables rather than the
// accelerator-specific swizzled metadata layouts. A is [M,K], B is [N,K].
Status mxfp8_gemm(const std::uint8_t* a_codes, const float* a_scales,
                  const std::uint8_t* b_codes, const float* b_scales, float* y,
                  long long m, long long n, long long k);
Status nvfp4_gemm(const std::uint8_t* a_packed,
                  const std::uint8_t* a_scale_codes, float a_global_scale,
                  const std::uint8_t* b_packed,
                  const std::uint8_t* b_scale_codes, float b_global_scale,
                  float* y, long long m, long long n, long long k);
// GPTQ act-order: packed weight K positions are multiplied by x[...,perm[k]].
Status qgemm_actorder(QuantFormat format, const void* packed_weights,
                      const float* x, const int* permutation, float* y,
                      long long m, long long n, long long k);
// Codes-only FP8 weights with one scale per [block_n,block_k] tile.
Status fp8_blockscale_gemm(const std::uint8_t* weights, const float* x,
                           const float* tile_scales, float* y, long long m,
                           long long n, long long k, long long block_n,
                           long long block_k, Float8Format format);
// Fused activation dynamic-int8 quantization followed by BitNet W2A8 GEMM.
Status bitnet_fused_gemm(const void* packed_weights, const float* x, float* y,
                         long long m, long long n, long long k);
Status qgemm_w2a8_fused(const void* packed_weights, const float* x, float* y,
                        long long m, long long n, long long k);

// Fused quantized vocabulary projection + greedy selection. packed_weights is
// [vocab,hidden], hidden_states [rows,hidden], and token_ids [rows]. Lowest id
// wins ties.
Status quantized_lm_head_argmax(QuantFormat format, const void* packed_weights,
                                const float* hidden_states, int* token_ids,
                                long long rows, long long vocab,
                                long long hidden);

Status quantized_lm_head_sample(QuantFormat format, const void* packed_weights,
                                const float* hidden_states, const float* bias,
                                int* token_ids, long long rows, long long vocab,
                                long long hidden, LmHeadSampling mode, int k,
                                float top_p, float temperature,
                                std::uint32_t seed);
Status lm_head_sample(const float* hidden_states, const float* weights,
                      const float* bias, int* token_ids, long long rows,
                      long long vocab, long long hidden, LmHeadSampling mode,
                      int k, float top_p, float temperature,
                      std::uint32_t seed);
Status quantized_lm_head_masked_topk(
    QuantFormat format, const void* packed_weights, const float* hidden_states,
    const float* bias, const std::uint8_t* allow_mask, int* token_ids,
    float* log_probabilities, long long rows, long long vocab, long long hidden,
    int top_k, bool normalize_allowed = true);
Status lm_head_masked_topk(const float* hidden_states, const float* weights,
                           const float* bias, const std::uint8_t* allow_mask,
                           int* token_ids, float* log_probabilities,
                           long long rows, long long vocab, long long hidden,
                           int top_k, bool normalize_allowed = true);
Status quantized_lm_head_candidates(
    QuantFormat format, const void* packed_weights, const float* hidden_states,
    const float* bias, const int* candidate_ids, const long long* offsets,
    int* token_ids, float* log_probabilities, long long rows, long long vocab,
    long long hidden, long long candidates, int top_k);
Status lm_head_candidates(const float* hidden_states, const float* weights,
                          const float* bias, const int* candidate_ids,
                          const long long* offsets, int* token_ids,
                          float* log_probabilities, long long rows,
                          long long vocab, long long hidden,
                          long long candidates, int top_k);
Status quantized_lm_head_beam_advance(
    QuantFormat format, const void* packed_weights, const float* hidden_states,
    const float* bias, const float* cumulative_log_probabilities,
    int* next_token, int* parent_beam, float* next_cumulative, long long batch,
    long long beam_width, long long vocab, long long hidden);

// Dense row-conditioned grammar selection. forbidden is [vocab,vocab].
Status lm_head_constrained(const float* hidden_states, const float* weights,
                           const float* bias, const std::uint8_t* forbidden,
                           const int* previous_token, int* selected_token,
                           float* selected_log_probability, long long rows,
                           long long vocab, long long hidden, int eos_id = -1,
                           bool forbid_eos = false);

}  // namespace quixicore_cpu
