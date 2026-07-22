#pragma once

#include <cstdint>

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu {

enum class Float8Format;
enum class LinearActivation;
enum class LmHeadSampling { kArgmax, kCategorical, kTopK, kTopP };

// Quantized matrix multiply with packed W [N,K], activations X [M,K], and
// output Y [M,N]: Y = X @ dequantize(W)^T. Activations and output are f32.
Status qgemm(QuantFormat format, const void* packed_weights, const float* x,
             float* y, long long m, long long n, long long k);

// Input-gradient seam for frozen packed weights: grad_x[M,K] =
// grad_y[M,N] @ dequantize(W[N,K]).
Status qgemm_backward_input(QuantFormat format, const void* packed_weights,
                            const float* grad_y, float* grad_x, long long m,
                            long long n, long long k);

// Quantized projection with optional output-channel bias and activation.
Status qgemm_epilogue(QuantFormat format, const void* packed_weights,
                      const float* x, const float* bias, float* y,
                      long long m, long long n, long long k,
                      LinearActivation activation);

// Prequantized integer GEMMs. X is token-major [M,K], W is [N,K].
Status int8_gemm(const std::int8_t* weights, const std::int8_t* x,
                 const float* weight_scale, const float* activation_scale,
                 const std::int32_t* weight_row_sum,
                 const int* activation_zero_point, float* y, long long m,
                 long long n, long long k, bool asymmetric = false);
Status bitnet_int8_gemm(const void* packed_weights, const std::int8_t* x,
                        const float* activation_scale, float* y, long long m,
                        long long n, long long k);

// Codes-only FP8 GEMM with rank-1 row scales.
Status fp8_scaled_gemm(const std::uint8_t* weights,
                       const std::uint8_t* x, const float* weight_scale,
                       const float* activation_scale, float* y, long long m,
                       long long n, long long k, Float8Format format);
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
Status bitnet_fused_gemm(const void* packed_weights, const float* x,
                         float* y, long long m, long long n, long long k);

// Fused quantized vocabulary projection + greedy selection. packed_weights is
// [vocab,hidden], hidden_states [rows,hidden], and token_ids [rows]. Lowest id
// wins ties.
Status quantized_lm_head_argmax(QuantFormat format, const void* packed_weights,
                                const float* hidden_states, int* token_ids,
                                long long rows, long long vocab,
                                long long hidden);

Status quantized_lm_head_sample(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias, int* token_ids,
    long long rows, long long vocab, long long hidden, LmHeadSampling mode,
    int k, float top_p, float temperature, std::uint32_t seed);
Status quantized_lm_head_masked_topk(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias,
    const std::uint8_t* allow_mask, int* token_ids, float* log_probabilities,
    long long rows, long long vocab, long long hidden, int top_k,
    bool normalize_allowed = true);
Status quantized_lm_head_candidates(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias, const int* candidate_ids,
    const long long* offsets, int* token_ids, float* log_probabilities,
    long long rows, long long vocab, long long hidden, long long candidates,
    int top_k);
Status quantized_lm_head_beam_advance(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias,
    const float* cumulative_log_probabilities, int* next_token,
    int* parent_beam, float* next_cumulative, long long batch,
    long long beam_width, long long vocab, long long hidden);

// Dense row-conditioned grammar selection. forbidden is [vocab,vocab].
Status lm_head_constrained(const float* hidden_states, const float* weights,
                           const float* bias, const std::uint8_t* forbidden,
                           const int* previous_token, int* selected_token,
                           float* selected_log_probability, long long rows,
                           long long vocab, long long hidden, int eos_id = -1,
                           bool forbid_eos = false);

}  // namespace quixicore_cpu
