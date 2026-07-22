#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

bool close(float lhs, float rhs, float tolerance = 2e-4f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

bool close_array(const float* lhs, const float* rhs, long long count,
                 float tolerance = 2e-4f) {
  for (long long i = 0; i < count; ++i) {
    if (!close(lhs[i], rhs[i], tolerance)) return false;
  }
  return true;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;

  const float q[] = {1.0f, 0.5f, -0.5f, 1.0f,
                     0.25f, 1.0f, 1.0f, -0.25f};
  const float k[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float v[] = {2.0f, -1.0f, 0.5f, 3.0f};
  float regular_out[8], staged_out[8], lse[4];
  ok &= require(attention(q, k, v, regular_out, 2, 1, 2, 2, 2, false) ==
                    Status::kOk &&
                    attention_with_lse(q, k, v, nullptr, staged_out, lse,
                                       2, 1, 2, 2, 2, false) == Status::kOk &&
                    close_array(regular_out, staged_out, 8),
                "attention forward + LSE");
  const float grad_out[] = {0.2f, -0.1f, 0.3f, 0.4f,
                            -0.5f, 0.7f, 0.6f, -0.2f};
  float delta[4], dq[8], dk[4], dv[4];
  float expected_dq[8], expected_dk[4], expected_dv[4];
  ok &= require(attention_backward_prep(staged_out, grad_out, delta, 2, 2,
                                        2) == Status::kOk &&
                    attention_backward_dq(q, k, v, grad_out, lse, delta, dq,
                                          2, 1, 2, 2, 2, false) ==
                        Status::kOk &&
                    attention_backward_dkv(q, k, v, grad_out, lse, delta,
                                           dk, dv, 2, 1, 2, 2, 2, false) ==
                        Status::kOk &&
                    attention_backward(q, k, v, grad_out, expected_dq,
                                       expected_dk, expected_dv, 2, 1, 2, 2,
                                       2, false) == Status::kOk &&
                    close_array(dq, expected_dq, 8) &&
                    close_array(dk, expected_dk, 4) &&
                    close_array(dv, expected_dv, 4),
                "public attention backward stages");

  const float prefix_out[] = {2.0f, 9.0f};
  const float suffix_out[] = {4.0f, 7.0f};
  const float prefix_lse[] = {0.0f, 0.0f};
  const float suffix_lse[] = {0.0f, 1.0f};
  float merged[2], merged_lse[2];
  ok &= require(merge_attention_states(
                        prefix_out, prefix_lse, suffix_out, suffix_lse,
                        merged, merged_lse, 2, 1, 1, 1) == Status::kOk &&
                    close(merged[0], 3.0f) && close(merged_lse[0], std::log(2.0f)) &&
                    close(merged[1], 7.0f) && close(merged_lse[1], 1.0f),
                "attention state merge");
  std::uint8_t merged_codes[2];
  float merged_fp8_lse[2];
  constexpr float kMergeScale = 0.25f;
  ok &= require(merge_attention_states_fp8(
                        prefix_out, prefix_lse, suffix_out, suffix_lse,
                        merged_codes, merged_fp8_lse, 2, 1, 1, 1,
                        kMergeScale) == Status::kOk &&
                    close(float8_decode(merged_codes[0],
                                        Float8Format::kE4M3FN) *
                              kMergeScale,
                          merged[0]) &&
                    close(float8_decode(merged_codes[1],
                                        Float8Format::kE4M3FN) *
                              kMergeScale,
                          merged[1]) &&
                    close_array(merged_fp8_lse, merged_lse, 2),
                "FP8 attention state merge");

  const float prefix_k0[] = {1.0f, 0.0f};
  const float prefix_v0[] = {1.0f, 2.0f};
  const float prefix_k1[] = {0.0f, 1.0f};
  const float prefix_v1[] = {3.0f, 4.0f};
  const float* prefix_ks[] = {prefix_k0, prefix_k1};
  const float* prefix_vs[] = {prefix_v0, prefix_v1};
  const long long prefix_lengths[] = {1, 1};
  const float suffix_k[] = {-1.0f, 0.5f};
  const float suffix_v[] = {5.0f, 6.0f};
  const int block_table[] = {0};
  const int context[] = {1};
  const float cascade_q[] = {0.5f, 1.0f};
  float cascade_out[2], flat_out[2];
  const float flat_k[] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.5f};
  const float flat_v[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  ok &= require(cascade_attention_multi(
                        cascade_q, prefix_ks, prefix_vs, prefix_lengths, 2,
                        suffix_k, suffix_v, block_table, context, cascade_out,
                        1, 1, 1, 1, 2, 1, 1) == Status::kOk &&
                    attention(cascade_q, flat_k, flat_v, flat_out, 1, 1, 1,
                              3, 2, false) == Status::kOk &&
                    close_array(cascade_out, flat_out, 2),
                "N-level cascade attention");

  const float decode_q[] = {0.5f, 1.0f};
  const float new_k[] = {0.0f, 1.0f};
  const float new_v[] = {3.0f, 4.0f};
  const float cosine[] = {1.0f};
  const float sine[] = {0.0f};
  const int positions[] = {0};
  const int append_at[] = {1};
  float decode_k_cache[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float decode_v_cache[] = {1.0f, 2.0f, 0.0f, 0.0f};
  float decode_out[2], decode_expected[2];
  ok &= require(decode_cache_attention(
                        decode_q, new_k, new_v, cosine, sine, positions,
                        append_at, nullptr, nullptr, decode_k_cache,
                        decode_v_cache, decode_out, 1, 1, 1, 2, 2, 1) ==
                        Status::kOk &&
                    attention(decode_q, decode_k_cache, decode_v_cache,
                              decode_expected, 1, 1, 1, 2, 2, false) ==
                        Status::kOk &&
                    close_array(decode_out, decode_expected, 2) &&
                    close_array(decode_k_cache + 2, new_k, 2),
                "decode cache attention");

  const float mla_q[] = {1, 2, 3, 4};
  const float mla_cos[] = {0};
  const float mla_sin[] = {1};
  const int mla_position[] = {0};
  float mla_q_out[4];
  ok &= require(mla_q_norm_rope(mla_q, mla_cos, mla_sin, mla_position,
                                nullptr, mla_q_out, 1, 1, 2, 2, 1, 0) ==
                        Status::kOk &&
                    close_array(mla_q_out,
                                std::vector<float>{1, 2, -4, 3}.data(), 4),
                "MLA Q norm + partial RoPE");
  const float latent[] = {5, 6};
  const float key_rope[] = {3, 4};
  const int mla_slot[] = {1};
  float mla_cache[8] = {};
  ok &= require(mla_kv_insert(latent, key_rope, mla_cos, mla_sin,
                              mla_position, mla_slot, nullptr, mla_cache, 1,
                              2, 2, 2, 1, 0) == Status::kOk &&
                    close_array(mla_cache + 4,
                                std::vector<float>{5, 6, -4, 3}.data(), 4),
                "MLA classic KV insert");

  const float dropout_input[] = {1, 2, 3, 4, 5, 6};
  float dropout_out[6], dropout_grad[6];
  ok &= require(dropout(dropout_input, dropout_out, 6, 0.25f, 17) ==
                        Status::kOk &&
                    dropout_backward(dropout_input, dropout_grad, 6, 0.25f,
                                     17) == Status::kOk &&
                    close_array(dropout_out, dropout_grad, 6),
                "dropout backward deterministic mask");

  const int ids[] = {2, 1, 2};
  const int sorted_ids[] = {1, 2, 2};
  const int permutation[] = {1, 0, 2};
  const float embedding_grad[] = {1, 2, 3, 4, 5, 6};
  float embedding_expected[8], embedding_sorted[8];
  ok &= require(embedding_backward(ids, embedding_grad, embedding_expected,
                                   4, 3, 2) == Status::kOk &&
                    embedding_backward_sorted(
                        sorted_ids, permutation, embedding_grad,
                        embedding_sorted, 4, 3, 2) == Status::kOk &&
                    close_array(embedding_expected, embedding_sorted, 8),
                "sorted embedding backward");

  const float fwht_input[] = {1, 2, 3, 4};
  const float fwht_sign[] = {1, -1, 1, -1};
  float fwht_output[4], fwht_inverse[4];
  ok &= require(fwht_rotate(fwht_input, fwht_sign, fwht_output, 1, 4) ==
                        Status::kOk &&
                    fwht_rotate(fwht_output, fwht_sign, fwht_inverse, 1, 4,
                                true) == Status::kOk &&
                    close_array(fwht_input, fwht_inverse, 4),
                "signed normalized FWHT round trip");

  const float ssd_c[] = {1, 2, 2, 1};
  const float ssd_b[] = {1, 0, 0, 1};
  const float ssd_x[] = {2, 3, 4, 5};
  const float ssd_log[] = {0, 0};
  float mamba_out[4], chunked_out[4];
  ok &= require(mamba2(ssd_c, ssd_b, ssd_x, ssd_log, mamba_out, 1, 1, 2,
                       2) == Status::kOk &&
                    ssd_chunked(ssd_c, ssd_b, ssd_x, ssd_log, chunked_out,
                                1, 1, 2, 2) == Status::kOk &&
                    close_array(mamba_out, chunked_out, 4),
                "explicit SSD chunked route");

  const float dense_hidden[] = {1, 2};
  const float dense_weights[] = {1, 0, 0, 1, 1, 1};
  const float dense_bias[] = {0, 0, 0.5f};
  int sampled = -1;
  ok &= require(lm_head_sample(dense_hidden, dense_weights, dense_bias,
                               &sampled, 1, 3, 2,
                               LmHeadSampling::kArgmax, 1, 0.0f, 1.0f, 0) ==
                        Status::kOk &&
                    sampled == 2,
                "dense LM-head sampling");
  const std::uint8_t allow_mask[] = {0xC0};
  int masked_token = -1;
  float masked_logp = 0;
  ok &= require(lm_head_masked_topk(
                        dense_hidden, dense_weights, dense_bias, allow_mask,
                        &masked_token, &masked_logp, 1, 3, 2, 1, true) ==
                        Status::kOk &&
                    masked_token == 1 && std::isfinite(masked_logp),
                "dense masked LM-head");
  const int candidates[] = {0, 2};
  const long long candidate_offsets[] = {0, 2};
  int candidate_token = -1;
  float candidate_logp = 0;
  ok &= require(lm_head_candidates(
                        dense_hidden, dense_weights, dense_bias, candidates,
                        candidate_offsets, &candidate_token, &candidate_logp,
                        1, 3, 2, 2, 1) == Status::kOk &&
                    candidate_token == 2 && std::isfinite(candidate_logp),
                "dense candidate LM-head");

  constexpr long long kQuantDim = 32;
  std::vector<float> quant_table(2 * kQuantDim);
  for (long long i = 0; i < 2 * kQuantDim; ++i) {
    quant_table[static_cast<std::size_t>(i)] = (i % 9 - 4) * 0.25f;
  }
  std::size_t table_bytes = 0;
  ok &= require(qgemv_packed_size(QuantFormat::kQ8_0, 2, kQuantDim,
                                  &table_bytes) == Status::kOk,
                "dequant gather packed size");
  std::vector<std::uint8_t> packed_table(table_bytes);
  std::vector<float> gathered(kQuantDim), unpacked(2 * kQuantDim);
  const int gather_id[] = {1};
  ok &= require(qgemv_pack(QuantFormat::kQ8_0, quant_table.data(), 2,
                           kQuantDim, packed_table.data()) == Status::kOk &&
                    qgemv_unpack(QuantFormat::kQ8_0, packed_table.data(), 2,
                                 kQuantDim, unpacked.data()) == Status::kOk &&
                    dequant_gather(packed_table.data(), gather_id,
                                   gathered.data(), 2, 1, kQuantDim,
                                   QuantFormat::kQ8_0, 2.0f) == Status::kOk,
                "dequant gather route");
  for (long long i = 0; i < kQuantDim; ++i) {
    ok &= require(close(gathered[static_cast<std::size_t>(i)],
                        2.0f * unpacked[kQuantDim + i]),
                  "dequant gather value");
  }

  const std::int8_t w8[] = {1, 2, -1, 3, -2, 1, 0, 2};
  const std::int8_t x8[] = {2, -1, 3, 1};
  const float ws[] = {0.5f, 0.25f};
  const float xs[] = {2.0f};
  float named_w8[2], generic_w8[2];
  ok &= require(qgemm_w8a8(w8, x8, ws, xs, named_w8, 1, 2, 4) ==
                        Status::kOk &&
                    int8_gemm(w8, x8, ws, xs, nullptr, nullptr, generic_w8,
                              1, 2, 4, false) == Status::kOk &&
                    close_array(named_w8, generic_w8, 2),
                "named W8A8 route");

  const float split_qkv[] = {1, 2, 3, 4, 5, 6};
  const float split_weight[] = {1, 1};
  const float split_cos[] = {1};
  const float split_sin[] = {0};
  const int split_position[] = {0};
  float packed_qkv[6], split_q[2], split_k[2], split_v[2];
  ok &= require(qk_norm_rope(
                        split_qkv, split_weight, split_weight, split_cos,
                        split_sin, split_position, packed_qkv, 1, 1, 1, 1,
                        2, 1) == Status::kOk &&
                    qk_norm_rope_split(
                        split_qkv, split_weight, split_weight, split_cos,
                        split_sin, split_position, split_q, split_k, split_v,
                        1, 1, 1, 1, 2, 1) == Status::kOk &&
                    close_array(packed_qkv, split_q, 2) &&
                    close_array(packed_qkv + 2, split_k, 2) &&
                    close_array(packed_qkv + 4, split_v, 2),
                "split QK-norm RoPE outputs");

  constexpr long long kSwinTokens = 2;
  constexpr long long kSwinDim = 32;
  std::vector<float> swin_qkv(kSwinTokens * 3 * kSwinDim, 0.0f);
  std::vector<float> swin_q(kSwinTokens * kSwinDim, 0.0f);
  std::vector<float> swin_k(kSwinTokens * kSwinDim, 0.0f);
  std::vector<float> swin_v(kSwinTokens * kSwinDim, 0.0f);
  for (long long token = 0; token < kSwinTokens; ++token) {
    swin_q[token * kSwinDim] = token == 0 ? 1.0f : 0.5f;
    swin_k[token * kSwinDim] = token == 0 ? 0.25f : 1.0f;
    for (long long d = 0; d < kSwinDim; ++d) {
      swin_v[token * kSwinDim + d] = static_cast<float>(token + d) * 0.1f;
      swin_qkv[(token * 3) * kSwinDim + d] =
          swin_q[token * kSwinDim + d];
      swin_qkv[(token * 3 + 1) * kSwinDim + d] =
          swin_k[token * kSwinDim + d];
      swin_qkv[(token * 3 + 2) * kSwinDim + d] =
          swin_v[token * kSwinDim + d];
    }
  }
  const float swin_bias[] = {0.1f, -0.2f, 0.0f, 0.3f};
  std::vector<float> swin_out(kSwinTokens * kSwinDim);
  std::vector<float> swin_expected(kSwinTokens * kSwinDim);
  ok &= require(swin_attention_d32(
                        swin_qkv.data(), swin_bias, nullptr, swin_out.data(),
                        1, kSwinTokens, 1) == Status::kOk &&
                    biased_attention(
                        swin_q.data(), swin_k.data(), swin_v.data(),
                        swin_bias, nullptr, swin_expected.data(), 1, 1,
                        kSwinTokens, kSwinTokens, kSwinDim) == Status::kOk &&
                    close_array(swin_out.data(), swin_expected.data(),
                                kSwinTokens * kSwinDim),
                "packed Swin attention");

  const float flux_x[] = {1, 2};
  const float flux_w[] = {1, 3, 2, 4};  // [K,N]
  const float flux_bias[] = {0.5f, -0.5f};
  float flux_out[2], flux_dense[2], flux_expected[2];
  ok &= require(flux_gelu(flux_x, flux_w, flux_bias, flux_out, 1, 2, 2,
                          GeluApprox::kErf) == Status::kOk &&
                    dense_gemm(flux_x, flux_w, flux_dense, 1, 2, 2) ==
                        Status::kOk &&
                    gelu(std::vector<float>{flux_dense[0] + 0.5f,
                                            flux_dense[1] - 0.5f}.data(),
                         flux_expected, 2, GeluApprox::kErf) == Status::kOk &&
                    close_array(flux_out, flux_expected, 2),
                "Flux GELU projection");
  const float decode_weight[] = {1, 2, 3, 4};  // [N,K]
  float decode_linear_out[2], decode_linear_expected[2];
  ok &= require(decode_linear(flux_x, decode_weight, flux_bias,
                              decode_linear_out, 1, 2, 2, false) ==
                        Status::kOk &&
                    linear_epilogue(flux_x, decode_weight, flux_bias, nullptr,
                                    decode_linear_expected, 1, 2, 2) ==
                        Status::kOk &&
                    close_array(decode_linear_out, decode_linear_expected, 2),
                "decode linear route");

  std::vector<float> qflux_weights(kQuantDim, 0.0f);
  std::vector<float> qflux_input(kQuantDim, 0.0f);
  for (long long i = 0; i < kQuantDim; ++i) {
    qflux_weights[static_cast<std::size_t>(i)] = (i % 5 - 2) * 0.25f;
    qflux_input[static_cast<std::size_t>(i)] = (i % 3 - 1) * 0.5f;
  }
  std::size_t qflux_bytes = 0;
  ok &= require(qgemv_packed_size(QuantFormat::kQ8_0, 1, kQuantDim,
                                  &qflux_bytes) == Status::kOk,
                "qflux packed size");
  std::vector<std::uint8_t> qflux_packed(qflux_bytes);
  const float qflux_bias_value[] = {0.25f};
  float qflux_out_value = 0, qflux_expected_value = 0;
  ok &= require(qgemv_pack(QuantFormat::kQ8_0, qflux_weights.data(), 1,
                           kQuantDim, qflux_packed.data()) == Status::kOk &&
                    qflux_gelu(QuantFormat::kQ8_0, qflux_packed.data(),
                               qflux_input.data(), qflux_bias_value,
                               &qflux_out_value, 1, 1, kQuantDim) ==
                        Status::kOk &&
                    qgemm_epilogue(QuantFormat::kQ8_0, qflux_packed.data(),
                                   qflux_input.data(), qflux_bias_value,
                                   &qflux_expected_value, 1, 1, kQuantDim,
                                   LinearActivation::kGeluTanh) ==
                        Status::kOk &&
                    close(qflux_out_value, qflux_expected_value),
                "quantized Flux GELU route");

  const float norm_x[] = {1, 2, 3, 4};
  const float norm_residual[] = {0, 0, 0, 0};
  const float norm_weight[] = {1, 1, 1, 1};
  std::uint8_t norm_codes[4];
  float norm_residual_out[4], norm_reference[4];
  constexpr float kStaticScale = 0.01f;
  ok &= require(rms_norm_add_quant_float8_static(
                        norm_x, norm_residual, norm_weight, norm_codes,
                        norm_residual_out, 1, 4, kStaticScale) == Status::kOk &&
                    rms_norm_add(norm_x, norm_residual, norm_weight,
                                 norm_reference, norm_residual_out, 1, 4) ==
                        Status::kOk,
                "static FP8 norm quantization");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(float8_decode(norm_codes[i],
                                     Float8Format::kE4M3FN) *
                            kStaticScale,
                        norm_reference[i], 0.08f),
                  "static FP8 norm quant value");
  }

  std::vector<float> ternary(kQuantDim, 1.0f);
  ternary[1] = 0.0f;
  ternary[2] = -1.0f;
  std::vector<std::uint8_t> ternary_packed(10);
  std::vector<float> ternary_dequant(kQuantDim);
  std::vector<std::int8_t> ternary_x(kQuantDim, 1);
  float w2_named = 0, w2_v2 = 0;
  ok &= require(ternary_pack(ternary.data(), ternary_packed.data(),
                             ternary_dequant.data(), 1, kQuantDim,
                             kQuantDim) == Status::kOk &&
                    qgemv_w2a8(ternary_packed.data(), ternary_x.data(), xs,
                               &w2_named, 1, kQuantDim) == Status::kOk &&
                    qgemv_w2a8_v2(ternary_packed.data(), ternary_x.data(),
                                  xs, &w2_v2, 1, kQuantDim) == Status::kOk &&
                    close(w2_named, w2_v2),
                "named W2A8 GEMV routes");

  const long long sampled64[] = {1, -1, 3, 4, -1, -1};
  const std::uint8_t discard64[] = {0, 1};
  const long long backup64[] = {8, 9};
  long long next64[2], valid64[2];
  ok &= require(eagle_prepare_next_token_int64(
                        sampled64, discard64, backup64, next64, valid64, 2,
                        3, 10) == Status::kOk &&
                    next64[0] == 3 && valid64[0] == 2 &&
                    next64[1] == 9 && valid64[1] == 0,
                "EAGLE int64 next-token metadata");
  const long long expand_input[] = {7, -1};
  const long long expand_offsets[] = {0, 2, 5};
  long long expanded64[5];
  ok &= require(eagle_expand_int64(expand_input, expand_offsets, expanded64,
                                   2, 5, -1, 4) == Status::kOk &&
                    expanded64[0] == 7 && expanded64[1] == 7 &&
                    expanded64[2] == 4 && expanded64[4] == 4,
                "EAGLE int64 expansion");
  long long sequence64[] = {3};
  const long long decode_position64[] = {2};
  const int eagle_blocks[] = {5, 6};
  long long clamped64[1], slots64[1];
  ok &= require(eagle_step_slot_mapping_int64(
                        sequence64, decode_position64, eagle_blocks,
                        clamped64, slots64, 1, 1, 2, 2, 8, -1) ==
                        Status::kOk &&
                    clamped64[0] == 3 && slots64[0] == 13 &&
                    sequence64[0] == 4,
                "EAGLE int64 slot metadata");

  const long long eagle_target_ids[] = {10, 11};
  const long long eagle_target_positions[] = {5, 6};
  const long long eagle_next[] = {20};
  const int eagle_query_start[] = {0, 2};
  const int eagle_query_end[] = {1};
  long long eagle_output_ids[4], eagle_output_positions[4];
  std::uint8_t eagle_rejected[4], eagle_masked[4];
  int eagle_new_indices[2], eagle_hidden_mapping[2];
  ok &= require(copy_and_expand_eagle(
                        eagle_target_ids, eagle_target_positions, eagle_next,
                        eagle_query_start, eagle_query_end, eagle_output_ids,
                        eagle_output_positions, eagle_rejected, eagle_masked,
                        eagle_new_indices, eagle_hidden_mapping, 1, 2, 4, -1,
                        99, 2, false) == Status::kOk &&
                    eagle_output_ids[0] == 10 &&
                    eagle_output_ids[1] == 11 &&
                    eagle_output_ids[2] == 20 &&
                    eagle_output_ids[3] == 99 &&
                    eagle_masked[3] == 1 && eagle_new_indices[0] == 2 &&
                    eagle_new_indices[1] == 3,
                "EAGLE padded copy and expansion");

  const std::uint8_t paged_codes[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float paged_scales[] = {0.5f, 1.0f, 1.5f, 2.0f};
  const int paged_table[] = {1, 0};
  const int paged_cumulative[] = {0, 2};
  std::uint8_t gathered_codes[4];
  float gathered_scales[2];
  ok &= require(indexer_k_gather_paged(
                        paged_codes, paged_scales, paged_table,
                        paged_cumulative, gathered_codes, gathered_scales, 2,
                        1, 2, 1, 2, 2, 2) == Status::kOk &&
                    gathered_codes[0] == 3 && gathered_codes[1] == 4 &&
                    gathered_codes[2] == 1 && gathered_codes[3] == 2 &&
                    close(gathered_scales[0], 1.0f) &&
                    close(gathered_scales[1], 0.5f),
                "paged lightning-indexer gather");

  const int query_lengths[] = {4};
  const int kv_lengths[] = {4};
  const int vertical_indexes[] = {0, 3};
  const int slash_indexes[] = {4, 100};
  int sparse_block_count[1], sparse_block_offset[2];
  int sparse_column_count[1], sparse_column_index[2];
  ok &= require(convert_vertical_slash_indexes(
                        query_lengths, kv_lengths, vertical_indexes,
                        slash_indexes, sparse_block_count,
                        sparse_block_offset, sparse_column_count,
                        sparse_column_index, 1, 1, 1, 2, 2, 4, 2, true) ==
                        Status::kOk &&
                    sparse_block_count[0] == 0 &&
                    sparse_column_count[0] == 2 &&
                    sparse_column_index[0] == 0 &&
                    sparse_column_index[1] == 3,
                "vertical/slash sparse index conversion");

  const float route_logits[] = {-1.0f, 2.0f, 0.0f};
  int scored_ids[2];
  float scored_weights[2];
  ok &= require(moe_route_scored(route_logits, scored_ids, scored_weights, 1,
                                 3, 2, 0, true, 1.0f) == Status::kOk &&
                    scored_ids[0] == 1 && scored_ids[1] == 2 &&
                    close(scored_weights[0] + scored_weights[1], 1.0f),
                "scored MoE routing");
  const int lora_topk[] = {1, 0};
  const int token_lora[] = {0, 1};
  const int lora_ids[] = {0, 1};
  const std::uint8_t lora_enabled[] = {1, 1};
  int lora_sorted[8], lora_experts[4], lora_post_pad[2];
  ok &= require(moe_lora_align(
                        lora_topk, token_lora, lora_ids, lora_enabled,
                        lora_sorted, lora_experts, lora_post_pad, 2, 2, 2, 1,
                        2, 4, 2) == Status::kOk &&
                    lora_post_pad[0] == 2 && lora_post_pad[1] == 2 &&
                    lora_sorted[0] == 0 && lora_experts[0] == 1 &&
                    lora_sorted[4] == 1 && lora_experts[2] == 0,
                "per-LoRA MoE alignment");

  const float fp4_values[] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f,
                              -0.5f, -1.0f, -2.0f, -6.0f};
  std::uint8_t fp4_packed[6];
  float fp4_unpacked[12];
  ok &= require(fp32_to_fp4x2(fp4_values, fp4_packed, 12) == Status::kOk &&
                    fp4x2_to_fp32(fp4_packed, fp4_unpacked, 12) ==
                        Status::kOk &&
                    close_array(fp4_values, fp4_unpacked, 12),
                "raw FP4x2 conversion");

  constexpr long long kMicroM = 2;
  constexpr long long kMicroN = 2;
  constexpr long long kMicroK = 32;
  std::vector<float> micro_a(kMicroM * kMicroK);
  std::vector<float> micro_b(kMicroN * kMicroK);
  for (long long i = 0; i < kMicroM * kMicroK; ++i) {
    micro_a[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 7) % 19 - 9) * 0.125f;
  }
  for (long long i = 0; i < kMicroN * kMicroK; ++i) {
    micro_b[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 5) % 17 - 8) * 0.1875f;
  }
  std::vector<std::uint8_t> mx_a(kMicroM * kMicroK);
  std::vector<std::uint8_t> mx_b(kMicroN * kMicroK);
  float mx_a_scales[kMicroM], mx_b_scales[kMicroN];
  std::vector<float> mx_a_dequant(kMicroM * kMicroK);
  std::vector<float> mx_b_dequant(kMicroN * kMicroK);
  float mx_out[kMicroM * kMicroN];
  float mx_expected[kMicroM * kMicroN] = {};
  ok &= require(mxfp8_quantize(micro_a.data(), mx_a.data(), mx_a_scales,
                               kMicroM, kMicroK) == Status::kOk &&
                    mxfp8_quantize(micro_b.data(), mx_b.data(), mx_b_scales,
                                   kMicroN, kMicroK) == Status::kOk &&
                    dequantize_float8(mx_a.data(), mx_a_scales,
                                      mx_a_dequant.data(), kMicroM, kMicroK,
                                      32) == Status::kOk &&
                    dequantize_float8(mx_b.data(), mx_b_scales,
                                      mx_b_dequant.data(), kMicroN, kMicroK,
                                      32) == Status::kOk &&
                    mxfp8_gemm(mx_a.data(), mx_a_scales, mx_b.data(),
                               mx_b_scales, mx_out, kMicroM, kMicroN,
                               kMicroK) == Status::kOk,
                "MXFP8 quantize and GEMM");
  for (long long row = 0; row < kMicroM; ++row) {
    for (long long column = 0; column < kMicroN; ++column) {
      for (long long input = 0; input < kMicroK; ++input) {
        mx_expected[row * kMicroN + column] +=
            mx_a_dequant[row * kMicroK + input] *
            mx_b_dequant[column * kMicroK + input];
      }
    }
  }
  ok &= require(close_array(mx_out, mx_expected, kMicroM * kMicroN, 1e-3f),
                "MXFP8 logical scale GEMM values");
  float mx_zero[32] = {};
  std::uint8_t mx_zero_codes[32];
  float mx_zero_scale = 0.0f;
  ok &= require(mxfp8_quantize(mx_zero, mx_zero_codes, &mx_zero_scale, 1,
                               32) == Status::kOk &&
                    close(mx_zero_scale, std::ldexp(1.0f, -39), 0.0f),
                "MXFP8 E8M0 minimum scale");

  constexpr long long kNvK = 16;
  std::vector<float> nv_a(kMicroM * kNvK);
  std::vector<float> nv_b(kMicroN * kNvK);
  for (long long i = 0; i < kMicroM * kNvK; ++i) {
    nv_a[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 3) % 15 - 7) * 0.25f;
    nv_b[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 11) % 13 - 6) * 0.375f;
  }
  std::uint8_t nv_a_packed[kMicroM * kNvK / 2];
  std::uint8_t nv_b_packed[kMicroN * kNvK / 2];
  std::uint8_t nv_a_scales[kMicroM], nv_b_scales[kMicroN];
  float nv_a_global = 0.0f, nv_b_global = 0.0f;
  float nv_out[kMicroM * kMicroN];
  float nv_expected[kMicroM * kMicroN] = {};
  ok &= require(nvfp4_quantize(nv_a.data(), nv_a_packed, nv_a_scales,
                               &nv_a_global, kMicroM, kNvK) == Status::kOk &&
                    nvfp4_quantize(nv_b.data(), nv_b_packed, nv_b_scales,
                                   &nv_b_global, kMicroN, kNvK) ==
                        Status::kOk &&
                    nvfp4_gemm(nv_a_packed, nv_a_scales, nv_a_global,
                               nv_b_packed, nv_b_scales, nv_b_global, nv_out,
                               kMicroM, kMicroN, kNvK) == Status::kOk,
                "NVFP4 quantize and GEMM");
  for (long long row = 0; row < kMicroM; ++row) {
    for (long long column = 0; column < kMicroN; ++column) {
      for (long long input = 0; input < kNvK; ++input) {
        const auto a_byte = nv_a_packed[row * (kNvK / 2) + input / 2];
        const auto b_byte = nv_b_packed[column * (kNvK / 2) + input / 2];
        const float a_code = fp4_e2m1_decode(
            static_cast<std::uint8_t>((a_byte >> (4 * (input & 1))) & 15));
        const float b_code = fp4_e2m1_decode(
            static_cast<std::uint8_t>((b_byte >> (4 * (input & 1))) & 15));
        nv_expected[row * kMicroN + column] +=
            a_code * b_code *
            float8_decode(nv_a_scales[row], Float8Format::kE4M3FN) *
            float8_decode(nv_b_scales[column], Float8Format::kE4M3FN) *
            nv_a_global * nv_b_global;
      }
    }
  }
  ok &= require(close_array(nv_out, nv_expected, kMicroM * kMicroN, 1e-3f),
                "NVFP4 logical scale GEMM values");
  float nv_gemv_out[kMicroN];
  float nv_gemv_expected[kMicroN] = {};
  ok &= require(nvfp4_gemv(nv_b_packed, nv_b_scales, nv_b_global,
                           nv_a.data(), nv_gemv_out, kMicroN, kNvK) ==
                        Status::kOk,
                "NVFP4 split-layout GEMV");
  for (long long row = 0; row < kMicroN; ++row) {
    for (long long input = 0; input < kNvK; ++input) {
      const auto byte = nv_b_packed[row * (kNvK / 2) + input / 2];
      nv_gemv_expected[row] +=
          fp4_e2m1_decode(static_cast<std::uint8_t>(
              (byte >> (4 * (input & 1))) & 15)) *
          float8_decode(nv_b_scales[row], Float8Format::kE4M3FN) *
          nv_b_global * nv_a[input];
    }
  }
  ok &= require(close_array(nv_gemv_out, nv_gemv_expected, kMicroN, 1e-3f),
                "NVFP4 split-layout GEMV values");

  std::uint8_t mx4_packed[kMicroN * kMicroK / 2];
  const std::uint8_t mx4_scales[] = {127, 128};
  ok &= require(fp32_to_fp4x2(micro_b.data(), mx4_packed,
                              kMicroN * kMicroK) == Status::kOk,
                "MXFP4 test packing");
  float mx4_out[kMicroN];
  float mx4_expected[kMicroN] = {};
  ok &= require(mxfp4_gemv(mx4_packed, mx4_scales, micro_a.data(), mx4_out,
                           kMicroN, kMicroK) == Status::kOk,
                "MXFP4 split-layout GEMV");
  for (long long row = 0; row < kMicroN; ++row) {
    const float scale = std::ldexp(1.0f, static_cast<int>(mx4_scales[row]) -
                                            127);
    for (long long input = 0; input < kMicroK; ++input) {
      const auto byte = mx4_packed[row * (kMicroK / 2) + input / 2];
      mx4_expected[row] +=
          fp4_e2m1_decode(static_cast<std::uint8_t>(
              (byte >> (4 * (input & 1))) & 15)) *
          scale * micro_a[input];
    }
  }
  ok &= require(close_array(mx4_out, mx4_expected, kMicroN, 1e-3f),
                "MXFP4 split-layout GEMV values");

  std::vector<std::uint8_t> moe_fp8_weights(2 * 4);
  const float moe_fp8_values[] = {1, 2, 3, 4, -1, 0.5f, 2, -2};
  for (int i = 0; i < 8; ++i) {
    moe_fp8_weights[static_cast<std::size_t>(i)] =
        float8_encode(moe_fp8_values[i], Float8Format::kE4M3FN);
  }
  const float moe_fp8_scales[] = {0.5f, 2.0f};
  const float moe_fp8_x[] = {1, -1, 2, 0.5f};
  const int one_expert[] = {0};
  float moe_fp8_out[2];
  ok &= require(moe_gemm_fp8(moe_fp8_x, moe_fp8_weights.data(),
                             moe_fp8_scales, one_expert, moe_fp8_out, 1, 1,
                             4, 2) == Status::kOk &&
                    close(moe_fp8_out[0], 3.5f) &&
                    close(moe_fp8_out[1], 3.0f),
                "FP8 weight-only MoE GEMM");

  std::uint32_t wna4_word = 0;
  float wna4_expected = 0.0f;
  const float wna4_x[] = {1, 1, 1, 1, 1, 1, 1, 1};
  const float wna4_scales[] = {0.5f, 1.0f};
  for (int input = 0; input < 8; ++input) {
    const int local = input < 4 ? 2 * input : 2 * (input - 4) + 1;
    const unsigned code = static_cast<unsigned>(8 + input);
    wna4_word |= code << (local * 4);
    wna4_expected += (static_cast<float>(code) - 8.0f) *
                     wna4_scales[input / 4];
  }
  float wna4_out = 0.0f;
  ok &= require(moe_gemm_wna16(
                        wna4_x, &wna4_word, wna4_scales, nullptr,
                        one_expert, &wna4_out, 1, 1, 8, 1, 4, 4) ==
                        Status::kOk &&
                    close(wna4_out, wna4_expected),
                "WNA16 int4 MoE GEMM");

  const int nv_experts[] = {0, 0};
  float moe_nv_out[kMicroM * kMicroN];
  const float moe_nv_alpha[] = {nv_a_global * nv_b_global};
  ok &= require(moe_gemm_nvfp4(
                        nv_a_packed, nv_b_packed, nv_a_scales, nv_b_scales,
                        moe_nv_alpha, nv_experts, moe_nv_out, kMicroM, 1,
                        kNvK, kMicroN) == Status::kOk &&
                    close_array(moe_nv_out, nv_out,
                                kMicroM * kMicroN, 1e-3f),
                "dual-operand NVFP4 MoE GEMM");
  ok &= require(mxfp8_quantize(micro_a.data(), mx_a.data(), mx_a_scales,
                               1, 31) == Status::kInvalidShape &&
                    nvfp4_quantize(nv_a.data(), nv_a_packed, nv_a_scales,
                                   &nv_a_global, 1, 15) ==
                        Status::kInvalidShape &&
                    merge_attention_states_fp8(
                        prefix_out, prefix_lse, suffix_out, suffix_lse,
                        merged_codes, merged_fp8_lse, 2, 1, 1, 1, 0.0f) ==
                        Status::kInvalidShape &&
                    moe_gemm_wna16(
                        wna4_x, &wna4_word, wna4_scales, nullptr,
                        one_expert, &wna4_out, 1, 1, 8, 1, 4, 2) ==
                        Status::kInvalidShape,
                "low-bit entrypoint validation");

  if (!ok) return 1;
  std::cout << "sibling entrypoint tests passed\n";
  return 0;
}
