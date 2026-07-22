#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"
#include "quixicore_cpu/rms_norm.h"

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ":"      \
                << __LINE__ << '\n';                                      \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

bool close(float actual, double expected, double atol = 2e-5,
           double rtol = 2e-4) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::fabs(static_cast<double>(actual) - expected) <=
             atol + rtol * std::fabs(expected);
}

double dot(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  double result = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) result += lhs[i] * rhs[i];
  return result;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;

  // Fused/backward norms, including feature gradients.
  {
    std::vector<float> x = {1.0f, -2.0f, 0.5f, 3.0f, 1.5f, -0.25f};
    const std::vector<float> weight = {1.1f, 0.7f, -0.5f};
    const std::vector<float> bias = {0.1f, -0.2f, 0.3f};
    const std::vector<float> grad = {0.5f, -1.0f, 0.25f, 0.2f, 0.4f, -0.7f};
    std::vector<float> dx(6), dw(3), db(3), y(6), residual(6);
    REQUIRE(layer_norm_backward(x.data(), weight.data(), grad.data(),
                                dx.data(), dw.data(), db.data(), 2, 3) ==
            Status::kOk);
    constexpr float h = 1e-3f;
    for (int i = 0; i < 6; ++i) {
      const float saved = x[i];
      x[i] = saved + h;
      REQUIRE(layer_norm(x.data(), weight.data(), bias.data(), y.data(), 2, 3) ==
              Status::kOk);
      const double plus = dot(y, grad);
      x[i] = saved - h;
      REQUIRE(layer_norm(x.data(), weight.data(), bias.data(), y.data(), 2, 3) ==
              Status::kOk);
      const double minus = dot(y, grad);
      x[i] = saved;
      REQUIRE(close(dx[i], (plus - minus) / (2 * h), 4e-4, 8e-4));
    }
    std::vector<float> rdx(6), rdw(3);
    REQUIRE(rms_norm_backward(x.data(), weight.data(), grad.data(), rdx.data(),
                              rdw.data(), 2, 3) == Status::kOk);
    for (int i = 0; i < 6; ++i) REQUIRE(std::isfinite(rdx[i]));
    const float r[] = {0.5f, 0.5f, 0.5f, -1.0f, 1.0f, 2.0f};
    REQUIRE(rms_norm_add(x.data(), r, weight.data(), y.data(), residual.data(),
                         2, 3) == Status::kOk);
    for (int i = 0; i < 6; ++i) REQUIRE(close(residual[i], x[i] + r[i]));
    REQUIRE(layer_norm_add(x.data(), r, weight.data(), bias.data(), y.data(),
                           residual.data(), 2, 3) == Status::kOk);
    std::vector<float> next(6);
    REQUIRE(rms_norm_residual_next(x.data(), weight.data(), r, weight.data(),
                                   residual.data(), next.data(), 2, 3) ==
            Status::kOk);
    for (float value : next) REQUIRE(std::isfinite(value));
  }

  // Rich cross entropy and deterministic logit transforms.
  {
    std::vector<float> logits = {1.0f, -0.5f, 2.0f};
    const int target[] = {2};
    float loss = 0.0f, lse = 0.0f;
    REQUIRE(cross_entropy_forward(logits.data(), target, &loss, &lse, 1, 3,
                                  -100, 0.1f, 0.01f, 2.5f) == Status::kOk);
    const float upstream[] = {0.7f};
    float gradient[3] = {};
    REQUIRE(cross_entropy_backward(logits.data(), target, upstream, gradient,
                                   1, 3, -100, 0.1f, 0.01f, 2.5f) ==
            Status::kOk);
    constexpr float h = 2e-3f;
    for (int i = 0; i < 3; ++i) {
      const float saved = logits[i];
      logits[i] = saved + h;
      float plus = 0, ignored = 0;
      REQUIRE(cross_entropy_forward(logits.data(), target, &plus, &ignored, 1,
                                    3, -100, 0.1f, 0.01f, 2.5f) == Status::kOk);
      logits[i] = saved - h;
      float minus = 0;
      REQUIRE(cross_entropy_forward(logits.data(), target, &minus, &ignored, 1,
                                    3, -100, 0.1f, 0.01f, 2.5f) == Status::kOk);
      logits[i] = saved;
      REQUIRE(close(gradient[i], 0.7 * (plus - minus) / (2 * h), 4e-4, 8e-4));
    }
    const float two_rows[] = {1, 2, 3, 4, 3, 2};
    int sampled[2], repeated[2];
    REQUIRE(typical_p_sample(two_rows, sampled, 2, 3, 0.8f, 1.0f, 9) ==
            Status::kOk);
    REQUIRE(typical_p_sample(two_rows, repeated, 2, 3, 0.8f, 1.0f, 9) ==
            Status::kOk);
    REQUIRE(std::equal(sampled, sampled + 2, repeated));
    const std::uint8_t mask[] = {0xA0, 0x60};
    float processed[6];
    REQUIRE(apply_token_bitmask(two_rows, mask, processed, 2, 3) == Status::kOk);
    REQUIRE(processed[0] == 1 && !std::isfinite(processed[1]) &&
            processed[2] == 3);
    const int bad[] = {1};
    REQUIRE(apply_bad_words(two_rows, bad, processed, 2, 3, 1) == Status::kOk);
    REQUIRE(!std::isfinite(processed[1]) && !std::isfinite(processed[4]));
    const int previous[] = {2, 2, 0, 1};
    const int lengths[] = {3, 1};
    REQUIRE(apply_repetition_penalty(two_rows, previous, lengths, processed, 2,
                                     3, 2, 2.0f, 0.5f, 0.25f) ==
            Status::kInvalidArgument);
    const int previous_ok[] = {2, 2, 1, 0};
    REQUIRE(apply_repetition_penalty(two_rows, previous_ok, lengths, processed,
                                     2, 3, 2, 2.0f, 0.5f, 0.25f) ==
            Status::kInvalidArgument);
    const int valid_lengths[] = {2, 1};
    REQUIRE(apply_repetition_penalty(two_rows, previous_ok, valid_lengths,
                                     processed, 2, 3, 2, 2.0f, 0.5f, 0.25f) ==
            Status::kOk);
    const float probabilities[] = {0.1f, 0.6f, 0.3f};
    REQUIRE(top_k_renorm(probabilities, processed, 1, 3, 2) == Status::kOk);
    REQUIRE(close(processed[0] + processed[1] + processed[2], 1.0));
    REQUIRE(processed[0] == 0.0f);
    REQUIRE(top_p_renorm(probabilities, processed, 1, 3, 0.7f) == Status::kOk);
    REQUIRE(close(processed[0] + processed[1] + processed[2], 1.0));
  }

  // Byte utilities, serving, and multimodal gradients.
  {
    const std::uint8_t bits[] = {1, 0, 1, 1, 0, 0, 0, 1, 1};
    std::uint8_t packed[2] = {};
    REQUIRE(packbits(bits, packed, 9, true) == Status::kOk);
    REQUIRE(packed[0] == 0xB1 && packed[1] == 0x80);
    const long long input_offsets[] = {0, 3, 9};
    const long long output_offsets[] = {0, 1, 2};
    REQUIRE(segment_packbits(bits, input_offsets, output_offsets, packed, 2, 9,
                             2, false) == Status::kOk);
    REQUIRE(packed[0] == 5 && packed[1] == 49);
    const float matrix[] = {1, 2, 3, 4, 5, 6};
    const int permutation[] = {2, 0, 1};
    float permuted[6] = {};
    REQUIRE(permute_cols(matrix, permutation, permuted, 2, 3) == Status::kOk);
    REQUIRE(permuted[0] == 3 && permuted[1] == 1 && permuted[5] == 5);

    const int ids[] = {1, 1, -1};
    const float grad[] = {1, 2, 3, 4, 8, 9};
    float table_grad[6];
    REQUIRE(embedding_backward(ids, grad, table_grad, 3, 3, 2) == Status::kOk);
    REQUIRE(table_grad[2] == 4 && table_grad[3] == 6);
    const float text[] = {1, 2, 3, 4, 5, 6};
    const float modal[] = {7, 8, 9, 10};
    const int span_offsets[] = {1, 2};
    const int span_lengths[] = {1, 1};
    const int modal_starts[] = {1, 0};
    int source[3];
    REQUIRE(build_multimodal_source_map(
                span_offsets, span_lengths, modal_starts, source, 2, 3) ==
            Status::kOk);
    REQUIRE(source[0] == -1 && source[1] == 1 && source[2] == 0);
    float merged[6];
    REQUIRE(merge_multimodal_spans(text, modal, source, merged, 3, 2, 2) ==
            Status::kOk);
    REQUIRE(merged[0] == 1 && merged[2] == 9 && merged[4] == 7);
    const int lengths[] = {2};
    const float unit_weight[] = {1, 1};
    float pooled[2];
    REQUIRE(mean_pool_rms_l2(text, unit_weight, lengths, pooled, 1, 3, 2) ==
            Status::kOk);
    REQUIRE(close(std::sqrt(pooled[0] * pooled[0] + pooled[1] * pooled[1]),
                  1.0));
  }

  // Vision composites on tiny shapes with independently obvious values.
  {
    const float image[] = {1, 2, 3, 4};  // B1,H2,W2,C1
    const float weight[] = {1, 1, 1, 1};
    const float bias[] = {0, 0, 0, 0};
    float merged[4];
    REQUIRE(patch_merge_layer_norm(image, weight, bias, merged, 1, 2, 2, 1) ==
            Status::kOk);
    const double mean = 2.5;
    const double inverse = 1.0 / std::sqrt(1.25 + 1e-5);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(close(merged[i], (image[i] - mean) * inverse));
    }
    const float projection[] = {1, 0, 0, 0, 0, 0, 0, 1};
    float projected[2];
    REQUIRE(space_to_depth_norm_linear(image, weight, bias, projection, nullptr,
                                       projected, 1, 2, 2, 1, 2, 2) ==
            Status::kOk);
    REQUIRE(close(projected[0], merged[0]) && close(projected[1], merged[3]));

    std::vector<float> hidden(256, 0.0f), w1(256 * 512, 0.0f), b1(256, 0.0f);
    std::vector<float> w2(7 * 256, 0.0f), b2(7), edge(7);
    b2[3] = 2.5f;
    REQUIRE(edge_mlp_256x7(hidden.data(), w1.data(), b1.data(), w2.data(),
                           b2.data(), edge.data(), 1, 1) == Status::kOk);
    REQUIRE(edge[3] == 2.5f && edge[0] == 0.0f);
  }

  // Attention backward finite difference and all remaining attention forms.
  {
    std::vector<float> q = {0.2f, -0.4f, 0.7f, 0.1f};
    std::vector<float> k = {0.3f, 0.5f, -0.2f, 0.8f};
    std::vector<float> v = {1.0f, -1.0f, 0.5f, 2.0f};
    const std::vector<float> go = {0.5f, -0.25f, 0.8f, 0.2f};
    std::vector<float> gq(4), gk(4), gv(4), output(4);
    REQUIRE(attention_backward(q.data(), k.data(), v.data(), go.data(),
                               gq.data(), gk.data(), gv.data(), 1, 1, 2, 2, 2,
                               true) == Status::kOk);
    auto objective = [&](std::vector<float>& query) {
      (void)attention(query.data(), k.data(), v.data(), output.data(), 1, 1,
                      2, 2, 2, true);
      return dot(output, go);
    };
    constexpr float h = 1e-3f;
    for (int i = 0; i < 4; ++i) {
      const float saved = q[i];
      q[i] += h;
      const double plus = objective(q);
      q[i] = saved - h;
      const double minus = objective(q);
      q[i] = saved;
      REQUIRE(close(gq[i], (plus - minus) / (2 * h), 4e-4, 8e-4));
    }
    REQUIRE(window_attention(q.data(), k.data(), v.data(), output.data(), 1, 1,
                             2, 2, 2, -1, 0) == Status::kOk);
    std::vector<float> causal(4);
    REQUIRE(attention(q.data(), k.data(), v.data(), causal.data(), 1, 1, 2, 2,
                      2, true) == Status::kOk);
    for (int i = 0; i < 4; ++i) REQUIRE(close(output[i], causal[i]));
    // packed [token,head,dim] is the same memory order for one head.
    const int cumulative[] = {0, 2};
    REQUIRE(varlen_attention(q.data(), k.data(), v.data(), cumulative,
                             cumulative, output.data(), 1, 1, 1, 2, true) ==
            Status::kOk);
    for (int i = 0; i < 4; ++i) REQUIRE(close(output[i], causal[i]));
    const float additive_bias[] = {0, -100, 0, 0};
    REQUIRE(biased_attention(q.data(), k.data(), v.data(), additive_bias,
                             nullptr, output.data(), 1, 1, 2, 2, 2) ==
            Status::kOk);
    REQUIRE(close(output[0], v[0], 1e-4, 1e-4));

    const float cos_table[] = {1, 0};
    const float sin_table[] = {0, 1};
    const int positions[] = {0, 1};
    REQUIRE(rope_table(q.data(), cos_table, sin_table, positions, output.data(),
                       2, 1, 2, 2, true) == Status::kOk);
    REQUIRE(close(output[0], q[0]) && close(output[1], q[1]));
    REQUIRE(close(output[2], -q[3]) && close(output[3], q[2]));
    const float packed_qkv[] = {1, 2, 3, 4, 5, 6};
    const float norm_weight[] = {1, 1};
    float qkv_out[6];
    const int position0[] = {0};
    REQUIRE(qk_norm_rope(packed_qkv, norm_weight, norm_weight, cos_table,
                         sin_table, position0, qkv_out, 1, 1, 1, 1, 2, 2) ==
            Status::kOk);
    REQUIRE(qkv_out[4] == 5 && qkv_out[5] == 6);
  }

  // Linear-attention, GDN, SSD/Mamba, and FFT semantics.
  {
    const float q[] = {1, 2};
    const float k[] = {3, 4};
    const float v[] = {5, 6};
    float out[2];
    REQUIRE(linear_attention_unnormalized(q, k, v, out, 1, 1, 2, 1) ==
            Status::kOk);
    REQUIRE(close(out[0], 39) && close(out[1], 78));
    REQUIRE(causal_linear_attention(q, k, v, out, 1, 1, 2, 1) == Status::kOk);
    REQUIRE(close(out[0], 15) && close(out[1], 78));
    const float cl[] = {0, static_cast<float>(std::log(0.5))};
    REQUIRE(decayed_linear_attention(q, k, v, cl, out, 1, 1, 2, 1) ==
            Status::kOk);
    REQUIRE(close(out[0], 15) && close(out[1], 63));
    REQUIRE(based_attention(q, k, v, out, 1, 1, 2, 1, 1) == Status::kOk);
    REQUIRE(close(out[0], 5 * (1 + 3 + 4.5)));
    REQUIRE(hedgehog_attention(q, k, v, out, 1, 1, 2, 1) == Status::kOk);
    REQUIRE(close(out[0], 11) && close(out[1], 11));

    const float gate[] = {1, 1};
    const float beta[] = {1, 1};
    float state[] = {0};
    const int cumulative[] = {0, 2};
    const int slot[] = {0};
    REQUIRE(gdn_recurrence(q, k, v, gate, beta, state, cumulative, slot, out,
                           1, 1, 1, 1, 1, 1, false) == Status::kOk);
    REQUIRE(close(out[0], 15) && close(out[1], -402));

    const float c[] = {1, 2};
    const float b[] = {3, 4};
    const float mx[] = {5, 6};
    const float mcl[] = {0, static_cast<float>(std::log(0.5))};
    REQUIRE(mamba2(c, b, mx, mcl, out, 1, 1, 2, 1) == Status::kOk);
    REQUIRE(close(out[0], 15) && close(out[1], 63));
    const float gy[] = {0.5f, -0.25f};
    float gc[2], gb[2], gx[2], gcl[2];
    REQUIRE(mamba2_backward(c, b, mx, mcl, gy, gc, gb, gx, gcl, 1, 1, 2, 1) ==
            Status::kOk);
    for (float value : gc) REQUIRE(std::isfinite(value));
    const float old_state[] = {2};
    const float alpha[] = {0.5f};
    const float one[] = {3};
    float next_state = 0;
    REQUIRE(ssd_decode(old_state, alpha, one, one, one, out, &next_state, 1, 1,
                       1) == Status::kOk);
    REQUIRE(close(next_state, 10) && close(out[0], 30));
    const float signal[] = {1, 2, 3, 4};
    const float convolution_kernel[] = {1, 0, 0, 0};
    float convolved[4];
    REQUIRE(fft_convolution(signal, convolution_kernel, convolved, 1, 1, 4) ==
            Status::kOk);
    REQUIRE(std::equal(signal, signal + 4, convolved));
  }

  // Extended matmul and complete portable MoE dataflow.
  {
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float bt[] = {1, 0, -1, 2, 1, 1};  // [N=2,K=3]
    float c[4];
    REQUIRE(dense_gemm_ex(a, bt, nullptr, c, 2, 2, 3, false, true) ==
            Status::kOk);
    REQUIRE(close(c[0], -2) && close(c[1], 7));
    const float bias[] = {1, -1};
    REQUIRE(linear_epilogue(a, bt, bias, nullptr, c, 2, 3, 2) == Status::kOk);
    REQUIRE(close(c[0], -1) && close(c[1], 6));
    REQUIRE(decode_swiglu(a, bt, bt, nullptr, nullptr, c, 2, 3, 2) ==
            Status::kOk);
    for (float value : c) REQUIRE(std::isfinite(value));
    const float b_kn[] = {1, 2, 0, 1, -1, 1};
    const float gate[] = {2, 0.5f};
    const float residual[] = {1, 1, 1, 1};
    REQUIRE(gemm_gate_residual(a, b_kn, nullptr, gate, residual, c, 2, 2, 3) ==
            Status::kOk);
    REQUIRE(close(c[0], -3) && close(c[1], 4.5));
    const float ar[] = {1, 2}, ai[] = {3, 4};
    const float br[] = {5, 6}, bi[] = {7, 8};
    float cr = 0, ci = 0;
    REQUIRE(complex_gemm(ar, ai, br, bi, &cr, &ci, 1, 1, 2) == Status::kOk);
    REQUIRE(close(cr, -36) && close(ci, 62));

    const float router[] = {4, 3, 2, 1};
    int ids[2];
    float routing_weights[2];
    REQUIRE(moe_route_grouped(router, nullptr, ids, routing_weights, 1, 4, 2,
                              2, 1, true) == Status::kOk);
    REQUIRE(ids[0] < 2 && ids[1] < 2);
    const int route_ids[] = {1, 0, 1, 0};
    int sorted[4], offsets[3], inverse[4];
    REQUIRE(moe_permute(route_ids, sorted, offsets, inverse, 2, 2, 2) ==
            Status::kOk);
    REQUIRE(offsets[0] == 0 && offsets[1] == 2 && offsets[2] == 4);
    const float tokens[] = {1, 2, 3, 4};
    float gathered[8];
    REQUIRE(moe_gather(tokens, sorted, gathered, 2, 4, 2) == Status::kOk);
    const float equal_weights[] = {0.5f, 0.5f, 0.25f, 0.75f};
    float finalized[4];
    REQUIRE(moe_finalize(gathered, inverse, equal_weights, finalized, 2, 2, 2) ==
            Status::kOk);
    REQUIRE(close(finalized[0], 1) && close(finalized[1], 2));
    const int row_experts[] = {0, 1};
    const float expert_matrix[] = {1, 0, 0, 1, 2, 0, 0, 2};
    float expert_out[4];
    REQUIRE(moe_grouped_gemm(tokens, expert_matrix, row_experts, expert_out, 2,
                             2, 2, 2) == Status::kOk);
    REQUIRE(expert_out[0] == 1 && expert_out[2] == 6);
    const float swiglu_weights[] = {1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1};
    float swiglu_out[4];
    REQUIRE(moe_grouped_swiglu(tokens, swiglu_weights, row_experts,
                               swiglu_out, 2, 2, 2, 2) == Status::kOk);
    for (float value : swiglu_out) REQUIRE(value > 0.0f);
  }

  // Packed serving paths, sparse metadata, KD, tau-tail, and masked optimizer.
  {
    const float table[] = {1, 2, 3, 4, 0, -1, -2, -3,
                           5, 6, 7, 8, 9, 10, 11, 12,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           -1, -2, -3, -4, 0, 1, 2, 3,
                           4, 5, 6, 7, 8, 9, 10, 11,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t packed_bytes = 0;
    REQUIRE(qgemv_packed_size(QuantFormat::kQ8_0, 2, 32, &packed_bytes) ==
            Status::kOk);
    std::vector<std::uint8_t> packed(packed_bytes);
    REQUIRE(qgemv_pack(QuantFormat::kQ8_0, table, 2, 32, packed.data()) ==
            Status::kOk);
    const int ids[] = {1, 0};
    float embedded[64];
    REQUIRE(quantized_embedding(packed.data(), ids, nullptr, embedded, 2, 2,
                                32, QuantFormat::kQ8_0) == Status::kOk);
    REQUIRE(close(embedded[0], -1, 0.1) && close(embedded[32], 1, 0.1));
    const long long offsets[] = {0, 2};
    float bag[32];
    REQUIRE(quantized_embedding_bag(
                packed.data(), ids, offsets, nullptr, bag, 2, 2, 1, 32,
                QuantFormat::kQ8_0, 1.0f, false, true) == Status::kOk);
    REQUIRE(close(bag[0], 0, 0.1));

    const float keys[] = {1, -2, 3, -4, 4, 3, 2, 1};
    const int mapping[] = {1, 0};
    std::uint8_t code_cache[8] = {};
    float scale_cache[4] = {};
    REQUIRE(indexer_k_quant_and_cache(keys, mapping, code_cache, scale_cache,
                                      2, 2, 4, 2, true) == Status::kOk);
    const int gather_slots[] = {1, 0};
    float gathered_keys[8];
    REQUIRE(indexer_k_gather(code_cache, scale_cache, gather_slots,
                             gathered_keys, 2, 2, 4, 2) == Status::kOk);
    for (int i = 0; i < 8; ++i) REQUIRE(close(gathered_keys[i], keys[i], 0.3));

    const int vertical[] = {0, -1};
    const int slash[] = {0, -1};
    const int contexts[] = {8};
    int mask[4];
    REQUIRE(minference_block_mask(vertical, slash, contexts, mask, 1, 1, 2,
                                   2, 4, 2, 2, 2, 1) == Status::kOk);
    REQUIRE(mask[0] == 1 && mask[3] == 1);

    const float teacher[] = {2, 1, 0, -1};
    const float student[] = {1.5f, 0.5f, 0.25f, -0.5f};
    float kd_loss, teacher_lse, student_lse;
    REQUIRE(kd_kl_dense_forward(teacher, student, &kd_loss, &teacher_lse,
                                &student_lse, 1, 4) == Status::kOk);
    float kd_grad[4];
    const float go = 1;
    REQUIRE(kd_kl_dense_backward(teacher, student, &teacher_lse, &student_lse,
                                 &go, kd_grad, 1, 4) == Status::kOk);
    REQUIRE(kd_loss >= 0 && close(std::accumulate(kd_grad, kd_grad + 4, 0.0),
                                  0, 1e-5));
    const int top_indices[] = {0, 1};
    const float top_probabilities[] = {0.6f, 0.3f};
    float top_loss, top_lse, top_grad[4];
    REQUIRE(kd_kl_topk_forward(student, top_indices, top_probabilities,
                               &top_loss, &top_lse, 1, 4, 2, 1.0f, true) ==
            Status::kOk);
    REQUIRE(kd_kl_topk_backward(student, top_indices, top_probabilities,
                                &top_lse, &go, top_grad, 1, 4, 2, 1.0f,
                                true) == Status::kOk);
    REQUIRE(std::isfinite(top_loss));

    const float qkv[] = {1, 2, 3, 4, 5, 6};
    const float gates[] = {0, 0};
    const float positional[] = {2};
    const int position[] = {0};
    float tailed[6];
    REQUIRE(tau_tail(qkv, gates, positional, position, tailed, 1, 1, 2, 1) ==
            Status::kOk);
    REQUIRE(close(tailed[0], 2) && close(tailed[2], 3) && close(tailed[4], 10));

    float parameters[] = {1, 2, 3, 4};
    const float gradients[] = {1, 1, 1, 1};
    float first[] = {0, 0, 0, 0};
    float second[] = {0, 0, 0, 0};
    const std::uint8_t update_mask[] = {1, 0};
    REQUIRE(adamw_masked(parameters, gradients, first, second, update_mask, 4,
                         2, 0, 0.1f, 0.9f, 0.99f, 1e-8f, 0.01f, 1) ==
            Status::kOk);
    REQUIRE(parameters[0] != 1 && parameters[2] == 3 && first[2] == 0);
  }

  // Quantized attention, RoPE cache seams, extended paged and cascade decode.
  {
    std::vector<float> q(64), k(64), v(64);
    for (int i = 0; i < 64; ++i) {
      q[i] = static_cast<float>((i % 7) - 3) / 4;
      k[i] = static_cast<float>((i % 5) - 2) / 3;
      v[i] = static_cast<float>((i % 9) - 4) / 5;
    }
    std::size_t bytes = 0;
    REQUIRE(qgemv_packed_size(QuantFormat::kQ8_0, 2, 32, &bytes) ==
            Status::kOk);
    std::vector<std::uint8_t> pk(bytes), pv(bytes);
    REQUIRE(qgemv_pack(QuantFormat::kQ8_0, k.data(), 2, 32, pk.data()) ==
            Status::kOk);
    REQUIRE(qgemv_pack(QuantFormat::kQ8_0, v.data(), 2, 32, pv.data()) ==
            Status::kOk);
    std::vector<float> quant_out(64), reference(64), ku(64), vu(64);
    REQUIRE(qgemv_unpack(QuantFormat::kQ8_0, pk.data(), 2, 32, ku.data()) ==
            Status::kOk);
    REQUIRE(qgemv_unpack(QuantFormat::kQ8_0, pv.data(), 2, 32, vu.data()) ==
            Status::kOk);
    REQUIRE(quantized_attention(q.data(), pk.data(), pv.data(),
                                quant_out.data(), 1, 1, 2, 32,
                                QuantFormat::kQ8_0, true) == Status::kOk);
    REQUIRE(attention(q.data(), ku.data(), vu.data(), reference.data(), 1, 1,
                      2, 2, 32, true) == Status::kOk);
    for (int i = 0; i < 64; ++i) REQUIRE(close(quant_out[i], reference[i], 2e-5));

    const float rq[] = {1, 2, 3, 4};
    const float cos[] = {1, 1};
    const float sin[] = {0, 0};
    const int positions[] = {0, 1};
    const float norm_weight[] = {1, 1};
    float rotated[4];
    REQUIRE(rope_q_norm(rq, cos, sin, positions, norm_weight, rotated, 2, 1,
                        2, 2, false) == Status::kOk);
    REQUIRE(std::equal(rq, rq + 4, rotated));
    const int slots[] = {1, 0};
    const float rv[] = {5, 6, 7, 8};
    float key_cache[4] = {}, value_cache[4] = {};
    REQUIRE(rope_kv_insert(rq, rv, cos, sin, positions, slots, nullptr,
                           key_cache, value_cache, 2, 2, 1, 2, 2, false) ==
            Status::kOk);
    REQUIRE(key_cache[0] == 3 && value_cache[2] == 5);

    const float decode_q[] = {1, 0};
    const int block_table[] = {0};
    const int context[] = {2};
    const int block_mask[] = {1};
    const float alibi[] = {0};
    const float sink[] = {-100};
    float decode_out[2];
    REQUIRE(paged_attention_advanced(
                decode_q, key_cache, value_cache, block_table, context,
                block_mask, alibi, sink, decode_out, 1, 1, 1, 1, 2, 2, 1,
                0.0f, 0, 0.0f) == Status::kOk);
    REQUIRE(std::isfinite(decode_out[0]));
    std::uint8_t fp8k[4], fp8v[4];
    for (int i = 0; i < 4; ++i) {
      fp8k[i] = float8_encode(key_cache[i], Float8Format::kE4M3FN);
      fp8v[i] = float8_encode(value_cache[i], Float8Format::kE4M3FN);
    }
    const float unit_scale[] = {1};
    float fp8_out[2];
    REQUIRE(paged_attention_fp8(
                decode_q, fp8k, fp8v, block_table, context, unit_scale,
                unit_scale, fp8_out, 1, 1, 1, 1, 2, 2, 1,
                Float8Format::kE4M3FN) == Status::kOk);
    REQUIRE(std::isfinite(fp8_out[0]));
    const float prefix_k[] = {1, 0};
    const float prefix_v[] = {9, 10};
    float cascade_out[2];
    REQUIRE(cascade_attention(decode_q, prefix_k, prefix_v, key_cache,
                              value_cache, block_table, context, cascade_out,
                              1, 1, 1, 1, 2, 1, 2, 1) == Status::kOk);
    REQUIRE(std::isfinite(cascade_out[0]));
    const std::uint8_t prefix_fp8_k[] = {
        float8_encode(1, Float8Format::kE4M3FN),
        float8_encode(0, Float8Format::kE4M3FN)};
    const std::uint8_t prefix_fp8_v[] = {
        float8_encode(9, Float8Format::kE4M3FN),
        float8_encode(10, Float8Format::kE4M3FN)};
    float cascade_fp8_out[2];
    REQUIRE(cascade_attention_fp8(
                decode_q, prefix_fp8_k, prefix_fp8_v, key_cache,
                value_cache, block_table, context, unit_scale, unit_scale,
                cascade_fp8_out, 1, 1, 1, 1, 2, 1, 2, 1,
                Float8Format::kE4M3FN) == Status::kOk);
    REQUIRE(close(cascade_fp8_out[0], cascade_out[0]) &&
            close(cascade_fp8_out[1], cascade_out[1]));
  }

  std::cout << "extended CPU kernel tests passed\n";
  return 0;
}
