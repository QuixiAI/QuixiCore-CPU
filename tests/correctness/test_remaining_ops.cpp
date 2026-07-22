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

bool close(float lhs, float rhs, float tolerance = 1e-4f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;

  const float glu_x[] = {2, -1, 3, 4};
  const float glu_grad[] = {5, 6};
  float glu_dx[4];
  ok &= require(glu_backward(glu_grad, glu_x, glu_dx, 1, 2,
                             GluMode::kReGlu) == Status::kOk &&
                    close(glu_dx[0], 15) && close(glu_dx[1], 0) &&
                    close(glu_dx[2], 10) && close(glu_dx[3], 0),
                "GLU backward");

  const float teacher[] = {2, 0, -1, 0, 1, 2};
  const float student[] = {1, 0, -1, 2, 1, 0};
  const int targets[] = {0, 2};
  float ce[2], kd[2], raw_lse[2], student_lse[2], teacher_lse[2];
  ok &= require(kd_ce_fused_forward(
                        teacher, student, targets, ce, kd, raw_lse,
                        student_lse, teacher_lse, 2, 3, 0.5f) == Status::kOk,
                "fused KD CE forward");
  float kd_only[2], kd_student_lse[2], kd_teacher_lse[2];
  ok &= require(kd_kl_dense_forward(
                        teacher, student, kd_only, kd_teacher_lse,
                        kd_student_lse, 2, 3, 0.5f) == Status::kOk &&
                    close(kd[0], kd_only[0]) &&
                    close(student_lse[1], kd_student_lse[1]),
                "fused KD CE composition");
  const float grad_ce[] = {1, 1};
  const float grad_kd[] = {0.5f, 0.5f};
  float kd_gradient[6];
  ok &= require(kd_ce_fused_backward(
                        teacher, student, targets, raw_lse, student_lse,
                        teacher_lse, grad_ce, grad_kd, kd_gradient, 2, 3,
                        0.5f) == Status::kOk,
                "fused KD CE backward");
  ok &= require(close(kd_gradient[0] + kd_gradient[1] + kd_gradient[2],
                      0.0f, 2e-5f),
                "fused KD CE gradient conservation");

  const int expert_ids[] = {0, 1, 1, 0};
  int sorted_rows[4], offsets[3], inverse[4];
  ok &= require(moe_permute(expert_ids, sorted_rows, offsets, inverse, 2, 2,
                            2) == Status::kOk,
                "MoE permute for schedule");
  int expert_tiles[2], gather_rows[4], inverse_padded[4], padded_offsets[3];
  ok &= require(moe_pad_schedule(
                        sorted_rows, offsets, inverse, expert_tiles,
                        gather_rows, inverse_padded, padded_offsets, 2, 2, 2,
                        2) == Status::kOk &&
                    padded_offsets[2] == 4 && expert_tiles[0] == 0 &&
                    expert_tiles[1] == 1 && inverse_padded[1] == 2,
                "MoE padded schedule");
  const float expert_out[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float weights[] = {0.25f, 0.75f, 0.5f, 0.5f};
  const float grad_out[] = {1, 2, 3, 4};
  float grad_expert[8], grad_weights[4];
  ok &= require(moe_finalize_backward(
                        grad_out, expert_out, inverse, weights, grad_expert,
                        grad_weights, 2, 2, 2) == Status::kOk &&
                    close(grad_expert[inverse[0] * 2], 0.25f) &&
                    close(grad_weights[0], 5.0f),
                "MoE finalize backward");
  const int gathered_indices[] = {0, 1, 0};
  const float gathered_gradient[] = {1, 2, 3, 4, 5, 6};
  float input_gradient[4];
  ok &= require(moe_gather_backward(gathered_gradient, gathered_indices,
                                    input_gradient, 2, 3, 2) == Status::kOk &&
                    close(input_gradient[0], 6) && close(input_gradient[3], 4),
                "MoE gather backward");
  const float grouped_x[] = {1, 2, 3, 4};
  const float grouped_w[] = {1, 0, 0, 1, 2, 0, 0, 2};
  const int row_experts[] = {0, 1};
  const float grouped_dy[] = {1, 2, 3, 4};
  float grouped_dx[4], grouped_dw[8];
  ok &= require(moe_grouped_gemm_backward_input(
                        grouped_dy, grouped_w, row_experts, grouped_dx, 2, 2,
                        2, 2) == Status::kOk &&
                    close(grouped_dx[0], 1) && close(grouped_dx[3], 8),
                "MoE grouped GEMM dx");
  ok &= require(moe_grouped_gemm_backward_weight(
                        grouped_x, grouped_dy, row_experts, grouped_dw, 2, 2,
                        2, 2) == Status::kOk &&
                    close(grouped_dw[0], 1) && close(grouped_dw[7], 16),
                "MoE grouped GEMM dw");

  constexpr long long kMoeRows = 2, kMoeExperts = 2, kMoeInput = 32;
  constexpr long long kMoeOutput = 2, kMoeIntermediate = 2;
  std::vector<float> moe_quant_x(kMoeRows * kMoeInput);
  std::vector<float> moe_quant_weights(
      kMoeExperts * 2 * kMoeIntermediate * kMoeInput);
  for (long long i = 0; i < static_cast<long long>(moe_quant_x.size()); ++i) {
    moe_quant_x[i] = static_cast<float>(i % 7 - 3) * 0.125f;
  }
  for (long long i = 0;
       i < static_cast<long long>(moe_quant_weights.size()); ++i) {
    moe_quant_weights[i] = static_cast<float>(i % 9 - 4) * 0.0625f;
  }
  std::size_t moe_expert_bytes = 0, moe_swiglu_expert_bytes = 0;
  ok &= require(qgemv_packed_size(QuantFormat::kQ8_0, kMoeOutput,
                                  kMoeInput, &moe_expert_bytes) ==
                        Status::kOk,
                "quantized MoE GEMM packed size");
  ok &= require(qgemv_packed_size(QuantFormat::kQ8_0,
                                  2 * kMoeIntermediate, kMoeInput,
                                  &moe_swiglu_expert_bytes) == Status::kOk,
                "quantized MoE SwiGLU packed size");
  std::vector<std::uint8_t> moe_packed(kMoeExperts * moe_expert_bytes);
  std::vector<std::uint8_t> moe_swiglu_packed(
      kMoeExperts * moe_swiglu_expert_bytes);
  for (long long expert = 0; expert < kMoeExperts; ++expert) {
    ok &= require(qgemv_pack(
                          QuantFormat::kQ8_0,
                          moe_quant_weights.data() +
                              expert * 2 * kMoeIntermediate * kMoeInput,
                          kMoeOutput, kMoeInput,
                          moe_packed.data() + expert * moe_expert_bytes) ==
                          Status::kOk,
                      "quantized MoE GEMM pack");
    ok &= require(qgemv_pack(
                          QuantFormat::kQ8_0,
                          moe_quant_weights.data() +
                              expert * 2 * kMoeIntermediate * kMoeInput,
                          2 * kMoeIntermediate, kMoeInput,
                          moe_swiglu_packed.data() +
                              expert * moe_swiglu_expert_bytes) == Status::kOk,
                      "quantized MoE SwiGLU pack");
  }
  const int moe_quant_experts[] = {0, 1};
  float moe_qgemm_out[kMoeRows * kMoeOutput];
  float moe_reference[2 * kMoeIntermediate];
  ok &= require(moe_grouped_qgemm(
                        moe_quant_x.data(), moe_packed.data(),
                        moe_quant_experts, nullptr, moe_qgemm_out, kMoeRows,
                        kMoeExperts, kMoeInput, kMoeOutput,
                        QuantFormat::kQ8_0, false) == Status::kOk,
                "quantized MoE grouped GEMM");
  for (long long row = 0; row < kMoeRows; ++row) {
    ok &= require(qgemv(QuantFormat::kQ8_0,
                        moe_packed.data() +
                            moe_quant_experts[row] * moe_expert_bytes,
                        moe_quant_x.data() + row * kMoeInput, moe_reference,
                        kMoeOutput, kMoeInput) == Status::kOk,
                  "quantized MoE oracle GEMV");
    for (long long output = 0; output < kMoeOutput; ++output) {
      ok &= require(close(moe_qgemm_out[row * kMoeOutput + output],
                          moe_reference[output]),
                    "quantized MoE grouped GEMM value");
    }
  }
  float moe_qswiglu_out[kMoeRows * kMoeIntermediate];
  ok &= require(moe_grouped_qswiglu(
                        moe_quant_x.data(), moe_swiglu_packed.data(),
                        moe_quant_experts, nullptr, moe_qswiglu_out,
                        kMoeRows, kMoeExperts, kMoeInput, kMoeIntermediate,
                        QuantFormat::kQ8_0, false, false) == Status::kOk,
                "quantized MoE SwiGLU");
  for (long long row = 0; row < kMoeRows; ++row) {
    qgemv(QuantFormat::kQ8_0,
          moe_swiglu_packed.data() +
              moe_quant_experts[row] * moe_swiglu_expert_bytes,
          moe_quant_x.data() + row * kMoeInput, moe_reference,
          2 * kMoeIntermediate, kMoeInput);
    for (long long item = 0; item < kMoeIntermediate; ++item) {
      const float expected =
          moe_reference[item] / (1.0f + std::exp(-moe_reference[item])) *
          moe_reference[kMoeIntermediate + item];
      ok &= require(close(moe_qswiglu_out[row * kMoeIntermediate + item],
                          expected),
                    "quantized MoE SwiGLU value");
    }
  }

  const float norm_x[] = {1, 2, 3, 4};
  const float residual[] = {1, 1, 1, 1};
  const float norm_weight[] = {1, 1, 1, 1};
  std::int8_t norm_codes[4];
  float norm_residual[4], norm_scales[1], norm_dequant[4], norm_reference[4];
  float residual_reference[4];
  ok &= require(rms_norm_add_quant_int8(
                        norm_x, residual, norm_weight, norm_codes,
                        norm_residual, norm_scales, 1, 4) == Status::kOk &&
                    rms_norm_add(norm_x, residual, norm_weight, norm_reference,
                                 residual_reference, 1, 4) == Status::kOk &&
                    dequantize_int8(norm_codes, norm_scales, norm_dequant, 1,
                                    4) == Status::kOk &&
                    close(norm_residual[3], 5),
                "norm int8 epilogue");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(norm_dequant[i], norm_reference[i], 0.02f),
                  "norm int8 reconstruction");
  }

  constexpr long long kN = 2, kK = 32;
  std::vector<float> projection(kN * kK), activation(kK);
  for (long long i = 0; i < kN * kK; ++i) projection[i] = (i % 5 - 2) * 0.1f;
  for (long long i = 0; i < kK; ++i) activation[i] = (i % 7 - 3) * 0.2f;
  std::size_t packed_bytes = 0;
  qgemv_packed_size(QuantFormat::kQ8_0, kN, kK, &packed_bytes);
  std::vector<std::uint8_t> packed(packed_bytes);
  qgemv_pack(QuantFormat::kQ8_0, projection.data(), kN, kK, packed.data());
  float up[2], gate[2], activated[2], q[2], k[2], v[2];
  ok &= require(qgemv_up_gate(QuantFormat::kQ8_0, packed.data(),
                              packed.data(), activation.data(), up, gate, kN,
                              kK) == Status::kOk &&
                    close(up[0], gate[0]),
                "fused up gate");
  ok &= require(qgemv_up_gate_activation(
                        QuantFormat::kQ8_0, packed.data(), packed.data(),
                        activation.data(), activated, kN, kK) == Status::kOk,
                "fused up gate activation");
  ok &= require(qgemv_qkv(
                        QuantFormat::kQ8_0, packed.data(), packed.data(),
                        packed.data(), activation.data(), q, k, v, kN, kN,
                        kK) == Status::kOk &&
                    close(q[1], k[1]) && close(k[1], v[1]),
                "fused QKV");
  int identity[kK];
  for (int i = 0; i < kK; ++i) identity[i] = i;
  float actorder[2];
  ok &= require(qgemm_actorder(QuantFormat::kQ8_0, packed.data(),
                               activation.data(), identity, actorder, 1, kN,
                               kK) == Status::kOk &&
                    close(actorder[0], up[0]),
                "act-order qgemm");
  const std::uint8_t fp8_weights[] = {
      float8_encode(1, Float8Format::kE4M3FN),
      float8_encode(2, Float8Format::kE4M3FN),
      float8_encode(3, Float8Format::kE4M3FN),
      float8_encode(4, Float8Format::kE4M3FN)};
  const float block_x[] = {1, 1};
  const float tile_scales[] = {0.5f, 2.0f};
  float block_y[2];
  ok &= require(fp8_blockscale_gemm(
                        fp8_weights, block_x, tile_scales, block_y, 1, 2, 2,
                        1, 2, Float8Format::kE4M3FN) == Status::kOk &&
                    close(block_y[0], 1.5f) && close(block_y[1], 14.0f),
                "FP8 blockscale GEMM");

  const int cumulative_lengths[] = {0, 1, 3};
  int varlen_padded_offsets[3], sequence_lengths_out[2], tile_sequence[2];
  int tile_local[2], tile_count = 0;
  ok &= require(varlen_build_worklist(
                        cumulative_lengths, varlen_padded_offsets,
                        sequence_lengths_out, tile_sequence, tile_local,
                        &tile_count, 2, 2, 2) == Status::kOk &&
                    varlen_padded_offsets[2] == 4 && tile_count == 2,
                "varlen worklist");
  const float packed_q[] = {1, 2, 3, 4, 5, 6};
  float padded_q[8], regathered[6];
  ok &= require(varlen_pad_q(packed_q, cumulative_lengths, varlen_padded_offsets,
                             padded_q, 2, 1, 2, 3, 4) == Status::kOk &&
                    close(padded_q[2], 0) && close(padded_q[4], 3),
                "varlen pad");
  ok &= require(varlen_regather_o(
                        padded_q, cumulative_lengths, varlen_padded_offsets,
                        regathered, 2, 1, 2, 3, 4) == Status::kOk,
                "varlen regather");
  for (int i = 0; i < 6; ++i) {
    ok &= require(close(regathered[i], packed_q[i]), "varlen round trip");
  }

  const float scan_u[] = {1, 1};
  const float scan_delta[] = {1, 1};
  const float scan_a[] = {0};
  const float scan_b[] = {1, 1};
  const float scan_c[] = {1, 1};
  const int scan_cumulative[] = {0, 2};
  const int cache_indices[] = {0};
  const std::uint8_t has_initial[] = {0};
  const int checkpoints[] = {1, -1};
  float state_pool[] = {0, 0};
  float scan_out[2];
  ok &= require(selective_scan_varlen(
                        scan_u, scan_delta, scan_a, scan_b, scan_c, nullptr,
                        nullptr, nullptr, scan_cumulative, cache_indices,
                        has_initial, checkpoints, state_pool, scan_out, 1, 2,
                        1, 2, 1, 1, false) == Status::kOk &&
                    close(scan_out[0], 1) && close(scan_out[1], 2) &&
                    close(state_pool[0], 2) && close(state_pool[1], 1),
                "stateful/APC selective scan");

  const float mla_kv[] = {1, 2, 3, 4};
  const int mla_slot[] = {1};
  std::uint8_t mla_codes[8] = {};
  float mla_scales[2] = {};
  ok &= require(mla_kv_insert_fp8(
                        mla_kv, mla_slot, mla_codes, mla_scales, 1, 2, 4, 4,
                        Float8Format::kE4M3FN, false) == Status::kOk,
                "MLA FP8 insert");
  ok &= require(mla_kv_insert_fp8(
                        mla_kv, mla_slot, mla_codes, mla_scales, 1, 2, 4, 3,
                        Float8Format::kE4M3FN, false) ==
                        Status::kInvalidShape,
                "MLA FP8 rejects partial groups");
  const int mla_blocks[] = {1};
  const int mla_context[] = {1};
  float mla_out[4], mla_sparse_out[4];
  ok &= require(mla_decode_fp8(
                        mla_kv, mla_codes, mla_scales, mla_blocks,
                        mla_context, mla_out, 2, 1, 1, 4, 1, 1, 4,
                        Float8Format::kE4M3FN) == Status::kOk,
                "MLA FP8 decode");
  const int mla_indices[] = {0};
  const int mla_topk[] = {1};
  ok &= require(mla_decode_fp8_sparse(
                        mla_kv, mla_codes, mla_scales, mla_blocks,
                        mla_indices, mla_topk, mla_sparse_out, 2, 1, 1, 4,
                        1, 1, 1, 4, Float8Format::kE4M3FN) == Status::kOk,
                "MLA FP8 sparse decode");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(mla_out[i], mla_sparse_out[i]),
                  "MLA dense sparse agreement");
  }

  if (!ok) return 1;
  std::cout << "remaining semantic kernel tests passed\n";
  return 0;
}
