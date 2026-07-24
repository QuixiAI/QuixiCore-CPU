#pragma once

#include <cstddef>
#include <cstdint>

#include "quixicore_cpu/float_storage.h"

namespace quixicore_cpu {

// Scale storage used by the canonical BaseQ2/3/4/5/6/8 sibling contract.
enum class BaseQScaleType { kBF16, kF16, kE8M0, kE4M3 };

// Non-owning canonical BaseQN matrix view. Codes are row-major and packed
// little-endian at bit index column*bits. Scales and optional affine biases are
// row-major [rows, columns/group_size] in scale_type storage.
struct BaseQTensorView {
  const std::uint8_t* codes = nullptr;
  std::size_t code_bytes = 0;
  const void* scales = nullptr;
  std::size_t scale_count = 0;
  const void* biases = nullptr;
  std::size_t bias_count = 0;
  long long rows = 0;
  long long columns = 0;
  int bits = 0;
  int group_size = 0;
  BaseQScaleType scale_type = BaseQScaleType::kBF16;
  bool symmetric = false;
};

Status base_q_dequant(BaseQTensorView weights, FloatStorageOutput output);
Status base_q_gemv(BaseQTensorView weights, FloatStorageInput x,
                   FloatStorageOutput y);
Status base_q_gemm(BaseQTensorView weights, FloatStorageInput x,
                   FloatStorageOutput y, long long m);

// Invalid token ids deliberately produce zero rows, matching the sibling
// packed-embedding contract.
Status base_q_embedding(BaseQTensorView weights, const int* ids,
                        long long tokens, FloatStorageOutput output);

Status base_q_gemv_qkv(BaseQTensorView q_weights, BaseQTensorView k_weights,
                       BaseQTensorView v_weights, FloatStorageInput x,
                       FloatStorageOutput q_output, FloatStorageOutput k_output,
                       FloatStorageOutput v_output);

// Fused gate/up projection with SiLU(gate)*up output.
Status base_q_gemv_swiglu(BaseQTensorView gate_weights,
                          BaseQTensorView up_weights, FloatStorageInput x,
                          FloatStorageOutput output);

// Greedy vocabulary selection over canonical BaseQN weights. Input is
// row-major [batch, columns] and token_ids is [batch]. Scores are rounded to
// the input storage type before comparison, matching the sibling GEMV-plus-
// argmax composition. Ties choose the lower token id.
Status base_q_lm_head_argmax(BaseQTensorView weights, FloatStorageInput x,
                             int* token_ids, long long batch);

// Grouped expert projection over the canonical 32-row padded MoE schedule.
// weights is a flattened [experts, output_rows, columns] BaseQN stack;
// expert_of_tile[total_rows/32] selects one expert per input tile. Output
// storage must match input storage, as in the sibling contract.
Status base_q_moe_gemm(BaseQTensorView weights, long long experts,
                       FloatStorageInput input, const int* expert_of_tile,
                       long long total_rows, FloatStorageOutput output);

// Fused expert gate/up projection and SiLU(gate)*up. The flattened weight row
// axis is [expert, gate(intermediate), up(intermediate)].
Status base_q_moe_swiglu(BaseQTensorView weights, long long experts,
                         FloatStorageInput input, const int* expert_of_tile,
                         long long total_rows, FloatStorageOutput output);

}  // namespace quixicore_cpu
