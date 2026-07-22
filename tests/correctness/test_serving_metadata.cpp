#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quantization.h"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

bool close(float lhs, float rhs, float tolerance = 0.05f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;

  const int parents[] = {-1, 0, 0, 1};
  int first_child[4], next_sibling[4], positions[4];
  ok &= require(build_dynamic_tree(parents, first_child, next_sibling,
                                   positions, 1, 4) == Status::kOk &&
                    first_child[0] == 1 && next_sibling[1] == 2 &&
                    first_child[1] == 3 && positions[3] == 2,
                "dynamic tree");
  const int drafts[] = {1, 2, 3};
  float target[4 * 5] = {};
  target[1] = 1.0f;
  target[5 + 3] = 1.0f;
  target[10] = 1.0f;
  target[15 + 2] = 1.0f;
  const int tree_valid[] = {1};
  int accepted_indices[4], accepted_tokens[4], accepted_count[1];
  ok &= require(speculative_verify_tree(
                        drafts, target, first_child, next_sibling, tree_valid,
                        accepted_indices, accepted_tokens, accepted_count, 1,
                        4, 5, 11) == Status::kOk &&
                    accepted_count[0] == 2 && accepted_indices[0] == 0 &&
                    accepted_indices[1] == 1 && accepted_indices[2] == 3 &&
                    accepted_tokens[0] == 1 && accepted_tokens[1] == 3 &&
                    accepted_tokens[2] == 2,
                "tree verification");

  const int verified[] = {1, 3, 2, -1, 4, 1, -1, -1};
  const int counts[] = {2, 1};
  const int sequence_lengths[] = {10, 20};
  int packed_tokens[8], packed_positions[8], cumulative[3];
  ok &= require(speculative_compact(
                        verified, counts, sequence_lengths, packed_tokens,
                        packed_positions, cumulative, 2, 3) == Status::kOk &&
                    cumulative[0] == 0 && cumulative[1] == 3 &&
                    cumulative[2] == 5 && packed_tokens[3] == 4 &&
                    packed_positions[4] == 21 && packed_tokens[5] == -1,
                "speculative compact");
  int new_lengths[2];
  ok &= require(speculative_update_kv_meta(sequence_lengths, counts,
                                           new_lengths, 2) == Status::kOk &&
                    new_lengths[0] == 13 && new_lengths[1] == 22,
                "speculative KV metadata");

  const int cumulative_drafts[] = {0, 2, 3};
  const int ragged_drafts[] = {1, 2, 3};
  const int target_argmax[] = {1, 4, 3};
  const int bonus[] = {0, 2};
  int rejection_out[6];
  ok &= require(rejection_greedy_sample(
                        cumulative_drafts, ragged_drafts, target_argmax,
                        bonus, nullptr, rejection_out, 2, 3, 2, false) ==
                        Status::kOk &&
                    rejection_out[0] == 1 && rejection_out[1] == 4 &&
                    rejection_out[3] == 3 && rejection_out[4] == 2,
                "ragged greedy rejection");
  float target_probs[3 * 5] = {};
  float draft_probs[3 * 5] = {};
  float inverse_noise[2 * 5];
  for (float& value : inverse_noise) value = 1.0f;
  target_probs[0 * 5 + 4] = 1.0f;
  target_probs[1 * 5 + 0] = 1.0f;
  target_probs[2 * 5 + 2] = 1.0f;
  int recovered[3];
  ok &= require(sample_recovered_tokens(
                        cumulative_drafts, ragged_drafts, draft_probs,
                        target_probs, inverse_noise, recovered, 2, 3, 5,
                        false) == Status::kOk &&
                    recovered[0] == 4 && recovered[1] == 0 &&
                    recovered[2] == 2,
                "recovered tokens");
  inverse_noise[0] = -1.0f;
  ok &= require(sample_recovered_tokens(
                        cumulative_drafts, ragged_drafts, draft_probs,
                        target_probs, inverse_noise, recovered, 2, 3, 5,
                        false) == Status::kInvalidArgument,
                "recovered tokens reject negative noise");
  inverse_noise[0] = 1.0f;
  const float uniforms[] = {0.5f, 0.5f, 0.5f};
  ok &= require(rejection_random_sample(
                        cumulative_drafts, ragged_drafts, nullptr,
                        target_probs, bonus, recovered, uniforms, nullptr,
                        rejection_out, 2, 3, 2, 5, true, false) ==
                        Status::kOk &&
                    rejection_out[0] == 4 && rejection_out[3] == 2,
                "ragged random rejection");

  const int valid_count[] = {2, 1};
  const int query_starts[] = {0, 3, 5};
  int sample_indices[2], rejected[2];
  ok &= require(eagle_prepare_inputs_padded(
                        cumulative_drafts, valid_count, query_starts,
                        sample_indices, rejected, 2) == Status::kOk &&
                    rejected[0] == 1 && sample_indices[0] == 1,
                "EAGLE input metadata");
  const int sampled[] = {1, 2, -1, 3, -1, -1};
  const std::uint8_t discard[] = {0, 1};
  const int backup[] = {4, 4};
  int next[2], valid[2];
  ok &= require(eagle_prepare_next_token_padded(
                        sampled, discard, backup, next, valid, 2, 3, 5) ==
                        Status::kOk &&
                    next[0] == 2 && valid[0] == 2 && next[1] == 4 &&
                    valid[1] == 0,
                "EAGLE next token metadata");
  const int invalid_sampled[] = {1, -2, -1, 3, -1, -1};
  ok &= require(eagle_prepare_next_token_padded(
                        invalid_sampled, discard, backup, next, valid, 2, 3,
                        5) == Status::kInvalidArgument,
                "EAGLE rejects invalid sampled token");
  const int step_positions[] = {3};
  const int step_blocks[] = {5, 6};
  const int step_lengths[] = {4};
  int clamped_positions[2], step_slots[2], step_new_lengths[1];
  ok &= require(eagle_step_slot_mapping_metadata(
                        step_positions, step_blocks, step_lengths,
                        clamped_positions, step_slots, step_new_lengths, 1, 2,
                        2, 4, 8, -1) == Status::kOk &&
                    clamped_positions[0] == 4 && step_slots[0] == 24 &&
                    step_new_lengths[0] == 5 &&
                    clamped_positions[1] == 0 && step_slots[1] == -1,
                "EAGLE step slot metadata");
  const int expand_input[] = {-1, 7};
  int expanded[3];
  ok &= require(eagle_expand_int32(expand_input, cumulative_drafts, expanded,
                                   2, 3, -1, 9) == Status::kOk &&
                    expanded[0] == 9 && expanded[1] == 9 && expanded[2] == 7,
                "EAGLE ragged expand");

  float key_cache[] = {1, 2, 3, 4};
  float value_cache[] = {10, 20, 30, 40};
  const long long copy_pairs[] = {0, 1};
  ok &= require(kv_cache_copy_blocks(key_cache, value_cache, copy_pairs, 1,
                                     2, 2) == Status::kOk &&
                    close(key_cache[2], 1) && close(value_cache[3], 20),
                "KV block copy");
  const int parent_beam[] = {1, 1};
  const int block_table[] = {4, 5, 6, 7};
  const int beam_lengths[] = {2, 2};
  long long beam_pairs[8];
  int remapped[4];
  ok &= require(beam_build_copy_pairs(parent_beam, block_table, beam_lengths,
                                      beam_pairs, 1, 2, 2, 2) == Status::kOk &&
                    beam_pairs[0] == 6 && beam_pairs[1] == 4 &&
                    beam_pairs[4] == -1,
                "beam copy pairs");
  ok &= require(beam_remap_block_table(block_table, parent_beam, remapped, 1,
                                       2, 2) == Status::kOk &&
                    remapped[0] == 6 && remapped[1] == 7 &&
                    remapped[2] == 6 && remapped[3] == 7,
                "beam block remap");

  const float key[] = {-2, 1, 3, -4};
  const float value[] = {1, -3, 2, 4};
  float key_scale = 0, value_scale = 0;
  ok &= require(kv_cache_scales(key, value, &key_scale, &value_scale, 4) ==
                        Status::kOk &&
                    close(key_scale, 4.0f / 240.0f, 1e-6f) &&
                    close(value_scale, 4.0f / 240.0f, 1e-6f),
                "KV scales");
  ok &= require(kv_cache_scale_update(key, value, key_scale, value_scale,
                                      nullptr, &value_scale, 4) ==
                        Status::kInvalidArgument,
                "KV scale update validates outputs");
  const int slots[] = {1};
  const float head_scales[] = {0.05f};
  std::uint8_t key_codes[8] = {}, value_codes[8] = {};
  float gathered_key[4], gathered_value[4];
  ok &= require(kv_cache_scatter_fp8(
                        key, value, slots, head_scales, head_scales,
                        key_codes, value_codes, 2, 1, 1, 4,
                        Float8Format::kE4M3FN) == Status::kOk,
                "FP8 KV scatter");
  ok &= require(kv_cache_gather_fp8(
                        key_codes, value_codes, slots, head_scales,
                        head_scales, gathered_key, gathered_value, 2, 1, 1,
                        4, Float8Format::kE4M3FN) == Status::kOk &&
                    close(gathered_key[0], key[0]) &&
                    close(gathered_value[3], value[3]),
                "FP8 KV gather");

  const float paged_q[] = {1, 0, 0, 0};
  const float standard_key[] = {1, 0, 0, 0, 0, 1, 0, 0};
  const float standard_value[] = {2, 3, 4, 5, 6, 7, 8, 9};
  float xcache_key[8], xcache_value[8];
  constexpr long long kPage = 2, kDim = 4, kVector = 2;
  for (long long position = 0; position < kPage; ++position) {
    for (long long dim = 0; dim < kDim; ++dim) {
      xcache_key[((dim / kVector) * kPage + position) * kVector +
                 dim % kVector] = standard_key[position * kDim + dim];
      xcache_value[dim * kPage + position] =
          standard_value[position * kDim + dim];
    }
  }
  const int paged_table[] = {0};
  const int paged_length[] = {2};
  float paged_reference[4], xcache_out[4];
  ok &= require(paged_attention(
                        paged_q, standard_key, standard_value, paged_table,
                        paged_length, paged_reference, 1, 1, 1, 1, kDim,
                        kPage, 1) == Status::kOk &&
                    paged_attention_xcache(
                        paged_q, xcache_key, xcache_value, paged_table,
                        paged_length, xcache_out, 1, 1, 1, 1, kDim, kPage, 1,
                        kVector) == Status::kOk,
                "xcache paged attention adapter");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(xcache_out[i], paged_reference[i]),
                  "xcache paged attention value");
  }

  if (!ok) return 1;
  std::cout << "serving metadata tests passed\n";
  return 0;
}
