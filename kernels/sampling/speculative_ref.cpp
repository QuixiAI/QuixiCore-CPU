#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common/validation.h"

namespace quixicore_cpu {
namespace {

double spec_uniform(std::uint32_t seed, std::uint64_t row,
                    std::uint64_t item) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^
                    (row * 0x9E3779B97F4A7C15ULL) ^ item ^
                    0xD1B54A32D192ED03ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) /
         9007199254740992.0;
}

bool valid_cumulative(const int* cumulative, long long batch,
                      long long total) {
  if (cumulative[0] != 0 || cumulative[batch] != total) return false;
  for (long long row = 0; row < batch; ++row) {
    if (cumulative[row] > cumulative[row + 1]) return false;
  }
  return true;
}

}  // namespace

Status build_dynamic_tree(const int* parents, int* first_child,
                          int* next_sibling, int* positions, long long batch,
                          long long nodes) {
  if (!detail::valid_product({batch, nodes})) return Status::kInvalidShape;
  if (!detail::all_nonnull(parents, first_child, next_sibling, positions)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    const long long base = request * nodes;
    if (parents[base] != -1) return Status::kInvalidArgument;
    std::fill_n(first_child + base, nodes, -1);
    std::fill_n(next_sibling + base, nodes, -1);
    for (long long child = 0; child < nodes; ++child) {
      const int parent = parents[base + child];
      if (child > 0 && (parent < 0 || parent >= child)) {
        return Status::kInvalidArgument;
      }
      int depth = 0;
      for (long long node = child; parents[base + node] >= 0;
           node = parents[base + node]) {
        if (++depth > nodes) return Status::kInvalidArgument;
      }
      positions[base + child] = depth;
      if (child == 0) continue;
      if (first_child[base + parent] == -1) {
        first_child[base + parent] = static_cast<int>(child);
      } else {
        int previous = first_child[base + parent];
        while (next_sibling[base + previous] != -1) {
          previous = next_sibling[base + previous];
        }
        next_sibling[base + previous] = static_cast<int>(child);
      }
    }
  }
  return Status::kOk;
}

Status speculative_verify_tree(
    const int* draft_tokens, const float* target_probs,
    const int* first_child, const int* next_sibling, const int* tree_valid,
    int* accepted_indices, int* accepted_tokens, int* accepted_count,
    long long batch, long long nodes, long long vocab, std::uint32_t seed) {
  if (!detail::valid_product({batch, nodes, vocab}) || nodes < 1) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(draft_tokens, target_probs, first_child,
                           next_sibling, tree_valid, accepted_indices,
                           accepted_tokens, accepted_count)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    const long long base = request * nodes;
    std::fill_n(accepted_indices + base, nodes, -1);
    std::fill_n(accepted_tokens + base, nodes, -1);
    accepted_indices[base] = 0;
    int accepted = 0, last = 0;
    bool residual = false;
    if (tree_valid[request] != 0) {
      for (long long step = 1; step < nodes; ++step) {
        const int first = first_child[base + last];
        if (first == -1) break;
        const double coin = spec_uniform(seed, request, step);
        double cumulative = 0.0;
        int selected = -1;
        for (int child = first; child != -1;
             child = next_sibling[base + child]) {
          if (child <= 0 || child >= nodes) return Status::kInvalidArgument;
          const int token = draft_tokens[request * (nodes - 1) + child - 1];
          if (token < 0 || token >= vocab) return Status::kInvalidArgument;
          const float p = target_probs[(base + last) * vocab + token];
          if (!std::isfinite(p) || p < 0.0f) return Status::kInvalidArgument;
          cumulative += p;
          if (coin <= cumulative) {
            selected = child;
            accepted_tokens[base + accepted] = token;
            ++accepted;
            accepted_indices[base + accepted] = child;
            last = child;
            break;
          }
        }
        if (selected == -1) {
          residual = true;
          break;
        }
      }
    }
    const float* probabilities = target_probs + (base + last) * vocab;
    double best = -std::numeric_limits<double>::infinity();
    int best_token = -1;
    for (long long token = 0; token < vocab; ++token) {
      const float p = probabilities[token];
      if (!std::isfinite(p) || p < 0.0f) return Status::kInvalidArgument;
      if (p <= 0.0f) continue;
      bool tried = false;
      if (residual) {
        for (int child = first_child[base + last]; child != -1;
             child = next_sibling[base + child]) {
          tried |= draft_tokens[request * (nodes - 1) + child - 1] == token;
        }
      }
      if (tried) continue;
      const double uniform = spec_uniform(seed + 0x2545F491u, request, token);
      const double score = std::log(p) - std::log(-std::log(uniform));
      if (score > best || (score == best && token < best_token)) {
        best = score;
        best_token = static_cast<int>(token);
      }
    }
    accepted_tokens[base + accepted] = best_token;
    accepted_count[request] = accepted;
  }
  return Status::kOk;
}

Status speculative_compact(const int* out_tokens, const int* accepted_count,
                           const int* sequence_lengths, int* packed_tokens,
                           int* packed_positions, int* cumulative_accepted,
                           long long batch, long long draft_length) {
  if (!detail::valid_product({batch, draft_length + 1})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(out_tokens, accepted_count, sequence_lengths,
                           packed_tokens, packed_positions,
                           cumulative_accepted)) {
    return Status::kInvalidArgument;
  }
  const long long stride = draft_length + 1;
  std::fill_n(packed_tokens, batch * stride, -1);
  std::fill_n(packed_positions, batch * stride, -1);
  int total = 0;
  for (long long request = 0; request < batch; ++request) {
    if (accepted_count[request] < 0 || accepted_count[request] > draft_length ||
        sequence_lengths[request] < 0) {
      return Status::kInvalidArgument;
    }
    cumulative_accepted[request] = total;
    const int count = accepted_count[request] + 1;
    for (int item = 0; item < count; ++item) {
      packed_tokens[total + item] = out_tokens[request * stride + item];
      packed_positions[total + item] = sequence_lengths[request] + item;
    }
    total += count;
  }
  cumulative_accepted[batch] = total;
  return Status::kOk;
}

Status speculative_update_kv_meta(const int* sequence_lengths,
                                  const int* accepted_count,
                                  int* new_sequence_lengths,
                                  long long batch) {
  if (batch <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(sequence_lengths, accepted_count,
                           new_sequence_lengths)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < batch; ++row) {
    if (sequence_lengths[row] < 0 || accepted_count[row] < 0) {
      return Status::kInvalidArgument;
    }
    new_sequence_lengths[row] = sequence_lengths[row] + accepted_count[row] + 1;
  }
  return Status::kOk;
}

Status rejection_greedy_sample(
    const int* cumulative_drafts, const int* draft_tokens,
    const int* target_argmax, const int* bonus_tokens,
    const std::uint8_t* greedy_mask, int* out_tokens, long long batch,
    long long total_drafts, long long max_draft, bool use_greedy_mask) {
  if (batch <= 0 || total_drafts < 0 || max_draft < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(cumulative_drafts, draft_tokens, target_argmax,
                           bonus_tokens, out_tokens) ||
      (use_greedy_mask && greedy_mask == nullptr) ||
      !valid_cumulative(cumulative_drafts, batch, total_drafts)) {
    return Status::kInvalidArgument;
  }
  const long long stride = max_draft + 1;
  for (long long request = 0; request < batch; ++request) {
    int* row = out_tokens + request * stride;
    std::fill_n(row, stride, -1);
    if (use_greedy_mask && greedy_mask[request] == 0) continue;
    const int start = cumulative_drafts[request];
    const int count = cumulative_drafts[request + 1] - start;
    if (count > max_draft) return Status::kInvalidArgument;
    bool rejected = false;
    for (int item = 0; item < count; ++item) {
      row[item] = target_argmax[start + item];
      if (draft_tokens[start + item] != row[item]) {
        rejected = true;
        break;
      }
    }
    if (!rejected) row[count] = bonus_tokens[request];
  }
  return Status::kOk;
}

Status rejection_random_sample(
    const int* cumulative_drafts, const int* draft_tokens,
    const float* draft_probs, const float* target_probs,
    const int* bonus_tokens, const int* recovered_tokens,
    const float* uniform_probs, const std::uint8_t* greedy_mask,
    int* out_tokens, long long batch, long long total_drafts,
    long long max_draft, long long vocab, bool no_draft_probs,
    bool use_greedy_mask) {
  if (batch <= 0 || total_drafts < 0 || max_draft < 0 || vocab <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(cumulative_drafts, draft_tokens, target_probs,
                           bonus_tokens, recovered_tokens, uniform_probs,
                           out_tokens) ||
      (!no_draft_probs && draft_probs == nullptr) ||
      (use_greedy_mask && greedy_mask == nullptr) ||
      !valid_cumulative(cumulative_drafts, batch, total_drafts)) {
    return Status::kInvalidArgument;
  }
  const long long stride = max_draft + 1;
  for (long long request = 0; request < batch; ++request) {
    int* row = out_tokens + request * stride;
    std::fill_n(row, stride, -1);
    if (use_greedy_mask && greedy_mask[request] != 0) continue;
    const int start = cumulative_drafts[request];
    const int count = cumulative_drafts[request + 1] - start;
    if (count > max_draft) return Status::kInvalidArgument;
    bool rejected = false;
    for (int item = 0; item < count; ++item) {
      const int index = start + item;
      const int token = draft_tokens[index];
      if (token < 0 || token >= vocab || !std::isfinite(uniform_probs[index]) ||
          uniform_probs[index] < 0.0f || uniform_probs[index] > 1.0f) {
        return Status::kInvalidArgument;
      }
      const float p = target_probs[static_cast<long long>(index) * vocab + token];
      const float q = no_draft_probs
                          ? 1.0f
                          : draft_probs[static_cast<long long>(index) * vocab + token];
      if (q > 0.0f && p / q >= uniform_probs[index]) {
        row[item] = token;
      } else {
        row[item] = recovered_tokens[index];
        rejected = true;
        break;
      }
    }
    if (!rejected) row[count] = bonus_tokens[request];
  }
  return Status::kOk;
}

Status sample_recovered_tokens(
    const int* cumulative_drafts, const int* draft_tokens,
    const float* draft_probs, const float* target_probs,
    const float* inverse_noise, int* recovered_tokens, long long batch,
    long long total_drafts, long long vocab, bool no_draft_probs) {
  if (batch <= 0 || total_drafts < 0 || vocab <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(cumulative_drafts, draft_tokens, target_probs,
                           inverse_noise, recovered_tokens) ||
      (!no_draft_probs && draft_probs == nullptr) ||
      !valid_cumulative(cumulative_drafts, batch, total_drafts)) {
    return Status::kInvalidArgument;
  }
  long long request = 0;
  for (long long item = 0; item < total_drafts; ++item) {
    while (request + 1 < batch && item >= cumulative_drafts[request + 1]) {
      ++request;
    }
    const int draft = draft_tokens[item];
    if (draft < 0 || draft >= vocab) return Status::kInvalidArgument;
    double best = -1.0;
    int selected = 0;
    for (long long token = 0; token < vocab; ++token) {
      const float target = target_probs[item * vocab + token];
      const float noise = inverse_noise[request * vocab + token];
      if (!std::isfinite(target) || target < 0.0f ||
          !std::isfinite(noise) || noise < 0.0f) {
        return Status::kInvalidArgument;
      }
      float draft_probability = 0.0f;
      if (!no_draft_probs) {
        draft_probability = draft_probs[item * vocab + token];
        if (!std::isfinite(draft_probability) || draft_probability < 0.0f) {
          return Status::kInvalidArgument;
        }
      }
      const float residual = no_draft_probs
                                 ? (token == draft ? 0.0f : target)
                                 : std::max(0.0f, target - draft_probability);
      const double score = residual * noise;
      if (score > best) {
        best = score;
        selected = static_cast<int>(token);
      }
    }
    recovered_tokens[item] = selected;
  }
  return Status::kOk;
}

Status eagle_prepare_inputs_padded(
    const int* cumulative_drafts, const int* valid_sampled_count,
    const int* query_start_locations, int* token_indices_to_sample,
    int* rejected_count, long long batch) {
  if (batch <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(cumulative_drafts, valid_sampled_count,
                           query_start_locations, token_indices_to_sample,
                           rejected_count)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    const int drafts = cumulative_drafts[request + 1] - cumulative_drafts[request];
    if (drafts < 0 || valid_sampled_count[request] < 0) {
      return Status::kInvalidArgument;
    }
    const int rejected = drafts > 0 ? drafts + 1 - valid_sampled_count[request] : 0;
    rejected_count[request] = rejected;
    token_indices_to_sample[request] =
        query_start_locations[request + 1] - 1 - rejected;
  }
  return Status::kOk;
}

Status eagle_prepare_next_token_padded(
    const int* sampled_tokens, const std::uint8_t* discard_mask,
    const int* backup_tokens, int* next_tokens, int* valid_sampled_count,
    long long batch, long long sampled_per_request, int vocab) {
  if (!detail::valid_product({batch, sampled_per_request}) || vocab <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(sampled_tokens, discard_mask, backup_tokens,
                           next_tokens, valid_sampled_count)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    if (backup_tokens[request] < 0 || backup_tokens[request] >= vocab) {
      return Status::kInvalidArgument;
    }
    int count = 0, last = -1;
    for (long long item = 0; item < sampled_per_request; ++item) {
      const int token = sampled_tokens[request * sampled_per_request + item];
      if (token < -1 || token >= vocab) return Status::kInvalidArgument;
      if (token != -1) {
        ++count;
        last = token;
      }
    }
    if (discard_mask[request]) count = 0;
    valid_sampled_count[request] = count;
    next_tokens[request] = count > 0 ? last : backup_tokens[request];
  }
  return Status::kOk;
}

Status eagle_step_slot_mapping_metadata(
    const int* positions, const int* block_table,
    const int* sequence_lengths, int* clamped_positions,
    int* slot_mapping, int* new_sequence_lengths, long long batch,
    long long input_batch, long long max_blocks, int block_size,
    int max_model_length, int pad_id) {
  if (!detail::valid_product({batch, input_batch, max_blocks}) ||
      input_batch < batch || block_size <= 0 || max_model_length <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(positions, block_table, sequence_lengths,
                           clamped_positions, slot_mapping,
                           new_sequence_lengths)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < input_batch; ++request) {
    if (request >= batch) {
      clamped_positions[request] = 0;
      slot_mapping[request] = pad_id;
      continue;
    }
    if (positions[request] < 0 || sequence_lengths[request] < 0) {
      return Status::kInvalidArgument;
    }
    const int new_position = positions[request] + 1;
    const bool exceeds = new_position >= max_model_length;
    const int clamped = exceeds ? 0 : new_position;
    const int block_number = std::min<int>(clamped / block_size,
                                           static_cast<int>(max_blocks - 1));
    const int block = block_table[request * max_blocks + block_number];
    if (!exceeds && block < 0) return Status::kInvalidArgument;
    clamped_positions[request] = clamped;
    slot_mapping[request] =
        exceeds ? pad_id : block * block_size + clamped % block_size;
    new_sequence_lengths[request] =
        exceeds ? 1 : std::min(sequence_lengths[request] + 1,
                               max_model_length);
  }
  return Status::kOk;
}

Status eagle_expand_int32(const int* input, const int* cumulative_tokens,
                          int* output, long long batch, long long total,
                          int replace_from, int replace_to) {
  if (batch <= 0 || total < 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(input, cumulative_tokens, output) ||
      !valid_cumulative(cumulative_tokens, batch, total)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    const int value = input[request] == replace_from ? replace_to : input[request];
    std::fill(output + cumulative_tokens[request],
              output + cumulative_tokens[request + 1], value);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
