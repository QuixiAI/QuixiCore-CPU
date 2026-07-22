#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cstdint>

#include "kernels/common/validation.h"

namespace quixicore_cpu {
namespace {

int save_blocks(int* output, int begin, int end, int block_size, int count,
                int kv_length, int capacity) {
  if (begin >= kv_length) return count;
  end = std::min(end, kv_length);
  for (int index = begin; index < end; index += block_size) {
    if (count < capacity) output[count] = index;
    ++count;
  }
  return count;
}

}  // namespace

Status eagle_prepare_next_token_int64(
    const long long* sampled_tokens, const std::uint8_t* discard_mask,
    const long long* backup_tokens, long long* next_tokens,
    long long* valid_sampled_count, long long batch,
    long long sampled_per_request, long long vocab) {
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
    long long count = 0;
    long long last = -1;
    for (long long item = 0; item < sampled_per_request; ++item) {
      const long long token =
          sampled_tokens[request * sampled_per_request + item];
      if (token < -1 || token >= vocab) return Status::kInvalidArgument;
      if (token != -1) {
        ++count;
        last = token;
      }
    }
    if (discard_mask[request] != 0) count = 0;
    valid_sampled_count[request] = count;
    next_tokens[request] = count > 0 ? last : backup_tokens[request];
  }
  return Status::kOk;
}

Status eagle_expand_int64(const long long* input,
                          const long long* cumulative_tokens,
                          long long* output, long long batch,
                          long long total, long long replace_from,
                          long long replace_to) {
  if (batch <= 0 || total < 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(input, cumulative_tokens, output) ||
      cumulative_tokens[0] != 0 || cumulative_tokens[batch] != total) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    if (cumulative_tokens[request] > cumulative_tokens[request + 1]) {
      return Status::kInvalidArgument;
    }
    const long long value =
        input[request] == replace_from ? replace_to : input[request];
    std::fill(output + cumulative_tokens[request],
              output + cumulative_tokens[request + 1], value);
  }
  return Status::kOk;
}

Status eagle_step_slot_mapping_int64(
    long long* sequence_lengths, const long long* positions,
    const int* block_table, long long* clamped_positions,
    long long* slot_mapping, long long batch, long long input_batch,
    long long max_blocks, long long block_size, long long max_model_length,
    long long pad_id) {
  if (!detail::valid_product({batch, input_batch, max_blocks}) ||
      input_batch < batch || block_size <= 0 || max_model_length <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(sequence_lengths, positions, block_table,
                           clamped_positions, slot_mapping)) {
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
    const long long new_position = positions[request] + 1;
    const bool exceeds = new_position >= max_model_length;
    const long long clamped = exceeds ? 0 : new_position;
    const long long block_number =
        std::min(clamped / block_size, max_blocks - 1);
    const int block = block_table[request * max_blocks + block_number];
    if (!exceeds && block < 0) return Status::kInvalidArgument;
    clamped_positions[request] = clamped;
    slot_mapping[request] =
        exceeds ? pad_id : block * block_size + clamped % block_size;
    sequence_lengths[request] =
        exceeds ? 1 : std::min(sequence_lengths[request] + 1,
                               max_model_length);
  }
  return Status::kOk;
}

Status copy_and_expand_eagle(
    const long long* target_token_ids, const long long* target_positions,
    const long long* next_token_ids, const int* query_start_locations,
    const int* query_end_locations, long long* output_input_ids,
    long long* output_positions, std::uint8_t* rejected_mask,
    std::uint8_t* masked_token_mask, int* new_token_indices,
    int* hidden_state_mapping, long long batch, long long input_tokens,
    long long output_tokens, long long padding_token_id,
    long long parallel_drafting_token_id,
    long long padding_slots_per_request, bool shift_input_ids) {
  if (batch <= 0 || input_tokens <= 0 || output_tokens <= 0 ||
      padding_slots_per_request <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(
          target_token_ids, target_positions, next_token_ids,
          query_start_locations, query_end_locations, output_input_ids,
          output_positions, rejected_mask, masked_token_mask,
          new_token_indices, hidden_state_mapping)) {
    return Status::kInvalidArgument;
  }
  if (query_start_locations[0] != 0 ||
      query_start_locations[batch] != input_tokens) {
    return Status::kInvalidArgument;
  }
  std::fill_n(output_input_ids, output_tokens, padding_token_id);
  std::fill_n(output_positions, output_tokens, 0);
  std::fill_n(rejected_mask, output_tokens, 0);
  std::fill_n(masked_token_mask, output_tokens, 0);
  std::fill_n(new_token_indices, batch * padding_slots_per_request, -1);
  std::fill_n(hidden_state_mapping, input_tokens, -1);
  for (long long request = 0; request < batch; ++request) {
    const long long query_start = query_start_locations[request];
    const long long next_query_start = query_start_locations[request + 1];
    const long long query_end = query_end_locations[request];
    if (query_start < 0 || next_query_start < query_start ||
        next_query_start > input_tokens || query_end < query_start ||
        query_end >= next_query_start) {
      return Status::kInvalidArgument;
    }
    const long long valid = shift_input_ids
                                ? query_end - query_start
                                : query_end - query_start + 1;
    const long long input_offset = shift_input_ids ? 1 : 0;
    const long long output_start =
        query_start + request *
                          (padding_slots_per_request -
                           (shift_input_ids ? 1 : 0));
    const long long rejected = next_query_start - query_end - 1;
    const long long count = valid + padding_slots_per_request + rejected;
    if (output_start < 0 || count < 0 ||
        output_start + count > output_tokens) {
      return Status::kInvalidArgument;
    }
    const long long start_position = target_positions[query_start];
    for (long long item = 0; item < count; ++item) {
      const long long output_index = output_start + item;
      const bool is_valid = item < valid;
      const bool is_bonus = item == valid;
      const bool is_parallel =
          item > valid && item < valid + padding_slots_per_request;
      const bool is_rejected =
          item >= valid + padding_slots_per_request;
      const long long input_index = std::min(
          query_start + input_offset + item, input_tokens - 1);
      long long token = padding_token_id;
      if (is_valid) token = target_token_ids[input_index];
      else if (is_bonus) token = next_token_ids[request];
      else if (is_parallel) token = parallel_drafting_token_id;
      output_input_ids[output_index] = token;
      output_positions[output_index] =
          is_rejected ? 0 : start_position + item;
      rejected_mask[output_index] = is_rejected ? 1 : 0;
      masked_token_mask[output_index] = is_parallel ? 1 : 0;
      if (is_bonus || is_parallel) {
        const long long local = item - valid;
        new_token_indices[request * padding_slots_per_request + local] =
            static_cast<int>(output_index);
      }
    }
    if (shift_input_ids) {
      const long long inputs = next_query_start - query_start;
      for (long long item = 0; item < inputs; ++item) {
        hidden_state_mapping[query_start + item] =
            static_cast<int>(output_start + item);
      }
    }
  }
  return Status::kOk;
}

Status indexer_k_gather_paged(
    const std::uint8_t* code_cache, const float* scale_cache,
    const int* block_table, const int* cumulative_lengths,
    std::uint8_t* output_codes, float* output_scales,
    long long cache_blocks, long long batch, long long max_blocks,
    long long cache_block_size, long long total_tokens,
    long long head_dim, long long quant_block_size) {
  if (!detail::valid_product({cache_blocks, cache_block_size, head_dim}) ||
      !detail::valid_product({batch, max_blocks}) || total_tokens < 0 ||
      quant_block_size <= 0 || head_dim % quant_block_size != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(code_cache, scale_cache, block_table,
                           cumulative_lengths, output_codes,
                           output_scales) ||
      cumulative_lengths[0] != 0 ||
      cumulative_lengths[batch] != total_tokens) {
    return Status::kInvalidArgument;
  }
  const long long groups = head_dim / quant_block_size;
  for (long long request = 0; request < batch; ++request) {
    if (cumulative_lengths[request] > cumulative_lengths[request + 1] ||
        cumulative_lengths[request + 1] - cumulative_lengths[request] >
            max_blocks * cache_block_size) {
      return Status::kInvalidArgument;
    }
    for (long long token = cumulative_lengths[request];
         token < cumulative_lengths[request + 1]; ++token) {
      const long long local = token - cumulative_lengths[request];
      const int block =
          block_table[request * max_blocks + local / cache_block_size];
      if (block < 0 || block >= cache_blocks) {
        return Status::kInvalidArgument;
      }
      const long long slot =
          static_cast<long long>(block) * cache_block_size +
          local % cache_block_size;
      std::copy_n(code_cache + slot * head_dim, head_dim,
                  output_codes + token * head_dim);
      std::copy_n(scale_cache + slot * groups, groups,
                  output_scales + token * groups);
    }
  }
  return Status::kOk;
}

Status convert_vertical_slash_indexes(
    const int* query_lengths, const int* kv_lengths,
    const int* vertical_indexes, const int* slash_indexes,
    int* block_count, int* block_offset, int* column_count,
    int* column_index, long long batch, long long heads,
    long long rows, long long vertical_width, long long slash_width,
    int block_size_m, int block_size_n, bool causal) {
  if (!detail::valid_product({batch, heads, rows, vertical_width,
                              slash_width}) ||
      block_size_m <= 0 || block_size_n <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(query_lengths, kv_lengths, vertical_indexes,
                           slash_indexes, block_count, block_offset,
                           column_count, column_index)) {
    return Status::kInvalidArgument;
  }
  const long long output_rows = batch * heads * rows;
  std::fill_n(block_count, output_rows, 0);
  std::fill_n(column_count, output_rows, 0);
  std::fill_n(block_offset, output_rows * slash_width, 0);
  std::fill_n(column_index, output_rows * vertical_width, 0);
  for (long long request = 0; request < batch; ++request) {
    const int query_length = query_lengths[request];
    const int kv_length = kv_lengths[request];
    if (query_length < 0 || kv_length < 0) return Status::kInvalidArgument;
    for (long long head = 0; head < heads; ++head) {
      const int* vertical =
          vertical_indexes + (request * heads + head) * vertical_width;
      const int* slash =
          slash_indexes + (request * heads + head) * slash_width;
      for (long long row = 0; row < rows; ++row) {
        const int start_m = static_cast<int>(row) * block_size_m;
        if (start_m >= query_length) continue;
        const int end_m = start_m + block_size_m;
        const long long output_row =
            (request * heads + head) * rows + row;
        int* row_blocks = block_offset + output_row * slash_width;
        int* row_columns = column_index + output_row * vertical_width;
        bool has_slash = true;
        int columns = 0;
        int blocks = 0;
        long long slash_cursor = 0;
        long long vertical_cursor = 0;
        int vertical_index = vertical[vertical_cursor++];
        int slash_index = slash[slash_cursor++];
        const int offset = kv_length - query_length;
        if (causal) {
          while (slash_index >= end_m + offset &&
                 slash_cursor < slash_width) {
            slash_index = slash[slash_cursor++];
          }
          if (slash_index > end_m + offset) has_slash = false;
          slash_index =
              std::max(offset + end_m - slash_index, block_size_m);
        } else {
          while (slash_index >= end_m + kv_length &&
                 slash_cursor < slash_width) {
            slash_index = slash[slash_cursor++];
          }
          if (slash_index > end_m + kv_length) has_slash = false;
          slash_index =
              std::max(kv_length + end_m - slash_index, block_size_m);
        }
        int range_start = slash_index - block_size_m;
        int range_end = slash_index;
        if (!has_slash) {
          if (causal) {
            range_start = offset + end_m;
            range_end = offset + end_m + block_size_n;
          } else {
            range_start = kv_length;
            range_end = kv_length + block_size_n;
          }
        }
        bool slash_finished = false;
        while (true) {
          if (vertical_index < range_end) {
            if (vertical_index < range_start && columns < vertical_width) {
              row_columns[columns++] = vertical_index;
            }
            if (vertical_cursor < vertical_width) {
              vertical_index = vertical[vertical_cursor++];
            } else {
              vertical_index = causal ? end_m + block_size_n + offset
                                      : end_m + block_size_n + kv_length;
            }
          } else {
            if (slash_cursor < slash_width &&
                (causal || slash[slash_cursor] >= start_m)) {
              slash_index =
                  causal
                      ? std::max(offset + end_m - slash[slash_cursor++],
                                 block_size_m)
                      : std::max(kv_length + end_m - slash[slash_cursor++],
                                 block_size_m);
            } else {
              if (vertical_cursor == vertical_width ||
                  (vertical_index > range_start && causal)) {
                if (vertical_cursor == vertical_width && !causal &&
                    vertical_index < kv_length && columns < vertical_width) {
                  row_columns[columns++] = vertical_index;
                }
                blocks = save_blocks(row_blocks, range_start, range_end,
                                     block_size_n, blocks, kv_length,
                                     static_cast<int>(slash_width));
                break;
              }
              if (causal) {
                range_start = offset + end_m;
                range_end = offset + end_m + block_size_n;
              } else {
                blocks = save_blocks(row_blocks, range_start, range_end,
                                     block_size_n, blocks, kv_length,
                                     static_cast<int>(slash_width));
                range_start = kv_length;
                range_end = kv_length + block_size_n;
              }
              slash_finished = true;
            }
            if (!slash_finished) {
              if (slash_index > range_end + block_size_m) {
                blocks = save_blocks(row_blocks, range_start, range_end,
                                     block_size_n, blocks, kv_length,
                                     static_cast<int>(slash_width));
                range_start = slash_index - block_size_m;
                range_end = slash_index;
              } else if (slash_index > range_end) {
                range_end += block_size_m;
              }
            }
          }
        }
        if (blocks > slash_width || columns > vertical_width) {
          return Status::kInvalidArgument;
        }
        block_count[output_row] = blocks;
        column_count[output_row] = columns;
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
