#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

constexpr long long kQ8Group = 32;

bool valid_storage(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

float load_storage(FloatStorageInput input, long long index) {
  if (input.type == FloatStorageType::kF32) {
    return static_cast<const float*>(input.data)[index];
  }
  const std::uint16_t bits =
      static_cast<const std::uint16_t*>(input.data)[index];
  return input.type == FloatStorageType::kF16 ? f16_to_float(bits)
                                              : bf16_to_float(bits);
}

void store_storage(FloatStorageOutput output, long long index, float value) {
  if (output.type == FloatStorageType::kF32) {
    static_cast<float*>(output.data)[index] = value;
  } else if (output.type == FloatStorageType::kF16) {
    static_cast<std::uint16_t*>(output.data)[index] = float_to_f16(value);
  } else {
    static_cast<std::uint16_t*>(output.data)[index] = float_to_bf16(value);
  }
}

bool valid_q8_shape(long long cache_blocks, long long rows, long long heads,
                    long long head_dim, long long page_size) {
  return detail::valid_product(
             {cache_blocks, rows, heads, head_dim, page_size}) &&
         head_dim % kQ8Group == 0;
}

long long code_row(long long slot, long long head, long long heads,
                   long long head_dim) {
  return (slot * heads + head) * head_dim;
}

long long scale_row(long long slot, long long head, long long heads,
                    long long groups) {
  return (slot * heads + head) * groups;
}

std::int8_t encode_q8(float value, float inverse) {
  const float rounded =
      std::copysign(std::floor(std::fabs(value * inverse) + 0.5f), value);
  return static_cast<std::int8_t>(std::clamp(rounded, -127.0f, 127.0f));
}

Status validate_paged_q8(const std::int8_t* key_codes,
                         const std::uint16_t* key_scales,
                         const std::int8_t* value_codes,
                         const std::uint16_t* value_scales,
                         const int* block_table, const int* context_lens,
                         long long cache_blocks, long long batch,
                         long long query_heads, long long kv_heads,
                         long long head_dim, long long page_size,
                         long long max_blocks, float scale, long long window) {
  if (!valid_q8_shape(cache_blocks, batch, kv_heads, head_dim, page_size) ||
      query_heads <= 0 || query_heads % kv_heads != 0 || max_blocks <= 0 ||
      !std::isfinite(scale) || scale < 0.0f || window < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key_codes, key_scales, value_codes, value_scales,
                           block_table, context_lens)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 || context > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (long long position = 0; position < context; ++position) {
      const int block =
          block_table[request * max_blocks + position / page_size];
      if (block >= cache_blocks) {
        return Status::kInvalidArgument;
      }
    }
  }
  return Status::kOk;
}

}  // namespace

Status kv_cache_scatter_q8_0_storage(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    std::int8_t* key_codes, std::uint16_t* key_scales, std::int8_t* value_codes,
    std::uint16_t* value_scales, long long cache_blocks, long long count,
    long long heads, long long head_dim, long long page_size) {
  if (!valid_q8_shape(cache_blocks, count, heads, head_dim, page_size) ||
      key.count != count * heads * head_dim || value.count != key.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage(key.type) || !valid_storage(value.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key.data, value.data, slots, key_codes, key_scales,
                           value_codes, value_scales)) {
    return Status::kInvalidArgument;
  }
  const long long cache_slots = cache_blocks * page_size;
  for (long long token = 0; token < count; ++token) {
    if (slots[token] >= cache_slots) return Status::kInvalidArgument;
  }
  const long long groups = head_dim / kQ8Group;
  const long long code_count = cache_slots * heads * head_dim;
  const long long scale_count = cache_slots * heads * groups;
  threading::parallel_ranges(
      code_count, 16384, [&](long long begin, long long end, int) {
        std::fill(key_codes + begin, key_codes + end, std::int8_t{0});
        std::fill(value_codes + begin, value_codes + end, std::int8_t{0});
        if (begin < scale_count) {
          const long long scale_end = std::min(end, scale_count);
          std::fill(key_scales + begin, key_scales + scale_end,
                    std::uint16_t{0});
          std::fill(value_scales + begin, value_scales + scale_end,
                    std::uint16_t{0});
        }
      });

  bool ordered_unique = true;
  int previous_slot = -1;
  for (long long token = 0; token < count; ++token) {
    if (slots[token] < 0) continue;
    if (slots[token] <= previous_slot) ordered_unique = false;
    previous_slot = slots[token];
  }
  std::atomic<bool> invalid{false};
  const bool f32_inputs = key.type == FloatStorageType::kF32 &&
                          value.type == FloatStorageType::kF32;
  const auto* key_f32 = static_cast<const float*>(key.data);
  const auto* value_f32 = static_cast<const float*>(value.data);
  const auto scatter_tasks = [&](long long begin, long long end, int) {
    for (long long task = begin; task < end; ++task) {
      if (invalid.load(std::memory_order_relaxed)) return;
      const long long token = task / heads;
      const long long head = task % heads;
      const int slot = slots[token];
      if (slot < 0) continue;
      const long long source = (token * heads + head) * head_dim;
      const long long code_base = code_row(slot, head, heads, head_dim);
      const long long scale_base = scale_row(slot, head, heads, groups);
      for (long long group = 0; group < groups; ++group) {
        alignas(64) float key_values[kQ8Group];
        alignas(64) float value_values[kQ8Group];
        float key_amax = 0.0f;
        float value_amax = 0.0f;
        if (f32_inputs) {
          for (long long lane = 0; lane < kQ8Group; ++lane) {
            const long long index = source + group * kQ8Group + lane;
            const float key_value = key_f32[index];
            const float value_value = value_f32[index];
            key_values[lane] = key_value;
            value_values[lane] = value_value;
            if (!std::isfinite(key_value) || !std::isfinite(value_value)) {
              invalid.store(true, std::memory_order_relaxed);
              return;
            }
            key_amax = std::max(key_amax, std::fabs(key_value));
            value_amax = std::max(value_amax, std::fabs(value_value));
          }
        } else {
          for (long long lane = 0; lane < kQ8Group; ++lane) {
            const long long index = source + group * kQ8Group + lane;
            const float key_value = load_storage(key, index);
            const float value_value = load_storage(value, index);
            key_values[lane] = key_value;
            value_values[lane] = value_value;
            if (!std::isfinite(key_value) || !std::isfinite(value_value)) {
              invalid.store(true, std::memory_order_relaxed);
              return;
            }
            key_amax = std::max(key_amax, std::fabs(key_value));
            value_amax = std::max(value_amax, std::fabs(value_value));
          }
        }
        const float key_scale = key_amax / 127.0f;
        const float value_scale = value_amax / 127.0f;
        const float key_inverse = key_scale > 0.0f ? 1.0f / key_scale : 0.0f;
        const float value_inverse =
            value_scale > 0.0f ? 1.0f / value_scale : 0.0f;
        key_scales[scale_base + group] = float_to_f16(key_scale);
        value_scales[scale_base + group] = float_to_f16(value_scale);
        for (long long lane = 0; lane < kQ8Group; ++lane) {
          const long long destination = code_base + group * kQ8Group + lane;
          key_codes[destination] = encode_q8(key_values[lane], key_inverse);
          value_codes[destination] =
              encode_q8(value_values[lane], value_inverse);
        }
      }
    }
  };
  if (ordered_unique) {
    threading::parallel_ranges(count * heads, 4,
                               [&](long long begin, long long end, int worker) {
                                 scatter_tasks(begin, end, worker);
                               });
  } else {
    scatter_tasks(0, count * heads, 0);
  }
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

Status kv_cache_scatter_q8_0(
    const float* key, const float* value, const int* slots,
    std::int8_t* key_codes, std::uint16_t* key_scales, std::int8_t* value_codes,
    std::uint16_t* value_scales, long long cache_blocks, long long count,
    long long heads, long long head_dim, long long page_size) {
  const long long elements = count * heads * head_dim;
  return kv_cache_scatter_q8_0_storage(
      FloatStorageInput{key, FloatStorageType::kF32, elements},
      FloatStorageInput{value, FloatStorageType::kF32, elements}, slots,
      key_codes, key_scales, value_codes, value_scales, cache_blocks, count,
      heads, head_dim, page_size);
}

Status kv_cache_gather_q8_0_storage(
    const std::int8_t* key_codes, const std::uint16_t* key_scales,
    const std::int8_t* value_codes, const std::uint16_t* value_scales,
    const int* block_table, const int* cumulative_lengths,
    FloatStorageOutput key_out, FloatStorageOutput value_out,
    long long cache_blocks, long long num_tokens, long long sequences,
    long long heads, long long head_dim, long long page_size,
    long long max_blocks) {
  if (!valid_q8_shape(cache_blocks, num_tokens, heads, head_dim, page_size) ||
      sequences <= 0 || max_blocks <= 0 ||
      key_out.count != num_tokens * heads * head_dim ||
      value_out.count != key_out.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage(key_out.type) || !valid_storage(value_out.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key_codes, key_scales, value_codes, value_scales,
                           block_table, cumulative_lengths, key_out.data,
                           value_out.data)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_lengths[0] != 0 ||
      cumulative_lengths[sequences] != num_tokens) {
    return Status::kInvalidArgument;
  }
  for (long long sequence = 0; sequence < sequences; ++sequence) {
    const int length =
        cumulative_lengths[sequence + 1] - cumulative_lengths[sequence];
    if (length < 0 || length > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (int local = 0; local < length; ++local) {
      const int block = block_table[sequence * max_blocks + local / page_size];
      if (block >= cache_blocks) return Status::kInvalidArgument;
    }
  }
  const long long groups = head_dim / kQ8Group;
  const bool f32_outputs = key_out.type == FloatStorageType::kF32 &&
                           value_out.type == FloatStorageType::kF32;
  auto* key_f32 = static_cast<float*>(key_out.data);
  auto* value_f32 = static_cast<float*>(value_out.data);
  threading::parallel_ranges(
      num_tokens * heads, 4, [&](long long begin, long long end, int) {
        for (long long task = begin; task < end; ++task) {
          const long long token = task / heads;
          const long long head = task % heads;
          const auto* upper = std::upper_bound(cumulative_lengths + 1,
                                               cumulative_lengths + sequences,
                                               static_cast<int>(token));
          const long long sequence = upper - cumulative_lengths - 1;
          const long long local = token - cumulative_lengths[sequence];
          const int block =
              block_table[sequence * max_blocks + local / page_size];
          const long long output_base = (token * heads + head) * head_dim;
          if (block < 0) {
            for (long long dim = 0; dim < head_dim; ++dim) {
              store_storage(key_out, output_base + dim, 0.0f);
              store_storage(value_out, output_base + dim, 0.0f);
            }
            continue;
          }
          const long long slot =
              static_cast<long long>(block) * page_size + local % page_size;
          const long long codes = code_row(slot, head, heads, head_dim);
          const long long scales = scale_row(slot, head, heads, groups);
          if (f32_outputs) {
            for (long long group = 0; group < groups; ++group) {
              const float key_scale = f16_to_float(key_scales[scales + group]);
              const float value_scale =
                  f16_to_float(value_scales[scales + group]);
              const long long group_base = group * kQ8Group;
              for (long long lane = 0; lane < kQ8Group; ++lane) {
                const long long dim = group_base + lane;
                key_f32[output_base + dim] = key_codes[codes + dim] * key_scale;
                value_f32[output_base + dim] =
                    value_codes[codes + dim] * value_scale;
              }
            }
          } else {
            for (long long group = 0; group < groups; ++group) {
              const float key_scale = f16_to_float(key_scales[scales + group]);
              const float value_scale =
                  f16_to_float(value_scales[scales + group]);
              const long long group_base = group * kQ8Group;
              for (long long lane = 0; lane < kQ8Group; ++lane) {
                const long long dim = group_base + lane;
                store_storage(key_out, output_base + dim,
                              key_codes[codes + dim] * key_scale);
                store_storage(value_out, output_base + dim,
                              value_codes[codes + dim] * value_scale);
              }
            }
          }
        }
      });
  return Status::kOk;
}

Status kv_cache_gather_q8_0(
    const std::int8_t* key_codes, const std::uint16_t* key_scales,
    const std::int8_t* value_codes, const std::uint16_t* value_scales,
    const int* block_table, const int* cumulative_lengths, float* key_out,
    float* value_out, long long cache_blocks, long long num_tokens,
    long long sequences, long long heads, long long head_dim,
    long long page_size, long long max_blocks) {
  const long long elements = num_tokens * heads * head_dim;
  return kv_cache_gather_q8_0_storage(
      key_codes, key_scales, value_codes, value_scales, block_table,
      cumulative_lengths,
      FloatStorageOutput{key_out, FloatStorageType::kF32, elements},
      FloatStorageOutput{value_out, FloatStorageType::kF32, elements},
      cache_blocks, num_tokens, sequences, heads, head_dim, page_size,
      max_blocks);
}

Status kv_cache_copy_blocks_q8_0(
    const std::int8_t* key_codes, const std::uint16_t* key_scales,
    const std::int8_t* value_codes, const std::uint16_t* value_scales,
    std::int8_t* key_codes_out, std::uint16_t* key_scales_out,
    std::int8_t* value_codes_out, std::uint16_t* value_scales_out,
    const long long* block_pairs, long long pair_count, long long cache_blocks,
    long long page_size, long long heads, long long head_dim) {
  if (!valid_q8_shape(cache_blocks, 1, heads, head_dim, page_size) ||
      pair_count < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key_codes, key_scales, value_codes, value_scales,
                           key_codes_out, key_scales_out, value_codes_out,
                           value_scales_out, block_pairs)) {
    return Status::kInvalidArgument;
  }
  for (long long pair = 0; pair < pair_count; ++pair) {
    if (block_pairs[2 * pair] >= cache_blocks ||
        block_pairs[2 * pair + 1] >= cache_blocks) {
      return Status::kInvalidArgument;
    }
  }
  const long long codes_per_block = page_size * heads * head_dim;
  const long long scales_per_block = codes_per_block / kQ8Group;
  const long long code_count = cache_blocks * codes_per_block;
  const long long scale_count = cache_blocks * scales_per_block;
  std::vector<std::int8_t> key_snapshot;
  std::vector<std::int8_t> value_snapshot;
  std::vector<std::uint16_t> key_scale_snapshot;
  std::vector<std::uint16_t> value_scale_snapshot;
  if (key_codes_out == key_codes) {
    key_snapshot.assign(key_codes, key_codes + code_count);
    key_codes = key_snapshot.data();
  }
  if (value_codes_out == value_codes) {
    value_snapshot.assign(value_codes, value_codes + code_count);
    value_codes = value_snapshot.data();
  }
  if (key_scales_out == key_scales) {
    key_scale_snapshot.assign(key_scales, key_scales + scale_count);
    key_scales = key_scale_snapshot.data();
  }
  if (value_scales_out == value_scales) {
    value_scale_snapshot.assign(value_scales, value_scales + scale_count);
    value_scales = value_scale_snapshot.data();
  }
  std::copy_n(key_codes, code_count, key_codes_out);
  std::copy_n(value_codes, code_count, value_codes_out);
  std::copy_n(key_scales, scale_count, key_scales_out);
  std::copy_n(value_scales, scale_count, value_scales_out);
  for (long long pair = 0; pair < pair_count; ++pair) {
    const long long source = block_pairs[2 * pair];
    const long long destination = block_pairs[2 * pair + 1];
    if (source < 0 || destination < 0) continue;
    std::copy_n(key_codes + source * codes_per_block, codes_per_block,
                key_codes_out + destination * codes_per_block);
    std::copy_n(value_codes + source * codes_per_block, codes_per_block,
                value_codes_out + destination * codes_per_block);
    std::copy_n(key_scales + source * scales_per_block, scales_per_block,
                key_scales_out + destination * scales_per_block);
    std::copy_n(value_scales + source * scales_per_block, scales_per_block,
                value_scales_out + destination * scales_per_block);
  }
  return Status::kOk;
}

Status paged_attention_q8_0(const float* q, const std::int8_t* key_codes,
                            const std::uint16_t* key_scales,
                            const std::int8_t* value_codes,
                            const std::uint16_t* value_scales,
                            const int* block_table, const int* context_lens,
                            float* out, long long cache_blocks, long long batch,
                            long long query_heads, long long kv_heads,
                            long long head_dim, long long page_size,
                            long long max_blocks, float scale,
                            long long window) {
  if (!detail::all_nonnull(q, out)) return Status::kInvalidArgument;
  const Status valid = validate_paged_q8(
      key_codes, key_scales, value_codes, value_scales, block_table,
      context_lens, cache_blocks, batch, query_heads, kv_heads, head_dim,
      page_size, max_blocks, scale, window);
  if (valid != Status::kOk) return valid;
  const float score_scale =
      scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
  const long long query_group = query_heads / kv_heads;
  const long long groups = head_dim / kQ8Group;
  threading::parallel_ranges(
      batch * query_heads, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long request = item / query_heads;
          const long long head = item % query_heads;
          const long long kv_head = head / query_group;
          const long long context = context_lens[request];
          const long long first =
              window > 0 ? std::max(0LL, context - window) : 0;
          const float* query = q + item * head_dim;
          float* output = out + item * head_dim;
          std::fill_n(output, head_dim, 0.0f);
          float maximum = -std::numeric_limits<float>::infinity();
          double denominator = 0.0;
          constexpr long long kScoreTile = 32;
          float scores[kScoreTile];
          long long slots[kScoreTile];
          for (long long tile = first; tile < context; tile += kScoreTile) {
            const long long tile_end = std::min(context, tile + kScoreTile);
            float tile_maximum = -std::numeric_limits<float>::infinity();
            long long valid_rows = 0;
            for (long long position = tile; position < tile_end; ++position) {
              const int block =
                  block_table[request * max_blocks + position / page_size];
              if (block < 0) continue;
              const long long slot = static_cast<long long>(block) * page_size +
                                     position % page_size;
              const long long codes =
                  code_row(slot, kv_head, kv_heads, head_dim);
              const long long scales =
                  scale_row(slot, kv_head, kv_heads, groups);
              double dot = 0.0;
              for (long long group = 0; group < groups; ++group) {
                const float group_scale =
                    f16_to_float(key_scales[scales + group]);
                double group_dot = 0.0;
                const long long group_base = group * kQ8Group;
                for (long long lane = 0; lane < kQ8Group; ++lane) {
                  group_dot += query[group_base + lane] *
                               key_codes[codes + group_base + lane];
                }
                dot += group_scale * group_dot;
              }
              scores[valid_rows] = static_cast<float>(dot * score_scale);
              slots[valid_rows] = slot;
              tile_maximum = std::max(tile_maximum, scores[valid_rows]);
              ++valid_rows;
            }
            if (valid_rows == 0) continue;
            const float next_maximum = std::max(maximum, tile_maximum);
            const double old_weight =
                denominator > 0.0 ? std::exp(maximum - next_maximum) : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0) {
              for (long long dim = 0; dim < head_dim; ++dim) {
                output[dim] *= static_cast<float>(old_weight);
              }
            }
            for (long long index = 0; index < valid_rows; ++index) {
              const double weight = std::exp(scores[index] - next_maximum);
              denominator += weight;
              const long long codes =
                  code_row(slots[index], kv_head, kv_heads, head_dim);
              const long long scales =
                  scale_row(slots[index], kv_head, kv_heads, groups);
              for (long long group = 0; group < groups; ++group) {
                const float weighted_scale = static_cast<float>(
                    weight * f16_to_float(value_scales[scales + group]));
                const long long group_base = group * kQ8Group;
                for (long long lane = 0; lane < kQ8Group; ++lane) {
                  output[group_base + lane] +=
                      weighted_scale * value_codes[codes + group_base + lane];
                }
              }
            }
            maximum = next_maximum;
          }
          if (denominator > 0.0) {
            const float inverse = static_cast<float>(1.0 / denominator);
            for (long long dim = 0; dim < head_dim; ++dim) {
              output[dim] *= inverse;
            }
          }
        }
      });
  return Status::kOk;
}

Status paged_attention_q8_0_storage(
    FloatStorageInput q, const std::int8_t* key_codes,
    const std::uint16_t* key_scales, const std::int8_t* value_codes,
    const std::uint16_t* value_scales, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks, float scale,
    long long window, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, query_heads, head_dim}) ||
      q.count != batch * query_heads * head_dim || out.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 1, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) -> Status {
        return paged_attention_q8_0(f32_inputs[0], key_codes, key_scales,
                                    value_codes, value_scales, block_table,
                                    context_lens, f32_outputs[0], cache_blocks,
                                    batch, query_heads, kv_heads, head_dim,
                                    page_size, max_blocks, scale, window);
      },
      workspace);
}

}  // namespace quixicore_cpu
