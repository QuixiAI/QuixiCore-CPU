#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/quantization.h"

namespace quixicore_cpu {

Status kv_cache_copy_blocks(float* key_cache, float* value_cache,
                            const long long* block_pairs,
                            long long pair_count, long long block_count,
                            long long elements_per_block) {
  if (!detail::valid_product({block_count, elements_per_block}) ||
      pair_count < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key_cache, value_cache, block_pairs)) {
    return Status::kInvalidArgument;
  }
  const long long elements = block_count * elements_per_block;
  std::vector<float> key_source(key_cache, key_cache + elements);
  std::vector<float> value_source(value_cache, value_cache + elements);
  for (long long pair = 0; pair < pair_count; ++pair) {
    const long long source = block_pairs[2 * pair];
    const long long destination = block_pairs[2 * pair + 1];
    if (source < 0 || destination < 0) continue;
    if (source >= block_count || destination >= block_count) {
      return Status::kInvalidArgument;
    }
    std::copy_n(key_source.data() + source * elements_per_block,
                elements_per_block,
                key_cache + destination * elements_per_block);
    std::copy_n(value_source.data() + source * elements_per_block,
                elements_per_block,
                value_cache + destination * elements_per_block);
  }
  return Status::kOk;
}

Status beam_build_copy_pairs(const int* parent_beam, const int* block_table,
                             const int* sequence_lengths,
                             long long* block_pairs, long long batch,
                             long long beam_width, long long max_blocks,
                             int block_size) {
  if (!detail::valid_product({batch, beam_width, max_blocks}) ||
      block_size <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(parent_beam, block_table, sequence_lengths,
                           block_pairs)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    for (long long beam = 0; beam < beam_width; ++beam) {
      const long long global_beam = request * beam_width + beam;
      const int parent = parent_beam[global_beam];
      if (parent < 0 || parent >= beam_width || sequence_lengths[global_beam] < 0) {
        return Status::kInvalidArgument;
      }
      const long long used_blocks =
          (sequence_lengths[global_beam] + block_size - 1) / block_size;
      for (long long column = 0; column < max_blocks; ++column) {
        const long long pair = global_beam * max_blocks + column;
        long long source = -1, destination = -1;
        if (parent != beam && column < used_blocks) {
          source = block_table[(request * beam_width + parent) * max_blocks +
                               column];
          destination = block_table[global_beam * max_blocks + column];
          if (source < 0 || destination < 0) source = destination = -1;
        }
        block_pairs[2 * pair] = source;
        block_pairs[2 * pair + 1] = destination;
      }
    }
  }
  return Status::kOk;
}

Status beam_remap_block_table(const int* block_table,
                              const int* parent_beam,
                              int* remapped_block_table, long long batch,
                              long long beam_width, long long max_blocks) {
  if (!detail::valid_product({batch, beam_width, max_blocks})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(block_table, parent_beam,
                           remapped_block_table)) {
    return Status::kInvalidArgument;
  }
  std::vector<int> source(block_table,
                          block_table + batch * beam_width * max_blocks);
  for (long long request = 0; request < batch; ++request) {
    for (long long beam = 0; beam < beam_width; ++beam) {
      const int parent = parent_beam[request * beam_width + beam];
      if (parent < 0 || parent >= beam_width) return Status::kInvalidArgument;
      std::copy_n(source.data() +
                      (request * beam_width + parent) * max_blocks,
                  max_blocks,
                  remapped_block_table +
                      (request * beam_width + beam) * max_blocks);
    }
  }
  return Status::kOk;
}

Status kv_cache_scales(const float* key, const float* value,
                       float* key_scale, float* value_scale,
                       long long count) {
  if (count <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(key, value, key_scale, value_scale)) {
    return Status::kInvalidArgument;
  }
  float key_max = 0.0f, value_max = 0.0f;
  for (long long item = 0; item < count; ++item) {
    if (!std::isfinite(key[item]) || !std::isfinite(value[item])) {
      return Status::kInvalidArgument;
    }
    key_max = std::max(key_max, std::fabs(key[item]));
    value_max = std::max(value_max, std::fabs(value[item]));
  }
  *key_scale = key_max / 240.0f;
  *value_scale = value_max / 240.0f;
  return Status::kOk;
}

Status kv_cache_scale_update(const float* key, const float* value,
                             float old_key_scale, float old_value_scale,
                             float* new_key_scale, float* new_value_scale,
                             long long count) {
  if (!detail::all_nonnull(new_key_scale, new_value_scale)) {
    return Status::kInvalidArgument;
  }
  if (!std::isfinite(old_key_scale) || old_key_scale < 0.0f ||
      !std::isfinite(old_value_scale) || old_value_scale < 0.0f) {
    return Status::kInvalidShape;
  }
  float key_scale = 0.0f, value_scale = 0.0f;
  const Status status =
      kv_cache_scales(key, value, &key_scale, &value_scale, count);
  if (status != Status::kOk) return status;
  *new_key_scale = std::max(old_key_scale, key_scale);
  *new_value_scale = std::max(old_value_scale, value_scale);
  return Status::kOk;
}

Status kv_cache_scatter_fp8(
    const float* key, const float* value, const int* slots,
    const float* key_scale, const float* value_scale,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    long long max_slots, long long count, long long heads,
    long long head_dim, Float8Format format) {
  if (!detail::valid_product({max_slots, count, heads, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key, value, slots, key_scale, value_scale,
                           key_cache, value_cache)) {
    return Status::kInvalidArgument;
  }
  const long long row_width = heads * head_dim;
  for (long long token = 0; token < count; ++token) {
    const int slot = slots[token];
    if (slot < 0) continue;
    if (slot >= max_slots) return Status::kInvalidArgument;
    for (long long head = 0; head < heads; ++head) {
      if (!std::isfinite(key_scale[head]) || key_scale[head] <= 0.0f ||
          !std::isfinite(value_scale[head]) || value_scale[head] <= 0.0f) {
        return Status::kInvalidArgument;
      }
      for (long long item = 0; item < head_dim; ++item) {
        const long long source = token * row_width + head * head_dim + item;
        const long long destination =
            static_cast<long long>(slot) * row_width + head * head_dim + item;
        key_cache[destination] = float8_encode(key[source] / key_scale[head],
                                               format);
        value_cache[destination] =
            float8_encode(value[source] / value_scale[head], format);
      }
    }
  }
  return Status::kOk;
}

Status kv_cache_gather_fp8(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const float* key_scale, const float* value_scale,
    float* key_out, float* value_out, long long max_slots,
    long long count, long long heads, long long head_dim,
    Float8Format format) {
  if (!detail::valid_product({max_slots, count, heads, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key_cache, value_cache, indices, key_scale,
                           value_scale, key_out, value_out)) {
    return Status::kInvalidArgument;
  }
  const long long row_width = heads * head_dim;
  for (long long token = 0; token < count; ++token) {
    const int slot = indices[token];
    if (slot < 0 || slot >= max_slots) return Status::kInvalidArgument;
    for (long long head = 0; head < heads; ++head) {
      for (long long item = 0; item < head_dim; ++item) {
        const long long source =
            static_cast<long long>(slot) * row_width + head * head_dim + item;
        const long long destination = token * row_width + head * head_dim + item;
        key_out[destination] =
            float8_decode(key_cache[source], format) * key_scale[head];
        value_out[destination] =
            float8_decode(value_cache[source], format) * value_scale[head];
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
