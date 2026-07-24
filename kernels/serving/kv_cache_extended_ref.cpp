#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quantization.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool valid_fp8_format(Float8Format format) {
  return format == Float8Format::kE4M3FN || format == Float8Format::kE5M2;
}

template <FloatStorageType Type>
float load_storage(const void* data, long long index) {
  if constexpr (Type == FloatStorageType::kF32) {
    return static_cast<const float*>(data)[index];
  } else if constexpr (Type == FloatStorageType::kF16) {
    return f16_to_float(static_cast<const std::uint16_t*>(data)[index]);
  } else {
    return bf16_to_float(static_cast<const std::uint16_t*>(data)[index]);
  }
}

float load_storage(FloatStorageInput input, long long index) {
  if (input.type == FloatStorageType::kF32) {
    return load_storage<FloatStorageType::kF32>(input.data, index);
  }
  return input.type == FloatStorageType::kF16
             ? load_storage<FloatStorageType::kF16>(input.data, index)
             : load_storage<FloatStorageType::kBF16>(input.data, index);
}

template <FloatStorageType Type>
void store_storage(void* data, long long index, float value) {
  if constexpr (Type == FloatStorageType::kF32) {
    static_cast<float*>(data)[index] = value;
  } else if constexpr (Type == FloatStorageType::kF16) {
    static_cast<std::uint16_t*>(data)[index] = float_to_f16(value);
  } else {
    static_cast<std::uint16_t*>(data)[index] = float_to_bf16(value);
  }
}

bool valid_storage_type(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

const std::array<float, 256>& fp8_decode_table(Float8Format format) {
  static const std::array<float, 256> e4m3 = [] {
    std::array<float, 256> table{};
    for (std::size_t code = 0; code < table.size(); ++code) {
      table[code] =
          float8_decode(static_cast<std::uint8_t>(code), Float8Format::kE4M3FN);
    }
    return table;
  }();
  static const std::array<float, 256> e5m2 = [] {
    std::array<float, 256> table{};
    for (std::size_t code = 0; code < table.size(); ++code) {
      table[code] =
          float8_decode(static_cast<std::uint8_t>(code), Float8Format::kE5M2);
    }
    return table;
  }();
  return format == Float8Format::kE5M2 ? e5m2 : e4m3;
}

template <FloatStorageType KeyType, FloatStorageType ValueType>
Status scatter_fp8_typed(const void* key, const void* value, const int* slots,
                         const float* key_scale, const float* value_scale,
                         std::uint8_t* key_cache, std::uint8_t* value_cache,
                         long long count, long long heads, long long head_dim,
                         Float8Format format) {
  const long long row_width = heads * head_dim;
  bool ordered_unique = true;
  int previous_slot = -1;
  for (long long token = 0; token < count; ++token) {
    if (slots[token] < 0) continue;
    if (slots[token] <= previous_slot) ordered_unique = false;
    previous_slot = slots[token];
  }
  std::atomic<bool> invalid{false};
  const auto scatter_tasks = [&](long long begin, long long end, int) {
    for (long long task = begin; task < end; ++task) {
      if (invalid.load(std::memory_order_relaxed)) return;
      const long long token = task / heads;
      const long long head = task % heads;
      const int slot = slots[token];
      if (slot < 0) continue;
      const float inverse_key =
          key_scale[head] > 0.0f ? 1.0f / key_scale[head] : 0.0f;
      const float inverse_value =
          value_scale[head] > 0.0f ? 1.0f / value_scale[head] : 0.0f;
      const long long source = token * row_width + head * head_dim;
      const long long destination =
          static_cast<long long>(slot) * row_width + head * head_dim;
      for (long long item = 0; item < head_dim; ++item) {
        const float key_value = load_storage<KeyType>(key, source + item);
        const float value_value = load_storage<ValueType>(value, source + item);
        if (!std::isfinite(key_value) || !std::isfinite(value_value)) {
          invalid.store(true, std::memory_order_relaxed);
          return;
        }
        key_cache[destination + item] =
            float8_encode(key_value * inverse_key, format);
        value_cache[destination + item] =
            float8_encode(value_value * inverse_value, format);
      }
    }
  };
  if (ordered_unique) {
    threading::parallel_ranges(count * heads, 8,
                               [&](long long begin, long long end, int worker) {
                                 scatter_tasks(begin, end, worker);
                               });
  } else {
    scatter_tasks(0, count * heads, 0);
  }
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

template <FloatStorageType KeyType, FloatStorageType ValueType>
void gather_fp8_typed(const std::uint8_t* key_cache,
                      const std::uint8_t* value_cache, const int* indices,
                      const float* key_scale, const float* value_scale,
                      void* key_out, void* value_out, long long count,
                      long long heads, long long head_dim,
                      const float* decode) {
  const long long row_width = heads * head_dim;
  threading::parallel_ranges(
      count * heads, 8, [&](long long begin, long long end, int) {
        for (long long task = begin; task < end; ++task) {
          const long long token = task / heads;
          const long long head = task % heads;
          const long long source_row =
              static_cast<long long>(indices[token]) * row_width;
          const long long destination_row = token * row_width;
          const long long source = source_row + head * head_dim;
          const long long destination = destination_row + head * head_dim;
          for (long long item = 0; item < head_dim; ++item) {
            store_storage<KeyType>(
                key_out, destination + item,
                decode[key_cache[source + item]] * key_scale[head]);
            store_storage<ValueType>(
                value_out, destination + item,
                decode[value_cache[source + item]] * value_scale[head]);
          }
        }
      });
}

template <FloatStorageType KeyType>
Status scatter_fp8_value_dispatch(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    const float* key_scale, const float* value_scale, std::uint8_t* key_cache,
    std::uint8_t* value_cache, long long count, long long heads,
    long long head_dim, Float8Format format) {
  if (value.type == FloatStorageType::kF32) {
    return scatter_fp8_typed<KeyType, FloatStorageType::kF32>(
        key.data, value.data, slots, key_scale, value_scale, key_cache,
        value_cache, count, heads, head_dim, format);
  }
  if (value.type == FloatStorageType::kF16) {
    return scatter_fp8_typed<KeyType, FloatStorageType::kF16>(
        key.data, value.data, slots, key_scale, value_scale, key_cache,
        value_cache, count, heads, head_dim, format);
  }
  return scatter_fp8_typed<KeyType, FloatStorageType::kBF16>(
      key.data, value.data, slots, key_scale, value_scale, key_cache,
      value_cache, count, heads, head_dim, format);
}

template <FloatStorageType KeyType>
void gather_fp8_value_dispatch(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const float* key_scale, const float* value_scale,
    FloatStorageOutput key_out, FloatStorageOutput value_out, long long count,
    long long heads, long long head_dim, const float* decode) {
  if (value_out.type == FloatStorageType::kF32) {
    gather_fp8_typed<KeyType, FloatStorageType::kF32>(
        key_cache, value_cache, indices, key_scale, value_scale, key_out.data,
        value_out.data, count, heads, head_dim, decode);
  } else if (value_out.type == FloatStorageType::kF16) {
    gather_fp8_typed<KeyType, FloatStorageType::kF16>(
        key_cache, value_cache, indices, key_scale, value_scale, key_out.data,
        value_out.data, count, heads, head_dim, decode);
  } else {
    gather_fp8_typed<KeyType, FloatStorageType::kBF16>(
        key_cache, value_cache, indices, key_scale, value_scale, key_out.data,
        value_out.data, count, heads, head_dim, decode);
  }
}

}  // namespace

Status kv_cache_copy_blocks(float* key_cache, float* value_cache,
                            const long long* block_pairs, long long pair_count,
                            long long block_count,
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
      if (parent < 0 || parent >= beam_width ||
          sequence_lengths[global_beam] < 0) {
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

Status beam_remap_block_table(const int* block_table, const int* parent_beam,
                              int* remapped_block_table, long long batch,
                              long long beam_width, long long max_blocks) {
  if (!detail::valid_product({batch, beam_width, max_blocks})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(block_table, parent_beam, remapped_block_table)) {
    return Status::kInvalidArgument;
  }
  std::vector<int> source(block_table,
                          block_table + batch * beam_width * max_blocks);
  for (long long request = 0; request < batch; ++request) {
    for (long long beam = 0; beam < beam_width; ++beam) {
      const int parent = parent_beam[request * beam_width + beam];
      if (parent < 0 || parent >= beam_width) return Status::kInvalidArgument;
      std::copy_n(
          source.data() + (request * beam_width + parent) * max_blocks,
          max_blocks,
          remapped_block_table + (request * beam_width + beam) * max_blocks);
    }
  }
  return Status::kOk;
}

Status kv_cache_scales(const float* key, const float* value, float* key_scale,
                       float* value_scale, long long count) {
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

Status kv_cache_scatter_fp8(const float* key, const float* value,
                            const int* slots, const float* key_scale,
                            const float* value_scale, std::uint8_t* key_cache,
                            std::uint8_t* value_cache, long long max_slots,
                            long long count, long long heads,
                            long long head_dim, Float8Format format) {
  if (!detail::valid_product({max_slots, count, heads, head_dim})) {
    return Status::kInvalidShape;
  }
  const long long elements = count * heads * head_dim;
  return kv_cache_scatter_fp8_storage(
      FloatStorageInput{key, FloatStorageType::kF32, elements},
      FloatStorageInput{value, FloatStorageType::kF32, elements}, slots,
      key_scale, value_scale, key_cache, value_cache, max_slots, count, heads,
      head_dim, format);
}

Status kv_cache_scatter_fp8_storage(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    const float* key_scale, const float* value_scale, std::uint8_t* key_cache,
    std::uint8_t* value_cache, long long max_slots, long long count,
    long long heads, long long head_dim, Float8Format format) {
  if (!detail::valid_product({max_slots, count, heads, head_dim}) ||
      key.count != count * heads * head_dim || value.count != key.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key.type) || !valid_storage_type(value.type) ||
      !valid_fp8_format(format)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key.data, value.data, slots, key_scale, value_scale,
                           key_cache, value_cache)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < count; ++token) {
    const int slot = slots[token];
    if (slot >= max_slots) return Status::kInvalidArgument;
  }
  for (long long head = 0; head < heads; ++head) {
    if (!std::isfinite(key_scale[head]) || key_scale[head] < 0.0f ||
        !std::isfinite(value_scale[head]) || value_scale[head] < 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  if (key.type == FloatStorageType::kF32) {
    return scatter_fp8_value_dispatch<FloatStorageType::kF32>(
        key, value, slots, key_scale, value_scale, key_cache, value_cache,
        count, heads, head_dim, format);
  }
  if (key.type == FloatStorageType::kF16) {
    return scatter_fp8_value_dispatch<FloatStorageType::kF16>(
        key, value, slots, key_scale, value_scale, key_cache, value_cache,
        count, heads, head_dim, format);
  }
  return scatter_fp8_value_dispatch<FloatStorageType::kBF16>(
      key, value, slots, key_scale, value_scale, key_cache, value_cache, count,
      heads, head_dim, format);
}

Status kv_cache_gather_fp8(const std::uint8_t* key_cache,
                           const std::uint8_t* value_cache, const int* indices,
                           const float* key_scale, const float* value_scale,
                           float* key_out, float* value_out,
                           long long max_slots, long long count,
                           long long heads, long long head_dim,
                           Float8Format format) {
  if (!detail::valid_product({max_slots, count, heads, head_dim})) {
    return Status::kInvalidShape;
  }
  const long long elements = count * heads * head_dim;
  return kv_cache_gather_fp8_storage(
      key_cache, value_cache, indices, key_scale, value_scale,
      FloatStorageOutput{key_out, FloatStorageType::kF32, elements},
      FloatStorageOutput{value_out, FloatStorageType::kF32, elements},
      max_slots, count, heads, head_dim, format);
}

Status kv_cache_gather_fp8_storage(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const float* key_scale, const float* value_scale,
    FloatStorageOutput key_out, FloatStorageOutput value_out,
    long long max_slots, long long count, long long heads, long long head_dim,
    Float8Format format) {
  if (!detail::valid_product({max_slots, count, heads, head_dim}) ||
      key_out.count != count * heads * head_dim ||
      value_out.count != key_out.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key_out.type) ||
      !valid_storage_type(value_out.type) || !valid_fp8_format(format)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key_cache, value_cache, indices, key_scale,
                           value_scale, key_out.data, value_out.data)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < count; ++token) {
    const int slot = indices[token];
    if (slot < 0 || slot >= max_slots) return Status::kInvalidArgument;
  }
  for (long long head = 0; head < heads; ++head) {
    if (!std::isfinite(key_scale[head]) || key_scale[head] < 0.0f ||
        !std::isfinite(value_scale[head]) || value_scale[head] < 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  const float* decode = fp8_decode_table(format).data();
  if (key_out.type == FloatStorageType::kF32) {
    gather_fp8_value_dispatch<FloatStorageType::kF32>(
        key_cache, value_cache, indices, key_scale, value_scale, key_out,
        value_out, count, heads, head_dim, decode);
  } else if (key_out.type == FloatStorageType::kF16) {
    gather_fp8_value_dispatch<FloatStorageType::kF16>(
        key_cache, value_cache, indices, key_scale, value_scale, key_out,
        value_out, count, heads, head_dim, decode);
  } else {
    gather_fp8_value_dispatch<FloatStorageType::kBF16>(
        key_cache, value_cache, indices, key_scale, value_scale, key_out,
        value_out, count, heads, head_dim, decode);
  }
  return Status::kOk;
}

Status kv_cache_scales_fp8_storage(FloatStorageInput key,
                                   FloatStorageInput value, float* key_scale,
                                   float* value_scale, long long count,
                                   long long heads, long long head_dim,
                                   Float8Format format,
                                   FloatStorageWorkspace*) {
  if (!detail::valid_product({count, heads, head_dim}) ||
      key.count != count * heads * head_dim || value.count != key.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key.type) || !valid_storage_type(value.type) ||
      !valid_fp8_format(format)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key.data, value.data, key_scale, value_scale)) {
    return Status::kInvalidArgument;
  }
  const float maximum = format == Float8Format::kE5M2 ? 57344.0f : 448.0f;
  std::atomic<bool> invalid{false};
  threading::parallel_ranges(
      heads, 1, [&](long long begin, long long end, int) {
        for (long long head = begin; head < end; ++head) {
          float key_max = 0.0f;
          float value_max = 0.0f;
          for (long long token = 0; token < count; ++token) {
            const long long base = (token * heads + head) * head_dim;
            for (long long item = 0; item < head_dim; ++item) {
              const float key_value = load_storage(key, base + item);
              const float value_value = load_storage(value, base + item);
              if (!std::isfinite(key_value) || !std::isfinite(value_value)) {
                invalid.store(true, std::memory_order_relaxed);
                return;
              }
              key_max = std::max(key_max, std::fabs(key_value));
              value_max = std::max(value_max, std::fabs(value_value));
            }
          }
          key_scale[head] = key_max / maximum;
          value_scale[head] = value_max / maximum;
        }
      });
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

}  // namespace quixicore_cpu
