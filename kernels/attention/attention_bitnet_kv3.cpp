#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quantization.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool valid_storage_type(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

bool valid_kv3_config(long long head_dim, const BitNetKv3Config& config) {
  const bool scale_type = config.scale_type == BitNetKv3ScaleType::kFP16 ||
                          config.scale_type == BitNetKv3ScaleType::kFP32;
  const bool signedness = config.signedness == BitNetKv3Signedness::kUnsigned ||
                          config.signedness == BitNetKv3Signedness::kSigned;
  const bool zero_mode =
      config.zero_point_mode == BitNetKv3ZeroPointMode::kNone ||
      config.zero_point_mode == BitNetKv3ZeroPointMode::kInteger;
  return (head_dim == 64 || head_dim == 128 || head_dim == 256) &&
         config.group_size > 0 && head_dim % config.group_size == 0 &&
         scale_type && signedness && zero_mode;
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

float load_scale(const void* scales, long long index, BitNetKv3ScaleType type) {
  if (type == BitNetKv3ScaleType::kFP32) {
    return static_cast<const float*>(scales)[index];
  }
  return f16_to_float(static_cast<const std::uint16_t*>(scales)[index]);
}

float store_scale(void* scales, long long index, BitNetKv3ScaleType type,
                  float value) {
  if (type == BitNetKv3ScaleType::kFP32) {
    static_cast<float*>(scales)[index] = value;
    return value;
  }
  const std::uint16_t bits = float_to_f16(value);
  static_cast<std::uint16_t*>(scales)[index] = bits;
  return f16_to_float(bits);
}

unsigned unpack3(const std::uint8_t* bytes, long long element) {
  const long long packet = (element >> 3) * 3;
  const int offset = static_cast<int>((element & 7) * 3);
  const unsigned raw = unsigned(bytes[packet]) |
                       (unsigned(bytes[packet + 1]) << 8) |
                       (unsigned(bytes[packet + 2]) << 16);
  return (raw >> offset) & 7u;
}

unsigned load3_packet(const std::uint8_t* bytes, long long first_element) {
  const long long packet = (first_element >> 3) * 3;
  return unsigned(bytes[packet]) | (unsigned(bytes[packet + 1]) << 8) |
         (unsigned(bytes[packet + 2]) << 16);
}

void pack3(std::uint8_t* bytes, long long element, unsigned code) {
  const long long packet = (element >> 3) * 3;
  const int offset = static_cast<int>((element & 7) * 3);
  const unsigned shifted = (code & 7u) << offset;
  bytes[packet] |= static_cast<std::uint8_t>(shifted);
  bytes[packet + 1] |= static_cast<std::uint8_t>(shifted >> 8);
  bytes[packet + 2] |= static_cast<std::uint8_t>(shifted >> 16);
}

int decode_code(unsigned raw, BitNetKv3Signedness signedness) {
  return signedness == BitNetKv3Signedness::kSigned && raw >= 4
             ? static_cast<int>(raw) - 8
             : static_cast<int>(raw);
}

unsigned encode_code(int code) { return static_cast<unsigned>(code) & 7u; }

Status validate_cache_metadata(
    const void* key_scale_cache, const void* value_scale_cache,
    const int* key_zero_cache, const int* value_zero_cache,
    const int* block_table, const int* context_lens, long long cache_blocks,
    long long batch, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, const BitNetKv3Config& config) {
  const long long groups = head_dim / config.group_size;
  const int qmin = config.signedness == BitNetKv3Signedness::kSigned ? -4 : 0;
  const int qmax = config.signedness == BitNetKv3Signedness::kSigned ? 3 : 7;
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 || context > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (long long position = 0; position < context; ++position) {
      const int physical =
          block_table[request * max_blocks + position / page_size];
      if (physical < 0 || physical >= cache_blocks) {
        return Status::kInvalidArgument;
      }
      const long long row =
          static_cast<long long>(physical) * page_size + position % page_size;
      for (long long head = 0; head < kv_heads; ++head) {
        const long long metadata = (row * kv_heads + head) * groups;
        for (long long group = 0; group < groups; ++group) {
          const long long index = metadata + group;
          const float key_scale =
              load_scale(key_scale_cache, index, config.scale_type);
          const float value_scale =
              load_scale(value_scale_cache, index, config.scale_type);
          if (!std::isfinite(key_scale) || key_scale < 0.0f ||
              !std::isfinite(value_scale) || value_scale < 0.0f) {
            return Status::kInvalidArgument;
          }
          if (config.zero_point_mode == BitNetKv3ZeroPointMode::kInteger &&
              (key_zero_cache[index] < qmin || key_zero_cache[index] > qmax ||
               value_zero_cache[index] < qmin ||
               value_zero_cache[index] > qmax)) {
            return Status::kInvalidArgument;
          }
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace

Status kv_cache_scatter_bitnet_kv3_storage(
    FloatStorageInput key, FloatStorageInput value, const int* slots,
    std::uint8_t* key_cache, std::uint8_t* value_cache, void* key_scale_cache,
    void* value_scale_cache, int* key_zero_cache, int* value_zero_cache,
    long long max_slots, long long count, long long heads, long long head_dim,
    const BitNetKv3Config& config) {
  if (!detail::valid_product({max_slots, count, heads, head_dim}) ||
      key.count != count * heads * head_dim || value.count != key.count ||
      !valid_kv3_config(head_dim, config)) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key.type) || !valid_storage_type(value.type)) {
    return Status::kUnsupportedFormat;
  }
  const bool integer_zero =
      config.zero_point_mode == BitNetKv3ZeroPointMode::kInteger;
  if (!detail::all_nonnull(key.data, value.data, slots, key_cache, value_cache,
                           key_scale_cache, value_scale_cache) ||
      (integer_zero &&
       !detail::all_nonnull(key_zero_cache, value_zero_cache))) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < count; ++token) {
    if (slots[token] >= max_slots) return Status::kInvalidArgument;
  }
  const long long packed_bytes = (head_dim * 3 + 7) / 8;
  const long long groups = head_dim / config.group_size;
  const int qmin = config.signedness == BitNetKv3Signedness::kSigned ? -4 : 0;
  const int qmax = config.signedness == BitNetKv3Signedness::kSigned ? 3 : 7;
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
      const long long source_base = (token * heads + head) * head_dim;
      const long long cache_row = static_cast<long long>(slot) * heads + head;
      auto* key_codes = key_cache + cache_row * packed_bytes;
      auto* value_codes = value_cache + cache_row * packed_bytes;
      std::fill_n(key_codes, packed_bytes, std::uint8_t{0});
      std::fill_n(value_codes, packed_bytes, std::uint8_t{0});
      for (long long group = 0; group < groups; ++group) {
        const long long source = source_base + group * config.group_size;
        float key_min = std::numeric_limits<float>::infinity();
        float key_max = -std::numeric_limits<float>::infinity();
        float value_min = std::numeric_limits<float>::infinity();
        float value_max = -std::numeric_limits<float>::infinity();
        for (long long item = 0; item < config.group_size; ++item) {
          const float key_value = load_storage(key, source + item);
          const float value_value = load_storage(value, source + item);
          if (!std::isfinite(key_value) || !std::isfinite(value_value)) {
            invalid.store(true, std::memory_order_relaxed);
            return;
          }
          key_min = std::min(key_min, key_value);
          key_max = std::max(key_max, key_value);
          value_min = std::min(value_min, value_value);
          value_max = std::max(value_max, value_value);
        }
        const auto parameters = [&](float minimum, float maximum, float* scale,
                                    int* zero) {
          if (integer_zero) {
            *scale = maximum == minimum ? 0.0f
                                        : (maximum - minimum) /
                                              static_cast<float>(qmax - qmin);
            *zero = *scale == 0.0f ? 0
                                   : std::clamp(static_cast<int>(std::nearbyint(
                                                    qmin - minimum / *scale)),
                                                qmin, qmax);
          } else if (config.signedness == BitNetKv3Signedness::kSigned) {
            *scale = std::max(maximum > 0.0f ? maximum / qmax : 0.0f,
                              minimum < 0.0f ? minimum / qmin : 0.0f);
            *zero = 0;
          } else {
            *scale = maximum > 0.0f ? maximum / qmax : 0.0f;
            *zero = 0;
          }
        };
        float key_scale = 0.0f;
        float value_scale = 0.0f;
        int key_zero = 0;
        int value_zero = 0;
        parameters(key_min, key_max, &key_scale, &key_zero);
        parameters(value_min, value_max, &value_scale, &value_zero);
        const long long metadata = cache_row * groups + group;
        key_scale = store_scale(key_scale_cache, metadata, config.scale_type,
                                key_scale);
        value_scale = store_scale(value_scale_cache, metadata,
                                  config.scale_type, value_scale);
        if (!std::isfinite(key_scale) || !std::isfinite(value_scale)) {
          invalid.store(true, std::memory_order_relaxed);
          return;
        }
        if (integer_zero) {
          key_zero_cache[metadata] = key_zero;
          value_zero_cache[metadata] = value_zero;
        }
        for (long long item = 0; item < config.group_size; ++item) {
          const long long dim = group * config.group_size + item;
          const float key_value = load_storage(key, source + item);
          const float value_value = load_storage(value, source + item);
          const int key_code = key_scale == 0.0f
                                   ? key_zero
                                   : std::clamp(static_cast<int>(std::nearbyint(
                                                    key_value / key_scale)) +
                                                    key_zero,
                                                qmin, qmax);
          const int value_code =
              value_scale == 0.0f ? value_zero
                                  : std::clamp(static_cast<int>(std::nearbyint(
                                                   value_value / value_scale)) +
                                                   value_zero,
                                               qmin, qmax);
          pack3(key_codes, dim, encode_code(key_code));
          pack3(value_codes, dim, encode_code(value_code));
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

Status kv_cache_scatter_bitnet_kv3(
    const float* key, const float* value, const int* slots,
    std::uint8_t* key_cache, std::uint8_t* value_cache, void* key_scale_cache,
    void* value_scale_cache, int* key_zero_cache, int* value_zero_cache,
    long long max_slots, long long count, long long heads, long long head_dim,
    const BitNetKv3Config& config) {
  const long long elements = count * heads * head_dim;
  return kv_cache_scatter_bitnet_kv3_storage(
      FloatStorageInput{key, FloatStorageType::kF32, elements},
      FloatStorageInput{value, FloatStorageType::kF32, elements}, slots,
      key_cache, value_cache, key_scale_cache, value_scale_cache,
      key_zero_cache, value_zero_cache, max_slots, count, heads, head_dim,
      config);
}

Status kv_cache_gather_bitnet_kv3_storage(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, FloatStorageOutput key_out,
    FloatStorageOutput value_out, long long max_slots, long long count,
    long long heads, long long head_dim, const BitNetKv3Config& config) {
  if (!detail::valid_product({max_slots, count, heads, head_dim}) ||
      key_out.count != count * heads * head_dim ||
      value_out.count != key_out.count || !valid_kv3_config(head_dim, config)) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key_out.type) ||
      !valid_storage_type(value_out.type)) {
    return Status::kUnsupportedFormat;
  }
  const bool integer_zero =
      config.zero_point_mode == BitNetKv3ZeroPointMode::kInteger;
  if (!detail::all_nonnull(key_cache, value_cache, indices, key_scale_cache,
                           value_scale_cache, key_out.data, value_out.data) ||
      (integer_zero &&
       !detail::all_nonnull(key_zero_cache, value_zero_cache))) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < count; ++row) {
    if (indices[row] < 0 || indices[row] >= max_slots) {
      return Status::kInvalidArgument;
    }
  }
  const long long packed_bytes = (head_dim * 3 + 7) / 8;
  const long long groups = head_dim / config.group_size;
  threading::parallel_ranges(
      count * heads, 4, [&](long long begin, long long end, int) {
        for (long long task = begin; task < end; ++task) {
          const long long row = task / heads;
          const long long head = task % heads;
          const long long slot = indices[row];
          const long long cache_row = slot * heads + head;
          const auto* key_codes = key_cache + cache_row * packed_bytes;
          const auto* value_codes = value_cache + cache_row * packed_bytes;
          const long long output_base = (row * heads + head) * head_dim;
          const long long metadata = cache_row * groups;
          for (long long group = 0; group < groups; ++group) {
            const int key_zero =
                integer_zero ? key_zero_cache[metadata + group] : 0;
            const int value_zero =
                integer_zero ? value_zero_cache[metadata + group] : 0;
            const float key_group_scale = load_scale(
                key_scale_cache, metadata + group, config.scale_type);
            const float value_group_scale = load_scale(
                value_scale_cache, metadata + group, config.scale_type);
            const long long group_base = group * config.group_size;
            if ((config.group_size & 7) == 0) {
              for (long long offset = 0; offset < config.group_size;
                   offset += 8) {
                const long long dim = group_base + offset;
                unsigned key_packet = load3_packet(key_codes, dim);
                unsigned value_packet = load3_packet(value_codes, dim);
                for (int lane = 0; lane < 8; ++lane) {
                  const float key_value =
                      (decode_code(key_packet & 7u, config.signedness) -
                       key_zero) *
                      key_group_scale;
                  const float value_value =
                      (decode_code(value_packet & 7u, config.signedness) -
                       value_zero) *
                      value_group_scale;
                  store_storage(key_out, output_base + dim + lane, key_value);
                  store_storage(value_out, output_base + dim + lane,
                                value_value);
                  key_packet >>= 3;
                  value_packet >>= 3;
                }
              }
            } else {
              for (long long offset = 0; offset < config.group_size; ++offset) {
                const long long dim = group_base + offset;
                const float key_value =
                    (decode_code(unpack3(key_codes, dim), config.signedness) -
                     key_zero) *
                    key_group_scale;
                const float value_value =
                    (decode_code(unpack3(value_codes, dim), config.signedness) -
                     value_zero) *
                    value_group_scale;
                store_storage(key_out, output_base + dim, key_value);
                store_storage(value_out, output_base + dim, value_value);
              }
            }
          }
        }
      });
  return Status::kOk;
}

Status kv_cache_gather_bitnet_kv3(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const int* indices, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, float* key_out, float* value_out,
    long long max_slots, long long count, long long heads, long long head_dim,
    const BitNetKv3Config& config) {
  const long long elements = count * heads * head_dim;
  return kv_cache_gather_bitnet_kv3_storage(
      key_cache, value_cache, indices, key_scale_cache, value_scale_cache,
      key_zero_cache, value_zero_cache,
      FloatStorageOutput{key_out, FloatStorageType::kF32, elements},
      FloatStorageOutput{value_out, FloatStorageType::kF32, elements},
      max_slots, count, heads, head_dim, config);
}

Status paged_attention_bitnet_kv3(
    const float* q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, const int* block_table,
    const int* context_lens, float* out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    const BitNetKv3Config& config, float scale, long long window) {
  if (!detail::valid_product({cache_blocks, batch, query_heads, kv_heads,
                              head_dim, page_size, max_blocks}) ||
      query_heads % kv_heads != 0 || !valid_kv3_config(head_dim, config) ||
      !std::isfinite(scale) || scale < 0.0f || window < 0) {
    return Status::kInvalidShape;
  }
  const bool integer_zero =
      config.zero_point_mode == BitNetKv3ZeroPointMode::kInteger;
  if (!detail::all_nonnull(q, key_cache, value_cache, key_scale_cache,
                           value_scale_cache, block_table, context_lens, out) ||
      (integer_zero &&
       !detail::all_nonnull(key_zero_cache, value_zero_cache))) {
    return Status::kInvalidArgument;
  }
  const Status metadata_status = validate_cache_metadata(
      key_scale_cache, value_scale_cache, key_zero_cache, value_zero_cache,
      block_table, context_lens, cache_blocks, batch, kv_heads, head_dim,
      page_size, max_blocks, config);
  if (metadata_status != Status::kOk) return metadata_status;

  const long long packed_bytes = (head_dim * 3 + 7) / 8;
  const long long groups = head_dim / config.group_size;
  const long long query_group = query_heads / kv_heads;
  const float score_scale =
      scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
  threading::parallel_ranges(
      batch * query_heads, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long request = item / query_heads;
          const long long qhead = item % query_heads;
          const long long kvhead = qhead / query_group;
          const long long context = context_lens[request];
          const long long start =
              window > 0 ? std::max(0LL, context - window) : 0;
          const float* query = q + item * head_dim;
          float* output = out + item * head_dim;
          alignas(64) float accumulator[256]{};
          float maximum = -std::numeric_limits<float>::infinity();
          double denominator = 0.0;
          constexpr long long kScoreTile = 16;
          float scores[kScoreTile];
          long long rows[kScoreTile];
          for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = std::min(context, tile + kScoreTile);
            float tile_maximum = -std::numeric_limits<float>::infinity();
            long long valid = 0;
            for (long long position = tile; position < tile_end; ++position) {
              const int physical =
                  block_table[request * max_blocks + position / page_size];
              const long long row =
                  static_cast<long long>(physical) * page_size +
                  position % page_size;
              const long long cache_row = row * kv_heads + kvhead;
              const auto* keys = key_cache + cache_row * packed_bytes;
              const long long metadata = cache_row * groups;
              double dot = 0.0;
              for (long long group = 0; group < groups; ++group) {
                const int zero =
                    integer_zero ? key_zero_cache[metadata + group] : 0;
                const float group_scale = load_scale(
                    key_scale_cache, metadata + group, config.scale_type);
                const long long group_base = group * config.group_size;
                if ((config.group_size & 7) == 0) {
                  for (long long offset = 0; offset < config.group_size;
                       offset += 8) {
                    const long long dim = group_base + offset;
                    unsigned packet = load3_packet(keys, dim);
                    for (int lane = 0; lane < 8; ++lane) {
                      const float key =
                          (decode_code(packet & 7u, config.signedness) - zero) *
                          group_scale;
                      dot += query[dim + lane] * key;
                      packet >>= 3;
                    }
                  }
                } else {
                  for (long long offset = 0; offset < config.group_size;
                       ++offset) {
                    const long long dim = group_base + offset;
                    const float key =
                        (decode_code(unpack3(keys, dim), config.signedness) -
                         zero) *
                        group_scale;
                    dot += query[dim] * key;
                  }
                }
              }
              scores[valid] = static_cast<float>(dot * score_scale);
              rows[valid] = row;
              tile_maximum = std::max(tile_maximum, scores[valid]);
              ++valid;
            }
            const float next_maximum = std::max(maximum, tile_maximum);
            const double old_weight =
                denominator > 0.0 ? std::exp(maximum - next_maximum) : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0) {
              for (long long dim = 0; dim < head_dim; ++dim) {
                accumulator[dim] *= static_cast<float>(old_weight);
              }
            }
            for (long long index = 0; index < valid; ++index) {
              const double weight = std::exp(scores[index] - next_maximum);
              denominator += weight;
              const long long cache_row = rows[index] * kv_heads + kvhead;
              const auto* values = value_cache + cache_row * packed_bytes;
              const long long metadata = cache_row * groups;
              for (long long group = 0; group < groups; ++group) {
                const int zero =
                    integer_zero ? value_zero_cache[metadata + group] : 0;
                const float weighted_scale = static_cast<float>(
                    weight * load_scale(value_scale_cache, metadata + group,
                                        config.scale_type));
                const long long group_base = group * config.group_size;
                if ((config.group_size & 7) == 0) {
                  for (long long offset = 0; offset < config.group_size;
                       offset += 8) {
                    const long long dim = group_base + offset;
                    unsigned packet = load3_packet(values, dim);
                    for (int lane = 0; lane < 8; ++lane) {
                      const int value =
                          decode_code(packet & 7u, config.signedness) - zero;
                      accumulator[dim + lane] += weighted_scale * value;
                      packet >>= 3;
                    }
                  }
                } else {
                  for (long long offset = 0; offset < config.group_size;
                       ++offset) {
                    const long long dim = group_base + offset;
                    const int value =
                        decode_code(unpack3(values, dim), config.signedness) -
                        zero;
                    accumulator[dim] += weighted_scale * value;
                  }
                }
              }
            }
            maximum = next_maximum;
          }
          if (denominator > 0.0) {
            const float inverse = static_cast<float>(1.0 / denominator);
            for (long long dim = 0; dim < head_dim; ++dim) {
              output[dim] = accumulator[dim] * inverse;
            }
          } else {
            std::fill_n(output, head_dim, 0.0f);
          }
        }
      });
  return Status::kOk;
}

Status paged_attention_bitnet_kv3_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const void* key_scale_cache,
    const void* value_scale_cache, const int* key_zero_cache,
    const int* value_zero_cache, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    const BitNetKv3Config& config, float scale, long long window,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, query_heads, head_dim}) ||
      q.count != batch * query_heads * head_dim || out.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 1, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) -> Status {
        return paged_attention_bitnet_kv3(
            f32_inputs[0], key_cache, value_cache, key_scale_cache,
            value_scale_cache, key_zero_cache, value_zero_cache, block_table,
            context_lens, f32_outputs[0], cache_blocks, batch, query_heads,
            kv_heads, head_dim, page_size, max_blocks, config, scale, window);
      },
      workspace);
}

}  // namespace quixicore_cpu
