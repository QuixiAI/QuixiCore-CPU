#include <algorithm>
#include <array>
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

constexpr long long kMxGroup = 32;
constexpr long long kMxBlockBytes = 33;

bool valid_mx_shape(long long max_slots, long long count, long long heads,
                    long long head_dim) {
  return detail::valid_product({max_slots, count, heads, head_dim}) &&
         (head_dim == 64 || head_dim == 128);
}

bool valid_storage_type(FloatStorageType type) {
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

const std::array<float, 256>& e8m0_decode_table() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (int code = 0; code < 255; ++code) {
      values[static_cast<std::size_t>(code)] = std::ldexp(1.0f, code - 127);
    }
    values[255] = std::numeric_limits<float>::quiet_NaN();
    return values;
  }();
  return table;
}

const std::array<float, 256>& e4m3_decode_table() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (int code = 0; code < 256; ++code) {
      values[static_cast<std::size_t>(code)] =
          float8_decode(static_cast<std::uint8_t>(code), Float8Format::kE4M3FN);
    }
    return values;
  }();
  return table;
}

float e8m0_decode(std::uint8_t code) { return e8m0_decode_table()[code]; }

std::uint8_t e8m0_encode_up(float requested) {
  if (!(requested > 0.0f)) return 0;
  const int exponent = static_cast<int>(std::ceil(std::log2(requested)));
  return static_cast<std::uint8_t>(std::clamp(exponent + 127, 0, 254));
}

long long mx_group_base(long long row, long long head, long long group,
                        long long heads, long long head_dim) {
  const long long groups = head_dim / kMxGroup;
  return ((row * heads + head) * groups + group) * kMxBlockBytes;
}

Status validate_mx_cache(const std::uint8_t* key_cache,
                         const std::uint8_t* value_cache,
                         const int* block_table, const int* context_lens,
                         long long cache_blocks, long long batch,
                         long long kv_heads, long long head_dim,
                         long long page_size, long long max_blocks) {
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
        for (long long group = 0; group < head_dim / kMxGroup; ++group) {
          const long long base =
              mx_group_base(row, head, group, kv_heads, head_dim);
          if (key_cache[base] == 255 || value_cache[base] == 255) {
            return Status::kInvalidArgument;
          }
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace

Status kv_cache_scatter_mxfp8_storage(FloatStorageInput key,
                                      FloatStorageInput value, const int* slots,
                                      std::uint8_t* key_cache,
                                      std::uint8_t* value_cache,
                                      long long max_slots, long long count,
                                      long long heads, long long head_dim) {
  if (!valid_mx_shape(max_slots, count, heads, head_dim) ||
      key.count != count * heads * head_dim || value.count != key.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key.type) || !valid_storage_type(value.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key.data, value.data, slots, key_cache,
                           value_cache)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < count; ++token) {
    if (slots[token] >= max_slots) return Status::kInvalidArgument;
  }
  const long long groups = head_dim / kMxGroup;
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
      for (long long group = 0; group < groups; ++group) {
        const long long source =
            (token * heads + head) * head_dim + group * kMxGroup;
        const long long destination =
            mx_group_base(slot, head, group, heads, head_dim);
        float key_max = 0.0f;
        float value_max = 0.0f;
        for (long long item = 0; item < kMxGroup; ++item) {
          const float key_value = load_storage(key, source + item);
          const float value_value = load_storage(value, source + item);
          if (!std::isfinite(key_value) || !std::isfinite(value_value)) {
            invalid.store(true, std::memory_order_relaxed);
            return;
          }
          key_max = std::max(key_max, std::fabs(key_value));
          value_max = std::max(value_max, std::fabs(value_value));
        }
        const std::uint8_t key_scale_code = e8m0_encode_up(key_max / 448.0f);
        const std::uint8_t value_scale_code =
            e8m0_encode_up(value_max / 448.0f);
        key_cache[destination] = key_scale_code;
        value_cache[destination] = value_scale_code;
        const float inverse_key =
            key_max > 0.0f ? 1.0f / e8m0_decode(key_scale_code) : 0.0f;
        const float inverse_value =
            value_max > 0.0f ? 1.0f / e8m0_decode(value_scale_code) : 0.0f;
        for (long long item = 0; item < kMxGroup; ++item) {
          key_cache[destination + 1 + item] =
              float8_encode(load_storage(key, source + item) * inverse_key,
                            Float8Format::kE4M3FN);
          value_cache[destination + 1 + item] =
              float8_encode(load_storage(value, source + item) * inverse_value,
                            Float8Format::kE4M3FN);
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

Status kv_cache_scatter_mxfp8(const float* key, const float* value,
                              const int* slots, std::uint8_t* key_cache,
                              std::uint8_t* value_cache, long long max_slots,
                              long long count, long long heads,
                              long long head_dim) {
  if (!valid_mx_shape(max_slots, count, heads, head_dim)) {
    return Status::kInvalidShape;
  }
  const long long elements = count * heads * head_dim;
  return kv_cache_scatter_mxfp8_storage(
      FloatStorageInput{key, FloatStorageType::kF32, elements},
      FloatStorageInput{value, FloatStorageType::kF32, elements}, slots,
      key_cache, value_cache, max_slots, count, heads, head_dim);
}

Status kv_cache_gather_mxfp8_storage(const std::uint8_t* key_cache,
                                     const std::uint8_t* value_cache,
                                     const int* indices,
                                     FloatStorageOutput key_out,
                                     FloatStorageOutput value_out,
                                     long long max_slots, long long count,
                                     long long heads, long long head_dim) {
  if (!valid_mx_shape(max_slots, count, heads, head_dim) ||
      key_out.count != count * heads * head_dim ||
      value_out.count != key_out.count) {
    return Status::kInvalidShape;
  }
  if (!valid_storage_type(key_out.type) ||
      !valid_storage_type(value_out.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(key_cache, value_cache, indices, key_out.data,
                           value_out.data)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < count; ++token) {
    if (indices[token] < 0 || indices[token] >= max_slots) {
      return Status::kInvalidArgument;
    }
  }
  const long long groups = head_dim / kMxGroup;
  const float* fp8_decode = e4m3_decode_table().data();
  std::atomic<bool> invalid{false};
  threading::parallel_ranges(
      count * heads, 4, [&](long long begin, long long end, int) {
        for (long long task = begin; task < end; ++task) {
          if (invalid.load(std::memory_order_relaxed)) return;
          const long long token = task / heads;
          const long long head = task % heads;
          for (long long group = 0; group < groups; ++group) {
            const long long source =
                mx_group_base(indices[token], head, group, heads, head_dim);
            if (key_cache[source] == 255 || value_cache[source] == 255) {
              invalid.store(true, std::memory_order_relaxed);
              return;
            }
            const float key_scale = e8m0_decode(key_cache[source]);
            const float value_scale = e8m0_decode(value_cache[source]);
            const long long destination =
                (token * heads + head) * head_dim + group * kMxGroup;
            for (long long item = 0; item < kMxGroup; ++item) {
              store_storage(
                  key_out, destination + item,
                  key_scale * fp8_decode[key_cache[source + 1 + item]]);
              store_storage(
                  value_out, destination + item,
                  value_scale * fp8_decode[value_cache[source + 1 + item]]);
            }
          }
        }
      });
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

Status kv_cache_gather_mxfp8(const std::uint8_t* key_cache,
                             const std::uint8_t* value_cache,
                             const int* indices, float* key_out,
                             float* value_out, long long max_slots,
                             long long count, long long heads,
                             long long head_dim) {
  if (!valid_mx_shape(max_slots, count, heads, head_dim)) {
    return Status::kInvalidShape;
  }
  const long long elements = count * heads * head_dim;
  return kv_cache_gather_mxfp8_storage(
      key_cache, value_cache, indices,
      FloatStorageOutput{key_out, FloatStorageType::kF32, elements},
      FloatStorageOutput{value_out, FloatStorageType::kF32, elements},
      max_slots, count, heads, head_dim);
}

Status paged_attention_mxfp8(const float* q, const std::uint8_t* key_cache,
                             const std::uint8_t* value_cache,
                             const int* block_table, const int* context_lens,
                             float* out, long long cache_blocks,
                             long long batch, long long query_heads,
                             long long kv_heads, long long head_dim,
                             long long page_size, long long max_blocks,
                             float scale, long long window) {
  if (!detail::valid_product({cache_blocks, batch, query_heads, kv_heads,
                              head_dim, page_size, max_blocks}) ||
      query_heads % kv_heads != 0 || (head_dim != 64 && head_dim != 128) ||
      !std::isfinite(scale) || scale < 0.0f || window < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, block_table, context_lens,
                           out)) {
    return Status::kInvalidArgument;
  }
  const Status cache_status = validate_mx_cache(
      key_cache, value_cache, block_table, context_lens, cache_blocks, batch,
      kv_heads, head_dim, page_size, max_blocks);
  if (cache_status != Status::kOk) return cache_status;
  const float score_scale =
      scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
  const long long query_group = query_heads / kv_heads;
  const long long groups = head_dim / kMxGroup;
  const float* fp8_decode = e4m3_decode_table().data();
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
          std::fill_n(output, head_dim, 0.0f);
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
              double dot = 0.0;
              for (long long group = 0; group < groups; ++group) {
                const long long base =
                    mx_group_base(row, kvhead, group, kv_heads, head_dim);
                const float group_scale = e8m0_decode(key_cache[base]);
                const long long query_base = group * kMxGroup;
                double group_dot = 0.0;
                for (long long d = 0; d < kMxGroup; ++d) {
                  group_dot += query[query_base + d] *
                               fp8_decode[key_cache[base + 1 + d]];
                }
                dot += group_scale * group_dot;
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
              for (long long d = 0; d < head_dim; ++d) {
                output[d] *= static_cast<float>(old_weight);
              }
            }
            for (long long index = 0; index < valid; ++index) {
              const double weight = std::exp(scores[index] - next_maximum);
              denominator += weight;
              for (long long group = 0; group < groups; ++group) {
                const long long base = mx_group_base(rows[index], kvhead, group,
                                                     kv_heads, head_dim);
                const float group_scale = e8m0_decode(value_cache[base]);
                const float scaled_weight =
                    static_cast<float>(weight * group_scale);
                const long long output_base = group * kMxGroup;
                for (long long d = 0; d < kMxGroup; ++d) {
                  output[output_base + d] +=
                      scaled_weight * fp8_decode[value_cache[base + 1 + d]];
                }
              }
            }
            maximum = next_maximum;
          }
          if (denominator > 0.0) {
            const float inverse = static_cast<float>(1.0 / denominator);
            for (long long d = 0; d < head_dim; ++d) output[d] *= inverse;
          }
        }
      });
  return Status::kOk;
}

Status paged_attention_mxfp8_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const int* block_table,
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
        return paged_attention_mxfp8(
            f32_inputs[0], key_cache, value_cache, block_table, context_lens,
            f32_outputs[0], cache_blocks, batch, query_heads, kv_heads,
            head_dim, page_size, max_blocks, scale, window);
      },
      workspace);
}

}  // namespace quixicore_cpu
