#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool valid_turboquant(long long head_size, int key_bits, int value_bits) {
  return (head_size == 64 || head_size == 128 || head_size == 256) &&
         key_bits >= 2 && key_bits <= 8 && value_bits >= 2 && value_bits <= 8;
}

unsigned unpack_bits(const std::uint8_t* bytes, long long element, int bits) {
  const long long bit_position = element * bits;
  const long long byte = bit_position / 8;
  const int offset = static_cast<int>(bit_position % 8);
  unsigned raw = bytes[byte];
  if (offset + bits > 8) raw |= unsigned(bytes[byte + 1]) << 8;
  return (raw >> offset) & ((1u << bits) - 1u);
}

template <int Bits>
inline unsigned unpack_bits_fast(const std::uint8_t* bytes, long long element,
                                 int runtime_bits) {
  if constexpr (Bits == 2) {
    return (bytes[element >> 2] >> ((element & 3) * 2)) & 3u;
  } else if constexpr (Bits == 4) {
    return (bytes[element >> 1] >> ((element & 1) * 4)) & 15u;
  } else if constexpr (Bits == 8) {
    return bytes[element];
  } else {
    return unpack_bits(bytes, element, runtime_bits);
  }
}

void fwht(float* values, long long count) {
  for (long long width = 1; width < count; width *= 2) {
    for (long long base = 0; base < count; base += 2 * width) {
      for (long long item = 0; item < width; ++item) {
        const float a = values[base + item];
        const float b = values[base + width + item];
        values[base + item] = a + b;
        values[base + width + item] = a - b;
      }
    }
  }
}

Status validate_turboquant_metadata(
    const float* key_scale_cache, const float* value_scale_cache,
    const float* key_zero_cache, const float* value_centroids,
    const float* signs, const int* block_table, const int* context_lens,
    long long cache_blocks, long long batch, long long kv_heads,
    long long head_size, long long page_size, long long max_blocks,
    int value_bits) {
  const int centroid_count = 1 << value_bits;
  for (int centroid = 0; centroid < centroid_count; ++centroid) {
    if (!std::isfinite(value_centroids[centroid]) ||
        (centroid > 0 &&
         value_centroids[centroid] < value_centroids[centroid - 1])) {
      return Status::kInvalidArgument;
    }
  }
  for (long long dim = 0; dim < head_size; ++dim) {
    if (!std::isfinite(signs[dim])) return Status::kInvalidArgument;
  }
  const long long groups = head_size / 32;
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
        const long long base = (row * kv_heads + head) * groups;
        for (long long group = 0; group < groups; ++group) {
          if (!std::isfinite(key_scale_cache[base + group]) ||
              key_scale_cache[base + group] < 0.0f ||
              !std::isfinite(value_scale_cache[base + group]) ||
              value_scale_cache[base + group] < 0.0f ||
              !std::isfinite(key_zero_cache[base + group])) {
            return Status::kInvalidArgument;
          }
        }
      }
    }
  }
  return Status::kOk;
}

template <int KeyBits, int ValueBits>
Status paged_attention_turboquant_impl(
    const float* q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const float* key_scale_cache,
    const float* value_scale_cache, const float* key_zero_cache,
    const float* value_centroids, const float* signs, const int* block_table,
    const int* context_lens, float* out, long long batch, long long query_heads,
    long long kv_heads, long long head_size, long long page_size,
    long long max_blocks, int runtime_key_bits, bool key_signed,
    int runtime_value_bits, float scale, long long window) {
  const int key_bits = KeyBits > 0 ? KeyBits : runtime_key_bits;
  const int value_bits = ValueBits > 0 ? ValueBits : runtime_value_bits;
  const long long groups = head_size / 32;
  const long long key_bytes = (head_size * key_bits + 7) / 8;
  const long long value_bytes = (head_size * value_bits + 7) / 8;
  const long long query_group = query_heads / kv_heads;
  const float score_scale =
      scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_size));
  const float normalization = 1.0f / std::sqrt(static_cast<float>(head_size));
  threading::parallel_ranges(
      batch * query_heads, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long request = item / query_heads;
          const long long qhead = item % query_heads;
          const long long kvhead = qhead / query_group;
          const long long context = context_lens[request];
          const long long start =
              window > 0 ? std::max(0LL, context - window) : 0;
          const float* query = q + item * head_size;
          float* output = out + item * head_size;
          alignas(64) float rotated_output[256]{};
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
              const auto* keys =
                  key_cache + (row * kv_heads + kvhead) * key_bytes;
              const long long metadata = (row * kv_heads + kvhead) * groups;
              double dot = 0.0;
              for (long long group = 0; group < groups; ++group) {
                const float group_scale = key_scale_cache[metadata + group];
                const float group_zero = key_zero_cache[metadata + group];
                const long long group_base = group * 32;
                for (long long offset = 0; offset < 32; ++offset) {
                  const long long dim = group_base + offset;
                  const unsigned code =
                      unpack_bits_fast<KeyBits>(keys, dim, runtime_key_bits);
                  const float quantized =
                      key_signed && key_bits == 8
                          ? static_cast<float>(static_cast<std::int8_t>(code))
                          : static_cast<float>(code);
                  dot += query[dim] * (quantized + group_zero) * group_scale;
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
              for (long long dim = 0; dim < head_size; ++dim) {
                rotated_output[dim] *= static_cast<float>(old_weight);
              }
            }
            for (long long index = 0; index < valid; ++index) {
              const double weight = std::exp(scores[index] - next_maximum);
              denominator += weight;
              const long long row = rows[index];
              const auto* values =
                  value_cache + (row * kv_heads + kvhead) * value_bytes;
              const long long metadata = (row * kv_heads + kvhead) * groups;
              for (long long group = 0; group < groups; ++group) {
                const float weighted_scale = static_cast<float>(
                    weight * value_scale_cache[metadata + group]);
                const long long group_base = group * 32;
                for (long long offset = 0; offset < 32; ++offset) {
                  const long long dim = group_base + offset;
                  const unsigned code = unpack_bits_fast<ValueBits>(
                      values, dim, runtime_value_bits);
                  rotated_output[dim] += weighted_scale * value_centroids[code];
                }
              }
            }
            maximum = next_maximum;
          }
          if (denominator > 0.0) {
            const float inverse = static_cast<float>(1.0 / denominator);
            for (long long dim = 0; dim < head_size; ++dim) {
              rotated_output[dim] *= inverse;
            }
            fwht(rotated_output, head_size);
            for (long long dim = 0; dim < head_size; ++dim) {
              output[dim] = rotated_output[dim] * normalization * signs[dim];
            }
          } else {
            std::fill_n(output, head_size, 0.0f);
          }
        }
      });
  return Status::kOk;
}

}  // namespace

Status turboquant_query_transform(const float* q, const float* signs,
                                  float* transformed, long long rows,
                                  long long heads, long long head_size) {
  if (!detail::valid_product({rows, heads, head_size}) ||
      (head_size != 64 && head_size != 128 && head_size != 256)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, signs, transformed)) {
    return Status::kInvalidArgument;
  }
  for (long long dim = 0; dim < head_size; ++dim) {
    if (!std::isfinite(signs[dim])) return Status::kInvalidArgument;
  }
  const float normalization = 1.0f / std::sqrt(static_cast<float>(head_size));
  threading::parallel_ranges(
      rows * heads, 1, [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long base = row * head_size;
          for (long long dim = 0; dim < head_size; ++dim) {
            transformed[base + dim] = q[base + dim] * signs[dim];
          }
          fwht(transformed + base, head_size);
          for (long long dim = 0; dim < head_size; ++dim) {
            transformed[base + dim] *= normalization;
          }
        }
      });
  return Status::kOk;
}

Status turboquant_query_transform_storage(FloatStorageInput q,
                                          const float* signs,
                                          FloatStorageOutput transformed,
                                          long long rows, long long heads,
                                          long long head_size,
                                          FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({rows, heads, head_size}) ||
      q.count != rows * heads * head_size || transformed.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q};
  const FloatStorageOutput outputs[] = {transformed};
  return with_float_storage(
      inputs, 1, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) -> Status {
        return turboquant_query_transform(f32_inputs[0], signs, f32_outputs[0],
                                          rows, heads, head_size);
      },
      workspace);
}

Status paged_attention_turboquant(
    const float* q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const float* key_scale_cache,
    const float* value_scale_cache, const float* key_zero_cache,
    const float* value_centroids, const float* signs, const int* block_table,
    const int* context_lens, float* out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_size, long long page_size, long long max_blocks,
    int key_bits, bool key_signed, int value_bits, float scale,
    long long window) {
  if (!detail::valid_product({cache_blocks, batch, query_heads, kv_heads,
                              head_size, page_size, max_blocks}) ||
      query_heads % kv_heads != 0 ||
      !valid_turboquant(head_size, key_bits, value_bits) ||
      !std::isfinite(scale) || scale < 0.0f || window < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, key_scale_cache,
                           value_scale_cache, key_zero_cache, value_centroids,
                           signs, block_table, context_lens, out)) {
    return Status::kInvalidArgument;
  }
  const Status metadata_status = validate_turboquant_metadata(
      key_scale_cache, value_scale_cache, key_zero_cache, value_centroids,
      signs, block_table, context_lens, cache_blocks, batch, kv_heads,
      head_size, page_size, max_blocks, value_bits);
  if (metadata_status != Status::kOk) return metadata_status;

  if (key_bits == 2 && value_bits == 2) {
    return paged_attention_turboquant_impl<2, 2>(
        q, key_cache, value_cache, key_scale_cache, value_scale_cache,
        key_zero_cache, value_centroids, signs, block_table, context_lens, out,
        batch, query_heads, kv_heads, head_size, page_size, max_blocks,
        key_bits, key_signed, value_bits, scale, window);
  }
  if (key_bits == 4 && value_bits == 4) {
    return paged_attention_turboquant_impl<4, 4>(
        q, key_cache, value_cache, key_scale_cache, value_scale_cache,
        key_zero_cache, value_centroids, signs, block_table, context_lens, out,
        batch, query_heads, kv_heads, head_size, page_size, max_blocks,
        key_bits, key_signed, value_bits, scale, window);
  }
  if (key_bits == 8 && value_bits == 8) {
    return paged_attention_turboquant_impl<8, 8>(
        q, key_cache, value_cache, key_scale_cache, value_scale_cache,
        key_zero_cache, value_centroids, signs, block_table, context_lens, out,
        batch, query_heads, kv_heads, head_size, page_size, max_blocks,
        key_bits, key_signed, value_bits, scale, window);
  }
  return paged_attention_turboquant_impl<0, 0>(
      q, key_cache, value_cache, key_scale_cache, value_scale_cache,
      key_zero_cache, value_centroids, signs, block_table, context_lens, out,
      batch, query_heads, kv_heads, head_size, page_size, max_blocks, key_bits,
      key_signed, value_bits, scale, window);
}

Status paged_attention_turboquant_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const float* key_scale_cache,
    const float* value_scale_cache, const float* key_zero_cache,
    const float* value_centroids, const float* signs, const int* block_table,
    const int* context_lens, FloatStorageOutput out, long long cache_blocks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_size, long long page_size, long long max_blocks,
    int key_bits, bool key_signed, int value_bits, float scale,
    long long window, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, query_heads, head_size}) ||
      q.count != batch * query_heads * head_size || out.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 1, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) -> Status {
        return paged_attention_turboquant(
            f32_inputs[0], key_cache, value_cache, key_scale_cache,
            value_scale_cache, key_zero_cache, value_centroids, signs,
            block_table, context_lens, f32_outputs[0], cache_blocks, batch,
            query_heads, kv_heads, head_size, page_size, max_blocks, key_bits,
            key_signed, value_bits, scale, window);
      },
      workspace);
}

}  // namespace quixicore_cpu
