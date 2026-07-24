#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "quixicore_cpu/ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/validation.h"
#include "kernels/quantization/gguf_ref.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

double dot_f32(const float* lhs, const float* rhs, long long count) {
  double total = 0.0;
  long long item = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  float64x2_t sum0 = vdupq_n_f64(0.0);
  float64x2_t sum1 = vdupq_n_f64(0.0);
  for (; item + 3 < count; item += 4) {
    const float32x4_t products =
        vmulq_f32(vld1q_f32(lhs + item), vld1q_f32(rhs + item));
    sum0 = vaddq_f64(sum0, vcvt_f64_f32(vget_low_f32(products)));
    sum1 = vaddq_f64(sum1, vcvt_f64_f32(vget_high_f32(products)));
  }
  total = vaddvq_f64(vaddq_f64(sum0, sum1));
#endif
  for (; item < count; ++item) total += lhs[item] * rhs[item];
  return total;
}

float capped_score(double score, float softcap) {
  return softcap > 0.0f
             ? softcap * std::tanh(static_cast<float>(score) / softcap)
             : static_cast<float>(score);
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

bool valid_paged(long long cache_blocks, long long batch, long long query_heads,
                 long long kv_heads, long long head_dim, long long page_size,
                 long long max_blocks, float scale, long long window,
                 float softcap) {
  return detail::valid_product({cache_blocks, batch, query_heads, kv_heads,
                                head_dim, page_size, max_blocks}) &&
         query_heads % kv_heads == 0 && std::isfinite(scale) && scale >= 0.0f &&
         window >= 0 && std::isfinite(softcap) && softcap >= 0.0f;
}

template <typename KeyValue>
Status paged_impl(const float* q, const KeyValue* key_cache,
                  const KeyValue* value_cache, const int* block_table,
                  const int* context_lens, const int* block_mask,
                  const float* alibi_slopes, const float* sinks,
                  const float* key_scale, const float* value_scale,
                  Float8Format fp8_format, float* out, long long cache_blocks,
                  long long batch, long long query_heads, long long kv_heads,
                  long long head_dim, long long page_size, long long max_blocks,
                  float scale, long long window, float softcap) {
  if (!valid_paged(cache_blocks, batch, query_heads, kv_heads, head_dim,
                   page_size, max_blocks, scale, window, softcap)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, block_table, context_lens,
                           out)) {
    return Status::kInvalidArgument;
  }
  const bool is_fp8 = key_scale != nullptr || value_scale != nullptr;
  if (is_fp8 && !detail::all_nonnull(key_scale, value_scale)) {
    return Status::kInvalidArgument;
  }
  if (is_fp8) {
    if (fp8_format != Float8Format::kE4M3FN &&
        fp8_format != Float8Format::kE5M2) {
      return Status::kUnsupportedFormat;
    }
    for (long long head = 0; head < kv_heads; ++head) {
      if (!std::isfinite(key_scale[head]) || key_scale[head] < 0.0f ||
          !std::isfinite(value_scale[head]) || value_scale[head] < 0.0f) {
        return Status::kInvalidArgument;
      }
    }
  }
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
    }
  }
  const float score_scale = scale > 0.0f ? scale : 1.0f / std::sqrt(head_dim);
  const float* fp8_table =
      is_fp8 ? fp8_decode_table(fp8_format).data() : nullptr;
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(
      batch * query_heads, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long request = item / query_heads;
          const long long qhead = item % query_heads;
          const long long kvhead = qhead / group;
          const long long context = context_lens[request];
          const long long start =
              window > 0 ? std::max(0LL, context - window) : 0;
          const float* query = q + item * head_dim;
          float* output = out + item * head_dim;
          const float head_score_scale =
              is_fp8 ? score_scale * key_scale[kvhead] : score_scale;
          const float head_value_scale = is_fp8 ? value_scale[kvhead] : 1.0f;
          std::fill_n(output, head_dim, 0.0f);
          float maximum = sinks != nullptr
                              ? sinks[qhead]
                              : -std::numeric_limits<float>::infinity();
          double denominator = sinks != nullptr ? 1.0 : 0.0;
          constexpr long long kScoreTile = 16;
          alignas(64) float scores[kScoreTile];
          long long bases[kScoreTile];
          for (long long tile_begin = start; tile_begin < context;
               tile_begin += kScoreTile) {
            const long long tile_end =
                std::min(tile_begin + kScoreTile, context);
            long long valid = 0;
            float tile_maximum = -std::numeric_limits<float>::infinity();
            for (long long position = tile_begin; position < tile_end;
                 ++position) {
              const long long logical_block = position / page_size;
              if (block_mask != nullptr &&
                  block_mask[(request * query_heads + qhead) * max_blocks +
                             logical_block] == 0) {
                continue;
              }
              const int physical =
                  block_table[request * max_blocks + logical_block];
              const long long base =
                  ((static_cast<long long>(physical) * page_size +
                    position % page_size) *
                       kv_heads +
                   kvhead) *
                  head_dim;
              double dot = 0.0;
              if constexpr (sizeof(KeyValue) == sizeof(std::uint8_t)) {
                for (long long dim = 0; dim < head_dim; ++dim) {
                  dot += query[dim] * fp8_table[static_cast<std::uint8_t>(
                                          key_cache[base + dim])];
                }
              } else {
                dot = dot_f32(query,
                              reinterpret_cast<const float*>(key_cache + base),
                              head_dim);
              }
              dot *= head_score_scale;
              if (alibi_slopes != nullptr) {
                dot += alibi_slopes[qhead] * (position - (context - 1));
              }
              const float score = capped_score(dot, softcap);
              scores[valid] = score;
              bases[valid] = base;
              tile_maximum = std::max(tile_maximum, score);
              ++valid;
            }
            if (valid == 0) continue;
            const float next_maximum = std::max(maximum, tile_maximum);
            const double old_weight =
                denominator > 0.0 ? std::exp(maximum - next_maximum) : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0) {
              const float scale_output = static_cast<float>(old_weight);
              for (long long dim = 0; dim < head_dim; ++dim) {
                output[dim] *= scale_output;
              }
            }
            for (long long index = 0; index < valid; ++index) {
              const double weight = std::exp(scores[index] - next_maximum);
              denominator += weight;
              const long long base = bases[index];
              if constexpr (sizeof(KeyValue) == sizeof(std::uint8_t)) {
                for (long long dim = 0; dim < head_dim; ++dim) {
                  const float value = fp8_table[static_cast<std::uint8_t>(
                      value_cache[base + dim])];
                  output[dim] += static_cast<float>(value * weight);
                }
              } else {
                for (long long dim = 0; dim < head_dim; ++dim) {
                  const float value =
                      static_cast<float>(value_cache[base + dim]);
                  output[dim] += static_cast<float>(value * weight);
                }
              }
            }
            maximum = next_maximum;
          }
          if (denominator > 0.0) {
            const float inverse =
                static_cast<float>(head_value_scale / denominator);
            for (long long dim = 0; dim < head_dim; ++dim)
              output[dim] *= inverse;
          }
        }
      });
  return Status::kOk;
}

}  // namespace

Status quantized_attention(const float* q, const void* packed_k,
                           const void* packed_v, float* out, long long batch,
                           long long heads, long long sequence,
                           long long head_dim, QuantFormat format,
                           bool causal) {
  if (!detail::valid_product({batch, heads, sequence, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, packed_k, packed_v, out)) {
    return Status::kInvalidArgument;
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  if (head_dim % block_size != 0) return Status::kInvalidShape;
  const std::size_t vector_bytes =
      static_cast<std::size_t>(head_dim / block_size) * block_bytes;
  const auto* keys = static_cast<const std::uint8_t*>(packed_k);
  const auto* values = static_cast<const std::uint8_t*>(packed_v);
  const float scale = 1.0f / std::sqrt(head_dim);
  threading::parallel_ranges(
      batch * heads * sequence, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long query_position = item % sequence;
          const long long prefix = item / sequence;
          const long long key_count = causal ? query_position + 1 : sequence;
          const float* query = q + item * head_dim;
          std::vector<double> scores(static_cast<std::size_t>(key_count));
          double maximum = -std::numeric_limits<double>::infinity();
          for (long long key_position = 0; key_position < key_count;
               ++key_position) {
            const std::uint8_t* key =
                keys +
                static_cast<std::size_t>(prefix * sequence + key_position) *
                    vector_bytes;
            double dot = 0.0;
            for (long long dim = 0; dim < head_dim; ++dim) {
              dot += query[dim] *
                     quant::gguf_dequant_element(
                         format,
                         key + static_cast<std::size_t>(dim / block_size) *
                                   block_bytes,
                         static_cast<int>(dim % block_size));
            }
            scores[key_position] = dot * scale;
            maximum = std::max(maximum, scores[key_position]);
          }
          double denominator = 0.0;
          for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          float* output = out + item * head_dim;
          std::fill_n(output, head_dim, 0.0f);
          for (long long key_position = 0; key_position < key_count;
               ++key_position) {
            const std::uint8_t* value =
                values +
                static_cast<std::size_t>(prefix * sequence + key_position) *
                    vector_bytes;
            const double probability = scores[key_position] / denominator;
            for (long long dim = 0; dim < head_dim; ++dim) {
              output[dim] += static_cast<float>(
                  probability *
                  quant::gguf_dequant_element(
                      format,
                      value + static_cast<std::size_t>(dim / block_size) *
                                  block_bytes,
                      static_cast<int>(dim % block_size)));
            }
          }
        }
      });
  return Status::kOk;
}

Status rope_q_norm(const float* q, const float* cosine, const float* sine,
                   const int* positions, const float* norm_weight, float* out,
                   long long tokens, long long heads, long long head_dim,
                   long long max_position, bool do_norm, bool gemma_weight,
                   float eps) {
  if (!detail::valid_product({tokens, heads, head_dim}) || head_dim % 2 != 0 ||
      max_position <= 0 || !std::isfinite(eps) || eps <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, cosine, sine, positions, out) ||
      (do_norm && norm_weight == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (positions[token] < 0 || positions[token] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  const long long half = head_dim / 2;
  for (long long token = 0; token < tokens; ++token) {
    for (long long head = 0; head < heads; ++head) {
      const float* input = q + (token * heads + head) * head_dim;
      float* output = out + (token * heads + head) * head_dim;
      double square_sum = 0.0;
      if (do_norm) {
        for (long long dim = 0; dim < head_dim; ++dim) {
          square_sum += input[dim] * input[dim];
        }
      }
      const float inverse =
          do_norm ? 1.0f / std::sqrt(square_sum / head_dim + eps) : 1.0f;
      for (long long dim = 0; dim < half; ++dim) {
        const float w0 =
            do_norm ? norm_weight[dim] + (gemma_weight ? 1.0f : 0.0f) : 1.0f;
        const float w1 =
            do_norm ? norm_weight[dim + half] + (gemma_weight ? 1.0f : 0.0f)
                    : 1.0f;
        const float first = input[dim] * inverse * w0;
        const float second = input[dim + half] * inverse * w1;
        const float c = cosine[positions[token] * half + dim];
        const float s = sine[positions[token] * half + dim];
        output[dim] = first * c - second * s;
        output[dim + half] = second * c + first * s;
      }
    }
  }
  return Status::kOk;
}

Status rope_kv_insert(const float* k, const float* v, const float* cosine,
                      const float* sine, const int* positions,
                      const int* slot_mapping, const float* norm_weight,
                      float* key_cache, float* value_cache, long long tokens,
                      long long slots, long long kv_heads, long long head_dim,
                      long long max_position, bool do_norm, bool gemma_weight,
                      float eps) {
  if (!detail::valid_product({tokens, slots, kv_heads, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(k, v, cosine, sine, positions, slot_mapping,
                           key_cache, value_cache)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (slot_mapping[token] >= slots || positions[token] < 0 ||
        positions[token] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  std::vector<float> rotated(
      static_cast<std::size_t>(tokens * kv_heads * head_dim));
  Status status = rope_q_norm(k, cosine, sine, positions, norm_weight,
                              rotated.data(), tokens, kv_heads, head_dim,
                              max_position, do_norm, gemma_weight, eps);
  if (status != Status::kOk) return status;
  for (long long token = 0; token < tokens; ++token) {
    const int slot = slot_mapping[token];
    if (slot < 0) continue;
    std::copy_n(rotated.data() + token * kv_heads * head_dim,
                kv_heads * head_dim,
                key_cache + static_cast<long long>(slot) * kv_heads * head_dim);
    std::copy_n(
        v + token * kv_heads * head_dim, kv_heads * head_dim,
        value_cache + static_cast<long long>(slot) * kv_heads * head_dim);
  }
  return Status::kOk;
}

Status paged_attention_advanced(
    const float* q, const float* key_cache, const float* value_cache,
    const int* block_table, const int* context_lens, const int* block_mask,
    const float* alibi_slopes, const float* sinks, float* out,
    long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, float scale, long long window, float softcap) {
  return paged_impl(q, key_cache, value_cache, block_table, context_lens,
                    block_mask, alibi_slopes, sinks, nullptr, nullptr,
                    Float8Format::kE4M3FN, out, cache_blocks, batch,
                    query_heads, kv_heads, head_dim, page_size, max_blocks,
                    scale, window, softcap);
}

Status paged_attention_fp8(const float* q, const std::uint8_t* key_cache,
                           const std::uint8_t* value_cache,
                           const int* block_table, const int* context_lens,
                           const float* key_scale, const float* value_scale,
                           float* out, long long cache_blocks, long long batch,
                           long long query_heads, long long kv_heads,
                           long long head_dim, long long page_size,
                           long long max_blocks, Float8Format format,
                           float scale, long long window, float softcap) {
  return paged_impl(q, key_cache, value_cache, block_table, context_lens,
                    nullptr, nullptr, nullptr, key_scale, value_scale, format,
                    out, cache_blocks, batch, query_heads, kv_heads, head_dim,
                    page_size, max_blocks, scale, window, softcap);
}

Status paged_attention_fp8_storage(
    FloatStorageInput q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const int* block_table,
    const int* context_lens, const float* key_scale, const float* value_scale,
    FloatStorageOutput out, long long cache_blocks, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, Float8Format format, float scale,
    long long window, float softcap, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, query_heads, head_dim}) ||
      q.count != batch * query_heads * head_dim || out.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 1, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) -> Status {
        return paged_attention_fp8(
            f32_inputs[0], key_cache, value_cache, block_table, context_lens,
            key_scale, value_scale, f32_outputs[0], cache_blocks, batch,
            query_heads, kv_heads, head_dim, page_size, max_blocks, format,
            scale, window, softcap);
      },
      workspace);
}

Status paged_attention_xcache(const float* q, const float* key_cache,
                              const float* value_cache, const int* block_table,
                              const int* context_lens, float* out,
                              long long cache_blocks, long long batch,
                              long long query_heads, long long kv_heads,
                              long long head_dim, long long page_size,
                              long long max_blocks, long long vector_width,
                              float scale) {
  if (!valid_paged(cache_blocks, batch, query_heads, kv_heads, head_dim,
                   page_size, max_blocks, scale, 0, 0.0f) ||
      vector_width <= 0 || head_dim % vector_width != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, block_table, context_lens,
                           out)) {
    return Status::kInvalidArgument;
  }
  const long long elements = cache_blocks * page_size * kv_heads * head_dim;
  std::vector<float> standard_key(static_cast<std::size_t>(elements));
  std::vector<float> standard_value(static_cast<std::size_t>(elements));
  for (long long block = 0; block < cache_blocks; ++block) {
    for (long long position = 0; position < page_size; ++position) {
      for (long long head = 0; head < kv_heads; ++head) {
        for (long long dim = 0; dim < head_dim; ++dim) {
          const long long standard =
              ((block * page_size + position) * kv_heads + head) * head_dim +
              dim;
          const long long split_key =
              ((((block * kv_heads + head) * (head_dim / vector_width) +
                 dim / vector_width) *
                    page_size +
                position) *
                   vector_width +
               dim % vector_width);
          const long long transposed_value =
              ((block * kv_heads + head) * head_dim + dim) * page_size +
              position;
          standard_key[standard] = key_cache[split_key];
          standard_value[standard] = value_cache[transposed_value];
        }
      }
    }
  }
  return paged_attention(q, standard_key.data(), standard_value.data(),
                         block_table, context_lens, out, cache_blocks, batch,
                         query_heads, kv_heads, head_dim, page_size, max_blocks,
                         scale);
}

Status cascade_attention(const float* q, const float* prefix_k,
                         const float* prefix_v, const float* key_cache,
                         const float* value_cache, const int* block_table,
                         const int* context_lens, float* out,
                         long long cache_blocks, long long batch,
                         long long query_heads, long long kv_heads,
                         long long head_dim, long long prefix_length,
                         long long page_size, long long max_blocks,
                         float scale) {
  if (!valid_paged(cache_blocks, batch, query_heads, kv_heads, head_dim,
                   page_size, max_blocks, scale, 0, 0.0f) ||
      prefix_length < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, prefix_k, prefix_v, key_cache, value_cache,
                           block_table, context_lens, out)) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    if (context_lens[request] < 0 ||
        context_lens[request] > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (long long position = 0; position < context_lens[request]; ++position) {
      const int physical =
          block_table[request * max_blocks + position / page_size];
      if (physical < 0 || physical >= cache_blocks) {
        return Status::kInvalidArgument;
      }
    }
  }
  const float score_scale = scale > 0.0f ? scale : 1.0f / std::sqrt(head_dim);
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(
      batch * query_heads, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long request = item / query_heads;
          const long long qhead = item % query_heads;
          const long long kvhead = qhead / group;
          const long long suffix = context_lens[request];
          const long long total = prefix_length + suffix;
          const float* query = q + item * head_dim;
          std::vector<double> scores(static_cast<std::size_t>(total));
          double maximum = -std::numeric_limits<double>::infinity();
          for (long long position = 0; position < total; ++position) {
            const float* key = nullptr;
            if (position < prefix_length) {
              key = prefix_k + (position * kv_heads + kvhead) * head_dim;
            } else {
              const long long local = position - prefix_length;
              const int physical =
                  block_table[request * max_blocks + local / page_size];
              key = key_cache + ((static_cast<long long>(physical) * page_size +
                                  local % page_size) *
                                     kv_heads +
                                 kvhead) *
                                    head_dim;
            }
            double dot = 0.0;
            for (long long dim = 0; dim < head_dim; ++dim)
              dot += query[dim] * key[dim];
            scores[position] = dot * score_scale;
            maximum = std::max(maximum, scores[position]);
          }
          double denominator = 0.0;
          for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          float* output = out + item * head_dim;
          std::fill_n(output, head_dim, 0.0f);
          for (long long position = 0; position < total; ++position) {
            const float* value = nullptr;
            if (position < prefix_length) {
              value = prefix_v + (position * kv_heads + kvhead) * head_dim;
            } else {
              const long long local = position - prefix_length;
              const int physical =
                  block_table[request * max_blocks + local / page_size];
              value =
                  value_cache + ((static_cast<long long>(physical) * page_size +
                                  local % page_size) *
                                     kv_heads +
                                 kvhead) *
                                    head_dim;
            }
            const double probability = scores[position] / denominator;
            for (long long dim = 0; dim < head_dim; ++dim) {
              output[dim] += static_cast<float>(probability * value[dim]);
            }
          }
        }
      });
  return Status::kOk;
}

Status cascade_attention_fp8(
    const float* q, const std::uint8_t* prefix_k, const std::uint8_t* prefix_v,
    const float* key_cache, const float* value_cache, const int* block_table,
    const int* context_lens, const float* key_scale, const float* value_scale,
    float* out, long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long prefix_length,
    long long page_size, long long max_blocks, Float8Format format,
    float scale) {
  if (!valid_paged(cache_blocks, batch, query_heads, kv_heads, head_dim,
                   page_size, max_blocks, scale, 0, 0.0f) ||
      prefix_length <= 0 ||
      !detail::valid_product({prefix_length, kv_heads, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, prefix_k, prefix_v, key_cache, value_cache,
                           block_table, context_lens, key_scale, value_scale,
                           out)) {
    return Status::kInvalidArgument;
  }
  for (long long head = 0; head < kv_heads; ++head) {
    if (!std::isfinite(key_scale[head]) || key_scale[head] <= 0.0f ||
        !std::isfinite(value_scale[head]) || value_scale[head] <= 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  const long long elements = prefix_length * kv_heads * head_dim;
  std::vector<float> decoded_k(static_cast<std::size_t>(elements));
  std::vector<float> decoded_v(static_cast<std::size_t>(elements));
  for (long long position = 0; position < prefix_length; ++position) {
    for (long long head = 0; head < kv_heads; ++head) {
      for (long long dim = 0; dim < head_dim; ++dim) {
        const long long index = (position * kv_heads + head) * head_dim + dim;
        decoded_k[index] =
            float8_decode(prefix_k[index], format) * key_scale[head];
        decoded_v[index] =
            float8_decode(prefix_v[index], format) * value_scale[head];
      }
    }
  }
  return cascade_attention(q, decoded_k.data(), decoded_v.data(), key_cache,
                           value_cache, block_table, context_lens, out,
                           cache_blocks, batch, query_heads, kv_heads, head_dim,
                           prefix_length, page_size, max_blocks, scale);
}

}  // namespace quixicore_cpu
