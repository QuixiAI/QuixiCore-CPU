#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float default_scale(long long width, float scale) {
  return scale == 0.0f
             ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(width)))
             : scale;
}

}  // namespace

Status rope(const float* x, float* y, long long tokens, long long heads,
            long long head_dim, float base, long long pos0) {
  if (!detail::valid_product({tokens, heads, head_dim}) ||
      head_dim % 2 != 0 || pos0 < 0 || pos0 > LLONG_MAX - tokens ||
      !std::isfinite(base) || base <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  const long long half = head_dim / 2;
  threading::parallel_ranges(tokens * heads, 8,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long token = item / heads;
      const long long offset = item * head_dim;
      const double position = static_cast<double>(pos0 + token);
      for (long long i = 0; i < half; ++i) {
        const double frequency =
            std::pow(static_cast<double>(base),
                     -2.0 * static_cast<double>(i) /
                         static_cast<double>(head_dim));
        const double angle = position * frequency;
        const float cosine = static_cast<float>(std::cos(angle));
        const float sine = static_cast<float>(std::sin(angle));
        const float first = x[offset + i];
        const float second = x[offset + half + i];
        y[offset + i] = first * cosine - second * sine;
        y[offset + half + i] = second * cosine + first * sine;
      }
    }
  });
  return Status::kOk;
}

Status rope_interleaved_to_split(const float* x, float* y, long long tokens,
                                 long long heads, long long head_dim,
                                 float base, long long pos0) {
  if (!detail::valid_product({tokens, heads, head_dim}) ||
      head_dim % 2 != 0 || pos0 < 0 || pos0 > LLONG_MAX - tokens ||
      !std::isfinite(base) || base <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) return Status::kInvalidArgument;
  const long long half = head_dim / 2;
  threading::parallel_ranges(tokens * heads, 8,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<float> scratch;
    if (x == y && static_cast<long long>(scratch.size()) < head_dim) {
      scratch.resize(static_cast<std::size_t>(head_dim));
    }
    for (long long item = begin; item < end; ++item) {
      const long long token = item / heads;
      const long long offset = item * head_dim;
      const float* source = x + offset;
      if (x == y) {
        std::copy_n(source, head_dim, scratch.data());
        source = scratch.data();
      }
      const double position = static_cast<double>(pos0 + token);
      for (long long pair = 0; pair < half; ++pair) {
        const double frequency =
            std::pow(static_cast<double>(base),
                     -2.0 * static_cast<double>(pair) /
                         static_cast<double>(head_dim));
        const double angle = position * frequency;
        const float cosine = static_cast<float>(std::cos(angle));
        const float sine = static_cast<float>(std::sin(angle));
        const float first = source[2 * pair];
        const float second = source[2 * pair + 1];
        y[offset + pair] = first * cosine - second * sine;
        y[offset + half + pair] = second * cosine + first * sine;
      }
    }
  });
  return Status::kOk;
}

Status attention(const float* q, const float* k, const float* v, float* out,
                 long long query_heads, long long kv_heads,
                 long long query_length, long long kv_length,
                 long long head_dim, bool causal) {
  if (!detail::valid_product({query_heads, query_length, head_dim}) ||
      !detail::valid_product({kv_heads, kv_length, head_dim}) ||
      query_heads % kv_heads != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) {
    return Status::kInvalidArgument;
  }
  const float scale = default_scale(head_dim, 0.0f);
  const long long queries = query_heads * query_length;
  threading::parallel_ranges(queries, 1,
                             [&](long long begin, long long end, int) {
    for (long long query_index = begin; query_index < end; ++query_index) {
      const long long head = query_index / query_length;
      const long long position = query_index % query_length;
      const long long kv_head = head / (query_heads / kv_heads);
      const float* query = q + query_index * head_dim;
      float* destination = out + query_index * head_dim;
      std::fill(destination, destination + head_dim, 0.0f);

      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      const long long causal_limit = position + (kv_length - query_length);
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) {
          continue;
        }
        const float* key =
            k + (kv_head * kv_length + key_position) * head_dim;
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          score += static_cast<double>(query[d]) * key[d];
        }
        score *= scale;
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        const float* value =
            v + (kv_head * kv_length + key_position) * head_dim;
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] = static_cast<float>(destination[d] * old_weight +
                                              value[d] * new_weight);
        }
        maximum = next_maximum;
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] *= inverse;
        }
      }
    }
  });
  return Status::kOk;
}

Status paged_attention(const float* q, const float* key_cache,
                       const float* value_cache, const int* block_table,
                       const int* context_lens, float* out,
                       long long cache_blocks, long long batch,
                       long long query_heads, long long kv_heads,
                       long long head_dim, long long page_size,
                       long long max_blocks, float scale, long long window) {
  if (!detail::valid_product({cache_blocks, page_size, kv_heads, head_dim}) ||
      !detail::valid_product({batch, query_heads, head_dim}) ||
      !detail::valid_product({batch, max_blocks}) ||
      query_heads % kv_heads != 0 || !std::isfinite(scale) || window < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, block_table,
                           context_lens, out)) {
    return Status::kInvalidArgument;
  }
  const float score_scale = default_scale(head_dim, scale);
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 ||
        static_cast<long long>(context) > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    const long long first =
        window > 0 ? std::max(0LL, static_cast<long long>(context) - window)
                   : 0;
    for (long long head = 0; head < query_heads; ++head) {
      const long long kv_head = head / (query_heads / kv_heads);
      const float* query =
          q + (request * query_heads + head) * head_dim;
      float* destination =
          out + (request * query_heads + head) * head_dim;
      std::fill(destination, destination + head_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long position = first; position < context; ++position) {
        const long long table_index = request * max_blocks + position / page_size;
        const int block = block_table[table_index];
        if (block < 0 || block >= cache_blocks) {
          return Status::kInvalidArgument;
        }
        const long long cache_offset =
            ((static_cast<long long>(block) * page_size + position % page_size) *
                 kv_heads +
             kv_head) *
            head_dim;
        const float* key = key_cache + cache_offset;
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          score += static_cast<double>(query[d]) * key[d];
        }
        score *= score_scale;
        const float* value = value_cache + cache_offset;
        if (score > maximum) {
          const double old_weight = std::exp(maximum - score);
          denominator = denominator * old_weight + 1.0;
          for (long long d = 0; d < head_dim; ++d) {
            destination[d] =
                static_cast<float>(destination[d] * old_weight + value[d]);
          }
          maximum = score;
        } else {
          const double weight = std::exp(score - maximum);
          denominator += weight;
          for (long long d = 0; d < head_dim; ++d) {
            destination[d] += static_cast<float>(value[d] * weight);
          }
        }
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] *= inverse;
        }
      }
    }
  }
  return Status::kOk;
}

Status mla_decode(const float* q, const float* kv_cache,
                  const int* block_table, const int* context_lens, float* out,
                  long long cache_blocks, long long batch, long long heads,
                  long long latent_dim, long long rope_dim, long long page_size,
                  long long max_blocks, float scale) {
  const long long width = latent_dim + rope_dim;
  if (latent_dim <= 0 || rope_dim < 0 || width <= 0 ||
      !detail::valid_product({cache_blocks, page_size, width}) ||
      !detail::valid_product({batch, heads, width}) ||
      !detail::valid_product({batch, heads, latent_dim}) ||
      !detail::valid_product({batch, max_blocks}) || !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, kv_cache, block_table, context_lens, out)) {
    return Status::kInvalidArgument;
  }
  const float score_scale = default_scale(width, scale);
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 ||
        static_cast<long long>(context) > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (long long head = 0; head < heads; ++head) {
      const float* query = q + (request * heads + head) * width;
      float* destination = out + (request * heads + head) * latent_dim;
      std::fill(destination, destination + latent_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long position = 0; position < context; ++position) {
        const int block =
            block_table[request * max_blocks + position / page_size];
        if (block < 0 || block >= cache_blocks) {
          return Status::kInvalidArgument;
        }
        const float* item =
            kv_cache +
            (static_cast<long long>(block) * page_size + position % page_size) *
                width;
        double score = 0.0;
        for (long long d = 0; d < width; ++d) {
          score += static_cast<double>(query[d]) * item[d];
        }
        score *= score_scale;
        if (score > maximum) {
          const double old_weight = std::exp(maximum - score);
          denominator = denominator * old_weight + 1.0;
          for (long long d = 0; d < latent_dim; ++d) {
            destination[d] =
                static_cast<float>(destination[d] * old_weight + item[d]);
          }
          maximum = score;
        } else {
          const double weight = std::exp(score - maximum);
          denominator += weight;
          for (long long d = 0; d < latent_dim; ++d) {
            destination[d] += static_cast<float>(item[d] * weight);
          }
        }
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        for (long long d = 0; d < latent_dim; ++d) {
          destination[d] *= inverse;
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
