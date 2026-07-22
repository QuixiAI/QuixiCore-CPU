#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/quantization.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool valid_attention_shape(long long query_heads, long long kv_heads,
                           long long query_length, long long kv_length,
                           long long head_dim) {
  return detail::valid_product({query_heads, query_length, head_dim}) &&
         detail::valid_product({kv_heads, kv_length, head_dim}) &&
         query_heads % kv_heads == 0;
}

double score_scale(long long head_dim, float scale) {
  return scale == 0.0f
             ? 1.0 / std::sqrt(static_cast<double>(head_dim))
             : static_cast<double>(scale);
}

double capped_score(double value, float softcap) {
  return softcap > 0.0f
             ? static_cast<double>(softcap) *
                   std::tanh(value / static_cast<double>(softcap))
             : value;
}

void rotate_interleaved(const float* source, const float* cosine,
                        const float* sine, float* destination,
                        long long width) {
  for (long long pair = 0; pair < width / 2; ++pair) {
    const float first = source[2 * pair];
    const float second = source[2 * pair + 1];
    destination[2 * pair] = first * cosine[pair] - second * sine[pair];
    destination[2 * pair + 1] = second * cosine[pair] + first * sine[pair];
  }
}

}  // namespace

Status attention_with_lse(
    const float* q, const float* k, const float* v, const float* sinks,
    float* out, float* logsumexp, long long query_heads, long long kv_heads,
    long long query_length, long long kv_length, long long head_dim,
    bool causal, float scale, float softcap) {
  if (!valid_attention_shape(query_heads, kv_heads, query_length, kv_length,
                             head_dim) ||
      !std::isfinite(scale) || scale < 0.0f || !std::isfinite(softcap)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out, logsumexp)) {
    return Status::kInvalidArgument;
  }
  const double multiplier = score_scale(head_dim, scale);
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(
      query_heads * query_length, 1,
      [&](long long begin, long long end, int) {
        std::vector<double> scores(static_cast<std::size_t>(kv_length));
        for (long long row = begin; row < end; ++row) {
          const long long head = row / query_length;
          const long long query_position = row % query_length;
          const long long kv_head = head / group;
          const long long causal_limit =
              query_position + kv_length - query_length;
          const float* query = q + row * head_dim;
          float* destination = out + row * head_dim;
          std::fill_n(destination, head_dim, 0.0f);
          double maximum = sinks == nullptr
                               ? -std::numeric_limits<double>::infinity()
                               : static_cast<double>(sinks[head]);
          for (long long position = 0; position < kv_length; ++position) {
            if (causal && position > causal_limit) {
              scores[static_cast<std::size_t>(position)] =
                  -std::numeric_limits<double>::infinity();
              continue;
            }
            const float* key =
                k + (kv_head * kv_length + position) * head_dim;
            double score = 0.0;
            for (long long d = 0; d < head_dim; ++d) {
              score += static_cast<double>(query[d]) * key[d];
            }
            score = capped_score(score * multiplier, softcap);
            scores[static_cast<std::size_t>(position)] = score;
            maximum = std::max(maximum, score);
          }
          if (!std::isfinite(maximum)) {
            logsumexp[row] = static_cast<float>(maximum);
            continue;
          }
          double denominator =
              sinks == nullptr ? 0.0 : std::exp(sinks[head] - maximum);
          for (long long position = 0; position < kv_length; ++position) {
            const double score = scores[static_cast<std::size_t>(position)];
            if (!std::isfinite(score)) continue;
            const double weight = std::exp(score - maximum);
            denominator += weight;
            const float* value =
                v + (kv_head * kv_length + position) * head_dim;
            for (long long d = 0; d < head_dim; ++d) {
              destination[d] += static_cast<float>(weight * value[d]);
            }
          }
          const float inverse = static_cast<float>(1.0 / denominator);
          for (long long d = 0; d < head_dim; ++d) destination[d] *= inverse;
          logsumexp[row] =
              static_cast<float>(maximum + std::log(denominator));
        }
      });
  return Status::kOk;
}

Status attention_backward_prep(const float* out, const float* grad_out,
                               float* delta, long long query_heads,
                               long long query_length, long long head_dim) {
  if (!detail::valid_product({query_heads, query_length, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(out, grad_out, delta)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(
      query_heads * query_length, 8,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          double sum = 0.0;
          for (long long d = 0; d < head_dim; ++d) {
            sum += static_cast<double>(out[row * head_dim + d]) *
                   grad_out[row * head_dim + d];
          }
          delta[row] = static_cast<float>(sum);
        }
      });
  return Status::kOk;
}

Status attention_backward_dq(
    const float* q, const float* k, const float* v, const float* grad_out,
    const float* logsumexp, const float* delta, float* grad_q,
    long long query_heads, long long kv_heads, long long query_length,
    long long kv_length, long long head_dim, bool causal) {
  if (!valid_attention_shape(query_heads, kv_heads, query_length, kv_length,
                             head_dim)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, grad_out, logsumexp, delta, grad_q)) {
    return Status::kInvalidArgument;
  }
  const double multiplier = score_scale(head_dim, 0.0f);
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(
      query_heads * query_length, 1,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long head = row / query_length;
          const long long query_position = row % query_length;
          const long long kv_head = head / group;
          const long long causal_limit =
              query_position + kv_length - query_length;
          const float* query = q + row * head_dim;
          const float* go = grad_out + row * head_dim;
          float* gq = grad_q + row * head_dim;
          std::fill_n(gq, head_dim, 0.0f);
          for (long long position = 0; position < kv_length; ++position) {
            if (causal && position > causal_limit) continue;
            const float* key =
                k + (kv_head * kv_length + position) * head_dim;
            const float* value =
                v + (kv_head * kv_length + position) * head_dim;
            double raw_score = 0.0;
            double grad_probability = 0.0;
            for (long long d = 0; d < head_dim; ++d) {
              raw_score += static_cast<double>(query[d]) * key[d];
              grad_probability += static_cast<double>(go[d]) * value[d];
            }
            const double probability =
                std::exp(raw_score * multiplier - logsumexp[row]);
            const double grad_score =
                probability * (grad_probability - delta[row]) * multiplier;
            for (long long d = 0; d < head_dim; ++d) {
              gq[d] += static_cast<float>(grad_score * key[d]);
            }
          }
        }
      });
  return Status::kOk;
}

Status attention_backward_dkv(
    const float* q, const float* k, const float* v, const float* grad_out,
    const float* logsumexp, const float* delta, float* grad_k, float* grad_v,
    long long query_heads, long long kv_heads, long long query_length,
    long long kv_length, long long head_dim, bool causal) {
  if (!valid_attention_shape(query_heads, kv_heads, query_length, kv_length,
                             head_dim)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, grad_out, logsumexp, delta, grad_k,
                           grad_v)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_k, kv_heads * kv_length * head_dim, 0.0f);
  std::fill_n(grad_v, kv_heads * kv_length * head_dim, 0.0f);
  const double multiplier = score_scale(head_dim, 0.0f);
  const long long group = query_heads / kv_heads;
  for (long long head = 0; head < query_heads; ++head) {
    const long long kv_head = head / group;
    for (long long query_position = 0; query_position < query_length;
         ++query_position) {
      const long long row = head * query_length + query_position;
      const long long causal_limit =
          query_position + kv_length - query_length;
      const float* query = q + row * head_dim;
      const float* go = grad_out + row * head_dim;
      for (long long position = 0; position < kv_length; ++position) {
        if (causal && position > causal_limit) continue;
        const long long offset =
            (kv_head * kv_length + position) * head_dim;
        const float* key = k + offset;
        const float* value = v + offset;
        double raw_score = 0.0;
        double grad_probability = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          raw_score += static_cast<double>(query[d]) * key[d];
          grad_probability += static_cast<double>(go[d]) * value[d];
        }
        const double probability =
            std::exp(raw_score * multiplier - logsumexp[row]);
        const double grad_score =
            probability * (grad_probability - delta[row]) * multiplier;
        for (long long d = 0; d < head_dim; ++d) {
          grad_k[offset + d] += static_cast<float>(grad_score * query[d]);
          grad_v[offset + d] += static_cast<float>(probability * go[d]);
        }
      }
    }
  }
  return Status::kOk;
}

Status merge_attention_states(
    const float* prefix_out, const float* prefix_logsumexp,
    const float* suffix_out, const float* suffix_logsumexp, float* out,
    float* output_logsumexp, long long tokens, long long heads,
    long long head_dim, long long prefix_tokens) {
  if (!detail::valid_product({tokens, heads, head_dim}) || prefix_tokens < 0 ||
      prefix_tokens > tokens) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(prefix_out, prefix_logsumexp, suffix_out,
                           suffix_logsumexp, out, output_logsumexp)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(tokens * heads, 8,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const long long token = row / heads;
      if (token >= prefix_tokens) {
        std::copy_n(suffix_out + row * head_dim, head_dim,
                    out + row * head_dim);
        output_logsumexp[row] = suffix_logsumexp[row];
        continue;
      }
      const double prefix_lse = prefix_logsumexp[row];
      const double suffix_lse = suffix_logsumexp[row];
      const double maximum = std::max(prefix_lse, suffix_lse);
      if (std::isinf(maximum)) {
        std::copy_n(prefix_out + row * head_dim, head_dim,
                    out + row * head_dim);
        output_logsumexp[row] = static_cast<float>(maximum);
        continue;
      }
      const double prefix_weight = std::exp(prefix_lse - maximum);
      const double suffix_weight = std::exp(suffix_lse - maximum);
      const double denominator = prefix_weight + suffix_weight;
      for (long long d = 0; d < head_dim; ++d) {
        out[row * head_dim + d] = static_cast<float>(
            (prefix_out[row * head_dim + d] * prefix_weight +
             suffix_out[row * head_dim + d] * suffix_weight) /
            denominator);
      }
      output_logsumexp[row] =
          static_cast<float>(maximum + std::log(denominator));
    }
  });
  return Status::kOk;
}

Status merge_attention_states_fp8(
    const float* prefix_out, const float* prefix_logsumexp,
    const float* suffix_out, const float* suffix_logsumexp,
    std::uint8_t* out_codes, float* output_logsumexp, long long tokens,
    long long heads, long long head_dim, long long prefix_tokens,
    float output_scale) {
  if (!detail::valid_product({tokens, heads, head_dim}) ||
      !std::isfinite(output_scale) || output_scale <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(prefix_out, prefix_logsumexp, suffix_out,
                           suffix_logsumexp, out_codes, output_logsumexp)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> merged(
      static_cast<std::size_t>(tokens * heads * head_dim));
  const Status status = merge_attention_states(
      prefix_out, prefix_logsumexp, suffix_out, suffix_logsumexp,
      merged.data(), output_logsumexp, tokens, heads, head_dim,
      prefix_tokens);
  if (status != Status::kOk) return status;
  const float inverse = 1.0f / output_scale;
  threading::parallel_ranges(
      tokens * heads * head_dim, 4096,
      [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          out_codes[item] = float8_encode(
              merged[static_cast<std::size_t>(item)] * inverse,
              Float8Format::kE4M3FN);
        }
      });
  return Status::kOk;
}

Status decode_cache_attention(
    const float* q, const float* new_k, const float* new_v,
    const float* cosine, const float* sine, const int* positions,
    const int* context_lengths, const float* q_weight,
    const float* k_weight, float* key_cache, float* value_cache, float* out,
    long long batch, long long query_heads, long long kv_heads,
    long long cache_length, long long head_dim, long long max_position,
    float eps, bool do_q_norm, bool do_k_norm, bool gemma_weight,
    float scale) {
  if (!detail::valid_product({batch, query_heads, head_dim}) ||
      !detail::valid_product({batch, kv_heads, cache_length, head_dim}) ||
      head_dim % 2 != 0 || query_heads % kv_heads != 0 ||
      max_position <= 0 || !std::isfinite(eps) || eps <= 0.0f ||
      !std::isfinite(scale) || scale < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, new_k, new_v, cosine, sine, positions,
                           context_lengths, key_cache, value_cache, out) ||
      (do_q_norm && q_weight == nullptr) ||
      (do_k_norm && k_weight == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (long long item = 0; item < batch; ++item) {
    if (positions[item] < 0 || positions[item] >= max_position ||
        context_lengths[item] < 0 ||
        context_lengths[item] >= cache_length) {
      return Status::kInvalidArgument;
    }
  }
  const long long half = head_dim / 2;
  // K/V append is independent of query-head scheduling.
  for (long long item = 0; item < batch; ++item) {
    const float* cos_row =
        cosine + static_cast<long long>(positions[item]) * half;
    const float* sin_row =
        sine + static_cast<long long>(positions[item]) * half;
    for (long long head = 0; head < kv_heads; ++head) {
      const long long source_offset = (item * kv_heads + head) * head_dim;
      const float* source = new_k + source_offset;
      double inverse = 1.0;
      if (do_k_norm) {
        double squares = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          squares += static_cast<double>(source[d]) * source[d];
        }
        inverse = 1.0 / std::sqrt(squares / head_dim + eps);
      }
      const long long cache_offset =
          ((item * kv_heads + head) * cache_length +
           context_lengths[item]) *
          head_dim;
      for (long long d = 0; d < half; ++d) {
        const double first_weight =
            do_k_norm ? k_weight[d] + (gemma_weight ? 1.0f : 0.0f) : 1.0;
        const double second_weight =
            do_k_norm
                ? k_weight[half + d] + (gemma_weight ? 1.0f : 0.0f)
                : 1.0;
        const double first = source[d] * inverse * first_weight;
        const double second = source[half + d] * inverse * second_weight;
        key_cache[cache_offset + d] =
            static_cast<float>(first * cos_row[d] - second * sin_row[d]);
        key_cache[cache_offset + half + d] =
            static_cast<float>(second * cos_row[d] + first * sin_row[d]);
      }
      std::copy_n(new_v + source_offset, head_dim,
                  value_cache + cache_offset);
    }
  }
  const double multiplier = score_scale(head_dim, scale);
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(batch * query_heads, 1,
                             [&](long long begin, long long end, int) {
    std::vector<float> rotated(static_cast<std::size_t>(head_dim));
    for (long long row = begin; row < end; ++row) {
      const long long item = row / query_heads;
      const long long head = row % query_heads;
      const long long kv_head = head / group;
      const float* source = q + row * head_dim;
      double inverse = 1.0;
      if (do_q_norm) {
        double squares = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          squares += static_cast<double>(source[d]) * source[d];
        }
        inverse = 1.0 / std::sqrt(squares / head_dim + eps);
      }
      const float* cos_row =
          cosine + static_cast<long long>(positions[item]) * half;
      const float* sin_row =
          sine + static_cast<long long>(positions[item]) * half;
      for (long long d = 0; d < half; ++d) {
        const double first_weight =
            do_q_norm ? q_weight[d] + (gemma_weight ? 1.0f : 0.0f) : 1.0;
        const double second_weight =
            do_q_norm
                ? q_weight[half + d] + (gemma_weight ? 1.0f : 0.0f)
                : 1.0;
        const double first = source[d] * inverse * first_weight;
        const double second = source[half + d] * inverse * second_weight;
        rotated[static_cast<std::size_t>(d)] =
            static_cast<float>(first * cos_row[d] - second * sin_row[d]);
        rotated[static_cast<std::size_t>(half + d)] =
            static_cast<float>(second * cos_row[d] + first * sin_row[d]);
      }
      float* destination = out + row * head_dim;
      std::fill_n(destination, head_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long token = 0; token <= context_lengths[item]; ++token) {
        const long long cache_offset =
            ((item * kv_heads + kv_head) * cache_length + token) * head_dim;
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          score += rotated[static_cast<std::size_t>(d)] *
                   key_cache[cache_offset + d];
        }
        score *= multiplier;
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] = static_cast<float>(
              destination[d] * old_weight +
              value_cache[cache_offset + d] * new_weight);
        }
        maximum = next_maximum;
      }
      const float inv_denominator = static_cast<float>(1.0 / denominator);
      for (long long d = 0; d < head_dim; ++d) {
        destination[d] *= inv_denominator;
      }
    }
  });
  return Status::kOk;
}

Status cascade_attention_multi(
    const float* q, const float* const* prefix_k,
    const float* const* prefix_v, const long long* prefix_lengths,
    long long levels, const float* key_cache, const float* value_cache,
    const int* block_table, const int* context_lens, float* out,
    long long cache_blocks, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, float scale) {
  if (levels < 0 ||
      !detail::valid_product({cache_blocks, page_size, kv_heads, head_dim}) ||
      !detail::valid_product({batch, query_heads, head_dim}) ||
      !detail::valid_product({batch, max_blocks}) ||
      query_heads % kv_heads != 0 || !std::isfinite(scale) || scale < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, block_table,
                           context_lens, out) ||
      (levels > 0 && !detail::all_nonnull(prefix_k, prefix_v,
                                          prefix_lengths))) {
    return Status::kInvalidArgument;
  }
  for (long long level = 0; level < levels; ++level) {
    if (prefix_lengths[level] < 0 ||
        (prefix_lengths[level] > 0 &&
         !detail::all_nonnull(prefix_k[level], prefix_v[level]))) {
      return Status::kInvalidArgument;
    }
  }
  const double multiplier = score_scale(head_dim, scale);
  const long long group = query_heads / kv_heads;
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 ||
        static_cast<long long>(context) > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (long long head = 0; head < query_heads; ++head) {
      const long long kv_head = head / group;
      const float* query = q + (request * query_heads + head) * head_dim;
      float* destination = out + (request * query_heads + head) * head_dim;
      std::fill_n(destination, head_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      auto consume = [&](const float* key, const float* value) {
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) score += query[d] * key[d];
        score *= multiplier;
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] = static_cast<float>(destination[d] * old_weight +
                                              value[d] * new_weight);
        }
        maximum = next_maximum;
      };
      for (long long level = 0; level < levels; ++level) {
        for (long long position = 0; position < prefix_lengths[level];
             ++position) {
          const long long offset =
              (position * kv_heads + kv_head) * head_dim;
          consume(prefix_k[level] + offset, prefix_v[level] + offset);
        }
      }
      for (long long position = 0; position < context; ++position) {
        const int block =
            block_table[request * max_blocks + position / page_size];
        if (block < 0 || block >= cache_blocks) {
          return Status::kInvalidArgument;
        }
        const long long offset =
            ((static_cast<long long>(block) * page_size +
              position % page_size) *
                 kv_heads +
             kv_head) *
            head_dim;
        consume(key_cache + offset, value_cache + offset);
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        for (long long d = 0; d < head_dim; ++d) destination[d] *= inverse;
      }
    }
  }
  return Status::kOk;
}

Status mla_q_norm_rope(
    const float* q, const float* cosine, const float* sine,
    const int* positions, const float* norm_weight, float* out,
    long long tokens, long long heads, long long nope_dim,
    long long rope_dim, long long max_position, int norm_mode, float eps) {
  const long long head_dim = nope_dim + rope_dim;
  if (!detail::valid_product({tokens, heads, head_dim}) || nope_dim < 0 ||
      rope_dim <= 0 || rope_dim % 2 != 0 || max_position <= 0 ||
      norm_mode < 0 || norm_mode > 2 || !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, cosine, sine, positions, out) ||
      (norm_mode == 2 && norm_weight == nullptr)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> normalized(static_cast<std::size_t>(head_dim));
  for (long long token = 0; token < tokens; ++token) {
    if (positions[token] < 0 || positions[token] >= max_position) {
      return Status::kInvalidArgument;
    }
    const float* cos_row = cosine +
        static_cast<long long>(positions[token]) * (rope_dim / 2);
    const float* sin_row = sine +
        static_cast<long long>(positions[token]) * (rope_dim / 2);
    for (long long head = 0; head < heads; ++head) {
      const long long offset = (token * heads + head) * head_dim;
      const float* source = q + offset;
      double inverse = 1.0;
      if (norm_mode != 0) {
        double squares = 0.0;
        for (long long d = 0; d < head_dim; ++d) {
          squares += static_cast<double>(source[d]) * source[d];
        }
        inverse = 1.0 / std::sqrt(squares / head_dim + eps);
      }
      for (long long d = 0; d < head_dim; ++d) {
        const double weight = norm_mode == 2 ? norm_weight[d] : 1.0;
        normalized[static_cast<std::size_t>(d)] =
            static_cast<float>(source[d] * inverse * weight);
      }
      std::copy_n(normalized.data(), nope_dim, out + offset);
      rotate_interleaved(normalized.data() + nope_dim, cos_row, sin_row,
                         out + offset + nope_dim, rope_dim);
    }
  }
  return Status::kOk;
}

Status mla_kv_insert(
    const float* latent, const float* key_rope, const float* cosine,
    const float* sine, const int* positions, const int* slot_mapping,
    const float* norm_weight, float* cache, long long tokens,
    long long slots, long long latent_dim, long long rope_dim,
    long long max_position, int norm_mode, float eps) {
  const long long width = latent_dim + rope_dim;
  if (!detail::valid_product({tokens, width}) || slots <= 0 ||
      latent_dim <= 0 || rope_dim <= 0 || rope_dim % 2 != 0 ||
      max_position <= 0 || norm_mode < 0 || norm_mode > 2 ||
      !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(latent, key_rope, cosine, sine, positions,
                           slot_mapping, cache) ||
      (norm_mode == 2 && norm_weight == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (positions[token] < 0 || positions[token] >= max_position ||
        slot_mapping[token] >= slots) {
      return Status::kInvalidArgument;
    }
  }
  for (long long token = 0; token < tokens; ++token) {
    if (slot_mapping[token] < 0) continue;
    float* destination =
        cache + static_cast<long long>(slot_mapping[token]) * width;
    const float* source = latent + token * latent_dim;
    double inverse = 1.0;
    if (norm_mode != 0) {
      double squares = 0.0;
      for (long long d = 0; d < latent_dim; ++d) {
        squares += static_cast<double>(source[d]) * source[d];
      }
      inverse = 1.0 / std::sqrt(squares / latent_dim + eps);
    }
    for (long long d = 0; d < latent_dim; ++d) {
      const double weight = norm_mode == 2 ? norm_weight[d] : 1.0;
      destination[d] = static_cast<float>(source[d] * inverse * weight);
    }
    const float* cos_row = cosine +
        static_cast<long long>(positions[token]) * (rope_dim / 2);
    const float* sin_row = sine +
        static_cast<long long>(positions[token]) * (rope_dim / 2);
    rotate_interleaved(key_rope + token * rope_dim, cos_row, sin_row,
                       destination + latent_dim, rope_dim);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
