#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool attention_shape(long long query_heads, long long kv_heads,
                     long long query_length, long long kv_length,
                     long long head_dim) {
  return detail::valid_product({query_heads, query_length, head_dim}) &&
         detail::valid_product({kv_heads, kv_length, head_dim}) &&
         query_heads % kv_heads == 0;
}

void rotate_table_row(const float* x, const float* cosine, const float* sine,
                      float* y, long long head_dim, bool interleaved) {
  const long long half = head_dim / 2;
  for (long long i = 0; i < half; ++i) {
    const long long first_index = interleaved ? 2 * i : i;
    const long long second_index = interleaved ? 2 * i + 1 : half + i;
    const float first = x[first_index];
    const float second = x[second_index];
    y[first_index] = first * cosine[i] - second * sine[i];
    y[second_index] = second * cosine[i] + first * sine[i];
  }
}

}  // namespace

Status attention_backward(const float* q, const float* k, const float* v,
                          const float* grad_out, float* grad_q, float* grad_k,
                          float* grad_v, long long query_heads,
                          long long kv_heads, long long query_length,
                          long long kv_length, long long head_dim,
                          bool causal) {
  if (!attention_shape(query_heads, kv_heads, query_length, kv_length,
                       head_dim)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, grad_out, grad_q, grad_k, grad_v)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_q, query_heads * query_length * head_dim, 0.0f);
  std::fill_n(grad_k, kv_heads * kv_length * head_dim, 0.0f);
  std::fill_n(grad_v, kv_heads * kv_length * head_dim, 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const long long group = query_heads / kv_heads;
  std::vector<double> probabilities(static_cast<std::size_t>(kv_length));
  std::vector<double> grad_probability(static_cast<std::size_t>(kv_length));
  for (long long head = 0; head < query_heads; ++head) {
    const long long kv_head = head / group;
    for (long long query_position = 0; query_position < query_length;
         ++query_position) {
      const float* query =
          q + (head * query_length + query_position) * head_dim;
      const float* go =
          grad_out + (head * query_length + query_position) * head_dim;
      float* gq = grad_q + (head * query_length + query_position) * head_dim;
      const long long causal_limit =
          query_position + kv_length - query_length;
      double maximum = -std::numeric_limits<double>::infinity();
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) {
          probabilities[key_position] = 0.0;
          continue;
        }
        const float* key =
            k + (kv_head * kv_length + key_position) * head_dim;
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) score += query[d] * key[d];
        probabilities[key_position] = score * scale;
        maximum = std::max(maximum, probabilities[key_position]);
      }
      double denominator = 0.0;
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) continue;
        probabilities[key_position] =
            std::exp(probabilities[key_position] - maximum);
        denominator += probabilities[key_position];
      }
      double correction = 0.0;
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) {
          grad_probability[key_position] = 0.0;
          continue;
        }
        probabilities[key_position] /= denominator;
        const float* value =
            v + (kv_head * kv_length + key_position) * head_dim;
        double dp = 0.0;
        for (long long d = 0; d < head_dim; ++d) dp += go[d] * value[d];
        grad_probability[key_position] = dp;
        correction += probabilities[key_position] * dp;
      }
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) continue;
        const double ds = probabilities[key_position] *
                          (grad_probability[key_position] - correction) * scale;
        const float* key =
            k + (kv_head * kv_length + key_position) * head_dim;
        float* gk =
            grad_k + (kv_head * kv_length + key_position) * head_dim;
        float* gv =
            grad_v + (kv_head * kv_length + key_position) * head_dim;
        for (long long d = 0; d < head_dim; ++d) {
          gq[d] += static_cast<float>(ds * key[d]);
          gk[d] += static_cast<float>(ds * query[d]);
          gv[d] += static_cast<float>(probabilities[key_position] * go[d]);
        }
      }
    }
  }
  return Status::kOk;
}

Status window_attention(const float* q, const float* k, const float* v,
                        float* out, long long query_heads,
                        long long kv_heads, long long query_length,
                        long long kv_length, long long head_dim,
                        long long left_window, long long right_window) {
  if (!attention_shape(query_heads, kv_heads, query_length, kv_length,
                       head_dim) ||
      left_window < -1 || right_window < -1) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) return Status::kInvalidArgument;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(query_heads * query_length, 1,
                             [&](long long begin, long long end, int) {
    for (long long index = begin; index < end; ++index) {
      const long long head = index / query_length;
      const long long query_position = index % query_length;
      const long long aligned = query_position + kv_length - query_length;
      const long long first = left_window < 0
                                  ? 0
                                  : std::max(0LL, aligned - left_window);
      const long long last = right_window < 0
                                 ? kv_length - 1
                                 : std::min(kv_length - 1,
                                            aligned + right_window);
      float* destination = out + index * head_dim;
      std::fill_n(destination, head_dim, 0.0f);
      if (first > last) continue;
      const float* query = q + index * head_dim;
      const long long kv_head = head / group;
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long position = first; position <= last; ++position) {
        const float* key =
            k + (kv_head * kv_length + position) * head_dim;
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) score += query[d] * key[d];
        score *= scale;
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        const float* value =
            v + (kv_head * kv_length + position) * head_dim;
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] = static_cast<float>(destination[d] * old_weight +
                                              value[d] * new_weight);
        }
        maximum = next_maximum;
      }
      for (long long d = 0; d < head_dim; ++d) {
        destination[d] = static_cast<float>(destination[d] / denominator);
      }
    }
  });
  return Status::kOk;
}

Status varlen_attention(const float* q, const float* k, const float* v,
                        const int* cumulative_q, const int* cumulative_k,
                        float* out, long long batch, long long query_heads,
                        long long kv_heads, long long head_dim, bool causal) {
  if (!detail::valid_product({batch, query_heads, head_dim}) || kv_heads <= 0 ||
      query_heads % kv_heads != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, cumulative_q, cumulative_k, out)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_q[0] != 0 || cumulative_k[0] != 0) {
    return Status::kInvalidArgument;
  }
  for (long long item = 0; item < batch; ++item) {
    if (cumulative_q[item + 1] < cumulative_q[item] ||
        cumulative_k[item + 1] < cumulative_k[item]) {
      return Status::kInvalidArgument;
    }
  }
  const long long group = query_heads / kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (long long item = 0; item < batch; ++item) {
    const long long q_begin = cumulative_q[item];
    const long long q_length = cumulative_q[item + 1] - q_begin;
    const long long k_begin = cumulative_k[item];
    const long long k_length = cumulative_k[item + 1] - k_begin;
    for (long long qp = 0; qp < q_length; ++qp) {
      for (long long head = 0; head < query_heads; ++head) {
        const float* query =
            q + ((q_begin + qp) * query_heads + head) * head_dim;
        float* destination =
            out + ((q_begin + qp) * query_heads + head) * head_dim;
        std::fill_n(destination, head_dim, 0.0f);
        const long long kv_head = head / group;
        const long long limit = qp + k_length - q_length;
        double maximum = -std::numeric_limits<double>::infinity();
        double denominator = 0.0;
        for (long long kp = 0; kp < k_length; ++kp) {
          if (causal && kp > limit) continue;
          const float* key =
              k + ((k_begin + kp) * kv_heads + kv_head) * head_dim;
          double score = 0.0;
          for (long long d = 0; d < head_dim; ++d) score += query[d] * key[d];
          score *= scale;
          const double next_maximum = std::max(maximum, score);
          const double old_weight = std::exp(maximum - next_maximum);
          const double new_weight = std::exp(score - next_maximum);
          denominator = denominator * old_weight + new_weight;
          const float* value =
              v + ((k_begin + kp) * kv_heads + kv_head) * head_dim;
          for (long long d = 0; d < head_dim; ++d) {
            destination[d] = static_cast<float>(destination[d] * old_weight +
                                                value[d] * new_weight);
          }
          maximum = next_maximum;
        }
        if (denominator > 0.0) {
          for (long long d = 0; d < head_dim; ++d) {
            destination[d] = static_cast<float>(destination[d] / denominator);
          }
        }
      }
    }
  }
  return Status::kOk;
}

Status biased_attention(const float* q, const float* k, const float* v,
                        const float* bias, const std::uint8_t* mask,
                        float* out, long long query_heads,
                        long long kv_heads, long long query_length,
                        long long kv_length, long long head_dim, float scale) {
  if (!attention_shape(query_heads, kv_heads, query_length, kv_length,
                       head_dim) ||
      !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) return Status::kInvalidArgument;
  const double score_scale =
      scale == 0.0f ? 1.0 / std::sqrt(static_cast<double>(head_dim)) : scale;
  const long long group = query_heads / kv_heads;
  for (long long head = 0; head < query_heads; ++head) {
    const long long kv_head = head / group;
    for (long long qp = 0; qp < query_length; ++qp) {
      const float* query = q + (head * query_length + qp) * head_dim;
      float* destination = out + (head * query_length + qp) * head_dim;
      std::fill_n(destination, head_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long kp = 0; kp < kv_length; ++kp) {
        if (mask != nullptr && mask[qp * kv_length + kp] == 0) continue;
        const float* key = k + (kv_head * kv_length + kp) * head_dim;
        double score = 0.0;
        for (long long d = 0; d < head_dim; ++d) score += query[d] * key[d];
        score *= score_scale;
        if (bias != nullptr) {
          score += bias[(head * query_length + qp) * kv_length + kp];
        }
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        const float* value = v + (kv_head * kv_length + kp) * head_dim;
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] = static_cast<float>(destination[d] * old_weight +
                                              value[d] * new_weight);
        }
        maximum = next_maximum;
      }
      if (denominator > 0.0) {
        for (long long d = 0; d < head_dim; ++d) {
          destination[d] = static_cast<float>(destination[d] / denominator);
        }
      }
    }
  }
  return Status::kOk;
}

Status rope_table(const float* x, const float* cosine, const float* sine,
                  const int* positions, float* y, long long tokens,
                  long long heads, long long head_dim,
                  long long max_position, bool interleaved) {
  if (!detail::valid_product({tokens, heads, head_dim}) || head_dim % 2 != 0 ||
      max_position <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, cosine, sine, positions, y)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (positions[token] < 0 || positions[token] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  const long long half = head_dim / 2;
  for (long long token = 0; token < tokens; ++token) {
    const float* cos_row = cosine + static_cast<long long>(positions[token]) * half;
    const float* sin_row = sine + static_cast<long long>(positions[token]) * half;
    for (long long head = 0; head < heads; ++head) {
      const long long offset = (token * heads + head) * head_dim;
      rotate_table_row(x + offset, cos_row, sin_row, y + offset, head_dim,
                       interleaved);
    }
  }
  return Status::kOk;
}

Status qk_norm_rope(const float* qkv, const float* q_weight,
                    const float* k_weight, const float* cosine,
                    const float* sine, const int* positions, float* y,
                    long long tokens, long long query_heads,
                    long long key_heads, long long value_heads,
                    long long head_dim, long long max_position, float eps,
                    bool interleaved) {
  const long long total_heads = query_heads + key_heads + value_heads;
  if (!detail::valid_product({tokens, total_heads, head_dim}) ||
      head_dim % 2 != 0 || !std::isfinite(eps) || eps < 0.0f ||
      max_position <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(qkv, q_weight, k_weight, cosine, sine, positions,
                           y)) {
    return Status::kInvalidArgument;
  }
  const long long half = head_dim / 2;
  std::vector<float> normalized(static_cast<std::size_t>(head_dim));
  for (long long token = 0; token < tokens; ++token) {
    if (positions[token] < 0 || positions[token] >= max_position) {
      return Status::kInvalidArgument;
    }
    const float* cos_row = cosine + static_cast<long long>(positions[token]) * half;
    const float* sin_row = sine + static_cast<long long>(positions[token]) * half;
    for (long long head = 0; head < query_heads + key_heads; ++head) {
      const long long offset = (token * total_heads + head) * head_dim;
      const float* source = qkv + offset;
      const float* weight = head < query_heads ? q_weight : k_weight;
      double squares = 0.0;
      for (long long d = 0; d < head_dim; ++d) {
        squares += double(source[d]) * source[d];
      }
      const double inverse = 1.0 / std::sqrt(squares / head_dim + eps);
      for (long long d = 0; d < head_dim; ++d) {
        normalized[d] = static_cast<float>(source[d] * inverse * weight[d]);
      }
      rotate_table_row(normalized.data(), cos_row, sin_row, y + offset,
                       head_dim, interleaved);
    }
    const long long value_offset =
        (token * total_heads + query_heads + key_heads) * head_dim;
    std::copy_n(qkv + value_offset, value_heads * head_dim, y + value_offset);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
