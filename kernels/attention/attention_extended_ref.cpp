#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

#if defined(__clang__) || defined(__GNUC__)
#define QUIXICORE_CPU_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define QUIXICORE_CPU_ALWAYS_INLINE inline
#endif

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

long long resolve_rotary_dim(long long head_dim, long long rotary_dim) {
  return rotary_dim == 0 ? head_dim : rotary_dim;
}

bool valid_rotary_shape(long long head_dim, long long rotary_dim,
                        long long max_position) {
  return head_dim > 0 && rotary_dim > 0 && rotary_dim <= head_dim &&
         rotary_dim % 2 == 0 && max_position > 0;
}

bool valid_mrope_sections(const int* sections, long long rotary_pairs,
                          bool interleaved) {
  if (sections == nullptr || sections[0] < 0 || sections[1] < 0 ||
      sections[2] < 0 ||
      static_cast<long long>(sections[0]) + sections[1] + sections[2] !=
          rotary_pairs) {
    return false;
  }
  return !interleaved || (sections[0] == (rotary_pairs + 2) / 3 &&
                          sections[1] == (rotary_pairs + 1) / 3 &&
                          sections[2] == rotary_pairs / 3);
}

QUIXICORE_CPU_ALWAYS_INLINE float inverse_rms(const float* x,
                                              long long head_dim, float eps) {
  float squares = 0.0f;
  for (long long d = 0; d < head_dim; ++d) {
    squares += x[d] * x[d];
  }
  return 1.0f / std::sqrt(squares / static_cast<float>(head_dim) + eps);
}

QUIXICORE_CPU_ALWAYS_INLINE void norm_rotate_positioned_row(
    const float* x, const float* weight, const float* cosine, const float* sine,
    float* y, long long head_dim, long long rotary_dim, bool interleaved,
    float inverse, float weight_offset) {
  const long long pairs = rotary_dim / 2;
  if (interleaved) {
    for (long long pair = 0; pair < pairs; ++pair) {
      const float first =
          x[2 * pair] * inverse * (weight[2 * pair] + weight_offset);
      const float second =
          x[2 * pair + 1] * inverse * (weight[2 * pair + 1] + weight_offset);
      y[2 * pair] = first * cosine[pair] - second * sine[pair];
      y[2 * pair + 1] = second * cosine[pair] + first * sine[pair];
    }
  } else {
    for (long long pair = 0; pair < pairs; ++pair) {
      const float first = x[pair] * inverse * (weight[pair] + weight_offset);
      const float second =
          x[pairs + pair] * inverse * (weight[pairs + pair] + weight_offset);
      y[pair] = first * cosine[pair] - second * sine[pair];
      y[pairs + pair] = second * cosine[pair] + first * sine[pair];
    }
  }
  for (long long d = rotary_dim; d < head_dim; ++d) {
    y[d] = x[d] * inverse * (weight[d] + weight_offset);
  }
}

QUIXICORE_CPU_ALWAYS_INLINE void norm_rotate_mrope_row(
    const float* x, const float* weight, const float* cosine, const float* sine,
    const int* positions, const int* sections, float* y, long long head_dim,
    long long rotary_dim, bool section_interleaved, float inverse,
    float weight_offset) {
  const long long pairs = rotary_dim / 2;
  if (section_interleaved) {
    for (long long first_pair = 0; first_pair < pairs; first_pair += 3) {
      const long long group = std::min(3LL, pairs - first_pair);
      for (long long axis = 0; axis < group; ++axis) {
        const long long pair = first_pair + axis;
        const long long table =
            static_cast<long long>(positions[axis]) * pairs + pair;
        const float first = x[pair] * inverse * (weight[pair] + weight_offset);
        const float second =
            x[pairs + pair] * inverse * (weight[pairs + pair] + weight_offset);
        y[pair] = first * cosine[table] - second * sine[table];
        y[pairs + pair] = second * cosine[table] + first * sine[table];
      }
    }
  } else {
    long long begin = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const long long end = begin + sections[axis];
      const long long table_base =
          static_cast<long long>(positions[axis]) * pairs;
      for (long long pair = begin; pair < end; ++pair) {
        const long long table = table_base + pair;
        const float first = x[pair] * inverse * (weight[pair] + weight_offset);
        const float second =
            x[pairs + pair] * inverse * (weight[pairs + pair] + weight_offset);
        y[pair] = first * cosine[table] - second * sine[table];
        y[pairs + pair] = second * cosine[table] + first * sine[table];
      }
      begin = end;
    }
  }
  for (long long d = rotary_dim; d < head_dim; ++d) {
    y[d] = x[d] * inverse * (weight[d] + weight_offset);
  }
}

#undef QUIXICORE_CPU_ALWAYS_INLINE

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
      const long long causal_limit = query_position + kv_length - query_length;
      double maximum = -std::numeric_limits<double>::infinity();
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) {
          probabilities[key_position] = 0.0;
          continue;
        }
        const float* key = k + (kv_head * kv_length + key_position) * head_dim;
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
        const float* key = k + (kv_head * kv_length + key_position) * head_dim;
        float* gk = grad_k + (kv_head * kv_length + key_position) * head_dim;
        float* gv = grad_v + (kv_head * kv_length + key_position) * head_dim;
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
                        float* out, long long query_heads, long long kv_heads,
                        long long query_length, long long kv_length,
                        long long head_dim, long long left_window,
                        long long right_window) {
  if (!attention_shape(query_heads, kv_heads, query_length, kv_length,
                       head_dim) ||
      left_window < -1 || right_window < -1) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) return Status::kInvalidArgument;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const long long group = query_heads / kv_heads;
  threading::parallel_ranges(
      query_heads * query_length, 1, [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long head = index / query_length;
          const long long query_position = index % query_length;
          const long long aligned = query_position + kv_length - query_length;
          const long long first =
              left_window < 0 ? 0 : std::max(0LL, aligned - left_window);
          const long long last =
              right_window < 0
                  ? kv_length - 1
                  : std::min(kv_length - 1, aligned + right_window);
          float* destination = out + index * head_dim;
          std::fill_n(destination, head_dim, 0.0f);
          if (first > last) continue;
          const float* query = q + index * head_dim;
          const long long kv_head = head / group;
          double maximum = -std::numeric_limits<double>::infinity();
          double denominator = 0.0;
          for (long long position = first; position <= last; ++position) {
            const float* key = k + (kv_head * kv_length + position) * head_dim;
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
                        const float* bias, const std::uint8_t* mask, float* out,
                        long long query_heads, long long kv_heads,
                        long long query_length, long long kv_length,
                        long long head_dim, float scale) {
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
                  long long heads, long long head_dim, long long max_position,
                  bool interleaved) {
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
    const float* cos_row =
        cosine + static_cast<long long>(positions[token]) * half;
    const float* sin_row =
        sine + static_cast<long long>(positions[token]) * half;
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
  return qk_norm_rope_positioned(qkv, q_weight, k_weight, cosine, sine,
                                 positions, y, tokens, query_heads, key_heads,
                                 value_heads, head_dim, head_dim, max_position,
                                 eps, interleaved, 0.0f, nullptr, false);
}

Status qk_norm_rope_positioned(
    const float* qkv, const float* q_weight, const float* k_weight,
    const float* cosine, const float* sine, const int* positions, float* y,
    long long tokens, long long query_heads, long long key_heads,
    long long value_heads, long long head_dim, long long rotary_dim,
    long long max_position, float eps, bool interleaved,
    float norm_weight_offset, const int* mrope_sections,
    bool section_interleaved) {
  const long long total_heads = query_heads + key_heads + value_heads;
  const long long rd = resolve_rotary_dim(head_dim, rotary_dim);
  const bool multimodal = mrope_sections != nullptr;
  if (query_heads <= 0 || key_heads <= 0 || value_heads < 0 ||
      !detail::valid_product({tokens, total_heads, head_dim}) ||
      !valid_rotary_shape(head_dim, rd, max_position) || !std::isfinite(eps) ||
      eps < 0.0f || !std::isfinite(norm_weight_offset) ||
      (multimodal && interleaved) || (!multimodal && section_interleaved) ||
      (multimodal &&
       !valid_mrope_sections(mrope_sections, rd / 2, section_interleaved))) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(qkv, q_weight, k_weight, cosine, sine, positions,
                           y)) {
    return Status::kInvalidArgument;
  }
  const long long position_count = (multimodal ? 3 : 1) * tokens;
  for (long long index = 0; index < position_count; ++index) {
    if (positions[index] < 0 || positions[index] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  const long long pairs = rd / 2;
  const long long norm_heads = query_heads + key_heads;
  if (num_threads() == 1) {
    for (long long token = 0; token < tokens; ++token) {
      for (long long head = 0; head < norm_heads; ++head) {
        const long long offset = (token * total_heads + head) * head_dim;
        const float* source = qkv + offset;
        const float* weight = head < query_heads ? q_weight : k_weight;
        const float inverse = inverse_rms(source, head_dim, eps);
        if (multimodal) {
          const int selected_positions[3] = {positions[token],
                                             positions[tokens + token],
                                             positions[2 * tokens + token]};
          norm_rotate_mrope_row(source, weight, cosine, sine,
                                selected_positions, mrope_sections, y + offset,
                                head_dim, rd, section_interleaved, inverse,
                                norm_weight_offset);
        } else {
          const long long table_offset =
              static_cast<long long>(positions[token]) * pairs;
          norm_rotate_positioned_row(source, weight, cosine + table_offset,
                                     sine + table_offset, y + offset, head_dim,
                                     rd, interleaved, inverse,
                                     norm_weight_offset);
        }
      }
      const long long value_offset =
          (token * total_heads + norm_heads) * head_dim;
      std::copy_n(qkv + value_offset, value_heads * head_dim, y + value_offset);
    }
    return Status::kOk;
  }
  threading::parallel_ranges(
      tokens * norm_heads, 4, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long token = item / norm_heads;
          const long long head = item % norm_heads;
          const long long offset = (token * total_heads + head) * head_dim;
          const float* source = qkv + offset;
          const float* weight = head < query_heads ? q_weight : k_weight;
          const float inverse = inverse_rms(source, head_dim, eps);
          if (multimodal) {
            const int selected_positions[3] = {positions[token],
                                               positions[tokens + token],
                                               positions[2 * tokens + token]};
            norm_rotate_mrope_row(source, weight, cosine, sine,
                                  selected_positions, mrope_sections,
                                  y + offset, head_dim, rd, section_interleaved,
                                  inverse, norm_weight_offset);
          } else {
            const long long table_offset =
                static_cast<long long>(positions[token]) * pairs;
            norm_rotate_positioned_row(source, weight, cosine + table_offset,
                                       sine + table_offset, y + offset,
                                       head_dim, rd, interleaved, inverse,
                                       norm_weight_offset);
          }
        }
      });
  threading::parallel_ranges(
      tokens, 16, [&](long long begin, long long end, int) {
        for (long long token = begin; token < end; ++token) {
          const long long value_offset =
              (token * total_heads + norm_heads) * head_dim;
          std::copy_n(qkv + value_offset, value_heads * head_dim,
                      y + value_offset);
        }
      });
  return Status::kOk;
}

Status qk_norm_rope_split(const float* qkv, const float* q_weight,
                          const float* k_weight, const float* cosine,
                          const float* sine, const int* positions, float* q_out,
                          float* k_out, float* v_out, long long tokens,
                          long long query_heads, long long key_heads,
                          long long value_heads, long long head_dim,
                          long long max_position, float eps, bool interleaved,
                          bool gemma_weight) {
  const long long total_heads = query_heads + key_heads + value_heads;
  if (!detail::valid_product({tokens, total_heads, head_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(qkv, q_weight, k_weight, cosine, sine, positions,
                           q_out, k_out, v_out)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> q_adjusted;
  std::vector<float> k_adjusted;
  const float* qw = q_weight;
  const float* kw = k_weight;
  if (gemma_weight) {
    q_adjusted.resize(static_cast<std::size_t>(head_dim));
    k_adjusted.resize(static_cast<std::size_t>(head_dim));
    for (long long d = 0; d < head_dim; ++d) {
      q_adjusted[static_cast<std::size_t>(d)] = q_weight[d] + 1.0f;
      k_adjusted[static_cast<std::size_t>(d)] = k_weight[d] + 1.0f;
    }
    qw = q_adjusted.data();
    kw = k_adjusted.data();
  }
  std::vector<float> packed(
      static_cast<std::size_t>(tokens * total_heads * head_dim));
  Status status = qk_norm_rope(
      qkv, qw, kw, cosine, sine, positions, packed.data(), tokens, query_heads,
      key_heads, value_heads, head_dim, max_position, eps, interleaved);
  if (status != Status::kOk) return status;
  for (long long token = 0; token < tokens; ++token) {
    const float* row = packed.data() + token * total_heads * head_dim;
    std::copy_n(row, query_heads * head_dim,
                q_out + token * query_heads * head_dim);
    std::copy_n(row + query_heads * head_dim, key_heads * head_dim,
                k_out + token * key_heads * head_dim);
    std::copy_n(row + (query_heads + key_heads) * head_dim,
                value_heads * head_dim, v_out + token * value_heads * head_dim);
  }
  return Status::kOk;
}

Status swin_attention_d32(const float* qkv, const float* relative_bias,
                          const float* mask, float* out, long long windows,
                          long long tokens, long long heads,
                          long long windows_per_image) {
  constexpr long long kHeadDim = 32;
  if (!detail::valid_product({windows, tokens, heads, kHeadDim}) ||
      windows_per_image < 0 ||
      (mask != nullptr &&
       (windows_per_image <= 0 || windows % windows_per_image != 0))) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(qkv, relative_bias, out)) {
    return Status::kInvalidArgument;
  }
  constexpr double kScale = 0.1767766952966369;  // 1/sqrt(32)
  threading::parallel_ranges(
      windows * heads * tokens, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long query_position = item % tokens;
          const long long head = (item / tokens) % heads;
          const long long window = item / (tokens * heads);
          const float* query =
              qkv + ((((window * tokens + query_position) * 3) * heads + head) *
                     kHeadDim);
          float* destination =
              out +
              ((window * tokens + query_position) * heads + head) * kHeadDim;
          std::fill_n(destination, kHeadDim, 0.0f);
          double maximum = -std::numeric_limits<double>::infinity();
          double denominator = 0.0;
          const long long mask_window =
              windows_per_image > 0 ? window % windows_per_image : 0;
          for (long long key_position = 0; key_position < tokens;
               ++key_position) {
            const float* key =
                qkv +
                ((((window * tokens + key_position) * 3 + 1) * heads + head) *
                 kHeadDim);
            double score = 0.0;
            for (long long d = 0; d < kHeadDim; ++d) {
              score += static_cast<double>(query[d]) * key[d];
            }
            score = score * kScale +
                    relative_bias[(head * tokens + query_position) * tokens +
                                  key_position];
            if (mask != nullptr) {
              score += mask[(mask_window * tokens + query_position) * tokens +
                            key_position];
            }
            const double next_maximum = std::max(maximum, score);
            const double old_weight = std::exp(maximum - next_maximum);
            const double new_weight = std::exp(score - next_maximum);
            denominator = denominator * old_weight + new_weight;
            const float* value =
                qkv +
                ((((window * tokens + key_position) * 3 + 2) * heads + head) *
                 kHeadDim);
            for (long long d = 0; d < kHeadDim; ++d) {
              destination[d] = static_cast<float>(destination[d] * old_weight +
                                                  value[d] * new_weight);
            }
            maximum = next_maximum;
          }
          const float inverse = static_cast<float>(1.0 / denominator);
          for (long long d = 0; d < kHeadDim; ++d) destination[d] *= inverse;
        }
      });
  return Status::kOk;
}

}  // namespace quixicore_cpu
