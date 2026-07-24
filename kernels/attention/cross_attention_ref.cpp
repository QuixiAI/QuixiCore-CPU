#include <algorithm>
#include <cmath>
#include <limits>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float cross_attention_dot(const float* query, const float* key,
                          long long head_dim) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (long long feature = 0; feature < head_dim; feature += 4) {
    sum0 += query[feature] * key[feature];
    sum1 += query[feature + 1] * key[feature + 1];
    sum2 += query[feature + 2] * key[feature + 2];
    sum3 += query[feature + 3] * key[feature + 3];
  }
  return (sum0 + sum1) + (sum2 + sum3);
}

}  // namespace

Status cross_attention(const float* q, const float* k, const float* v,
                       const int* key_lengths, const float* bias, float* out,
                       long long batch, long long query_heads,
                       long long kv_heads, long long query_length,
                       long long key_length, long long head_dim, float scale,
                       float softcap) {
  if (!detail::valid_product(
          {batch, query_heads, query_length, head_dim}) ||
      !detail::valid_product({batch, kv_heads, key_length, head_dim}) ||
      query_heads % kv_heads != 0 ||
      !(head_dim == 64 || head_dim == 128 || head_dim == 256) ||
      !std::isfinite(softcap) || softcap < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, key_lengths, out)) {
    return Status::kInvalidArgument;
  }
  const double score_scale =
      scale > 0.0f ? scale : 1.0 / std::sqrt(static_cast<double>(head_dim));
  if (!std::isfinite(score_scale)) return Status::kInvalidArgument;
  const long long head_group = query_heads / kv_heads;
  const long long queries_per_batch = query_heads * query_length;
  threading::parallel_ranges(
      batch * queries_per_batch, 1,
      [&](long long begin, long long end, int) {
    for (long long query_row = begin; query_row < end; ++query_row) {
      const long long item = query_row / queries_per_batch;
      const long long remainder = query_row - item * queries_per_batch;
      const long long query_head = remainder / query_length;
      const long long valid_keys =
          std::clamp<long long>(key_lengths[item], 0, key_length);
      const long long kv_head = query_head / head_group;
      const float* query = q + query_row * head_dim;
      float* destination = out + query_row * head_dim;
      std::fill_n(destination, head_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long key_position = 0; key_position < valid_keys;
           ++key_position) {
        const long long kv_row =
            ((item * kv_heads + kv_head) * key_length + key_position);
        const float* key = k + kv_row * head_dim;
        double score = static_cast<double>(
                           cross_attention_dot(query, key, head_dim)) *
                       score_scale;
        if (bias != nullptr) {
          score += bias[query_row * key_length + key_position];
        }
        if (softcap > 0.0f) {
          score = static_cast<double>(softcap) *
                  std::tanh(score / static_cast<double>(softcap));
        }
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        const float* value = v + kv_row * head_dim;
        for (long long feature = 0; feature < head_dim; ++feature) {
          destination[feature] = static_cast<float>(
              destination[feature] * old_weight + value[feature] * new_weight);
        }
        maximum = next_maximum;
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        for (long long feature = 0; feature < head_dim; ++feature) {
          destination[feature] *= inverse;
        }
      }
    }
  });
  return Status::kOk;
}

Status cross_attention_storage(
    FloatStorageInput q, FloatStorageInput k, FloatStorageInput v,
    const int* key_lengths, const float* bias, FloatStorageOutput out,
    long long batch, long long query_heads, long long kv_heads,
    long long query_length, long long key_length, long long head_dim,
    float scale, float softcap, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product(
          {batch, query_heads, query_length, head_dim}) ||
      !detail::valid_product({batch, kv_heads, key_length, head_dim}) ||
      q.count != batch * query_heads * query_length * head_dim ||
      k.count != batch * kv_heads * key_length * head_dim ||
      v.count != k.count || out.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q, k, v};
  return with_float_storage(
      inputs, 3, &out, 1,
      [&](const float* const* values, float* const* outputs) {
        return cross_attention(values[0], values[1], values[2], key_lengths,
                               bias, outputs[0], batch, query_heads, kv_heads,
                               query_length, key_length, head_dim, scale,
                               softcap);
      },
      workspace);
}

}  // namespace quixicore_cpu
