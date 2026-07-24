#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float relative_score(const float* scaled_query, const float* key,
                     const float* relative, long long head_dim,
                     float key_scale) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (long long feature = 0; feature < head_dim; feature += 4) {
    const float r0 = relative == nullptr ? 0.0f : relative[feature];
    const float r1 = relative == nullptr ? 0.0f : relative[feature + 1];
    const float r2 = relative == nullptr ? 0.0f : relative[feature + 2];
    const float r3 = relative == nullptr ? 0.0f : relative[feature + 3];
    sum0 += scaled_query[feature] * (key[feature] * key_scale + r0);
    sum1 += scaled_query[feature + 1] *
            (key[feature + 1] * key_scale + r1);
    sum2 += scaled_query[feature + 2] *
            (key[feature + 2] * key_scale + r2);
    sum3 += scaled_query[feature + 3] *
            (key[feature + 3] * key_scale + r3);
  }
  return (sum0 + sum1) + (sum2 + sum3);
}

float softplus(float value) {
  return std::max(value, 0.0f) + std::log1p(std::exp(-std::fabs(value)));
}

}  // namespace

Status audio_relative_attention(
    const float* q, const float* k, const float* v,
    const float* relative_k, const float* per_dim_scale, const int* lengths,
    float* out, long long batch, long long length, long long heads,
    long long head_dim, long long relative_positions, long long chunk_size,
    long long left_context, long long right_context, float q_scale,
    float k_scale, float softcap) {
  if (!detail::valid_product({batch, length, heads, head_dim}) ||
      !detail::valid_product({relative_positions, heads, head_dim}) ||
      !(head_dim == 64 || head_dim == 128 || head_dim == 256) ||
      chunk_size <= 0 || left_context <= 0 || right_context < 0 ||
      !std::isfinite(q_scale) || !std::isfinite(k_scale) ||
      !std::isfinite(softcap) || softcap < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, relative_k, per_dim_scale, lengths,
                           out)) {
    return Status::kInvalidArgument;
  }
  const float log_two = std::log(2.0f);
  const float used_q_scale =
      q_scale > 0.0f
          ? q_scale
          : 1.0f / (std::sqrt(static_cast<float>(head_dim)) * log_two);
  const float used_k_scale = k_scale > 0.0f ? k_scale : 1.0f / log_two;
  std::array<float, 256> query_dimension_scale{};
  for (long long feature = 0; feature < head_dim; ++feature) {
    query_dimension_scale[feature] =
        used_q_scale * softplus(per_dim_scale[feature]);
  }
  const long long rows_per_batch = length * heads;
  threading::parallel_ranges(
      batch * rows_per_batch, 1,
      [&](long long begin, long long end, int) {
        std::array<float, 256> scaled_query{};
        for (long long row = begin; row < end; ++row) {
          const long long item = row / rows_per_batch;
          const long long remainder = row - item * rows_per_batch;
          const long long query_position = remainder / heads;
          const long long head = remainder - query_position * heads;
          const long long valid_length =
              std::clamp<long long>(lengths[item], 0, length);
          float* destination = out + row * head_dim;
          std::fill_n(destination, head_dim, 0.0f);
          if (query_position >= valid_length) continue;

          const float* query = q + row * head_dim;
          for (long long feature = 0; feature < head_dim; ++feature) {
            scaled_query[feature] =
                query[feature] * query_dimension_scale[feature];
          }
          const long long query_in_chunk = query_position % chunk_size;
          const long long block_start =
              (query_position / chunk_size) * chunk_size;
          const long long context_start = block_start - (left_context - 1);
          const long long context_length =
              chunk_size + left_context - 1 + right_context;
          float maximum = -std::numeric_limits<float>::infinity();
          float denominator = 0.0f;
          for (long long context = 0; context < context_length; ++context) {
            const long long key_position = context_start + context;
            if (key_position < 0 || key_position >= valid_length) continue;
            const long long kv_row =
                ((item * length + key_position) * heads + head);
            const long long relative_index = context - query_in_chunk;
            const float* relative =
                relative_index >= 0 && relative_index < relative_positions
                    ? relative_k +
                          (relative_index * heads + head) * head_dim
                    : nullptr;
            float score = relative_score(scaled_query.data(),
                                         k + kv_row * head_dim, relative,
                                         head_dim, used_k_scale);
            if (softcap > 0.0f) score = softcap * std::tanh(score / softcap);
            const float* value = v + kv_row * head_dim;
            if (score > maximum) {
              const float previous_scale =
                  std::isfinite(maximum) ? std::exp(maximum - score) : 0.0f;
              denominator = denominator * previous_scale + 1.0f;
              for (long long feature = 0; feature < head_dim; ++feature) {
                destination[feature] =
                    destination[feature] * previous_scale + value[feature];
              }
              maximum = score;
            } else {
              const float probability = std::exp(score - maximum);
              denominator += probability;
              for (long long feature = 0; feature < head_dim; ++feature) {
                destination[feature] += probability * value[feature];
              }
            }
          }
          if (denominator > 0.0f) {
            const float inverse = 1.0f / denominator;
            for (long long feature = 0; feature < head_dim; ++feature) {
              destination[feature] *= inverse;
            }
          }
        }
      });
  return Status::kOk;
}

Status audio_relative_attention_storage(
    FloatStorageInput q, FloatStorageInput k, FloatStorageInput v,
    FloatStorageInput relative_k, const float* per_dim_scale,
    const int* lengths, FloatStorageOutput out, long long batch,
    long long length, long long heads, long long head_dim,
    long long relative_positions, long long chunk_size,
    long long left_context, long long right_context, float q_scale,
    float k_scale, float softcap, FloatStorageWorkspace* workspace) {
  const long long tensor_count = batch * length * heads * head_dim;
  const long long relative_count = relative_positions * heads * head_dim;
  if (!detail::valid_product({batch, length, heads, head_dim}) ||
      !detail::valid_product({relative_positions, heads, head_dim}) ||
      q.count != tensor_count || k.count != tensor_count ||
      v.count != tensor_count || out.count != tensor_count ||
      relative_k.count != relative_count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q, k, v, relative_k};
  return with_float_storage(
      inputs, 4, &out, 1,
      [&](const float* const* values, float* const* outputs) {
        return audio_relative_attention(
            values[0], values[1], values[2], values[3], per_dim_scale,
            lengths, outputs[0], batch, length, heads, head_dim,
            relative_positions, chunk_size, left_context, right_context,
            q_scale, k_scale, softcap);
      },
      workspace);
}

}  // namespace quixicore_cpu
