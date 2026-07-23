#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool recurrent_shape(long long sequences, long long tokens,
                     long long heads, long long dim) {
  return detail::valid_product({sequences, tokens, heads, dim}) &&
         detail::valid_product({sequences, heads, dim, dim});
}

}  // namespace

Status gated_linear_attention(
    const float* key, const float* value, const float* query,
    const float* gate, const float* initial_state, float* output,
    float* final_state, long long sequences, long long tokens_per_sequence,
    long long heads, long long head_dim, float scale) {
  if (!recurrent_shape(sequences, tokens_per_sequence, heads, head_dim) ||
      !std::isfinite(scale)) return Status::kInvalidShape;
  if (!detail::all_nonnull(key, value, query, gate, initial_state, output,
                           final_state)) return Status::kInvalidArgument;
  const long long token_stride = heads * head_dim;
  const long long matrix_size = head_dim * head_dim;
  threading::parallel_ranges(sequences * heads, 1,
      [&](long long begin, long long end, int) {
    for (long long task = begin; task < end; ++task) {
      const long long sequence = task / heads;
      const long long head = task % heads;
      float* state = final_state + task * matrix_size;
      const float* initial = initial_state + task * matrix_size;
      if (state != initial) std::copy_n(initial, matrix_size, state);
      for (long long token = 0; token < tokens_per_sequence; ++token) {
        const long long token_offset =
            (sequence * tokens_per_sequence + token) * token_stride +
            head * head_dim;
        std::fill_n(output + token_offset, head_dim, 0.0f);
        for (long long i = 0; i < head_dim; ++i) {
          const float key_value = key[token_offset + i];
          const float query_value = query[token_offset + i] * scale;
          const float gate_value = gate[token_offset + i];
          float* state_row = state + i * head_dim;
          for (long long j = 0; j < head_dim; ++j) {
            state_row[j] = state_row[j] * gate_value +
                           key_value * value[token_offset + j];
            output[token_offset + j] += state_row[j] * query_value;
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status rwkv_wkv6(const float* key, const float* value,
                 const float* receptance, const float* time_first,
                 const float* time_decay, const float* initial_state,
                 float* output, float* final_state, long long sequences,
                 long long tokens_per_sequence, long long heads,
                 long long head_dim) {
  if (!recurrent_shape(sequences, tokens_per_sequence, heads, head_dim))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(key, value, receptance, time_first, time_decay,
                           initial_state, output, final_state)) {
    return Status::kInvalidArgument;
  }
  const long long token_stride = heads * head_dim;
  const long long matrix_size = head_dim * head_dim;
  threading::parallel_ranges(sequences * heads, 1,
      [&](long long begin, long long end, int) {
    for (long long task = begin; task < end; ++task) {
      const long long sequence = task / heads;
      const long long head = task % heads;
      float* state = final_state + task * matrix_size;
      const float* initial = initial_state + task * matrix_size;
      if (state != initial) std::copy_n(initial, matrix_size, state);
      for (long long token = 0; token < tokens_per_sequence; ++token) {
        const long long offset =
            (sequence * tokens_per_sequence + token) * token_stride +
            head * head_dim;
        std::fill_n(output + offset, head_dim, 0.0f);
        for (long long i = 0; i < head_dim; ++i) {
          const float k = key[offset + i];
          const float r = receptance[offset + i];
          const float first = time_first[head * head_dim + i];
          const float decay = time_decay[offset + i];
          float* state_row = state + i * head_dim;
          for (long long j = 0; j < head_dim; ++j) {
            const float kv = k * value[offset + j];
            const float previous = state_row[j];
            output[offset + j] += (previous + kv * first) * r;
            state_row[j] = previous * decay + kv;
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status rwkv_wkv7(const float* receptance, const float* decay,
                 const float* key, const float* value, const float* a,
                 const float* b, const float* initial_state, float* output,
                 float* final_state, long long sequences,
                 long long tokens_per_sequence, long long heads,
                 long long head_dim) {
  if (!recurrent_shape(sequences, tokens_per_sequence, heads, head_dim))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(receptance, decay, key, value, a, b, initial_state,
                           output, final_state)) return Status::kInvalidArgument;
  const long long token_stride = heads * head_dim;
  const long long matrix_size = head_dim * head_dim;
  threading::parallel_ranges(sequences * heads, 1,
      [&](long long begin, long long end, int) {
    for (long long task = begin; task < end; ++task) {
      const long long sequence = task / heads;
      const long long head = task % heads;
      float* state = final_state + task * matrix_size;
      const float* initial = initial_state + task * matrix_size;
      if (state != initial) std::copy_n(initial, matrix_size, state);
      for (long long token = 0; token < tokens_per_sequence; ++token) {
        const long long offset =
            (sequence * tokens_per_sequence + token) * token_stride +
            head * head_dim;
        for (long long i = 0; i < head_dim; ++i) {
          float* state_row = state + i * head_dim;
          float state_a = 0.0f;
          for (long long j = 0; j < head_dim; ++j) {
            state_a += a[offset + j] * state_row[j];
          }
          float result = 0.0f;
          for (long long j = 0; j < head_dim; ++j) {
            state_row[j] = state_row[j] * decay[offset + j] +
                           value[offset + i] * key[offset + j] +
                           state_a * b[offset + j];
            result += state_row[j] * receptance[offset + j];
          }
          output[offset + i] = result;
        }
      }
    }
  });
  return Status::kOk;
}

Status dsv4_hc_comb(const float* mixes, const float* scale,
                    const float* base, float* comb, long long tokens,
                    float eps, int iterations) {
  constexpr long long kConnections = 4;
  constexpr long long kOffset = 8;
  constexpr long long kMixSize = 24;
  if (!detail::valid_product({tokens, kMixSize}) || !std::isfinite(eps) ||
      eps < 0.0f || iterations <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(mixes, scale, base, comb))
    return Status::kInvalidArgument;
  if (!std::isfinite(scale[2])) return Status::kInvalidArgument;
  threading::parallel_ranges(tokens, 1,
      [&](long long begin, long long end, int) {
    for (long long token = begin; token < end; ++token) {
      float matrix[16];
      for (long long source = 0; source < kConnections; ++source) {
        float maximum = -INFINITY;
        for (long long destination = 0; destination < kConnections;
             ++destination) {
          const long long index = destination + kConnections * source;
          matrix[index] = mixes[token * kMixSize + kOffset + index] * scale[2] +
                          base[kOffset + index];
          maximum = std::max(maximum, matrix[index]);
        }
        float sum = 0.0f;
        for (long long destination = 0; destination < kConnections;
             ++destination) {
          const long long index = destination + kConnections * source;
          matrix[index] = std::exp(matrix[index] - maximum);
          sum += matrix[index];
        }
        for (long long destination = 0; destination < kConnections;
             ++destination) {
          const long long index = destination + kConnections * source;
          matrix[index] = matrix[index] / sum + eps;
        }
      }
      auto normalize_columns = [&] {
        for (long long destination = 0; destination < kConnections;
             ++destination) {
          float sum = eps;
          for (long long source = 0; source < kConnections; ++source)
            sum += matrix[destination + kConnections * source];
          for (long long source = 0; source < kConnections; ++source)
            matrix[destination + kConnections * source] /= sum;
        }
      };
      auto normalize_rows = [&] {
        for (long long source = 0; source < kConnections; ++source) {
          float sum = eps;
          for (long long destination = 0; destination < kConnections;
               ++destination)
            sum += matrix[destination + kConnections * source];
          for (long long destination = 0; destination < kConnections;
               ++destination)
            matrix[destination + kConnections * source] /= sum;
        }
      };
      normalize_columns();
      for (int iteration = 1; iteration < iterations; ++iteration) {
        normalize_rows();
        normalize_columns();
      }
      std::copy_n(matrix, 16, comb + token * 16);
    }
  });
  return Status::kOk;
}

Status dsv4_hc_pre(const float* x, const float* weights, float* output,
                   long long tokens, long long embedding) {
  constexpr long long kConnections = 4;
  if (!detail::valid_product({tokens, kConnections, embedding}))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(x, weights, output)) return Status::kInvalidArgument;
  threading::parallel_ranges(tokens * embedding, 16,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long token = item / embedding;
      const long long dimension = item % embedding;
      float sum = 0.0f;
      for (long long connection = 0; connection < kConnections; ++connection) {
        sum += x[(token * kConnections + connection) * embedding + dimension] *
               weights[token * kConnections + connection];
      }
      output[item] = sum;
    }
  });
  return Status::kOk;
}

Status dsv4_hc_post(const float* x, const float* residual,
                    const float* post, const float* comb, float* output,
                    long long tokens, long long embedding) {
  constexpr long long kConnections = 4;
  if (!detail::valid_product({tokens, kConnections, embedding}))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(x, residual, post, comb, output))
    return Status::kInvalidArgument;
  threading::parallel_ranges(tokens * kConnections, 1,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long token = item / kConnections;
      const long long destination = item % kConnections;
      for (long long dimension = 0; dimension < embedding; ++dimension) {
        float sum = x[token * embedding + dimension] *
                    post[token * kConnections + destination];
        for (long long source = 0; source < kConnections; ++source) {
          sum += residual[(token * kConnections + source) * embedding +
                          dimension] *
                 comb[token * 16 + source * kConnections + destination];
        }
        output[(token * kConnections + destination) * embedding + dimension] =
            sum;
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
