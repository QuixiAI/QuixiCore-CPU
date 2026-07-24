#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool valid_gdn_metadata(const int* cumulative_lengths, const int* slot_mapping,
                        long long requests, long long slots,
                        bool allow_negative_slots, long long* total_tokens) {
  if (!detail::all_nonnull(cumulative_lengths, slot_mapping, total_tokens) ||
      requests <= 0 || slots <= 0 || cumulative_lengths[0] != 0) {
    return false;
  }
  for (long long request = 0; request < requests; ++request) {
    if (cumulative_lengths[request + 1] < cumulative_lengths[request] ||
        slot_mapping[request] >= slots ||
        (!allow_negative_slots && slot_mapping[request] < 0)) {
      return false;
    }
  }
  *total_tokens = cumulative_lengths[requests];
  return *total_tokens >= 0;
}

float stable_sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-value));
  }
  const float exponential = std::exp(value);
  return exponential / (1.0f + exponential);
}

float stable_softplus(float value) {
  if (value > 20.0f) return value;
  if (value < -20.0f) return std::exp(value);
  return std::log1p(std::exp(value));
}

bool positive_slots_are_unique(const int* slot_mapping, long long requests) {
  for (long long left = 0; left < requests; ++left) {
    if (slot_mapping[left] < 0) continue;
    for (long long right = left + 1; right < requests; ++right) {
      if (slot_mapping[left] == slot_mapping[right]) return false;
    }
  }
  return true;
}

void gdn_recur_rows(const float* q, const float* k, const float* v,
                    const float* decay, const float* beta,
                    const int* cumulative_lengths, const int* slot_mapping,
                    float* out, float* state_pool_out, long long key_heads,
                    long long value_heads, long long key_dim,
                    long long value_dim, bool load_initial, long long begin,
                    long long end) {
  const long long head_group = value_heads / key_heads;
  const long long rows_per_request = value_heads * value_dim;
  for (long long task = begin; task < end; ++task) {
    const long long request = task / rows_per_request;
    const long long row = task % rows_per_request;
    const long long value_head = row / value_dim;
    const long long value_index = row % value_dim;
    const long long key_head = value_head / head_group;
    float* state =
        state_pool_out +
        ((static_cast<long long>(slot_mapping[request]) * value_heads +
          value_head) *
             value_dim +
         value_index) *
            key_dim;
    if (!load_initial) std::fill_n(state, key_dim, 0.0f);
    for (long long token = cumulative_lengths[request];
         token < cumulative_lengths[request + 1]; ++token) {
      const float* key = k + (token * key_heads + key_head) * key_dim;
      const float* query = q + (token * key_heads + key_head) * key_dim;
      const float gate = decay[token * value_heads + value_head];
      double memory = 0.0;
      for (long long dim = 0; dim < key_dim; ++dim) {
        state[dim] *= gate;
        memory += static_cast<double>(state[dim]) * key[dim];
      }
      const double correction =
          (v[(token * value_heads + value_head) * value_dim + value_index] -
           memory) *
          beta[token * value_heads + value_head];
      double result = 0.0;
      for (long long dim = 0; dim < key_dim; ++dim) {
        state[dim] += static_cast<float>(key[dim] * correction);
        result += static_cast<double>(state[dim]) * query[dim];
      }
      out[(token * value_heads + value_head) * value_dim + value_index] =
          static_cast<float>(result);
    }
  }
}

void gdn_recur_serial(const float* q, const float* k, const float* v,
                      const float* decay, const float* beta,
                      const int* cumulative_lengths, const int* slot_mapping,
                      float* out, float* state_pool_out, long long requests,
                      long long key_heads, long long value_heads,
                      long long key_dim, long long value_dim,
                      bool load_initial) {
  const long long head_group = value_heads / key_heads;
  for (long long request = 0; request < requests; ++request) {
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    const long long slot = slot_mapping[request];
    for (long long value_head = 0; value_head < value_heads; ++value_head) {
      const long long key_head = value_head / head_group;
      for (long long value_index = 0; value_index < value_dim; ++value_index) {
        float* state =
            state_pool_out +
            ((slot * value_heads + value_head) * value_dim + value_index) *
                key_dim;
        if (!load_initial) std::fill_n(state, key_dim, 0.0f);
        for (long long token = start; token < end; ++token) {
          const float* key = k + (token * key_heads + key_head) * key_dim;
          const float* query = q + (token * key_heads + key_head) * key_dim;
          const float gate = decay[token * value_heads + value_head];
          double memory = 0.0;
          for (long long dim = 0; dim < key_dim; ++dim) {
            state[dim] *= gate;
            memory += static_cast<double>(state[dim]) * key[dim];
          }
          const double correction =
              (v[(token * value_heads + value_head) * value_dim + value_index] -
               memory) *
              beta[token * value_heads + value_head];
          double result = 0.0;
          for (long long dim = 0; dim < key_dim; ++dim) {
            state[dim] += static_cast<float>(key[dim] * correction);
            result += static_cast<double>(state[dim]) * query[dim];
          }
          out[(token * value_heads + value_head) * value_dim + value_index] =
              static_cast<float>(result);
        }
      }
    }
  }
}

void gdn_conv_channels(const float* x, const float* weight,
                       const int* cumulative_lengths,
                       const int* slot_mapping, float* out,
                       float* state_pool_out, long long channels,
                       long long kernel_size, bool load_initial,
                       bool apply_silu, long long begin, long long end) {
  const long long history_size = kernel_size - 1;
  for (long long task = begin; task < end; ++task) {
    const long long request = task / channels;
    const long long channel = task % channels;
    const long long start = cumulative_lengths[request];
    const long long finish = cumulative_lengths[request + 1];
    const int slot = slot_mapping[request];
    if (slot < 0) {
      for (long long token = start; token < finish; ++token) {
        out[token * channels + channel] = 0.0f;
      }
      continue;
    }
    float history[7]{};
    float* state =
        state_pool_out +
        (static_cast<long long>(slot) * channels + channel) * history_size;
    if (load_initial) std::copy_n(state, history_size, history);
    const float* channel_weight = weight + channel * kernel_size;
    if (kernel_size == 4) {
      float h0 = history[0];
      float h1 = history[1];
      float h2 = history[2];
      for (long long token = start; token < finish; ++token) {
        const float current = x[token * channels + channel];
        float value = current * channel_weight[3];
        value += h0 * channel_weight[0];
        value += h1 * channel_weight[1];
        value += h2 * channel_weight[2];
        if (apply_silu) value *= stable_sigmoid(value);
        out[token * channels + channel] = value;
        h0 = h1;
        h1 = h2;
        h2 = current;
      }
      state[0] = h0;
      state[1] = h1;
      state[2] = h2;
      continue;
    }
    for (long long token = start; token < finish; ++token) {
      const float current = x[token * channels + channel];
      float value = current * channel_weight[kernel_size - 1];
      for (long long item = 0; item < history_size; ++item) {
        value += history[item] * channel_weight[item];
      }
      if (apply_silu) value *= stable_sigmoid(value);
      out[token * channels + channel] = value;
      for (long long item = 0; item + 1 < history_size; ++item) {
        history[item] = history[item + 1];
      }
      history[history_size - 1] = current;
    }
    std::copy_n(history, history_size, state);
  }
}

void gdn_conv_serial(const float* x, const float* weight,
                     const int* cumulative_lengths, const int* slot_mapping,
                     float* out, float* state_pool_out, long long requests,
                     long long channels, long long kernel_size,
                     bool load_initial, bool apply_silu) {
  const long long history_size = kernel_size - 1;
  for (long long request = 0; request < requests; ++request) {
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    const int slot = slot_mapping[request];
    if (slot < 0) {
      std::fill(out + start * channels, out + end * channels, 0.0f);
      continue;
    }
    for (long long channel = 0; channel < channels; ++channel) {
      float history[7]{};
      float* state =
          state_pool_out +
          (static_cast<long long>(slot) * channels + channel) * history_size;
      if (load_initial) std::copy_n(state, history_size, history);
      const float* channel_weight = weight + channel * kernel_size;
      if (kernel_size == 4) {
        float h0 = history[0];
        float h1 = history[1];
        float h2 = history[2];
        for (long long token = start; token < end; ++token) {
          const float current = x[token * channels + channel];
          float value = current * channel_weight[3];
          value += h0 * channel_weight[0];
          value += h1 * channel_weight[1];
          value += h2 * channel_weight[2];
          if (apply_silu) value *= stable_sigmoid(value);
          out[token * channels + channel] = value;
          h0 = h1;
          h1 = h2;
          h2 = current;
        }
        state[0] = h0;
        state[1] = h1;
        state[2] = h2;
        continue;
      }
      for (long long token = start; token < end; ++token) {
        const float current = x[token * channels + channel];
        float value = current * channel_weight[kernel_size - 1];
        for (long long item = 0; item < history_size; ++item) {
          value += history[item] * channel_weight[item];
        }
        if (apply_silu) value *= stable_sigmoid(value);
        out[token * channels + channel] = value;
        for (long long item = 0; item + 1 < history_size; ++item) {
          history[item] = history[item + 1];
        }
        history[history_size - 1] = current;
      }
      std::copy_n(history, history_size, state);
    }
  }
}

void gdn_qkv_tokens(const float* mixed, float* q, float* k, float* v,
                    long long key_heads, long long value_heads,
                    long long key_dim, long long value_dim, float eps,
                    float q_scale, float k_scale, long long begin,
                    long long end) {
  const long long qk_width = key_heads * key_dim;
  const long long value_width = value_heads * value_dim;
  const long long channels = 2 * qk_width + value_width;
  for (long long token = begin; token < end; ++token) {
    const float* source = mixed + token * channels;
    for (long long head = 0; head < key_heads; ++head) {
      for (int is_key = 0; is_key < 2; ++is_key) {
        const float* input = source + is_key * qk_width + head * key_dim;
        float* output = (is_key ? k : q) + (token * key_heads + head) * key_dim;
        float sum_squares = 0.0f;
        for (long long dim = 0; dim < key_dim; ++dim) {
          sum_squares += input[dim] * input[dim];
        }
        const float scale =
            (is_key ? k_scale : q_scale) /
            std::sqrt(sum_squares / static_cast<float>(key_dim) + eps);
        for (long long dim = 0; dim < key_dim; ++dim) {
          output[dim] = input[dim] * scale;
        }
      }
    }
    std::copy_n(source + 2 * qk_width, value_width, v + token * value_width);
  }
}

void gdn_gate_rows(const float* a, const float* b, const float* a_log,
                   const float* dt_bias, float* decay, float* beta,
                   long long value_heads, long long begin, long long end) {
  for (long long token = begin; token < end; ++token) {
    const long long offset = token * value_heads;
    for (long long head = 0; head < value_heads; ++head) {
      const long long index = offset + head;
      const float alpha = a[index] + dt_bias[head];
      decay[index] = std::exp(-std::exp(a_log[head]) * stable_softplus(alpha));
      beta[index] = stable_sigmoid(b[index]);
    }
  }
}

void gdn_norm_rows(const float* y, const float* z, const float* weight,
                   float* out, long long value_dim, float eps, long long begin,
                   long long end) {
  for (long long row = begin; row < end; ++row) {
    const long long offset = row * value_dim;
    float sum_squares = 0.0f;
    for (long long dim = 0; dim < value_dim; ++dim) {
      sum_squares += y[offset + dim] * y[offset + dim];
    }
    const float inverse =
        1.0f / std::sqrt(sum_squares / static_cast<float>(value_dim) + eps);
    for (long long dim = 0; dim < value_dim; ++dim) {
      const float gate = z[offset + dim];
      out[offset + dim] =
          y[offset + dim] * inverse * weight[dim] * gate * stable_sigmoid(gate);
    }
  }
}

}  // namespace

Status gdn_recur(const float* q, const float* k, const float* v,
                 const float* decay, const float* beta, const float* state_pool,
                 const int* cumulative_lengths, const int* slot_mapping,
                 float* out, float* state_pool_out, long long requests,
                 long long slots, long long key_heads, long long value_heads,
                 long long key_dim, long long value_dim, bool load_initial) {
  if (key_heads <= 0 || value_heads <= 0 || value_heads % key_heads != 0 ||
      (key_dim != 64 && key_dim != 128) || value_dim <= 0 ||
      !detail::valid_product({slots, value_heads, value_dim, key_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, decay, beta, state_pool, cumulative_lengths,
                           slot_mapping, out, state_pool_out) ||
      state_pool_out == state_pool) {
    return Status::kInvalidArgument;
  }
  long long total_tokens = 0;
  if (!valid_gdn_metadata(cumulative_lengths, slot_mapping, requests, slots,
                          false, &total_tokens) ||
      !detail::valid_product(
          {total_tokens, key_heads, value_heads, key_dim, value_dim})) {
    return Status::kInvalidArgument;
  }
  const long long state_count = slots * value_heads * value_dim * key_dim;
  std::copy_n(state_pool, state_count, state_pool_out);
  const bool unique_slots = positive_slots_are_unique(slot_mapping, requests);
  if (num_threads() == 1 || !unique_slots) {
    gdn_recur_serial(q, k, v, decay, beta, cumulative_lengths, slot_mapping,
                     out, state_pool_out, requests, key_heads, value_heads,
                     key_dim, value_dim, load_initial);
  } else {
    const long long tasks = requests * value_heads * value_dim;
    threading::parallel_ranges(tasks, 8,
                               [&](long long begin, long long end, int) {
      gdn_recur_rows(q, k, v, decay, beta, cumulative_lengths, slot_mapping,
                     out, state_pool_out, key_heads, value_heads, key_dim,
                     value_dim, load_initial, begin, end);
    });
  }
  return Status::kOk;
}

Status gdn_short_conv(const float* x, const float* weight,
                      const float* state_pool, const int* cumulative_lengths,
                      const int* slot_mapping, float* out,
                      float* state_pool_out, long long requests,
                      long long slots, long long channels,
                      long long kernel_size, bool load_initial,
                      bool apply_silu) {
  if (channels <= 0 || kernel_size < 2 || kernel_size > 8 ||
      !detail::valid_product({slots, channels, kernel_size - 1})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, state_pool, cumulative_lengths,
                           slot_mapping, out, state_pool_out) ||
      state_pool_out == state_pool) {
    return Status::kInvalidArgument;
  }
  long long total_tokens = 0;
  if (!valid_gdn_metadata(cumulative_lengths, slot_mapping, requests, slots,
                          true, &total_tokens) ||
      !detail::valid_product({total_tokens, channels, kernel_size})) {
    return Status::kInvalidArgument;
  }
  const long long history_size = kernel_size - 1;
  const long long state_count = slots * channels * history_size;
  std::copy_n(state_pool, state_count, state_pool_out);
  const bool unique_slots = positive_slots_are_unique(slot_mapping, requests);
  if (num_threads() == 1 || !unique_slots) {
    gdn_conv_serial(x, weight, cumulative_lengths, slot_mapping, out,
                    state_pool_out, requests, channels, kernel_size,
                    load_initial, apply_silu);
  } else {
    const long long tasks = requests * channels;
    threading::parallel_ranges(tasks, 64,
                               [&](long long begin, long long end, int) {
      gdn_conv_channels(x, weight, cumulative_lengths, slot_mapping, out,
                        state_pool_out, channels, kernel_size, load_initial,
                        apply_silu, begin, end);
    });
  }
  return Status::kOk;
}

Status gdn_qkv_prepare(const float* mixed, float* q, float* k, float* v,
                       long long tokens, long long key_heads,
                       long long value_heads, long long key_dim,
                       long long value_dim, float eps, float q_scale,
                       float k_scale) {
  if (!detail::valid_product(
          {tokens, key_heads, value_heads, key_dim, value_dim}) ||
      value_heads % key_heads != 0 || (key_dim != 64 && key_dim != 128) ||
      (value_dim != 64 && value_dim != 128) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(mixed, q, k, v)) {
    return Status::kInvalidArgument;
  }
  if (std::isnan(q_scale)) {
    q_scale = 1.0f / static_cast<float>(key_dim);
  }
  if (std::isnan(k_scale)) {
    k_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  }
  if (!std::isfinite(q_scale) || !std::isfinite(k_scale)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(tokens, 4,
                             [&](long long begin, long long end, int) {
    gdn_qkv_tokens(mixed, q, k, v, key_heads, value_heads, key_dim, value_dim,
                   eps, q_scale, k_scale, begin, end);
  });
  return Status::kOk;
}

Status gdn_gate_beta(const float* a, const float* b, const float* a_log,
                     const float* dt_bias, float* decay, float* beta,
                     long long tokens, long long value_heads) {
  if (!detail::valid_product({tokens, value_heads})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, a_log, dt_bias, decay, beta)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(tokens, 32,
                             [&](long long begin, long long end, int) {
    gdn_gate_rows(a, b, a_log, dt_bias, decay, beta, value_heads, begin, end);
  });
  return Status::kOk;
}

Status gdn_gated_rmsnorm(const float* y, const float* z, const float* weight,
                         float* out, long long tokens, long long value_heads,
                         long long value_dim, float eps) {
  if (!detail::valid_product({tokens, value_heads, value_dim}) ||
      (value_dim != 64 && value_dim != 128) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(y, z, weight, out)) {
    return Status::kInvalidArgument;
  }
  const long long rows = tokens * value_heads;
  threading::parallel_ranges(rows, 4,
                             [&](long long begin, long long end, int) {
    gdn_norm_rows(y, z, weight, out, value_dim, eps, begin, end);
  });
  return Status::kOk;
}

Status sigmoid_mul_storage(FloatStorageInput gate_logits,
                           FloatStorageInput values, FloatStorageOutput out,
                           FloatStorageWorkspace* workspace) {
  if (gate_logits.count <= 0 || values.count != gate_logits.count ||
      out.count != gate_logits.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {gate_logits, values};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 2, outputs, 1,
      [count = gate_logits.count](const float* const* f32_inputs,
                                  float* const* f32_outputs) {
        return sigmoid_mul(f32_inputs[0], f32_inputs[1], f32_outputs[0], count);
      },
      workspace);
}

Status sigmoid_mul_backward_storage(FloatStorageInput grad_out,
                                    FloatStorageInput gate_logits,
                                    FloatStorageInput values,
                                    FloatStorageOutput grad_gate,
                                    FloatStorageOutput grad_values,
                                    FloatStorageWorkspace* workspace) {
  if (grad_out.count <= 0 || gate_logits.count != grad_out.count ||
      values.count != grad_out.count || grad_gate.count != grad_out.count ||
      grad_values.count != grad_out.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {grad_out, gate_logits, values};
  const FloatStorageOutput outputs[] = {grad_gate, grad_values};
  return with_float_storage(
      inputs, 3, outputs, 2,
      [count = grad_out.count](const float* const* f32_inputs,
                               float* const* f32_outputs) {
        return sigmoid_mul_backward(f32_inputs[0], f32_inputs[1], f32_inputs[2],
                                    f32_outputs[0], f32_outputs[1], count);
      },
      workspace);
}

Status gdn_recur_storage(FloatStorageInput q, FloatStorageInput k,
                         FloatStorageInput v, FloatStorageInput decay,
                         FloatStorageInput beta, const float* state_pool,
                         const int* cumulative_lengths, const int* slot_mapping,
                         FloatStorageOutput out, float* state_pool_out,
                         long long requests, long long slots,
                         long long key_heads, long long value_heads,
                         long long key_dim, long long value_dim,
                         bool load_initial, FloatStorageWorkspace* workspace) {
  if (!detail::all_nonnull(cumulative_lengths, slot_mapping)) {
    return Status::kInvalidArgument;
  }
  long long total_tokens = 0;
  if (!valid_gdn_metadata(cumulative_lengths, slot_mapping, requests, slots,
                          false, &total_tokens)) {
    return Status::kInvalidArgument;
  }
  if (!detail::valid_product(
          {total_tokens, key_heads, value_heads, key_dim, value_dim}) ||
      q.count != total_tokens * key_heads * key_dim || k.count != q.count ||
      v.count != total_tokens * value_heads * value_dim ||
      decay.count != total_tokens * value_heads || beta.count != decay.count ||
      out.count != v.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q, k, v, decay, beta};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 5, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) {
        return gdn_recur(f32_inputs[0], f32_inputs[1], f32_inputs[2],
                         f32_inputs[3], f32_inputs[4], state_pool,
                         cumulative_lengths, slot_mapping, f32_outputs[0],
                         state_pool_out, requests, slots, key_heads,
                         value_heads, key_dim, value_dim, load_initial);
      },
      workspace);
}

Status gdn_short_conv_storage(
    FloatStorageInput x, FloatStorageInput weight, const float* state_pool,
    const int* cumulative_lengths, const int* slot_mapping,
    FloatStorageOutput out, float* state_pool_out, long long requests,
    long long slots, long long channels, long long kernel_size,
    bool load_initial, bool apply_silu, FloatStorageWorkspace* workspace) {
  if (!detail::all_nonnull(cumulative_lengths, slot_mapping)) {
    return Status::kInvalidArgument;
  }
  long long total_tokens = 0;
  if (!valid_gdn_metadata(cumulative_lengths, slot_mapping, requests, slots,
                          true, &total_tokens)) {
    return Status::kInvalidArgument;
  }
  if (!detail::valid_product({total_tokens, channels, kernel_size}) ||
      x.count != total_tokens * channels ||
      weight.count != channels * kernel_size || out.count != x.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {x, weight};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 2, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) {
        return gdn_short_conv(f32_inputs[0], f32_inputs[1], state_pool,
                              cumulative_lengths, slot_mapping, f32_outputs[0],
                              state_pool_out, requests, slots, channels,
                              kernel_size, load_initial, apply_silu);
      },
      workspace);
}

Status gdn_qkv_prepare_storage(FloatStorageInput mixed, FloatStorageOutput q,
                               FloatStorageOutput k, FloatStorageOutput v,
                               long long tokens, long long key_heads,
                               long long value_heads, long long key_dim,
                               long long value_dim, float eps, float q_scale,
                               float k_scale,
                               FloatStorageWorkspace* workspace) {
  if (!detail::valid_product(
          {tokens, key_heads, value_heads, key_dim, value_dim})) {
    return Status::kInvalidShape;
  }
  const long long qk_count = tokens * key_heads * key_dim;
  const long long value_count = tokens * value_heads * value_dim;
  if (mixed.count != 2 * qk_count + value_count || q.count != qk_count ||
      k.count != qk_count || v.count != value_count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {mixed};
  const FloatStorageOutput outputs[] = {q, k, v};
  return with_float_storage(
      inputs, 1, outputs, 3,
      [&](const float* const* f32_inputs, float* const* f32_outputs) {
        return gdn_qkv_prepare(f32_inputs[0], f32_outputs[0], f32_outputs[1],
                               f32_outputs[2], tokens, key_heads, value_heads,
                               key_dim, value_dim, eps, q_scale, k_scale);
      },
      workspace);
}

Status gdn_gate_beta_storage(FloatStorageInput a, FloatStorageInput b,
                             const float* a_log, const float* dt_bias,
                             float* decay, float* beta, long long tokens,
                             long long value_heads,
                             FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({tokens, value_heads}) ||
      a.count != tokens * value_heads || b.count != a.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {a, b};
  return with_float_storage(
      inputs, 2, nullptr, 0,
      [&](const float* const* f32_inputs, float* const*) {
        return gdn_gate_beta(f32_inputs[0], f32_inputs[1], a_log, dt_bias,
                             decay, beta, tokens, value_heads);
      },
      workspace);
}

Status gdn_gated_rmsnorm_storage(FloatStorageInput y, FloatStorageInput z,
                                 FloatStorageInput weight,
                                 FloatStorageOutput out, long long tokens,
                                 long long value_heads, long long value_dim,
                                 float eps, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({tokens, value_heads, value_dim}) ||
      y.count != tokens * value_heads * value_dim || z.count != y.count ||
      weight.count != value_dim || out.count != y.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {y, z, weight};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 3, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) {
        return gdn_gated_rmsnorm(f32_inputs[0], f32_inputs[1], f32_inputs[2],
                                 f32_outputs[0], tokens, value_heads, value_dim,
                                 eps);
      },
      workspace);
}

}  // namespace quixicore_cpu
