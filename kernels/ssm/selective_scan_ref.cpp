#include "quixicore_cpu/ops.h"

#include <cmath>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status selective_scan(const float* u, const float* delta, const float* a,
                      const float* b, const float* c, const float* d, float* y,
                      long long channels, long long sequence,
                      long long state_size) {
  if (!detail::valid_product({channels, sequence}) ||
      !detail::valid_product({channels, state_size}) ||
      !detail::valid_product({sequence, state_size})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(u, delta, a, b, c, d, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(channels, 1,
                             [&](long long begin, long long end, int) {
    std::vector<double> hidden(static_cast<std::size_t>(state_size));
    for (long long channel = begin; channel < end; ++channel) {
      std::fill(hidden.begin(), hidden.end(), 0.0);
      for (long long token = 0; token < sequence; ++token) {
        const double input = u[channel * sequence + token];
        const double step = delta[channel * sequence + token];
        double output = static_cast<double>(d[channel]) * input;
        for (long long state = 0; state < state_size; ++state) {
          const long long channel_state = channel * state_size + state;
          const long long token_state = token * state_size + state;
          hidden[static_cast<std::size_t>(state)] =
              std::exp(step * a[channel_state]) *
                  hidden[static_cast<std::size_t>(state)] +
              step * b[token_state] * input;
          output += c[token_state] * hidden[static_cast<std::size_t>(state)];
        }
        y[channel * sequence + token] = static_cast<float>(output);
      }
    }
  });
  return Status::kOk;
}

Status selective_scan_varlen(
    const float* u, const float* delta, const float* a, const float* b,
    const float* c, const float* d, const float* delta_bias,
    const float* z, const int* cumulative_lengths,
    const int* cache_indices, const std::uint8_t* has_initial_state,
    const int* checkpoint_slots, float* state_pool, float* out,
    long long batch, long long slots, long long channels,
    long long total_tokens, long long state_size, long long groups,
    bool delta_softplus, int null_slot) {
  if (!detail::valid_product({batch, slots, channels, total_tokens,
                              state_size, groups}) ||
      channels % groups != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(u, delta, a, b, c, cumulative_lengths,
                           cache_indices, has_initial_state, state_pool,
                           out)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_lengths[0] != 0 ||
      cumulative_lengths[batch] != total_tokens) {
    return Status::kInvalidArgument;
  }
  const long long channels_per_group = channels / groups;
  for (long long request = 0; request < batch; ++request) {
    const int begin = cumulative_lengths[request];
    const int end = cumulative_lengths[request + 1];
    const int slot = cache_indices[request];
    if (begin < 0 || end < begin || end > total_tokens ||
        (slot != null_slot && (slot < 0 || slot >= slots))) {
      return Status::kInvalidArgument;
    }
    if (slot == null_slot) {
      for (long long channel = 0; channel < channels; ++channel) {
        std::fill(out + channel * total_tokens + begin,
                  out + channel * total_tokens + end, 0.0f);
      }
      continue;
    }
    for (long long channel = 0; channel < channels; ++channel) {
      const long long group = channel / channels_per_group;
      std::vector<double> hidden(static_cast<std::size_t>(state_size), 0.0);
      if (has_initial_state[request]) {
        for (long long state = 0; state < state_size; ++state) {
          hidden[state] = state_pool[
              (static_cast<long long>(slot) * channels + channel) * state_size +
              state];
        }
      }
      for (int token = begin; token < end; ++token) {
        double step = delta[channel * total_tokens + token] +
                      (delta_bias != nullptr ? delta_bias[channel] : 0.0f);
        if (delta_softplus) {
          step = step > 20.0 ? step : std::log1p(std::exp(step));
        }
        const double input = u[channel * total_tokens + token];
        double output_value = d != nullptr ? d[channel] * input : 0.0;
        for (long long state = 0; state < state_size; ++state) {
          const long long channel_state = channel * state_size + state;
          const long long token_state =
              (group * state_size + state) * total_tokens + token;
          hidden[state] = std::exp(step * a[channel_state]) * hidden[state] +
                          step * b[token_state] * input;
          output_value += c[token_state] * hidden[state];
        }
        if (z != nullptr) {
          const double gate = z[channel * total_tokens + token];
          output_value *= gate / (1.0 + std::exp(-gate));
        }
        out[channel * total_tokens + token] = static_cast<float>(output_value);
        if (checkpoint_slots != nullptr && checkpoint_slots[token] >= 0) {
          const int checkpoint = checkpoint_slots[token];
          if (checkpoint >= slots) return Status::kInvalidArgument;
          for (long long state = 0; state < state_size; ++state) {
            state_pool[(static_cast<long long>(checkpoint) * channels + channel) *
                           state_size +
                       state] = static_cast<float>(hidden[state]);
          }
        }
      }
      for (long long state = 0; state < state_size; ++state) {
        state_pool[(static_cast<long long>(slot) * channels + channel) *
                       state_size +
                   state] = static_cast<float>(hidden[state]);
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
