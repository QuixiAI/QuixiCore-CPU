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

}  // namespace quixicore_cpu
