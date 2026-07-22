#include "quixicore_cpu/ops.h"

#include <cmath>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status layer_norm(const float* x, const float* weight, const float* bias,
                  float* y, long long rows, long long hidden, float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 8, [&](long long r0, long long r1, int) {
    for (long long row = r0; row < r1; ++row) {
      const float* in = x + row * hidden;
      float* out = y + row * hidden;
      // Welford avoids catastrophic cancellation for rows with a large mean
      // and small variance while retaining deterministic iteration order.
      double mean = 0.0;
      double m2 = 0.0;
      for (long long i = 0; i < hidden; ++i) {
        const double value = in[i];
        const double delta = value - mean;
        mean += delta / static_cast<double>(i + 1);
        m2 += delta * (value - mean);
      }
      const double variance = m2 / static_cast<double>(hidden);
      const float inverse =
          static_cast<float>(1.0 / std::sqrt(variance + eps));
      for (long long i = 0; i < hidden; ++i) {
        const float normalized = static_cast<float>(
            (static_cast<double>(in[i]) - mean) * inverse);
        out[i] = normalized * weight[i] + (bias != nullptr ? bias[i] : 0.0f);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
