// Portable scalar RMSNorm reference. Sum of squares accumulates in float64
// so the reference is a near-oracle; optimized variants must still meet the
// contract tolerance against the true float64 oracle, not against this.

#include "kernels/norms/rms_norm.h"

#include <cmath>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::norms {
namespace {

// Free function with by-value arguments so the hot loops run on true
// locals; see the codegen note in src/threading/thread_pool.h.
void rms_rows_ref(const float* x, const float* weight, float* y,
                  long long hidden, float eps, long long r0, long long r1) {
  for (long long r = r0; r < r1; ++r) {
    const float* xr = x + r * hidden;
    float* yr = y + r * hidden;
    double sumsq = 0.0;
    for (long long j = 0; j < hidden; ++j) {
      sumsq += static_cast<double>(xr[j]) * xr[j];
    }
    const float scale = static_cast<float>(
        1.0 / std::sqrt(sumsq / static_cast<double>(hidden) +
                        static_cast<double>(eps)));
    for (long long j = 0; j < hidden; ++j) {
      // Normalize before applying weight. xr*weight can overflow even when
      // the mathematically equivalent normalized result is finite.
      yr[j] = (xr[j] * scale) * weight[j];
    }
  }
}

}  // namespace

void rms_norm_ref(const float* x, const float* weight, float* y,
                  long long rows, long long hidden, float eps) {
  threading::parallel_ranges(rows, 8, [&](long long r0, long long r1, int) {
    rms_rows_ref(x, weight, y, hidden, eps, r0, r1);
  });
}

}  // namespace quixicore_cpu::norms
