// Portable scalar RMSNorm reference. Sum of squares accumulates in float64
// so the reference is a near-oracle; optimized variants must still meet the
// contract tolerance against the true float64 oracle, not against this.

#include "kernels/norms/rms_norm.h"

#include <cmath>

namespace quixicore_cpu::norms {

void rms_norm_ref(const float* x, const float* weight, float* y,
                  long long rows, long long hidden, float eps) {
  for (long long r = 0; r < rows; ++r) {
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
      yr[j] = xr[j] * weight[j] * scale;
    }
  }
}

}  // namespace quixicore_cpu::norms
