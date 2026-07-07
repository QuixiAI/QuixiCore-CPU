#pragma once

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Row-wise RMSNorm: y[r][j] = x[r][j] * weight[j] / sqrt(mean_j(x[r]^2) + eps)
// over `rows` rows of length `hidden`, f32 in/out. Deterministic (umbrella
// policy for the norms family). The kernel variant is resolved once per
// process from runtime CPU features; set QUIXICORE_CPU_RMS_NORM_VARIANT to
// force a named variant for testing and benchmarking.
Status rms_norm(const float* x, const float* weight, float* y, long long rows,
                long long hidden, float eps);

// Name of the variant rms_norm resolves to ("ref", "neon", ...).
const char* rms_norm_variant();

}  // namespace quixicore_cpu
