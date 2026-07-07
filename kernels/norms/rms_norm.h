#pragma once

// Kernel-internal interface for RMSNorm. The public API and variant
// dispatch live in src/dispatch/rms_norm.cpp; this header is not installed.

namespace quixicore_cpu::norms {

using RmsNormFn = void (*)(const float* x, const float* weight, float* y,
                           long long rows, long long hidden, float eps);

// Portable scalar reference; float64 sum-of-squares accumulation.
void rms_norm_ref(const float* x, const float* weight, float* y,
                  long long rows, long long hidden, float eps);

#if defined(__aarch64__) || defined(_M_ARM64)
// NEON variant (Advanced SIMD is baseline on aarch64; no extra build
// flags): f32 four-accumulator sum of squares, vectorized scale pass.
void rms_norm_neon(const float* x, const float* weight, float* y,
                   long long rows, long long hidden, float eps);
#endif

}  // namespace quixicore_cpu::norms
