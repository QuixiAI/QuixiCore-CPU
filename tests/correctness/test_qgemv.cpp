// Contract correctness for qgemv (q8_0): argument validation, pack/unpack
// roundtrip bounds, the family oracle out = dequantize(wq) @ x at umbrella
// tolerances, per-variant checks, determinism.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/qgemv.h"

// Tests must fail in any build configuration, so no assert()/NDEBUG.
#define REQUIRE(cond)                                                     \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ \
                << '\n';                                                  \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

class Rng {
 public:
  explicit Rng(uint32_t seed) : state_(seed) {}
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  uint32_t state_;
};

bool all_close(const float* out, const std::vector<double>& ref, double atol,
               double rtol) {
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(out[i]) || !std::isfinite(ref[i]) ||
        std::fabs(static_cast<double>(out[i]) - ref[i]) >
            atol + rtol * std::fabs(ref[i])) {
      return false;
    }
  }
  return true;
}

bool global_relative_below(const float* out, const std::vector<double>& ref,
                           double tolerance) {
  double max_abs = 0.0;
  double max_ref = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(out[i]) || !std::isfinite(ref[i])) {
      return false;
    }
    max_abs = std::max(
        max_abs, std::fabs(static_cast<double>(out[i]) - ref[i]));
    max_ref = std::max(max_ref, std::fabs(ref[i]));
  }
  return max_abs / (max_ref + 1e-9) < tolerance;
}

}  // namespace

int main() {
  using quixicore_cpu::QuantFormat;
  using quixicore_cpu::Status;

  // fp16 conversion sanity.
  REQUIRE(quixicore_cpu::fp16_to_fp32(quixicore_cpu::fp32_to_fp16(1.0f)) ==
          1.0f);
  REQUIRE(quixicore_cpu::fp16_to_fp32(quixicore_cpu::fp32_to_fp16(0.5f)) ==
          0.5f);
  REQUIRE(quixicore_cpu::fp16_to_fp32(quixicore_cpu::fp32_to_fp16(0.0f)) ==
          0.0f);
  REQUIRE(quixicore_cpu::fp16_to_fp32(quixicore_cpu::fp32_to_fp16(65504.0f)) ==
          65504.0f);
  REQUIRE(quixicore_cpu::fp32_to_fp16(std::ldexp(1.0f, -14)) == 0x0400u);
  REQUIRE(quixicore_cpu::fp32_to_fp16(std::ldexp(1.0f, -15)) == 0x0200u);
  REQUIRE(quixicore_cpu::fp32_to_fp16(std::ldexp(1.0f, -24)) == 0x0001u);
  REQUIRE(quixicore_cpu::fp32_to_fp16(std::ldexp(1.0f, -25)) == 0x0000u);
  REQUIRE(quixicore_cpu::fp32_to_fp16(-std::ldexp(1.0f, -24)) == 0x8001u);
  {
    const float v = 0.0123456f;
    const float rt =
        quixicore_cpu::fp16_to_fp32(quixicore_cpu::fp32_to_fp16(v));
    REQUIRE(std::fabs(rt - v) / v < 1e-3f);  // half has ~11 bits of mantissa
  }

  // Argument validation.
  float dummy = 0.0f;
  size_t size = 0;
  REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, 4, 33, &size) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, 0, 32, &size) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, nullptr, nullptr, nullptr,
                               4, 33) == Status::kInvalidShape);
  REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, 1, 32,
                                           nullptr) ==
          Status::kInvalidArgument);
  REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, 1LL << 62, 128,
                                           &size) == Status::kInvalidShape);
  REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, nullptr, &dummy, &dummy, 1,
                               32) == Status::kInvalidArgument);

  // Packed size formula: 34 bytes per 32 weights.
  REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, 4, 64, &size) ==
          Status::kOk);
  REQUIRE(size == 4 * 2 * 34);
  REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, 4, 64,
                                           &size) == Status::kOk);
  REQUIRE(size == 4 * 2 * 18);

  // Packing must reject non-finite weights before any float-to-int cast.
  {
    std::vector<float> bad(32, 1.0f);
    std::vector<uint8_t> packed(34);
    bad[7] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, bad.data(), 1, 32,
                                      packed.data()) ==
            Status::kInvalidArgument);
    bad[7] = std::numeric_limits<float>::infinity();
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, bad.data(), 1, 32,
                                      packed.data()) ==
            Status::kInvalidArgument);
    bad[7] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, bad.data(), 1, 32,
                                      packed.data()) ==
            Status::kInvalidArgument);
  }

  // q4_0 weight-only GEMV uses the same full-precision-activation contract.
  {
    const long long n = 33;
    const long long k = 512;
    Rng rng(0xD1B54A32u);
    std::vector<float> weights(static_cast<std::size_t>(n * k));
    std::vector<float> x(static_cast<std::size_t>(k));
    for (auto& value : weights) {
      value = rng.next();
    }
    for (auto& value : x) {
      value = rng.next();
    }
    REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, n, k,
                                             &size) == Status::kOk);
    std::vector<std::uint8_t> packed(size);
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, weights.data(), n,
                                      k, packed.data()) == Status::kOk);
    std::vector<float> dequantized(static_cast<std::size_t>(n * k));
    REQUIRE(quixicore_cpu::qgemv_unpack(QuantFormat::kQ4_0, packed.data(), n,
                                        k, dequantized.data()) == Status::kOk);
    std::vector<float> output(static_cast<std::size_t>(n));
    REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ4_0, packed.data(), x.data(),
                                 output.data(), n, k) == Status::kOk);
    std::vector<double> reference(static_cast<std::size_t>(n));
    for (long long row = 0; row < n; ++row) {
      double sum = 0.0;
      for (long long column = 0; column < k; ++column) {
        sum += dequantized[static_cast<std::size_t>(row * k + column)] *
               static_cast<double>(x[static_cast<std::size_t>(column)]);
      }
      reference[static_cast<std::size_t>(row)] = sum;
    }
    REQUIRE(all_close(output.data(), reference, 1e-5, 1e-4));
    REQUIRE(std::string(quixicore_cpu::qgemv_variant(QuantFormat::kQ4_0)) ==
            "ref");
  }

  // A scale below the minimum normal fp16 value is still a valid fp16
  // subnormal. This used to pack d=0 and erase the entire block.
  {
    std::vector<float> w(32, 1e-3f);
    std::vector<float> x(32, 1.0f);
    std::vector<float> dq(32);
    std::vector<float> y(1);
    std::vector<uint8_t> packed(34);
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, w.data(), 1, 32,
                                      packed.data()) == Status::kOk);
    REQUIRE(quixicore_cpu::qgemv_unpack(QuantFormat::kQ8_0, packed.data(), 1,
                                        32, dq.data()) == Status::kOk);
    REQUIRE(dq[0] != 0.0f);
    REQUIRE(std::fabs(dq[0] - w[0]) < 1e-5f);
    REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, packed.data(), x.data(),
                                 y.data(), 1, 32) == Status::kOk);
    REQUIRE(std::fabs(y[0] - 32.0f * dq[0]) < 1e-6f);
  }

  // The contract path exposes fp32-activation variants only; the
  // activation-quantizing dotprod_i8 path is never publicly selectable.
  const std::string variant =
      quixicore_cpu::qgemv_variant(QuantFormat::kQ8_0);
#if defined(__aarch64__) || defined(_M_ARM64)
  const bool expect_neon = quixicore_cpu::cpu_features().neon;
#else
  const bool expect_neon = false;
#endif
  REQUIRE(variant == (expect_neon ? "neon" : "ref"));

  const std::vector<std::pair<long long, long long>> shapes = {
      {1, 32}, {3, 64}, {5, 96}, {33, 512}, {129, 2048}};

  for (const auto& [n, k] : shapes) {
    Rng rng(0x9E3779B9u ^ static_cast<uint32_t>(n * 131 + k));
    std::vector<float> w(static_cast<size_t>(n * k));
    std::vector<float> x(static_cast<size_t>(k));
    for (auto& v : w) {
      v = rng.next();
    }
    for (auto& v : x) {
      v = rng.next();
    }

    REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, n, k,
                                             &size) == Status::kOk);
    std::vector<uint8_t> packed(size);
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, w.data(), n, k,
                                      packed.data()) == Status::kOk);
    const auto* blocks =
        reinterpret_cast<const quixicore_cpu::quant::BlockQ8_0*>(
            packed.data());

    // Pack/unpack roundtrip: q8_0 step is amax/127 per block, so the
    // element error is bounded by ~0.5% of the block amax (plus fp16 scale
    // rounding). Check against 1% of the global amax.
    std::vector<float> dq(static_cast<size_t>(n * k));
    REQUIRE(quixicore_cpu::qgemv_unpack(QuantFormat::kQ8_0, packed.data(), n,
                                        k, dq.data()) == Status::kOk);
    float amax = 0.0f;
    double max_round = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
      amax = std::fmax(amax, std::fabs(w[i]));
      max_round = std::fmax(max_round,
                            std::fabs(static_cast<double>(w[i]) - dq[i]));
    }
    REQUIRE(max_round < 0.01 * amax);

    std::vector<float> y(static_cast<size_t>(n));
    REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, packed.data(), x.data(),
                                 y.data(), n, k) == Status::kOk);

    // Determinism: bit-identical on rerun (umbrella policy for the
    // quantization family).
    std::vector<float> y2(static_cast<size_t>(n));
    REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, packed.data(), x.data(),
                                 y2.data(), n, k) == Status::kOk);
    REQUIRE(std::memcmp(y.data(), y2.data(), n * sizeof(float)) == 0);

    // Family oracles, repo error convention max|diff| / (max|ref| + 1e-9):
    //   acc_dq   — float64 dequantize(wq) @ x, THE qgemv contract oracle,
    //   acc_orig — float64 GEMV over the original weights (total error at
    //              the umbrella quantized tolerance).
    std::vector<double> acc_dq(static_cast<size_t>(n));
    std::vector<double> acc_orig(static_cast<size_t>(n));
    for (long long i = 0; i < n; ++i) {
      double sum_dq = 0.0;
      double sum_orig = 0.0;
      for (long long j = 0; j < k; ++j) {
        sum_dq += static_cast<double>(dq[static_cast<size_t>(i * k + j)]) *
                  x[static_cast<size_t>(j)];
        sum_orig += static_cast<double>(w[static_cast<size_t>(i * k + j)]) *
                    x[static_cast<size_t>(j)];
      }
      acc_dq[static_cast<size_t>(i)] = sum_dq;
      acc_orig[static_cast<size_t>(i)] = sum_orig;
    }
    // Public entry: fp32-activation contract numerics on every platform.
    REQUIRE(all_close(y.data(), acc_dq, 1e-5, 1e-4));
    REQUIRE(global_relative_below(y.data(), acc_orig, 0.03));

    // Scalar reference directly.
    std::vector<float> y_ref(static_cast<size_t>(n));
    quixicore_cpu::quant::q8_0_gemv_ref(blocks, x.data(), y_ref.data(), n, k);
    REQUIRE(all_close(y_ref.data(), acc_dq, 1e-5, 1e-4));

#if defined(__aarch64__) || defined(_M_ARM64)
    if (expect_neon) {
      // NEON contract variant: same oracle bound, and the dispatched public
      // entry must be exactly this variant.
      std::vector<float> y_neon(static_cast<size_t>(n));
      quixicore_cpu::quant::q8_0_gemv_neon(blocks, x.data(), y_neon.data(), n,
                                           k);
      REQUIRE(all_close(y_neon.data(), acc_dq, 1e-5, 1e-4));
      REQUIRE(std::memcmp(y.data(), y_neon.data(), n * sizeof(float)) == 0);
    }
#endif

#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    if (quixicore_cpu::cpu_features().dotprod) {
      // dotprod_i8 (internal experiment): quantizes activations to int8
      // blocks; its tight oracle is float64 GEMV over dequantized weights AND
      // dequantized quantized activations (the int dot itself is exact).
      std::vector<float> yd(static_cast<size_t>(n));
      quixicore_cpu::quant::q8_0_gemv_dotprod(blocks, x.data(), yd.data(), n,
                                              k);

      // Replicate the activation quantization spec: d = amax/127, nearbyint.
      std::vector<double> dqx(static_cast<size_t>(k));
      for (long long b = 0; b < k / 32; ++b) {
        float xmax = 0.0f;
        for (long long j = 0; j < 32; ++j) {
          xmax =
              std::fmax(xmax, std::fabs(x[static_cast<size_t>(b * 32 + j)]));
        }
        const float d = xmax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (long long j = 0; j < 32; ++j) {
          const float q =
              std::nearbyint(x[static_cast<size_t>(b * 32 + j)] * id);
          dqx[static_cast<size_t>(b * 32 + j)] =
              static_cast<double>(d) *
              (q < -127.0f ? -127.0f : (q > 127.0f ? 127.0f : q));
        }
      }
      std::vector<double> acc_tight(static_cast<size_t>(n));
      for (long long i = 0; i < n; ++i) {
        double sum = 0.0;
        for (long long j = 0; j < k; ++j) {
          sum += static_cast<double>(dq[static_cast<size_t>(i * k + j)]) *
                 dqx[static_cast<size_t>(j)];
        }
        acc_tight[static_cast<size_t>(i)] = sum;
      }
      REQUIRE(all_close(yd.data(), acc_tight, 1e-5, 1e-5));
      REQUIRE(global_relative_below(yd.data(), acc_orig, 0.03));

      std::vector<float> yd2(static_cast<size_t>(n));
      quixicore_cpu::quant::q8_0_gemv_dotprod(blocks, x.data(), yd2.data(), n,
                                              k);
      REQUIRE(std::memcmp(yd.data(), yd2.data(), n * sizeof(float)) == 0);
    }
#endif
  }

  return 0;
}
