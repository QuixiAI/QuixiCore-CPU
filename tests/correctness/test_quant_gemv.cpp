// Contract correctness for quant_gemv (q8_0): argument validation, pack/
// unpack roundtrip bounds, GEMV against a float64 oracle at the umbrella
// tolerances (registry/tolerances.yaml: quantized rtol 0.03; fp32-accum
// error vs the dequantized oracle must be far tighter), determinism.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/quant_gemv.h"

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
  {
    const float v = 0.0123456f;
    const float rt =
        quixicore_cpu::fp16_to_fp32(quixicore_cpu::fp32_to_fp16(v));
    REQUIRE(std::fabs(rt - v) / v < 1e-3f);  // half has ~11 bits of mantissa
  }

  // Argument validation.
  size_t size = 0;
  REQUIRE(quixicore_cpu::quant_gemv_packed_size(QuantFormat::kQ8_0, 4, 33,
                                                &size) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::quant_gemv_packed_size(QuantFormat::kQ8_0, 0, 32,
                                                &size) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::quant_gemv(QuantFormat::kQ8_0, nullptr, nullptr,
                                    nullptr, 4, 33) == Status::kInvalidShape);

  // Packed size formula: 34 bytes per 32 weights.
  REQUIRE(quixicore_cpu::quant_gemv_packed_size(QuantFormat::kQ8_0, 4, 64,
                                                &size) == Status::kOk);
  REQUIRE(size == 4 * 2 * 34);

  const std::string variant =
      quixicore_cpu::quant_gemv_variant(QuantFormat::kQ8_0);
  const bool expect_dotprod =
#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
      quixicore_cpu::cpu_features().dotprod;
#else
      false;
#endif
  REQUIRE(variant == (expect_dotprod ? "dotprod" : "ref"));

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

    REQUIRE(quixicore_cpu::quant_gemv_packed_size(QuantFormat::kQ8_0, n, k,
                                                  &size) == Status::kOk);
    std::vector<uint8_t> packed(size);
    REQUIRE(quixicore_cpu::quant_gemv_pack(QuantFormat::kQ8_0, w.data(), n, k,
                                           packed.data()) == Status::kOk);

    // Pack/unpack roundtrip: q8_0 step is amax/127 per block, so the
    // element error is bounded by ~0.5% of the block amax (plus fp16 scale
    // rounding). Check against 1% of the global amax.
    std::vector<float> dq(static_cast<size_t>(n * k));
    REQUIRE(quixicore_cpu::quant_gemv_unpack(QuantFormat::kQ8_0, packed.data(),
                                             n, k, dq.data()) == Status::kOk);
    float amax = 0.0f;
    double max_round = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
      amax = std::fmax(amax, std::fabs(w[i]));
      max_round = std::fmax(max_round,
                            std::fabs(static_cast<double>(w[i]) - dq[i]));
    }
    REQUIRE(max_round < 0.01 * amax);

    std::vector<float> y(static_cast<size_t>(n));
    REQUIRE(quixicore_cpu::quant_gemv(QuantFormat::kQ8_0, packed.data(),
                                      x.data(), y.data(), n, k) ==
            Status::kOk);

    // Determinism: bit-identical on rerun (umbrella policy for the
    // quantization family).
    std::vector<float> y2(static_cast<size_t>(n));
    REQUIRE(quixicore_cpu::quant_gemv(QuantFormat::kQ8_0, packed.data(),
                                      x.data(), y2.data(), n, k) ==
            Status::kOk);
    REQUIRE(std::memcmp(y.data(), y2.data(), n * sizeof(float)) == 0);

    // Float64 oracles, repo error convention max|diff| / (max|ref| + 1e-9):
    //   acc_dq   — GEMV over dequantized weights, f32 activations (isolates
    //              kernel accumulation error for f32-activation paths),
    //   acc_orig — GEMV over the original weights (contract bound: total
    //              quantization + kernel error at the quantized tolerance).
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
    const auto max_rel = [n](const float* out, const std::vector<double>& ref) {
      double max_abs = 0.0;
      double max_r = 0.0;
      for (long long i = 0; i < n; ++i) {
        max_abs = std::fmax(
            max_abs, std::fabs(out[static_cast<size_t>(i)] -
                               ref[static_cast<size_t>(i)]));
        max_r = std::fmax(max_r, std::fabs(ref[static_cast<size_t>(i)]));
      }
      return max_abs / (max_r + 1e-9);
    };

    // Public entry (whatever variant dispatch resolved): the contract bound
    // applies to every variant. The tight f32-activation oracle only makes
    // sense for f32-activation paths — for activation-quantizing variants
    // (dotprod) it measures quantization error against a possibly tiny
    // output and is checked instead via bit-exact equivalence to the direct
    // variant call plus that variant's own oracle below.
    REQUIRE(max_rel(y.data(), acc_orig) < 0.03);
    if (!expect_dotprod) {
      REQUIRE(max_rel(y.data(), acc_dq) < 1e-4);
    }

    // The scalar reference, called directly, always meets the tight bound.
    std::vector<float> y_ref(static_cast<size_t>(n));
    quixicore_cpu::qgemv::q8_0_gemv_ref(
        reinterpret_cast<const quixicore_cpu::qgemv::BlockQ8_0*>(
            packed.data()),
        x.data(), y_ref.data(), n, k);
    REQUIRE(max_rel(y_ref.data(), acc_dq) < 1e-4);
    REQUIRE(max_rel(y_ref.data(), acc_orig) < 0.03);

#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    if (expect_dotprod) {
      // The dotprod variant quantizes activations to int8 blocks; its tight
      // oracle is float64 GEMV over dequantized weights AND dequantized
      // quantized activations (the int dot itself is exact).
      std::vector<float> yd(static_cast<size_t>(n));
      quixicore_cpu::qgemv::q8_0_gemv_dotprod(
          reinterpret_cast<const quixicore_cpu::qgemv::BlockQ8_0*>(
              packed.data()),
          x.data(), yd.data(), n, k);

      // The dispatched public entry must be exactly this variant.
      REQUIRE(std::memcmp(y.data(), yd.data(), n * sizeof(float)) == 0);

      // Replicate the activation quantization spec: d = amax/127, nearbyint.
      std::vector<double> dqx(static_cast<size_t>(k));
      for (long long b = 0; b < k / 32; ++b) {
        float amax = 0.0f;
        for (long long j = 0; j < 32; ++j) {
          amax = std::fmax(amax, std::fabs(x[static_cast<size_t>(b * 32 + j)]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (long long j = 0; j < 32; ++j) {
          const float q = std::nearbyint(x[static_cast<size_t>(b * 32 + j)] * id);
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
      REQUIRE(max_rel(yd.data(), acc_tight) < 1e-5);
      REQUIRE(max_rel(yd.data(), acc_orig) < 0.03);

      // Determinism for the dotprod path.
      std::vector<float> yd2(static_cast<size_t>(n));
      quixicore_cpu::qgemv::q8_0_gemv_dotprod(
          reinterpret_cast<const quixicore_cpu::qgemv::BlockQ8_0*>(
              packed.data()),
          x.data(), yd2.data(), n, k);
      REQUIRE(std::memcmp(yd.data(), yd2.data(), n * sizeof(float)) == 0);
    }
#endif
  }

  return 0;
}
