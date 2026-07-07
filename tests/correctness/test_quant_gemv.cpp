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

  REQUIRE(std::string(quixicore_cpu::quant_gemv_variant(QuantFormat::kQ8_0)) ==
          "ref");

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

    // Oracle 1 (tight): float64 GEMV over the dequantized weights isolates
    // the kernel's own accumulation error.
    // Oracle 2 (contract): float64 GEMV over the original weights bounds
    // total quantization + kernel error at the quantized tolerance.
    // Both use the repo error convention: max|diff| / (max|ref| + 1e-9).
    double max_abs_dq = 0.0;
    double max_ref_dq = 0.0;
    double max_abs_orig = 0.0;
    double max_ref_orig = 0.0;
    for (long long i = 0; i < n; ++i) {
      double acc_dq = 0.0;
      double acc_orig = 0.0;
      for (long long j = 0; j < k; ++j) {
        acc_dq += static_cast<double>(dq[static_cast<size_t>(i * k + j)]) *
                  x[static_cast<size_t>(j)];
        acc_orig += static_cast<double>(w[static_cast<size_t>(i * k + j)]) *
                    x[static_cast<size_t>(j)];
      }
      const double out = y[static_cast<size_t>(i)];
      max_abs_dq = std::fmax(max_abs_dq, std::fabs(out - acc_dq));
      max_ref_dq = std::fmax(max_ref_dq, std::fabs(acc_dq));
      max_abs_orig = std::fmax(max_abs_orig, std::fabs(out - acc_orig));
      max_ref_orig = std::fmax(max_ref_orig, std::fabs(acc_orig));
    }
    REQUIRE(max_abs_dq / (max_ref_dq + 1e-9) < 1e-4);
    REQUIRE(max_abs_orig / (max_ref_orig + 1e-9) < 0.03);
  }

  return 0;
}
