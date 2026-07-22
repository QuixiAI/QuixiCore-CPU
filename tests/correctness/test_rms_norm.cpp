// Contract correctness for rms_norm: argument validation, float64 oracle at
// fp32 tolerance across shapes with vector tails, variant equivalence,
// determinism (umbrella policy: norms family must be deterministic).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "kernels/norms/rms_norm.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/rms_norm.h"

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

}  // namespace

int main() {
  using quixicore_cpu::Status;
  constexpr float kEps = 1e-5f;

  // Argument validation.
  float dummy = 0.0f;
  REQUIRE(quixicore_cpu::rms_norm(&dummy, &dummy, &dummy, 0, 4, kEps) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::rms_norm(&dummy, &dummy, &dummy, 1, 0, kEps) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::rms_norm(&dummy, &dummy, &dummy, 1, 4, -1.0f) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::rms_norm(&dummy, &dummy, &dummy, 1, 4,
                                  std::numeric_limits<float>::infinity()) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::rms_norm(nullptr, &dummy, &dummy, 1, 4, kEps) ==
          Status::kInvalidArgument);
  REQUIRE(quixicore_cpu::rms_norm(&dummy, &dummy, &dummy, 1LL << 62, 4,
                                  kEps) == Status::kInvalidShape);

  const std::string variant = quixicore_cpu::rms_norm_variant();
#if defined(__aarch64__) || defined(_M_ARM64)
  const bool expect_neon = quixicore_cpu::cpu_features().neon;
#else
  const bool expect_neon = false;
#endif
  REQUIRE(variant == (expect_neon ? "neon" : "ref"));

  const std::vector<std::pair<long long, long long>> shapes = {
      {1, 1}, {1, 7}, {2, 32}, {3, 777}, {4, 2048}, {1, 4096}};

  for (const auto& [rows, hidden] : shapes) {
    Rng rng(0xB5297A4Du ^ static_cast<uint32_t>(rows * 977 + hidden));
    std::vector<float> x(static_cast<size_t>(rows * hidden));
    std::vector<float> w(static_cast<size_t>(hidden));
    std::vector<float> y(static_cast<size_t>(rows * hidden));
    for (auto& v : x) {
      v = rng.next();
    }
    for (auto& v : w) {
      v = rng.next();
    }

    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y.data(), rows,
                                    hidden, kEps) == Status::kOk);

    // Determinism.
    std::vector<float> y2(static_cast<size_t>(rows * hidden));
    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y2.data(), rows,
                                    hidden, kEps) == Status::kOk);
    REQUIRE(std::memcmp(y.data(), y2.data(),
                        static_cast<size_t>(rows * hidden) * sizeof(float)) ==
            0);

    // Float64 oracle, repo convention max|diff| / (max|ref| + 1e-9).
    std::vector<double> oracle(static_cast<size_t>(rows * hidden));
    for (long long r = 0; r < rows; ++r) {
      double sumsq = 0.0;
      for (long long j = 0; j < hidden; ++j) {
        const double v = x[static_cast<size_t>(r * hidden + j)];
        sumsq += v * v;
      }
      const double scale =
          1.0 / std::sqrt(sumsq / static_cast<double>(hidden) +
                          static_cast<double>(kEps));
      for (long long j = 0; j < hidden; ++j) {
        oracle[static_cast<size_t>(r * hidden + j)] =
            static_cast<double>(x[static_cast<size_t>(r * hidden + j)]) *
            w[static_cast<size_t>(j)] * scale;
      }
    }
    // Public entry at the umbrella fp32 tolerance.
    REQUIRE(all_close(y.data(), oracle, 1e-6, 1e-5));

    // Scalar reference directly (float64 accumulation: near-oracle).
    std::vector<float> y_ref(static_cast<size_t>(rows * hidden));
    quixicore_cpu::norms::rms_norm_ref(x.data(), w.data(), y_ref.data(), rows,
                                       hidden, kEps);
    REQUIRE(all_close(y_ref.data(), oracle, 1e-7, 1e-6));

#if defined(__aarch64__) || defined(_M_ARM64)
    if (expect_neon) {
      std::vector<float> y_neon(static_cast<size_t>(rows * hidden));
      quixicore_cpu::norms::rms_norm_neon(x.data(), w.data(), y_neon.data(),
                                          rows, hidden, kEps);
      REQUIRE(all_close(y_neon.data(), oracle, 1e-6, 1e-5));
      // The dispatched public entry must be exactly this variant.
      REQUIRE(std::memcmp(y.data(), y_neon.data(),
                          static_cast<size_t>(rows * hidden) *
                              sizeof(float)) == 0);
    }
#endif

    // A zero row stays finite and zero (eps guards the denominator).
    std::vector<float> xz(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> yz(static_cast<size_t>(hidden), -1.0f);
    REQUIRE(quixicore_cpu::rms_norm(xz.data(), w.data(), yz.data(), 1, hidden,
                                    kEps) == Status::kOk);
    for (long long j = 0; j < hidden; ++j) {
      REQUIRE(yz[static_cast<size_t>(j)] == 0.0f);
    }
  }

  // The optimized f32 fast reduction falls back to f64 only for exceptional
  // magnitude rows, avoiding overflow/underflow without taxing model values.
  for (const float magnitude : {1e20f, 1e30f}) {
    constexpr long long kHidden = 32;
    std::vector<float> x(static_cast<size_t>(kHidden), magnitude);
    std::vector<float> w(static_cast<size_t>(kHidden), 1.0f);
    std::vector<float> y(static_cast<size_t>(kHidden));
    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y.data(), 1, kHidden,
                                    kEps) == Status::kOk);
    for (const float value : y) {
      REQUIRE(std::isfinite(value));
      REQUIRE(std::fabs(value - 1.0f) < 1e-5f);
    }
  }
  {
    constexpr long long kHidden = 32;
    std::vector<float> x(static_cast<size_t>(kHidden), 1e-30f);
    std::vector<float> w(static_cast<size_t>(kHidden), 1.0f);
    std::vector<float> y(static_cast<size_t>(kHidden));
    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y.data(), 1, kHidden,
                                    0.0f) == Status::kOk);
    for (const float value : y) {
      REQUIRE(std::isfinite(value));
      REQUIRE(std::fabs(value - 1.0f) < 1e-5f);
    }
  }
  {
    constexpr long long kHidden = 32;
    std::vector<float> x(static_cast<size_t>(kHidden), 1e30f);
    std::vector<float> w(static_cast<size_t>(kHidden), 1e20f);
    std::vector<float> y(static_cast<size_t>(kHidden));
    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y.data(), 1, kHidden,
                                    kEps) == Status::kOk);
    for (const float value : y) {
      REQUIRE(std::isfinite(value));
      REQUIRE(std::fabs(value / 1e20f - 1.0f) < 1e-5f);
    }
  }

  return 0;
}
