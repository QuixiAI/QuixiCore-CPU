// Contract correctness for qgemv_w8a8 (q4_0 weight x int8 activation): the
// family oracle out = dequantize(wq) @ x at the umbrella `quantized` tolerance
// (the extra error over qgemv comes from quantizing the activations), argument
// and format validation, and determinism.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/qgemv_w8a8.h"

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
    const double diff = std::fabs(static_cast<double>(out[i]) - ref[i]);
    if (diff > atol + rtol * std::fabs(ref[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

using namespace quixicore_cpu;

int main() {
  const long long n = 96;
  const long long k = 256;
  Rng rng(1234);
  std::vector<float> weights(static_cast<size_t>(n * k));
  std::vector<float> x(static_cast<size_t>(k));
  for (auto& v : weights) v = rng.next();
  for (auto& v : x) v = rng.next();

  long long bytes = 0;
  REQUIRE(qgemv_w8a8_packed_size(QuantFormat::kQ4_0, n, k, &bytes) ==
          Status::kOk);
  std::vector<uint8_t> packed(static_cast<size_t>(bytes));
  REQUIRE(qgemv_w8a8_pack(QuantFormat::kQ4_0, weights.data(), n, k,
                          packed.data()) == Status::kOk);

  // Oracle: dequantize the packed weights (what the kernel actually multiplies)
  // and do a full-precision matvec in f64. The residual is the activation
  // quantization error, which must stay within the quantized tolerance tier.
  std::vector<float> dequantized(static_cast<size_t>(n * k));
  quant::q4_0_unpack_ref(
      reinterpret_cast<const quant::BlockQ4_0*>(packed.data()), n, k,
      dequantized.data());
  std::vector<double> ref(static_cast<size_t>(n), 0.0);
  for (long long i = 0; i < n; ++i) {
    double sum = 0.0;
    for (long long j = 0; j < k; ++j) {
      sum += static_cast<double>(dequantized[static_cast<size_t>(i * k + j)]) *
             static_cast<double>(x[static_cast<size_t>(j)]);
    }
    ref[static_cast<size_t>(i)] = sum;
  }

  std::vector<float> y(static_cast<size_t>(n), 0.0f);
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, packed.data(), x.data(), y.data(), n,
                     k) == Status::kOk);
  REQUIRE(all_close(y.data(), ref, 3e-2, 3e-2));

  // Determinism.
  std::vector<float> y2(static_cast<size_t>(n), 0.0f);
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, packed.data(), x.data(), y2.data(), n,
                     k) == Status::kOk);
  for (long long i = 0; i < n; ++i) {
    REQUIRE(y[static_cast<size_t>(i)] == y2[static_cast<size_t>(i)]);
  }

  // Format + argument validation.
  REQUIRE(qgemv_w8a8(QuantFormat::kQ8_0, packed.data(), x.data(), y.data(), n,
                     k) == Status::kUnsupportedFormat);
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, packed.data(), x.data(), y.data(), n,
                     k + 1) == Status::kInvalidShape);
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, nullptr, x.data(), y.data(), n, k) ==
          Status::kInvalidArgument);

  std::cout << "qgemv_w8a8 (q4_0 weight x int8 activation): all checks passed\n";
  return 0;
}
