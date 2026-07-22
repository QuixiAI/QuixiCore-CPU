// Contract correctness for qgemv_w8a8 (q4_0 weight x int8 activation): the
// exact oracle out = dequantize(wq) @ dequantize(blockwise_int8(x)), argument
// and format validation, and determinism.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/qgemv_w8a8.h"
#include "quixicore_cpu/threading.h"

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

  std::vector<double> quantized_x(static_cast<std::size_t>(k));
  for (long long block = 0; block < k / 32; ++block) {
    float maximum = 0.0f;
    for (long long element = 0; element < 32; ++element) {
      maximum = std::fmax(
          maximum, std::fabs(x[static_cast<std::size_t>(block * 32 + element)]));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (long long element = 0; element < 32; ++element) {
      const float rounded = std::nearbyint(
          x[static_cast<std::size_t>(block * 32 + element)] * inverse);
      const float clamped = rounded < -127.0f
                                ? -127.0f
                                : (rounded > 127.0f ? 127.0f : rounded);
      quantized_x[static_cast<std::size_t>(block * 32 + element)] =
          static_cast<double>(scale) * clamped;
    }
  }

  for (const QuantFormat format : {QuantFormat::kQ4_0, QuantFormat::kQ8_0}) {
    long long bytes = 0;
    REQUIRE(qgemv_w8a8_packed_size(format, n, k, &bytes) == Status::kOk);
    std::vector<uint8_t> packed(static_cast<size_t>(bytes));
    REQUIRE(qgemv_w8a8_pack(format, weights.data(), n, k, packed.data()) ==
            Status::kOk);

    // Tight oracle: dequantize both packed weights and the specified
    // per-block int8 activations, then multiply in f64.
    std::vector<float> dequantized(static_cast<size_t>(n * k));
    REQUIRE(qgemv_unpack(format, packed.data(), n, k, dequantized.data()) ==
            Status::kOk);
    std::vector<double> ref(static_cast<size_t>(n), 0.0);
    for (long long row = 0; row < n; ++row) {
      double sum = 0.0;
      for (long long column = 0; column < k; ++column) {
        sum += static_cast<double>(
                   dequantized[static_cast<size_t>(row * k + column)]) *
               quantized_x[static_cast<size_t>(column)];
      }
      ref[static_cast<size_t>(row)] = sum;
    }

    std::vector<float> y(static_cast<size_t>(n), 0.0f);
    REQUIRE(qgemv_w8a8(format, packed.data(), x.data(), y.data(), n, k) ==
            Status::kOk);
    REQUIRE(all_close(y.data(), ref, 1e-4, 1e-4));

    std::vector<float> y2(static_cast<size_t>(n), 0.0f);
    quixicore_cpu::set_num_threads(4);
    REQUIRE(qgemv_w8a8(format, packed.data(), x.data(), y2.data(), n, k) ==
            Status::kOk);
    quixicore_cpu::set_num_threads(1);
    REQUIRE(std::memcmp(y.data(), y2.data(), n * sizeof(float)) == 0);

    const std::string variant = qgemv_w8a8_variant(format);
    REQUIRE(variant == "ref" || variant == "dotprod");
  }

  long long bytes = 0;
  REQUIRE(qgemv_w8a8_packed_size(QuantFormat::kQ4_0, n, k, &bytes) ==
          Status::kOk);
  std::vector<uint8_t> packed(static_cast<size_t>(bytes));
  REQUIRE(qgemv_w8a8_pack(QuantFormat::kQ4_0, weights.data(), n, k,
                          packed.data()) == Status::kOk);
  std::vector<float> output(static_cast<size_t>(n));
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, packed.data(), x.data(), output.data(),
                     n, k + 1) == Status::kInvalidShape);
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, nullptr, x.data(), output.data(), n,
                     k) == Status::kInvalidArgument);
  x[7] = std::numeric_limits<float>::quiet_NaN();
  REQUIRE(qgemv_w8a8(QuantFormat::kQ4_0, packed.data(), x.data(), output.data(),
                     n, k) == Status::kInvalidArgument);

  std::cout << "qgemv_w8a8 (q4_0/q8_0 weight x int8 activation): all checks "
               "passed\n";
  return 0;
}
