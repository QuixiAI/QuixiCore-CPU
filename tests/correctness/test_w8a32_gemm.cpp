#include "quixicore_cpu/qgemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <=
         3e-5f * std::max(1.0f, std::fabs(expected));
}

bool test_shape(long long m, long long n, long long k) {
  std::vector<std::int8_t> weights(static_cast<std::size_t>(n * k));
  std::vector<float> scales(static_cast<std::size_t>(n));
  std::vector<float> x(static_cast<std::size_t>(m * k));
  std::vector<float> y(static_cast<std::size_t>(m * n));
  for (long long i = 0; i < n * k; ++i) {
    weights[static_cast<std::size_t>(i)] =
        static_cast<std::int8_t>((i * 37 + 11) % 255 - 127);
  }
  for (long long i = 0; i < n; ++i) {
    scales[static_cast<std::size_t>(i)] = 0.001f * (i + 1);
  }
  for (long long i = 0; i < m * k; ++i) {
    x[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 29 + 7) % 101 - 50) / 37.0f;
  }
  if (quixicore_cpu::qgemm_w8a32(weights.data(), scales.data(), x.data(),
                                  y.data(), m, n, k) !=
      quixicore_cpu::Status::kOk) {
    return false;
  }
  for (long long row = 0; row < m; ++row) {
    for (long long output = 0; output < n; ++output) {
      float expected = 0.0f;
      for (long long column = 0; column < k; ++column) {
        expected += x[static_cast<std::size_t>(row * k + column)] *
                    static_cast<float>(
                        weights[static_cast<std::size_t>(output * k + column)]);
      }
      expected *= scales[static_cast<std::size_t>(output)];
      if (!close(y[static_cast<std::size_t>(row * n + output)], expected)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  for (long long k : {1LL, 7LL, 8LL, 15LL, 16LL, 63LL, 64LL, 65LL,
                      100LL, 1408LL}) {
    if (!test_shape(5, 5, k)) {
      std::cerr << "FAIL: W8A32 shape K=" << k << '\n';
      return 1;
    }
  }
  std::cout << "W8A32 GEMM tests passed ("
            << quixicore_cpu::qgemm_w8a32_variant() << ")\n";
  return 0;
}
