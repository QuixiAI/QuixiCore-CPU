#include "quixicore_cpu/qgemm.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "quixicore_cpu/cpu_features.h"

namespace {

std::uint32_t state = 0x9E3779B9u;

std::uint32_t random_u32() {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

bool check_shape(long long m, long long n, long long k, bool asymmetric) {
  std::vector<std::int8_t> weights(static_cast<std::size_t>(n * k));
  std::vector<std::int8_t> x(static_cast<std::size_t>(m * k));
  std::vector<float> weight_scale(static_cast<std::size_t>(n));
  std::vector<float> activation_scale(static_cast<std::size_t>(m));
  std::vector<std::int32_t> row_sum(static_cast<std::size_t>(n));
  std::vector<int> zero_point(static_cast<std::size_t>(m));
  for (std::int8_t& value : weights) {
    value = static_cast<std::int8_t>(random_u32() & 255u);
  }
  for (std::int8_t& value : x) {
    value = static_cast<std::int8_t>(int(random_u32() % 255u) - 127);
  }
  if (!weights.empty()) weights[0] = -128;
  for (long long output = 0; output < n; ++output) {
    weight_scale[output] = 0.001f * static_cast<float>(output + 1);
    std::int32_t sum = 0;
    for (long long input = 0; input < k; ++input) {
      sum += weights[output * k + input];
    }
    row_sum[output] = sum;
  }
  for (long long row = 0; row < m; ++row) {
    activation_scale[row] = 0.002f * static_cast<float>(row + 1);
    zero_point[row] = asymmetric ? int(row) - 2 : 0;
  }
  std::vector<float> output(static_cast<std::size_t>(m * n));
  if (quixicore_cpu::int8_gemm(
          weights.data(), x.data(), weight_scale.data(),
          activation_scale.data(), asymmetric ? row_sum.data() : nullptr,
          asymmetric ? zero_point.data() : nullptr, output.data(), m, n, k,
          asymmetric) != quixicore_cpu::Status::kOk) {
    std::cerr << "FAIL: int8_gemm rejected M=" << m << " N=" << n
              << " K=" << k << '\n';
    return false;
  }
  for (long long row = 0; row < m; ++row) {
    for (long long column = 0; column < n; ++column) {
      std::int64_t dot = 0;
      for (long long input = 0; input < k; ++input) {
        dot += static_cast<int>(weights[column * k + input]) *
               static_cast<int>(x[row * k + input]);
      }
      if (asymmetric) {
        dot -= static_cast<std::int64_t>(zero_point[row]) * row_sum[column];
      }
      const float expected = static_cast<float>(dot) * weight_scale[column] *
                             activation_scale[row];
      if (output[row * n + column] != expected) {
        std::cerr << "FAIL: int8_gemm "
                  << quixicore_cpu::int8_gemm_variant() << " M=" << m
                  << " N=" << n << " K=" << k << " at [" << row << ','
                  << column << "]: " << output[row * n + column] << " != "
                  << expected << '\n';
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (const char* forced = std::getenv("QUIXICORE_CPU_INT8_GEMM_VARIANT")) {
    const quixicore_cpu::CpuFeatures& features = quixicore_cpu::cpu_features();
    const bool available =
        std::strcmp(forced, "ref") == 0 ||
        (std::strcmp(forced, "avx2") == 0 && features.avx2) ||
        (std::strcmp(forced, "dotprod") == 0 && features.dotprod) ||
        (std::strcmp(forced, "i8mm") == 0 && features.i8mm) ||
        (std::strcmp(forced, "avx512_vnni") == 0 && features.avx512_vnni);
    if (available &&
        std::strcmp(quixicore_cpu::int8_gemm_variant(), forced) != 0) {
      std::cerr << "FAIL: requested int8 GEMM route " << forced
                << " but selected " << quixicore_cpu::int8_gemm_variant()
                << '\n';
      return 1;
    }
  }
  constexpr long long kSizes[] = {1, 15, 16, 17, 63, 64, 65, 100, 1408};
  bool ok = true;
  for (long long m = 1; m <= 5 && ok; ++m) {
    for (long long n = 1; n <= 5 && ok; ++n) {
      for (long long k : kSizes) {
        ok &= check_shape(m, n, k, false);
        ok &= check_shape(m, n, k, true);
        if (!ok) break;
      }
    }
  }
  if (ok) {
    std::cout << "int8 GEMM (" << quixicore_cpu::int8_gemm_variant()
              << "): exact\n";
  }
  return ok ? 0 : 1;
}
