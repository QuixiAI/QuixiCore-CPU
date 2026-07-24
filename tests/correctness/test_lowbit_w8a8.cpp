#include "quixicore_cpu/lowbit.h"
#include "quixicore_cpu/quantization.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "quixicore_cpu/cpu_features.h"

namespace {

std::uint32_t state = 0xA341316Cu;

std::uint32_t random_u32() {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

bool check(long long m, long long n, long long k) {
  std::vector<float> weights(static_cast<std::size_t>(n * k));
  std::vector<float> x(static_cast<std::size_t>(m * k));
  for (float& value : weights) {
    value = (float(int(random_u32() % 2001u) - 1000) / 1000.0f) * 0.2f;
  }
  for (float& value : x) {
    value = float(int(random_u32() % 4001u) - 2000) / 500.0f;
  }
  std::size_t bytes = 0;
  std::size_t scale_count = 0;
  if (quixicore_cpu::lowbit_packed_size(
          quixicore_cpu::LowBitFormat::kInt4Row, n, k, 0, &bytes,
          &scale_count) != quixicore_cpu::Status::kOk) {
    return false;
  }
  std::vector<std::uint8_t> packed(bytes);
  std::vector<float> weight_scales(scale_count);
  if (quixicore_cpu::lowbit_pack(
          quixicore_cpu::LowBitFormat::kInt4Row, weights.data(),
          packed.data(), weight_scales.data(), n, k) !=
      quixicore_cpu::Status::kOk) {
    return false;
  }
  std::vector<float> output(static_cast<std::size_t>(m * n));
  if (quixicore_cpu::lowbit_gemm_w8a8(
          packed.data(), weight_scales.data(), x.data(), output.data(), m, n,
          k) != quixicore_cpu::Status::kOk) {
    return false;
  }
  std::vector<std::int8_t> quantized(static_cast<std::size_t>(m * k));
  std::vector<float> activation_scales(static_cast<std::size_t>(m));
  if (quixicore_cpu::quantize_int8(
          x.data(), quantized.data(), activation_scales.data(), m, k, k) !=
      quixicore_cpu::Status::kOk) {
    return false;
  }
  const long long row_bytes = (k + 1) / 2;
  for (long long row = 0; row < m; ++row) {
    for (long long column = 0; column < n; ++column) {
      std::int32_t dot = 0;
      const std::uint8_t* weight = packed.data() + column * row_bytes;
      for (long long input = 0; input < k; ++input) {
        const int code =
            int((weight[input >> 1] >> (4 * (input & 1))) & 15) - 8;
        dot += code * int(quantized[row * k + input]);
      }
      const float expected = float(dot) * weight_scales[column] *
                             activation_scales[row];
      if (output[row * n + column] != expected) {
        std::cerr << "FAIL: lowbit W8A8 "
                  << quixicore_cpu::lowbit_gemm_w8a8_variant() << " M="
                  << m << " N=" << n << " K=" << k << " at [" << row
                  << ',' << column << "]: " << output[row * n + column]
                  << " != " << expected << '\n';
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (const char* forced =
          std::getenv("QUIXICORE_CPU_LOWBIT_W8A8_VARIANT")) {
    const quixicore_cpu::CpuFeatures& features = quixicore_cpu::cpu_features();
    const bool available =
        std::strcmp(forced, "ref") == 0 ||
        (std::strcmp(forced, "avx2") == 0 && features.avx2) ||
        (std::strcmp(forced, "dotprod") == 0 && features.dotprod) ||
        (std::strcmp(forced, "i8mm") == 0 && features.i8mm) ||
        (std::strcmp(forced, "avx512_vnni") == 0 && features.avx512_vnni);
    if (available &&
        std::strcmp(quixicore_cpu::lowbit_gemm_w8a8_variant(), forced) != 0) {
      std::cerr << "FAIL: requested low-bit W4A8 route " << forced
                << " but selected "
                << quixicore_cpu::lowbit_gemm_w8a8_variant() << '\n';
      return 1;
    }
  }
  bool ok = true;
  for (long long k : {17LL, 32LL, 63LL, 64LL, 65LL, 100LL, 1408LL}) {
    ok &= check(5, 5, k);
  }
  if (ok) {
    std::cout << "low-bit W4A8 ("
              << quixicore_cpu::lowbit_gemm_w8a8_variant() << "): exact\n";
  }
  return ok ? 0 : 1;
}
