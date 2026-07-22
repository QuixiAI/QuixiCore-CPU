#include "kernels/quantization/lowbit.h"

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void lowbit_w8a8_ref(const std::uint8_t* packed, const float* weight_scales,
                     const std::int8_t* x, const float* activation_scales,
                     float* y, long long m, long long n, long long k) {
  const long long row_bytes = (k + 1) / 2;
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      const std::uint8_t* weights = packed + output * row_bytes;
      const std::int8_t* input = x + row * k;
      std::int32_t dot = 0;
      long long inner = 0;
      for (; inner + 1 < k; inner += 2) {
        const std::uint8_t byte = weights[inner >> 1];
        dot += (int(byte & 15) - 8) * int(input[inner]);
        dot += (int(byte >> 4) - 8) * int(input[inner + 1]);
      }
      if (inner < k) {
        dot += (int(weights[inner >> 1] & 15) - 8) * int(input[inner]);
      }
      y[item] = static_cast<float>(dot) * weight_scales[output] *
                activation_scales[row];
    }
  });
}

}  // namespace quixicore_cpu::quant
