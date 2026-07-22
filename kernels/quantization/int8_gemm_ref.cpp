#include "kernels/quantization/int8_gemm.h"

#include <cstdint>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void int8_gemm_ref_kernel(
    const std::int8_t* weights, const std::int8_t* x,
    const float* weight_scale, const float* activation_scale,
    const std::int32_t* weight_row_sum, const int* activation_zero_point,
    float* y, long long m, long long n, long long k, bool asymmetric) {
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      std::int64_t dot = 0;
      for (long long input = 0; input < k; ++input) {
        dot += static_cast<int>(weights[output * k + input]) *
               static_cast<int>(x[row * k + input]);
      }
      if (asymmetric) {
        dot -= static_cast<std::int64_t>(activation_zero_point[row]) *
               weight_row_sum[output];
      }
      y[item] = static_cast<float>(dot) * weight_scale[output] *
                activation_scale[row];
    }
  });
}

}  // namespace quixicore_cpu::quant
