#include "kernels/quantization/w8a32_gemm.h"

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void w8a32_gemm_ref(const std::int8_t* weights, const float* weight_scales,
                    const float* x, float* y, long long m, long long n,
                    long long k) {
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::int8_t* weight_row = weights + output * k;
      const float scale = weight_scales[output];
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        float sum = 0.0f;
        for (long long column = 0; column < k; ++column) {
          sum += input[column] * static_cast<float>(weight_row[column]);
        }
        y[row * n + output] = sum * scale;
      }
    }
  });
}

}  // namespace quixicore_cpu::quant
