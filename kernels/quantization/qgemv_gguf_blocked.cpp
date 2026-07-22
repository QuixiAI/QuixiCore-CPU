#include "kernels/quantization/gguf_ref.h"

#include <cstddef>
#include <cstdint>

#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void gguf_gemv_blocked_ref(QuantFormat format, const void* packed,
                           const float* x, float* y, long long n,
                           long long k) {
  long long block_size = 0;
  std::size_t block_bytes = 0;
  (void)gguf_format_info(format, &block_size, &block_bytes);
  const long long blocks_per_row = k / block_size;
  const auto* bytes = static_cast<const std::uint8_t*>(packed);
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    alignas(64) float decoded[256];
    for (long long row = begin; row < end; ++row) {
      double total = 0.0;
      for (long long block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* source =
            bytes + (row * blocks_per_row + block) * block_bytes;
        gguf_dequant_block_ref(format, source, decoded);
        const float* input = x + block * block_size;
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        int column = 0;
        for (; column + 3 < block_size; column += 4) {
          sum0 += static_cast<double>(decoded[column]) * input[column];
          sum1 += static_cast<double>(decoded[column + 1]) * input[column + 1];
          sum2 += static_cast<double>(decoded[column + 2]) * input[column + 2];
          sum3 += static_cast<double>(decoded[column + 3]) * input[column + 3];
        }
        for (; column < block_size; ++column) {
          sum0 += static_cast<double>(decoded[column]) * input[column];
        }
        total += (sum0 + sum1) + (sum2 + sum3);
      }
      y[row] = static_cast<float>(total);
    }
  });
}

}  // namespace quixicore_cpu::quant
