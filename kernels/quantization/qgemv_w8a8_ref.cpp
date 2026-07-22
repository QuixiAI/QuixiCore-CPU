// Portable reference for the activation-quantizing q4_0 GEMV (qgemv_w8a8).
//
// The integer dot is exact for the specified blockwise-int8 activations. This
// scalar path is the correctness anchor; ISA variants are validated against
// an independently dequantized weight-and-activation oracle.

#include "kernels/quantization/qgemv_w8a8.h"

#include <cmath>
#include <vector>

#include "kernels/common/fp16.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

void quantize_activations(const float* x, long long k,
                          std::vector<int8_t>& quantized,
                          std::vector<float>& scales) {
  const long long blocks = k / kQ4_0BlockSize;
  quantized.resize(static_cast<std::size_t>(k));
  scales.resize(static_cast<std::size_t>(blocks));
  for (long long block = 0; block < blocks; ++block) {
    const float* input = x + block * kQ4_0BlockSize;
    float maximum = 0.0f;
    for (long long element = 0; element < kQ4_0BlockSize; ++element) {
      maximum = std::fmax(maximum, std::fabs(input[element]));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    scales[static_cast<std::size_t>(block)] = scale;
    for (long long element = 0; element < kQ4_0BlockSize; ++element) {
      const float value = std::nearbyint(input[element] * inverse);
      quantized[static_cast<std::size_t>(block * kQ4_0BlockSize + element)] =
          static_cast<int8_t>(value < -127.0f
                                  ? -127.0f
                                  : (value > 127.0f ? 127.0f : value));
    }
  }
}

}  // namespace

bool q4_0_pack_ref(const float* weights, long long n, long long k,
                   BlockQ4_0* packed) {
  const long long blocks = k / kQ4_0BlockSize;
  for (long long r = 0; r < n; ++r) {
    for (long long b = 0; b < blocks; ++b) {
      const float* wb = weights + r * k + b * kQ4_0BlockSize;
      float amax = 0.0f;
      float vmax = 0.0f;  // signed value with the largest magnitude
      for (long long j = 0; j < kQ4_0BlockSize; ++j) {
        const float v = wb[j];
        if (!std::isfinite(v)) {
          return false;  // no undefined float->int conversion on NaN/inf
        }
        if (std::fabs(v) > amax) {
          amax = std::fabs(v);
          vmax = v;
        }
      }
      const float d = vmax / -8.0f;
      const float id = d != 0.0f ? 1.0f / d : 0.0f;
      BlockQ4_0& blk = packed[r * blocks + b];
      blk.d = fp32_to_fp16(d);
      for (int j = 0; j < 16; ++j) {
        int q0 = static_cast<int>(std::lround(wb[j] * id)) + 8;
        int q1 = static_cast<int>(std::lround(wb[j + 16] * id)) + 8;
        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
        blk.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
      }
    }
  }
  return true;
}

void q4_0_unpack_ref(const BlockQ4_0* packed, long long n, long long k,
                     float* weights) {
  const long long blocks = k / kQ4_0BlockSize;
  for (long long r = 0; r < n; ++r) {
    for (long long b = 0; b < blocks; ++b) {
      const BlockQ4_0& blk = packed[r * blocks + b];
      const float d = fp16_to_fp32(blk.d);
      float* wb = weights + r * k + b * kQ4_0BlockSize;
      for (int j = 0; j < 16; ++j) {
        wb[j] = static_cast<float>((blk.qs[j] & 0x0F) - 8) * d;
        wb[j + 16] = static_cast<float>((blk.qs[j] >> 4) - 8) * d;
      }
    }
  }
}

void q4_0_gemv_ref(const BlockQ4_0* packed, const float* x, float* y,
                   long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;
  threading::parallel_ranges(n, 32, [&](long long begin, long long end, int) {
    for (long long row_index = begin; row_index < end; ++row_index) {
      const BlockQ4_0* row = packed + row_index * blocks;
      float sum = 0.0f;
      for (long long block = 0; block < blocks; ++block) {
        const float* input = x + block * kQ4_0BlockSize;
        float block_sum = 0.0f;
        for (int element = 0; element < 16; ++element) {
          block_sum +=
              static_cast<float>((row[block].qs[element] & 0x0F) - 8) *
              input[element];
          block_sum +=
              static_cast<float>((row[block].qs[element] >> 4) - 8) *
              input[element + 16];
        }
        sum += fp16_to_fp32(row[block].d) * block_sum;
      }
      y[row_index] = sum;
    }
  });
}

void q4_0_gemv_w8a8_ref(const BlockQ4_0* packed, const float* x, float* y,
                        long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;

  // Quantize the activations to per-block int8 once (amortized over all rows).
  thread_local std::vector<int8_t> qx;
  thread_local std::vector<float> dx;
  quantize_activations(x, k, qx, dx);
  const int8_t* qx_data = qx.data();
  const float* dx_data = dx.data();
  threading::parallel_ranges(n, 32, [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) {
      const BlockQ4_0* row = packed + i * blocks;
      float sumf = 0.0f;
      for (long long b = 0; b < blocks; ++b) {
        const uint8_t* qs = row[b].qs;
        const int8_t* qxb = qx_data + b * kQ4_0BlockSize;
        int sumi = 0;
        for (int j = 0; j < 16; ++j) {
          const int v0 = static_cast<int>(qs[j] & 0x0F) - 8;
          const int v1 = static_cast<int>(qs[j] >> 4) - 8;
          sumi += v0 * static_cast<int>(qxb[j]);
          sumi += v1 * static_cast<int>(qxb[j + 16]);
        }
        sumf += static_cast<float>(sumi) * fp16_to_fp32(row[b].d) *
                dx_data[b];
      }
      y[i] = sumf;
    }
  });
}

void q8_0_gemv_w8a8_ref(const BlockQ8_0* packed, const float* x, float* y,
                        long long n, long long k) {
  const long long blocks = k / kQ8_0BlockSize;
  thread_local std::vector<int8_t> qx;
  thread_local std::vector<float> dx;
  quantize_activations(x, k, qx, dx);
  const int8_t* qx_data = qx.data();
  const float* dx_data = dx.data();
  threading::parallel_ranges(n, 32, [&](long long begin, long long end, int) {
    for (long long row_index = begin; row_index < end; ++row_index) {
      const BlockQ8_0* row = packed + row_index * blocks;
      float sum = 0.0f;
      for (long long block = 0; block < blocks; ++block) {
        const int8_t* input = qx_data + block * kQ8_0BlockSize;
        int dot = 0;
        for (long long element = 0; element < kQ8_0BlockSize; ++element) {
          dot += static_cast<int>(row[block].qs[element]) *
                 static_cast<int>(input[element]);
        }
        sum += static_cast<float>(dot) * fp16_to_fp32(row[block].d) *
               dx_data[block];
      }
      y[row_index] = sum;
    }
  });
}

}  // namespace quixicore_cpu::quant
