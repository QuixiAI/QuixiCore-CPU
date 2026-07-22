#include "quixicore_cpu/qgemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quantization.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float activation(float value, LinearActivation kind) {
  constexpr float kInvSqrt2 = 0.7071067811865475f;
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  switch (kind) {
    case LinearActivation::kNone:
      return value;
    case LinearActivation::kGeluErf:
      return 0.5f * value * (1.0f + std::erf(value * kInvSqrt2));
    case LinearActivation::kGeluTanh:
      return 0.5f * value *
             (1.0f + std::tanh(kSqrt2OverPi *
                               (value + 0.044715f * value * value * value)));
    case LinearActivation::kSilu:
      return value / (1.0f + std::exp(-value));
  }
  return value;
}

float half_at(const std::uint8_t* bytes) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, bytes, sizeof(bits));
  return fp16_to_fp32(bits);
}

}  // namespace

Status qgemm_backward_input(QuantFormat format, const void* packed_weights,
                            const float* grad_y, float* grad_x, long long m,
                            long long n, long long k) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_weights, grad_y, grad_x)) {
    return Status::kInvalidArgument;
  }
  std::size_t packed_size = 0;
  Status status = qgemv_packed_size(format, n, k, &packed_size);
  if (status != Status::kOk) return status;
  (void)packed_size;
  std::vector<float> weights(static_cast<std::size_t>(n * k));
  status = qgemv_unpack(format, packed_weights, n, k, weights.data());
  if (status != Status::kOk) return status;
  threading::parallel_ranges(m, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long input = 0; input < k; ++input) {
        double sum = 0.0;
        for (long long output = 0; output < n; ++output) {
          sum += grad_y[row * n + output] * weights[output * k + input];
        }
        grad_x[row * k + input] = static_cast<float>(sum);
      }
    }
  });
  return Status::kOk;
}

Status qgemm_epilogue(QuantFormat format, const void* packed_weights,
                      const float* x, const float* bias, float* y,
                      long long m, long long n, long long k,
                      LinearActivation kind) {
  Status status = qgemm(format, packed_weights, x, y, m, n, k);
  if (status != Status::kOk) return status;
  threading::parallel_ranges(m * n, 4096,
                             [&](long long begin, long long end, int) {
    for (long long index = begin; index < end; ++index) {
      const long long output = index % n;
      y[index] = activation(y[index] + (bias != nullptr ? bias[output] : 0.0f),
                            kind);
    }
  });
  return Status::kOk;
}

Status qflux_gelu(QuantFormat format, const void* packed_weights,
                  const float* x, const float* bias, float* y, long long m,
                  long long n, long long k) {
  return qgemm_epilogue(format, packed_weights, x, bias, y, m, n, k,
                        LinearActivation::kGeluTanh);
}

Status bitnet_int8_gemm(const void* packed_weights, const std::int8_t* x,
                        const float* activation_scale, float* y, long long m,
                        long long n, long long k) {
  if (!detail::valid_product({m, n, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, x, activation_scale, y)) {
    return Status::kInvalidArgument;
  }
  const auto* packed = static_cast<const std::uint8_t*>(packed_weights);
  const long long blocks = k / 32;
  for (long long row = 0; row < m; ++row) {
    if (!std::isfinite(activation_scale[row])) {
      return Status::kInvalidArgument;
    }
  }
  for (long long output = 0; output < n; ++output) {
    for (long long block = 0; block < blocks; ++block) {
      if (!std::isfinite(half_at(
              packed + (output * blocks + block) * 10))) {
        return Status::kInvalidArgument;
      }
    }
  }
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      double result = 0.0;
      for (long long block = 0; block < blocks; ++block) {
        const std::uint8_t* source =
            packed + (output * blocks + block) * 10;
        std::int32_t dot = 0;
        for (int input = 0; input < 32; ++input) {
          const int code =
              (source[2 + input / 4] >> (2 * (input % 4))) & 3;
          dot += (code - 1) *
                 static_cast<int>(x[row * k + block * 32 + input]);
        }
        result += half_at(source) * dot;
      }
      y[item] = static_cast<float>(result * activation_scale[row]);
    }
  });
  return Status::kOk;
}

Status qgemm_w8a8(const std::int8_t* weights, const std::int8_t* x,
                  const float* weight_scale, const float* activation_scale,
                  float* y, long long m, long long n, long long k) {
  return int8_gemm(weights, x, weight_scale, activation_scale, nullptr,
                   nullptr, y, m, n, k, false);
}

Status qgemm_w8a8_azp(
    const std::int8_t* weights, const std::int8_t* x,
    const float* weight_scale, const float* activation_scale,
    const std::int32_t* weight_row_sum, const int* activation_zero_point,
    float* y, long long m, long long n, long long k) {
  return int8_gemm(weights, x, weight_scale, activation_scale, weight_row_sum,
                   activation_zero_point, y, m, n, k, true);
}

Status qgemm_w2a8(const void* packed_weights, const std::int8_t* x,
                  const float* activation_scale, float* y, long long m,
                  long long n, long long k) {
  return bitnet_int8_gemm(packed_weights, x, activation_scale, y, m, n, k);
}

Status qgemv_w2a8(const void* packed_weights, const std::int8_t* x,
                  const float* activation_scale, float* y, long long n,
                  long long k) {
  return bitnet_int8_gemm(packed_weights, x, activation_scale, y, 1, n, k);
}

Status qgemv_w2a8_v2(const void* packed_weights, const std::int8_t* x,
                     const float* activation_scale, float* y, long long n,
                     long long k) {
  return bitnet_int8_gemm(packed_weights, x, activation_scale, y, 1, n, k);
}

Status fp8_scaled_gemm(const std::uint8_t* weights,
                       const std::uint8_t* x, const float* weight_scale,
                       const float* activation_scale, float* y, long long m,
                       long long n, long long k, Float8Format format) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(weights, x, weight_scale, activation_scale, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(m * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / n;
      const long long output = item % n;
      double sum = 0.0;
      for (long long input = 0; input < k; ++input) {
        sum += float8_decode(weights[output * k + input], format) *
               float8_decode(x[row * k + input], format);
      }
      y[item] = static_cast<float>(sum * weight_scale[output] *
                                   activation_scale[row]);
    }
  });
  return Status::kOk;
}

Status qgemm_actorder(QuantFormat format, const void* packed_weights,
                      const float* x, const int* permutation, float* y,
                      long long m, long long n, long long k) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_weights, x, permutation, y)) {
    return Status::kInvalidArgument;
  }
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(k), 0);
  for (long long item = 0; item < k; ++item) {
    if (permutation[item] < 0 || permutation[item] >= k ||
        seen[permutation[item]]++) {
      return Status::kInvalidArgument;
    }
  }
  std::vector<float> permuted(static_cast<std::size_t>(m * k));
  for (long long row = 0; row < m; ++row) {
    for (long long item = 0; item < k; ++item) {
      permuted[row * k + item] = x[row * k + permutation[item]];
    }
  }
  return qgemm(format, packed_weights, permuted.data(), y, m, n, k);
}

Status fp8_blockscale_gemm(const std::uint8_t* weights, const float* x,
                           const float* tile_scales, float* y, long long m,
                           long long n, long long k, long long block_n,
                           long long block_k, Float8Format format) {
  if (!detail::valid_product({m, n, k, block_n, block_k})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(weights, x, tile_scales, y)) {
    return Status::kInvalidArgument;
  }
  const long long scale_columns = (k + block_k - 1) / block_k;
  for (long long row = 0; row < m; ++row) {
    for (long long output = 0; output < n; ++output) {
      double sum = 0.0;
      for (long long input = 0; input < k; ++input) {
        const float scale = tile_scales[(output / block_n) * scale_columns +
                                        input / block_k];
        if (!std::isfinite(scale)) return Status::kInvalidArgument;
        sum += x[row * k + input] *
               float8_decode(weights[output * k + input], format) * scale;
      }
      y[row * n + output] = static_cast<float>(sum);
    }
  }
  return Status::kOk;
}

Status bitnet_fused_gemm(const void* packed_weights, const float* x,
                         float* y, long long m, long long n, long long k) {
  if (!detail::valid_product({m, n, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, x, y)) {
    return Status::kInvalidArgument;
  }
  std::vector<std::int8_t> codes(static_cast<std::size_t>(m * k));
  std::vector<float> scales(static_cast<std::size_t>(m));
  Status status = quantize_int8(x, codes.data(), scales.data(), m, k, k);
  return status == Status::kOk
             ? bitnet_int8_gemm(packed_weights, codes.data(), scales.data(), y,
                                m, n, k)
             : status;
}

Status qgemm_w2a8_fused(const void* packed_weights, const float* x, float* y,
                        long long m, long long n, long long k) {
  return bitnet_fused_gemm(packed_weights, x, y, m, n, k);
}

}  // namespace quixicore_cpu
