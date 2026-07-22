#include "quixicore_cpu/quantization.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/ops.h"

namespace quixicore_cpu {

Status rms_norm_add_quant_int8(
    const float* x, const float* residual, const float* weight,
    std::int8_t* codes, float* residual_out, float* scales, long long rows,
    long long hidden, float eps, long long group_size) {
  if (!detail::valid_product({rows, hidden})) return Status::kInvalidShape;
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  Status status = rms_norm_add(x, residual, weight, normalized.data(),
                               residual_out, rows, hidden, eps);
  return status == Status::kOk
             ? quantize_int8(normalized.data(), codes, scales, rows, hidden,
                             group_size)
             : status;
}

Status layer_norm_add_quant_int8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::int8_t* codes, float* residual_out,
    float* scales, long long rows, long long hidden, float eps,
    long long group_size) {
  if (!detail::valid_product({rows, hidden})) return Status::kInvalidShape;
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  Status status = layer_norm_add(x, residual, weight, bias, normalized.data(),
                                 residual_out, rows, hidden, eps);
  return status == Status::kOk
             ? quantize_int8(normalized.data(), codes, scales, rows, hidden,
                             group_size)
             : status;
}

Status rms_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    std::uint8_t* codes, float* residual_out, float* scales, long long rows,
    long long hidden, float eps, long long group_size,
    bool power_of_two_scale, Float8Format format) {
  if (!detail::valid_product({rows, hidden})) return Status::kInvalidShape;
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  Status status = rms_norm_add(x, residual, weight, normalized.data(),
                               residual_out, rows, hidden, eps);
  return status == Status::kOk
             ? quantize_float8(normalized.data(), codes, scales, rows, hidden,
                               group_size, format, power_of_two_scale)
             : status;
}

Status layer_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out,
    float* scales, long long rows, long long hidden, float eps,
    long long group_size, bool power_of_two_scale, Float8Format format) {
  if (!detail::valid_product({rows, hidden})) return Status::kInvalidShape;
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  Status status = layer_norm_add(x, residual, weight, bias, normalized.data(),
                                 residual_out, rows, hidden, eps);
  return status == Status::kOk
             ? quantize_float8(normalized.data(), codes, scales, rows, hidden,
                               group_size, format, power_of_two_scale)
             : status;
}

Status rms_norm_add_quant_float8_static(
    const float* x, const float* residual, const float* weight,
    std::uint8_t* codes, float* residual_out, long long rows,
    long long hidden, float scale, float eps, Float8Format format) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(scale) ||
      scale <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(codes)) return Status::kInvalidArgument;
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  Status status = rms_norm_add(x, residual, weight, normalized.data(),
                               residual_out, rows, hidden, eps);
  if (status != Status::kOk) return status;
  for (long long index = 0; index < rows * hidden; ++index) {
    codes[index] = float8_encode(normalized[index] / scale, format);
  }
  return Status::kOk;
}

Status layer_norm_add_quant_float8_static(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out,
    long long rows, long long hidden, float scale, float eps,
    Float8Format format) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(scale) ||
      scale <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(codes)) return Status::kInvalidArgument;
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  Status status = layer_norm_add(x, residual, weight, bias, normalized.data(),
                                 residual_out, rows, hidden, eps);
  if (status != Status::kOk) return status;
  for (long long index = 0; index < rows * hidden; ++index) {
    codes[index] = float8_encode(normalized[index] / scale, format);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
