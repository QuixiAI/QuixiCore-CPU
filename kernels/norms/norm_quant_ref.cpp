#include "quixicore_cpu/quantization.h"

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

}  // namespace quixicore_cpu
