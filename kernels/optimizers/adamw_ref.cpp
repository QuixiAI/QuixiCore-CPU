#include "quixicore_cpu/ops.h"

#include <cmath>

#include "kernels/common/validation.h"

namespace quixicore_cpu {

Status adamw(float* parameters, const float* gradients, float* first_moment,
             float* second_moment, long long count, float learning_rate,
             float beta1, float beta2, float eps, float weight_decay,
             long long step) {
  if (!detail::valid_product({count}) || step <= 0 ||
      !std::isfinite(learning_rate) || learning_rate < 0.0f ||
      !std::isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
      !std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
      !std::isfinite(eps) || eps < 0.0f ||
      !std::isfinite(weight_decay) || weight_decay < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(parameters, gradients, first_moment,
                           second_moment)) {
    return Status::kInvalidArgument;
  }
  const double first_correction = 1.0 - std::pow(beta1, step);
  const double second_correction = 1.0 - std::pow(beta2, step);
  for (long long i = 0; i < count; ++i) {
    const float gradient = gradients[i];
    first_moment[i] = beta1 * first_moment[i] + (1.0f - beta1) * gradient;
    second_moment[i] =
        beta2 * second_moment[i] + (1.0f - beta2) * gradient * gradient;
    const double corrected_first = first_moment[i] / first_correction;
    const double corrected_second = second_moment[i] / second_correction;
    const double update = corrected_first / (std::sqrt(corrected_second) + eps) +
                          weight_decay * parameters[i];
    parameters[i] -= static_cast<float>(learning_rate * update);
  }
  return Status::kOk;
}

Status adamw_masked(float* parameters, const float* gradients,
                    float* first_moment, float* second_moment,
                    const std::uint8_t* mask, long long count,
                    long long segment_size, int mask_mode,
                    float learning_rate, float beta1, float beta2, float eps,
                    float weight_decay, long long step) {
  if (!detail::valid_product({count, segment_size}) || step <= 0 ||
      (mask_mode != 0 && mask_mode != 1) ||
      !std::isfinite(learning_rate) || learning_rate < 0.0f ||
      !std::isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
      !std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
      !std::isfinite(eps) || eps < 0.0f ||
      !std::isfinite(weight_decay) || weight_decay < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(parameters, gradients, first_moment,
                           second_moment, mask)) {
    return Status::kInvalidArgument;
  }
  const double first_correction = 1.0 - std::pow(beta1, step);
  const double second_correction = 1.0 - std::pow(beta2, step);
  for (long long i = 0; i < count; ++i) {
    const bool active = mask[i / segment_size] != 0;
    if (!active && mask_mode == 0) continue;
    const float gradient = gradients[i];
    first_moment[i] = beta1 * first_moment[i] + (1.0f - beta1) * gradient;
    second_moment[i] =
        beta2 * second_moment[i] + (1.0f - beta2) * gradient * gradient;
    const double corrected_first = first_moment[i] / first_correction;
    const double corrected_second = second_moment[i] / second_correction;
    const double update = corrected_first / (std::sqrt(corrected_second) + eps) +
                          (active ? weight_decay * parameters[i] : 0.0f);
    parameters[i] -= static_cast<float>(learning_rate * update);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
