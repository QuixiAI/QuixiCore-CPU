#include "quixicore_cpu/ops.h"

#include <cmath>

#include "kernels/common/validation.h"

namespace quixicore_cpu {
namespace {

constexpr float kInvSqrt2 = 0.70710678118654752440f;
constexpr float kSqrt2OverPi = 0.79788456080286535588f;
constexpr float kGeluCoeff = 0.044715f;

float gelu_value(float x, GeluApprox approx) {
  if (approx == GeluApprox::kErf) {
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
  }
  return 0.5f * x *
         (1.0f + std::tanh(kSqrt2OverPi *
                           (x + kGeluCoeff * x * x * x)));
}

float gelu_derivative(float x, GeluApprox approx) {
  if (approx == GeluApprox::kErf) {
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    return 0.5f * (1.0f + std::erf(x * kInvSqrt2)) +
           x * kInvSqrt2Pi * std::exp(-0.5f * x * x);
  }
  const float x2 = x * x;
  const float u = kSqrt2OverPi * (x + kGeluCoeff * x * x2);
  const float t = std::tanh(u);
  const float du = kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x2);
  return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

float sigmoid(float x) {
  if (x >= 0.0f) {
    const float e = std::exp(-x);
    return 1.0f / (1.0f + e);
  }
  const float e = std::exp(x);
  return e / (1.0f + e);
}

}  // namespace

Status gelu(const float* x, float* y, long long count, GeluApprox approx) {
  if (!detail::valid_product({count})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  for (long long i = 0; i < count; ++i) {
    y[i] = gelu_value(x[i], approx);
  }
  return Status::kOk;
}

Status gelu_backward(const float* grad_out, const float* x, float* grad_in,
                     long long count, GeluApprox approx) {
  if (!detail::valid_product({count})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_out, x, grad_in)) {
    return Status::kInvalidArgument;
  }
  for (long long i = 0; i < count; ++i) {
    grad_in[i] = grad_out[i] * gelu_derivative(x[i], approx);
  }
  return Status::kOk;
}

Status silu(const float* x, float* y, long long count) {
  if (!detail::valid_product({count})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  for (long long i = 0; i < count; ++i) {
    y[i] = x[i] * sigmoid(x[i]);
  }
  return Status::kOk;
}

Status glu(const float* x, float* y, long long rows, long long dim,
           GluMode mode) {
  if (!detail::valid_product({rows, dim, 2})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    const float* gate = x + row * 2 * dim;
    const float* value = gate + dim;
    float* out = y + row * dim;
    for (long long i = 0; i < dim; ++i) {
      float activated = 0.0f;
      switch (mode) {
        case GluMode::kSwiGlu:
          activated = gate[i] * sigmoid(gate[i]);
          break;
        case GluMode::kGeGlu:
          activated = gelu_value(gate[i], GeluApprox::kErf);
          break;
        case GluMode::kReGlu:
          activated = gate[i] > 0.0f ? gate[i] : 0.0f;
          break;
        case GluMode::kGlu:
          activated = sigmoid(gate[i]);
          break;
      }
      out[i] = activated * value[i];
    }
  }
  return Status::kOk;
}

Status glu_backward(const float* grad_out, const float* x, float* grad_in,
                    long long rows, long long dim, GluMode mode) {
  if (!detail::valid_product({rows, dim, 2})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_out, x, grad_in)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    const float* gate = x + row * 2 * dim;
    const float* value = gate + dim;
    float* grad_gate = grad_in + row * 2 * dim;
    float* grad_value = grad_gate + dim;
    for (long long item = 0; item < dim; ++item) {
      float activated = 0.0f, derivative = 0.0f;
      switch (mode) {
        case GluMode::kSwiGlu: {
          const float probability = sigmoid(gate[item]);
          activated = gate[item] * probability;
          derivative = probability + gate[item] * probability *
                                         (1.0f - probability);
          break;
        }
        case GluMode::kGeGlu:
          activated = gelu_value(gate[item], GeluApprox::kErf);
          derivative = gelu_derivative(gate[item], GeluApprox::kErf);
          break;
        case GluMode::kReGlu:
          activated = gate[item] > 0.0f ? gate[item] : 0.0f;
          derivative = gate[item] > 0.0f ? 1.0f : 0.0f;
          break;
        case GluMode::kGlu: {
          activated = sigmoid(gate[item]);
          derivative = activated * (1.0f - activated);
          break;
        }
      }
      const float gradient = grad_out[row * dim + item];
      grad_gate[item] = gradient * value[item] * derivative;
      grad_value[item] = gradient * activated;
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
