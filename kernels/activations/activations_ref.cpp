#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

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

float gelu_quick_value(float x) { return x * sigmoid(1.702f * x); }

float gelu_quick_derivative(float x) {
  const float probability = sigmoid(1.702f * x);
  return probability + 1.702f * x * probability * (1.0f - probability);
}

float softplus_value(float x) {
  return x > 20.0f ? x : std::log(1.0f + std::exp(x));
}

bool valid_unary_op(UnaryOp op) {
  return op >= UnaryOp::kAbs && op <= UnaryOp::kTrunc;
}

bool valid_glu_mode(GluMode mode) {
  return mode >= GluMode::kSwiGlu && mode <= GluMode::kGeGluQuick;
}

float unary_value(float x, UnaryOp op, float alpha_n, float alpha_p,
                  float beta, float eps) {
  switch (op) {
    case UnaryOp::kAbs:
      return std::fabs(x);
    case UnaryOp::kSign:
      return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    case UnaryOp::kNegate:
      return -x;
    case UnaryOp::kStep:
      return x > 0.0f ? 1.0f : 0.0f;
    case UnaryOp::kTanh:
      return std::tanh(x);
    case UnaryOp::kElu:
      return x > 0.0f ? x : std::expm1(x);
    case UnaryOp::kRelu:
      return x > 0.0f ? x : 0.0f;
    case UnaryOp::kSigmoid:
      return sigmoid(x);
    case UnaryOp::kGelu:
      return gelu_value(x, GeluApprox::kTanh);
    case UnaryOp::kGeluQuick:
      return gelu_quick_value(x);
    case UnaryOp::kSilu:
      return x * sigmoid(x);
    case UnaryOp::kHardSwish:
      return x * std::clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
    case UnaryOp::kHardSigmoid:
      return std::clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
    case UnaryOp::kExp:
      return std::exp(x);
    case UnaryOp::kExpm1:
      return std::exp(x) - 1.0f;
    case UnaryOp::kSoftplus:
      return softplus_value(x);
    case UnaryOp::kGeluErf:
      return gelu_value(x, GeluApprox::kErf);
    case UnaryOp::kXiElu:
      if (x > 0.0f) return alpha_p * x * x + beta * x;
      return (std::expm1(std::min(x, eps)) - x) * alpha_n + beta * x;
    case UnaryOp::kFloor:
      return std::floor(x);
    case UnaryOp::kCeil:
      return std::ceil(x);
    case UnaryOp::kRound:
      return std::round(x);
    case UnaryOp::kTrunc:
      return std::trunc(x);
  }
  return 0.0f;
}

template <UnaryOp Op>
void apply_unary_mode(const float* x, float* y, long long count,
                      float alpha_n, float alpha_p, float beta, float eps,
                      long long min_per_chunk) {
  threading::parallel_ranges(count, min_per_chunk,
      [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) {
      y[i] = unary_value(x[i], Op, alpha_n, alpha_p, beta, eps);
    }
  });
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

Status silu_backward(const float* grad_out, const float* x, float* grad_in,
                     long long count) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(grad_out, x, grad_in)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(count, 16384,
      [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) {
      const float probability = sigmoid(x[i]);
      grad_in[i] = grad_out[i] * probability *
                   (1.0f + x[i] * (1.0f - probability));
    }
  });
  return Status::kOk;
}

Status unary(const float* x, float* y, long long count, UnaryOp op,
             XiEluParams xielu) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, y) || !valid_unary_op(op)) {
    return Status::kInvalidArgument;
  }
  float alpha_n = 0.0f;
  float alpha_p = 0.0f;
  if (op == UnaryOp::kXiElu) {
    if (!std::isfinite(xielu.alpha_n) || !std::isfinite(xielu.alpha_p) ||
        !std::isfinite(xielu.beta) || !std::isfinite(xielu.eps)) {
      return Status::kInvalidArgument;
    }
    alpha_n = xielu.beta + softplus_value(xielu.alpha_n);
    alpha_p = softplus_value(xielu.alpha_p);
  }
  constexpr long long kMinPerChunk = 16384;
#define QUIXICORE_CPU_UNARY_CASE(name)                                      \
  case UnaryOp::name:                                                       \
    apply_unary_mode<UnaryOp::name>(x, y, count, alpha_n, alpha_p,           \
                                     xielu.beta, xielu.eps, kMinPerChunk);   \
    break
  switch (op) {
    QUIXICORE_CPU_UNARY_CASE(kAbs);
    QUIXICORE_CPU_UNARY_CASE(kSign);
    QUIXICORE_CPU_UNARY_CASE(kNegate);
    QUIXICORE_CPU_UNARY_CASE(kStep);
    QUIXICORE_CPU_UNARY_CASE(kTanh);
    QUIXICORE_CPU_UNARY_CASE(kElu);
    QUIXICORE_CPU_UNARY_CASE(kRelu);
    QUIXICORE_CPU_UNARY_CASE(kSigmoid);
    QUIXICORE_CPU_UNARY_CASE(kGelu);
    QUIXICORE_CPU_UNARY_CASE(kGeluQuick);
    QUIXICORE_CPU_UNARY_CASE(kSilu);
    QUIXICORE_CPU_UNARY_CASE(kHardSwish);
    QUIXICORE_CPU_UNARY_CASE(kHardSigmoid);
    QUIXICORE_CPU_UNARY_CASE(kExp);
    QUIXICORE_CPU_UNARY_CASE(kExpm1);
    QUIXICORE_CPU_UNARY_CASE(kSoftplus);
    QUIXICORE_CPU_UNARY_CASE(kGeluErf);
    QUIXICORE_CPU_UNARY_CASE(kXiElu);
    QUIXICORE_CPU_UNARY_CASE(kFloor);
    QUIXICORE_CPU_UNARY_CASE(kCeil);
    QUIXICORE_CPU_UNARY_CASE(kRound);
    QUIXICORE_CPU_UNARY_CASE(kTrunc);
  }
#undef QUIXICORE_CPU_UNARY_CASE
  return Status::kOk;
}

Status glu(const float* x, float* y, long long rows, long long dim,
           GluMode mode) {
  if (!detail::valid_product({rows, dim, 2})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y) || !valid_glu_mode(mode)) {
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
          activated = gelu_value(gate[i], GeluApprox::kTanh);
          break;
        case GluMode::kReGlu:
          activated = gate[i] > 0.0f ? gate[i] : 0.0f;
          break;
        case GluMode::kGlu:
          activated = sigmoid(gate[i]);
          break;
        case GluMode::kGeGluErf:
          activated = gelu_value(gate[i], GeluApprox::kErf);
          break;
        case GluMode::kGeGluQuick:
          activated = gelu_quick_value(gate[i]);
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
  if (!detail::all_nonnull(grad_out, x, grad_in) || !valid_glu_mode(mode)) {
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
          activated = gelu_value(gate[item], GeluApprox::kTanh);
          derivative = gelu_derivative(gate[item], GeluApprox::kTanh);
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
        case GluMode::kGeGluErf:
          activated = gelu_value(gate[item], GeluApprox::kErf);
          derivative = gelu_derivative(gate[item], GeluApprox::kErf);
          break;
        case GluMode::kGeGluQuick:
          activated = gelu_quick_value(gate[item]);
          derivative = gelu_quick_derivative(gate[item]);
          break;
      }
      const float gradient = grad_out[row * dim + item];
      grad_gate[item] = gradient * value[item] * derivative;
      grad_value[item] = gradient * activated;
    }
  }
  return Status::kOk;
}

Status swiglu_oai(const float* gate, const float* value, float* y,
                  long long rows, long long dim, float alpha, float limit) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(gate, value, y) || !std::isfinite(alpha) ||
      !std::isfinite(limit) || limit < 0.0f) {
    return Status::kInvalidArgument;
  }
  const long long count = rows * dim;
  threading::parallel_ranges(count, 16384,
      [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) {
      const float x = std::min(gate[i], limit);
      const float v = std::clamp(value[i], -limit, limit);
      y[i] = x / (1.0f + std::exp(-alpha * x)) * (v + 1.0f);
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
