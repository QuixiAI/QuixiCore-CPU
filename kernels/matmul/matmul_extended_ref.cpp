#include <cmath>

#include "kernels/common/validation.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float activate(float value, LinearActivation activation) {
  switch (activation) {
    case LinearActivation::kNone:
      return value;
    case LinearActivation::kGeluErf:
      return static_cast<float>(0.5 * value *
                                (1.0 + std::erf(value / std::sqrt(2.0))));
    case LinearActivation::kGeluTanh: {
      constexpr double kCoefficient = 0.044715;
      constexpr double kSqrtTwoOverPi = 0.7978845608028654;
      const double x = value;
      return static_cast<float>(
          0.5 * x *
          (1.0 + std::tanh(kSqrtTwoOverPi * (x + kCoefficient * x * x * x))));
    }
    case LinearActivation::kSilu:
      return value / (1.0f + std::exp(-value));
    case LinearActivation::kRelu2:
      return value > 0.0f ? value * value : 0.0f;
  }
  return value;
}

}  // namespace

Status dense_gemm_ex(const float* a, const float* b, const float* addend,
                     float* c, long long m, long long n, long long k,
                     bool transpose_a, bool transpose_b, float alpha,
                     float beta) {
  if (!detail::valid_product({m, n, k}) || !std::isfinite(alpha) ||
      !std::isfinite(beta)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, c) || (beta != 0.0f && addend == nullptr)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(m, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long column = 0; column < n; ++column) {
        double accumulator = 0.0;
        for (long long inner = 0; inner < k; ++inner) {
          const float av =
              transpose_a ? a[inner * m + row] : a[row * k + inner];
          const float bv =
              transpose_b ? b[column * k + inner] : b[inner * n + column];
          accumulator += double(av) * bv;
        }
        const double prior = addend == nullptr ? 0.0 : addend[row * n + column];
        c[row * n + column] =
            static_cast<float>(alpha * accumulator + beta * prior);
      }
    }
  });
  return Status::kOk;
}

Status linear_epilogue(const float* x, const float* weight, const float* bias,
                       const float* residual, float* y, long long rows,
                       long long input_dim, long long output_dim,
                       LinearActivation activation) {
  if (!detail::valid_product({rows, input_dim}) ||
      !detail::valid_product({rows, output_dim}) ||
      !detail::valid_product({input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, y)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long output = 0; output < output_dim; ++output) {
        double accumulator = bias == nullptr ? 0.0 : bias[output];
        const float* wr = weight + output * input_dim;
        for (long long input = 0; input < input_dim; ++input) {
          accumulator += double(x[row * input_dim + input]) * wr[input];
        }
        if (residual != nullptr)
          accumulator += residual[row * output_dim + output];
        y[row * output_dim + output] =
            activate(static_cast<float>(accumulator), activation);
      }
    }
  });
  return Status::kOk;
}

Status decode_swiglu(const float* x, const float* gate_weight,
                     const float* up_weight, const float* gate_bias,
                     const float* up_bias, float* y, long long rows,
                     long long input_dim, long long output_dim) {
  if (!detail::valid_product({rows, input_dim}) ||
      !detail::valid_product({rows, output_dim}) ||
      !detail::valid_product({input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, gate_weight, up_weight, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long output = 0; output < output_dim; ++output) {
        double gate = gate_bias == nullptr ? 0.0 : gate_bias[output];
        double up = up_bias == nullptr ? 0.0 : up_bias[output];
        for (long long input = 0; input < input_dim; ++input) {
          const float value = x[row * input_dim + input];
          gate += double(value) * gate_weight[output * input_dim + input];
          up += double(value) * up_weight[output * input_dim + input];
        }
        const double silu_gate = gate / (1.0 + std::exp(-gate));
        y[row * output_dim + output] = static_cast<float>(silu_gate * up);
      }
    }
  });
  return Status::kOk;
}

Status gemm_gate_residual(const float* x, const float* weight,
                          const float* bias, const float* gate,
                          const float* residual, float* y, long long rows,
                          long long output_dim, long long input_dim) {
  if (!detail::valid_product({rows, input_dim}) ||
      !detail::valid_product({rows, output_dim}) ||
      !detail::valid_product({input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, y)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long output = 0; output < output_dim; ++output) {
        double accumulator = bias == nullptr ? 0.0 : bias[output];
        for (long long input = 0; input < input_dim; ++input) {
          accumulator += double(x[row * input_dim + input]) *
                         weight[input * output_dim + output];
        }
        accumulator *= gate == nullptr ? 1.0 : gate[output];
        if (residual != nullptr)
          accumulator += residual[row * output_dim + output];
        y[row * output_dim + output] = static_cast<float>(accumulator);
      }
    }
  });
  return Status::kOk;
}

Status flux_gelu(const float* x, const float* weight, const float* bias,
                 float* y, long long rows, long long output_dim,
                 long long input_dim, GeluApprox approx) {
  Status status = dense_gemm(x, weight, y, rows, output_dim, input_dim);
  if (status != Status::kOk) return status;
  threading::parallel_ranges(
      rows * output_dim, 4096, [&](long long begin, long long end, int) {
        const LinearActivation activation = approx == GeluApprox::kErf
                                                ? LinearActivation::kGeluErf
                                                : LinearActivation::kGeluTanh;
        for (long long index = begin; index < end; ++index) {
          y[index] = activate(
              y[index] + (bias == nullptr ? 0.0f : bias[index % output_dim]),
              activation);
        }
      });
  return Status::kOk;
}

Status flux_gate(const float* x, const float* weight, const float* bias,
                 const float* gate, const float* residual, float* y,
                 long long rows, long long output_dim, long long input_dim) {
  return gemm_gate_residual(x, weight, bias, gate, residual, y, rows,
                            output_dim, input_dim);
}

Status decode_linear(const float* x, const float* weight, const float* bias,
                     float* y, long long rows, long long input_dim,
                     long long output_dim, bool use_gelu) {
  return linear_epilogue(
      x, weight, bias, nullptr, y, rows, input_dim, output_dim,
      use_gelu ? LinearActivation::kGeluTanh : LinearActivation::kNone);
}

Status decode_linear_residual(const float* x, const float* weight,
                              const float* bias, const float* residual,
                              float* y, long long rows, long long input_dim,
                              long long output_dim) {
  return linear_epilogue(x, weight, bias, residual, y, rows, input_dim,
                         output_dim, LinearActivation::kNone);
}

Status complex_gemm(const float* a_real, const float* a_imag,
                    const float* b_real, const float* b_imag, float* c_real,
                    float* c_imag, long long m, long long n, long long k) {
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(a_real, a_imag, b_real, b_imag, c_real, c_imag)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(m, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long column = 0; column < n; ++column) {
        double real = 0.0;
        double imag = 0.0;
        for (long long inner = 0; inner < k; ++inner) {
          const long long ai = row * k + inner;
          const long long bi = inner * n + column;
          real +=
              double(a_real[ai]) * b_real[bi] - double(a_imag[ai]) * b_imag[bi];
          imag +=
              double(a_real[ai]) * b_imag[bi] + double(a_imag[ai]) * b_real[bi];
        }
        c_real[row * n + column] = static_cast<float>(real);
        c_imag[row * n + column] = static_cast<float>(imag);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
