#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>

#include "kernels/common/validation.h"
#include "quixicore_cpu/rms_norm.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status layer_norm_backward(const float* x, const float* weight,
                           const float* grad_out, float* grad_x,
                           float* grad_weight, float* grad_bias,
                           long long rows, long long hidden, float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, grad_out, grad_x, grad_weight)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_weight, hidden, 0.0f);
  if (grad_bias != nullptr) std::fill_n(grad_bias, hidden, 0.0f);

  // The feature-gradient reductions are deterministic in row order. dX is
  // parallel because each row is independent.
  for (long long row = 0; row < rows; ++row) {
    const float* xr = x + row * hidden;
    const float* gr = grad_out + row * hidden;
    double mean = 0.0;
    for (long long i = 0; i < hidden; ++i) mean += xr[i];
    mean /= static_cast<double>(hidden);
    double variance = 0.0;
    for (long long i = 0; i < hidden; ++i) {
      const double centered = static_cast<double>(xr[i]) - mean;
      variance += centered * centered;
    }
    const double rstd = 1.0 / std::sqrt(variance / hidden + eps);
    for (long long i = 0; i < hidden; ++i) {
      const double normalized = (static_cast<double>(xr[i]) - mean) * rstd;
      grad_weight[i] += static_cast<float>(gr[i] * normalized);
      if (grad_bias != nullptr) grad_bias[i] += gr[i];
    }
  }
  threading::parallel_ranges(rows, 4,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* xr = x + row * hidden;
      const float* gr = grad_out + row * hidden;
      float* dxr = grad_x + row * hidden;
      double mean = 0.0;
      for (long long i = 0; i < hidden; ++i) mean += xr[i];
      mean /= static_cast<double>(hidden);
      double variance = 0.0;
      for (long long i = 0; i < hidden; ++i) {
        const double centered = static_cast<double>(xr[i]) - mean;
        variance += centered * centered;
      }
      const double rstd = 1.0 / std::sqrt(variance / hidden + eps);
      double mean_g = 0.0;
      double mean_gx = 0.0;
      for (long long i = 0; i < hidden; ++i) {
        const double normalized = (static_cast<double>(xr[i]) - mean) * rstd;
        const double g = static_cast<double>(gr[i]) * weight[i];
        mean_g += g;
        mean_gx += g * normalized;
      }
      mean_g /= hidden;
      mean_gx /= hidden;
      for (long long i = 0; i < hidden; ++i) {
        const double normalized = (static_cast<double>(xr[i]) - mean) * rstd;
        const double g = static_cast<double>(gr[i]) * weight[i];
        dxr[i] = static_cast<float>(rstd * (g - mean_g - normalized * mean_gx));
      }
    }
  });
  return Status::kOk;
}

Status rms_norm_backward(const float* x, const float* weight,
                         const float* grad_out, float* grad_x,
                         float* grad_weight, long long rows, long long hidden,
                         float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, grad_out, grad_x, grad_weight)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_weight, hidden, 0.0f);
  for (long long row = 0; row < rows; ++row) {
    const float* xr = x + row * hidden;
    const float* gr = grad_out + row * hidden;
    double squares = 0.0;
    for (long long i = 0; i < hidden; ++i) squares += double(xr[i]) * xr[i];
    const double rstd = 1.0 / std::sqrt(squares / hidden + eps);
    for (long long i = 0; i < hidden; ++i) {
      grad_weight[i] += static_cast<float>(gr[i] * xr[i] * rstd);
    }
  }
  threading::parallel_ranges(rows, 4,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* xr = x + row * hidden;
      const float* gr = grad_out + row * hidden;
      float* dxr = grad_x + row * hidden;
      double squares = 0.0;
      double dot = 0.0;
      for (long long i = 0; i < hidden; ++i) {
        squares += double(xr[i]) * xr[i];
        dot += double(gr[i]) * weight[i] * xr[i];
      }
      const double rstd = 1.0 / std::sqrt(squares / hidden + eps);
      const double correction = rstd * rstd * rstd * dot / hidden;
      for (long long i = 0; i < hidden; ++i) {
        dxr[i] = static_cast<float>(rstd * gr[i] * weight[i] -
                                    correction * xr[i]);
      }
    }
  });
  return Status::kOk;
}

Status rms_norm_add(const float* x, const float* residual,
                    const float* weight, float* y, float* residual_out,
                    long long rows, long long hidden, float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, residual, weight, y, residual_out)) {
    return Status::kInvalidArgument;
  }
  const long long count = rows * hidden;
  for (long long i = 0; i < count; ++i) residual_out[i] = x[i] + residual[i];
  return rms_norm(residual_out, weight, y, rows, hidden, eps);
}

Status layer_norm_add(const float* x, const float* residual,
                      const float* weight, const float* bias, float* y,
                      float* residual_out, long long rows, long long hidden,
                      float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, residual, weight, y, residual_out)) {
    return Status::kInvalidArgument;
  }
  const long long count = rows * hidden;
  for (long long i = 0; i < count; ++i) residual_out[i] = x[i] + residual[i];
  return layer_norm(residual_out, weight, bias, y, rows, hidden, eps);
}

Status rms_norm_residual_next(const float* x, const float* post_weight,
                              const float* residual,
                              const float* next_weight, float* residual_out,
                              float* next_out, long long rows,
                              long long hidden, float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, post_weight, residual, next_weight,
                           residual_out, next_out)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 4,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* xr = x + row * hidden;
      const float* rr = residual + row * hidden;
      float* ro = residual_out + row * hidden;
      float* no = next_out + row * hidden;
      double sum = 0.0;
      for (long long i = 0; i < hidden; ++i) sum += double(xr[i]) * xr[i];
      const double post_rstd = 1.0 / std::sqrt(sum / hidden + eps);
      double next_sum = 0.0;
      for (long long i = 0; i < hidden; ++i) {
        ro[i] = static_cast<float>(rr[i] + xr[i] * post_rstd * post_weight[i]);
        next_sum += double(ro[i]) * ro[i];
      }
      const double next_rstd = 1.0 / std::sqrt(next_sum / hidden + eps);
      for (long long i = 0; i < hidden; ++i) {
        no[i] = static_cast<float>(ro[i] * next_rstd * next_weight[i]);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
