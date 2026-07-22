#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status mamba2(const float* c, const float* b, const float* x,
              const float* cumulative_log, float* y, long long batch,
              long long heads, long long sequence, long long dim) {
  if (!detail::valid_product({batch, heads, sequence, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(c, b, x, cumulative_log, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long bh = begin; bh < end; ++bh) {
      const long long offset = bh * sequence * dim;
      const float* decay = cumulative_log + bh * sequence;
      for (long long target = 0; target < sequence; ++target) {
        float* destination = y + offset + target * dim;
        std::fill_n(destination, dim, 0.0f);
        for (long long source = 0; source <= target; ++source) {
          double similarity = 0.0;
          for (long long d = 0; d < dim; ++d) {
            similarity += double(c[offset + target * dim + d]) *
                          b[offset + source * dim + d];
          }
          const double coefficient =
              similarity * std::exp(static_cast<double>(decay[target] -
                                                        decay[source]));
          for (long long d = 0; d < dim; ++d) {
            destination[d] +=
                static_cast<float>(coefficient * x[offset + source * dim + d]);
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status mamba2_backward(const float* c, const float* b, const float* x,
                       const float* cumulative_log, const float* grad_y,
                       float* grad_c, float* grad_b, float* grad_x,
                       float* grad_cumulative_log, long long batch,
                       long long heads, long long sequence, long long dim) {
  if (!detail::valid_product({batch, heads, sequence, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(c, b, x, cumulative_log, grad_y, grad_c, grad_b,
                           grad_x, grad_cumulative_log)) {
    return Status::kInvalidArgument;
  }
  const long long tensor_count = batch * heads * sequence * dim;
  const long long decay_count = batch * heads * sequence;
  std::fill_n(grad_c, tensor_count, 0.0f);
  std::fill_n(grad_b, tensor_count, 0.0f);
  std::fill_n(grad_x, tensor_count, 0.0f);
  std::fill_n(grad_cumulative_log, decay_count, 0.0f);
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long bh = begin; bh < end; ++bh) {
      const long long offset = bh * sequence * dim;
      const long long decay_offset = bh * sequence;
      for (long long target = 0; target < sequence; ++target) {
        for (long long source = 0; source <= target; ++source) {
          double similarity = 0.0;
          double grad_dot_x = 0.0;
          for (long long d = 0; d < dim; ++d) {
            similarity += double(c[offset + target * dim + d]) *
                          b[offset + source * dim + d];
            grad_dot_x += double(grad_y[offset + target * dim + d]) *
                          x[offset + source * dim + d];
          }
          const double decay = std::exp(
              static_cast<double>(cumulative_log[decay_offset + target] -
                                  cumulative_log[decay_offset + source]));
          const double score_grad = decay * grad_dot_x;
          for (long long d = 0; d < dim; ++d) {
            grad_c[offset + target * dim + d] +=
                static_cast<float>(score_grad * b[offset + source * dim + d]);
            grad_b[offset + source * dim + d] +=
                static_cast<float>(score_grad * c[offset + target * dim + d]);
            grad_x[offset + source * dim + d] += static_cast<float>(
                decay * similarity * grad_y[offset + target * dim + d]);
          }
          const float decay_gradient =
              static_cast<float>(decay * similarity * grad_dot_x);
          grad_cumulative_log[decay_offset + target] += decay_gradient;
          grad_cumulative_log[decay_offset + source] -= decay_gradient;
        }
      }
    }
  });
  return Status::kOk;
}

Status ssd_chunked(const float* c, const float* b, const float* x,
                   const float* cumulative_log, float* y, long long batch,
                   long long heads, long long sequence, long long dim) {
  return mamba2(c, b, x, cumulative_log, y, batch, heads, sequence, dim);
}

Status ssd_chunked_backward(
    const float* c, const float* b, const float* x,
    const float* cumulative_log, const float* grad_y, float* grad_c,
    float* grad_b, float* grad_x, float* grad_cumulative_log,
    long long batch, long long heads, long long sequence, long long dim) {
  return mamba2_backward(c, b, x, cumulative_log, grad_y, grad_c, grad_b,
                         grad_x, grad_cumulative_log, batch, heads, sequence,
                         dim);
}

Status ssd_decode(const float* state, const float* alpha, const float* x,
                  const float* k, const float* q, float* y, float* next_state,
                  long long batch, long long heads, long long dim) {
  if (!detail::valid_product({batch, heads, dim, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(state, alpha, x, k, q, y, next_state)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long bh = begin; bh < end; ++bh) {
      const float* state_row = state + bh * dim * dim;
      float* next = next_state + bh * dim * dim;
      const float* xr = x + bh * dim;
      const float* kr = k + bh * dim;
      const float* qr = q + bh * dim;
      float* destination = y + bh * dim;
      for (long long value = 0; value < dim; ++value) {
        double result = 0.0;
        for (long long key = 0; key < dim; ++key) {
          const long long index = value * dim + key;
          next[index] =
              alpha[bh] * state_row[index] + xr[value] * kr[key];
          result += double(next[index]) * qr[key];
        }
        destination[value] = static_cast<float>(result);
      }
    }
  });
  return Status::kOk;
}

Status fft_convolution(const float* x, const float* kernel, float* out,
                       long long batch, long long heads, long long length) {
  if (!detail::valid_product({batch, heads, length})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, kernel, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long bh = begin; bh < end; ++bh) {
      const long long head = bh % heads;
      const float* xr = x + bh * length;
      const float* kr = kernel + head * length;
      float* destination = out + bh * length;
      for (long long target = 0; target < length; ++target) {
        double accumulator = 0.0;
        for (long long source = 0; source < length; ++source) {
          long long kernel_index = target - source;
          if (kernel_index < 0) kernel_index += length;
          accumulator += double(xr[source]) * kr[kernel_index];
        }
        destination[target] = static_cast<float>(accumulator);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
