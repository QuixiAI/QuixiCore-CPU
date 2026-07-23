#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status mamba2(const float* c, const float* b, const float* x,
              const float* cumulative_log, float* y, long long batch,
              long long heads, long long sequence, long long dim) {
  if (!detail::valid_product({batch, heads, sequence, dim}) ||
      !detail::valid_product({dim, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(c, b, x, cumulative_log, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<float> state;
    thread_local std::vector<float> projected;
    for (long long bh = begin; bh < end; ++bh) {
      const long long offset = bh * sequence * dim;
      const float* decay = cumulative_log + bh * sequence;
      state.assign(static_cast<std::size_t>(dim * dim), 0.0);
      projected.resize(static_cast<std::size_t>(dim));
      for (long long target = 0; target < sequence; ++target) {
        const float transition =
            target == 0
                ? 0.0f
                : std::exp(decay[target] - decay[target - 1]);
        const float* bt = b + offset + target * dim;
        const float* xt = x + offset + target * dim;
        const float* ct = c + offset + target * dim;
        std::fill(projected.begin(), projected.end(), 0.0);
        for (long long key = 0; key < dim; ++key) {
          float* state_row = state.data() + key * dim;
          const float coefficient = ct[key];
          for (long long value = 0; value < dim; ++value) {
            state_row[value] =
                (target == 0 ? 0.0f : transition * state_row[value]) +
                bt[key] * xt[value];
            projected[static_cast<std::size_t>(value)] +=
                coefficient * state_row[value];
          }
        }
        float* destination = y + offset + target * dim;
        for (long long value = 0; value < dim; ++value) {
          destination[value] = projected[static_cast<std::size_t>(value)];
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
  const bool radix2 = (length & (length - 1)) == 0;
  if (!radix2 || length < 64) {
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

  using Complex = std::complex<float>;
  const auto transform = [](Complex* values, long long count, bool inverse) {
    for (long long item = 1, reversed = 0; item < count; ++item) {
      long long bit = count >> 1;
      while (reversed & bit) {
        reversed ^= bit;
        bit >>= 1;
      }
      reversed ^= bit;
      if (item < reversed) std::swap(values[item], values[reversed]);
    }
    constexpr float kPi = 3.14159265358979323846f;
    for (long long width = 2; width <= count;) {
      const float angle = (inverse ? 2.0f : -2.0f) * kPi /
                          static_cast<float>(width);
      const Complex root(std::cos(angle), std::sin(angle));
      for (long long base = 0; base < count; base += width) {
        Complex phase(1.0, 0.0);
        for (long long item = 0; item < width / 2; ++item) {
          const Complex even = values[base + item];
          const Complex odd = values[base + item + width / 2] * phase;
          values[base + item] = even + odd;
          values[base + item + width / 2] = even - odd;
          phase *= root;
        }
      }
      if (width == count) break;
      width <<= 1;
    }
    if (inverse) {
      const float scale = 1.0f / static_cast<float>(count);
      for (long long item = 0; item < count; ++item) values[item] *= scale;
    }
  };

  if (batch == 1) {
    threading::parallel_ranges(heads, 1,
                               [&](long long begin, long long end, int) {
      thread_local std::vector<Complex> spectrum;
      thread_local std::vector<Complex> signal;
      spectrum.resize(static_cast<std::size_t>(length));
      signal.resize(static_cast<std::size_t>(length));
      for (long long head = begin; head < end; ++head) {
        for (long long item = 0; item < length; ++item) {
          spectrum[static_cast<std::size_t>(item)] =
              Complex(kernel[head * length + item], 0.0f);
          signal[static_cast<std::size_t>(item)] =
              Complex(x[head * length + item], 0.0f);
        }
        transform(spectrum.data(), length, false);
        transform(signal.data(), length, false);
        for (long long item = 0; item < length; ++item) {
          signal[static_cast<std::size_t>(item)] *=
              spectrum[static_cast<std::size_t>(item)];
        }
        transform(signal.data(), length, true);
        for (long long item = 0; item < length; ++item) {
          out[head * length + item] =
              signal[static_cast<std::size_t>(item)].real();
        }
      }
    });
    return Status::kOk;
  }

  std::vector<Complex> kernel_spectrum(
      static_cast<std::size_t>(heads * length));
  threading::parallel_ranges(heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long head = begin; head < end; ++head) {
      Complex* spectrum = kernel_spectrum.data() + head * length;
      for (long long item = 0; item < length; ++item) {
        spectrum[item] = Complex(kernel[head * length + item], 0.0);
      }
      transform(spectrum, length, false);
    }
  });
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<Complex> signal;
    for (long long bh = begin; bh < end; ++bh) {
      const long long head = bh % heads;
      const float* xr = x + bh * length;
      float* destination = out + bh * length;
      signal.resize(static_cast<std::size_t>(length));
      for (long long item = 0; item < length; ++item) {
        signal[static_cast<std::size_t>(item)] = Complex(xr[item], 0.0);
      }
      transform(signal.data(), length, false);
      const Complex* spectrum = kernel_spectrum.data() + head * length;
      for (long long item = 0; item < length; ++item) {
        signal[static_cast<std::size_t>(item)] *= spectrum[item];
      }
      transform(signal.data(), length, true);
      for (long long item = 0; item < length; ++item) {
        destination[item] =
            static_cast<float>(signal[static_cast<std::size_t>(item)].real());
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
