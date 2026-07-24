#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/float_storage.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

template <typename Function>
Status unary_apply(const float* x, float* out, long long count,
                   Function function) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(count, 16384,
      [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) out[i] = function(x[i]);
  });
  return Status::kOk;
}

template <typename Function>
Status binary_apply(const float* x, const float* y, float* out,
                    long long count, Function function) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, y, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(count, 16384,
      [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) out[i] = function(x[i], y[i]);
  });
  return Status::kOk;
}

long long positive_mod(long long value, long long modulus) {
  const long long remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

float bf16_bits_to_float(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

}  // namespace

Status add_scalar(const float* x, float value, float* out, long long count) {
  if (!std::isfinite(value)) return Status::kInvalidArgument;
  return unary_apply(x, out, count, [value](float v) { return v + value; });
}

Status subtract(const float* x, const float* y, float* out, long long count) {
  return binary_apply(x, y, out, count, [](float a, float b) { return a - b; });
}

Status multiply(const float* x, const float* y, float* out, long long count) {
  return binary_apply(x, y, out, count, [](float a, float b) { return a * b; });
}

Status divide(const float* x, const float* y, float* out, long long count) {
  return binary_apply(x, y, out, count, [](float a, float b) { return a / b; });
}

Status scale(const float* x, float value, float* out, long long count) {
  if (!std::isfinite(value)) return Status::kInvalidArgument;
  return unary_apply(x, out, count, [value](float v) { return v * value; });
}

Status clamp(const float* x, float minimum, float maximum, float* out,
             long long count) {
  if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
      minimum > maximum) {
    return Status::kInvalidShape;
  }
  return unary_apply(x, out, count, [minimum, maximum](float v) {
    return std::clamp(v, minimum, maximum);
  });
}

Status value_clip(const float* x, float minimum, float maximum, float* out,
                  long long count) {
  if (std::isnan(minimum) || std::isnan(maximum) || minimum > maximum) {
    return Status::kInvalidShape;
  }
  return unary_apply(x, out, count, [minimum, maximum](float value) {
    return std::clamp(value, minimum, maximum);
  });
}

Status value_clip_storage(FloatStorageInput x, FloatStorageOutput out,
                          float minimum, float maximum,
                          FloatStorageWorkspace* workspace) {
  if (x.count <= 0 || x.count != out.count) return Status::kInvalidShape;
  if (std::isnan(minimum) || std::isnan(maximum) || minimum > maximum) {
    return Status::kInvalidShape;
  }
  if (x.type == FloatStorageType::kBF16 &&
      out.type == FloatStorageType::kBF16) {
    if (!detail::all_nonnull(x.data, out.data)) {
      return Status::kInvalidArgument;
    }
    const auto* source = static_cast<const std::uint16_t*>(x.data);
    auto* destination = static_cast<std::uint16_t*>(out.data);
    const std::uint16_t minimum_bits = float_to_bf16(minimum);
    const std::uint16_t maximum_bits = float_to_bf16(maximum);
    threading::parallel_ranges(
        x.count, 16384, [&](long long begin, long long end, int) {
          for (long long i = begin; i < end; ++i) {
            const float value = bf16_bits_to_float(source[i]);
            destination[i] = value < minimum
                                 ? minimum_bits
                                 : (value > maximum ? maximum_bits : source[i]);
          }
        });
    return Status::kOk;
  }
  return with_float_storage(
      &x, 1, &out, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return value_clip(inputs[0], minimum, maximum, outputs[0], x.count);
      },
      workspace);
}

Status square(const float* x, float* out, long long count) {
  return unary_apply(x, out, count, [](float v) { return v * v; });
}

Status square_root(const float* x, float* out, long long count) {
  return unary_apply(x, out, count, [](float v) { return std::sqrt(v); });
}

Status logarithm(const float* x, float* out, long long count) {
  return unary_apply(x, out, count, [](float v) { return std::log(v); });
}

Status sine(const float* x, float* out, long long count) {
  return unary_apply(x, out, count, [](float v) { return std::sin(v); });
}

Status cosine(const float* x, float* out, long long count) {
  return unary_apply(x, out, count, [](float v) { return std::cos(v); });
}

Status leaky_relu(const float* x, float negative_slope, float* out,
                  long long count) {
  if (!std::isfinite(negative_slope)) return Status::kInvalidArgument;
  return unary_apply(x, out, count, [negative_slope](float v) {
    return v >= 0.0f ? v : negative_slope * v;
  });
}

Status fill(float* out, long long count, float value) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (out == nullptr || !std::isfinite(value)) return Status::kInvalidArgument;
  std::fill_n(out, count, value);
  return Status::kOk;
}

Status arange(float start, float step, float* out, long long count) {
  if (!detail::valid_product({count}) || !std::isfinite(step) || step == 0.0f) {
    return Status::kInvalidShape;
  }
  if (out == nullptr || !std::isfinite(start)) return Status::kInvalidArgument;
  for (long long i = 0; i < count; ++i) out[i] = start + step * i;
  return Status::kOk;
}

Status cumulative_sum(const float* x, float* out, long long rows,
                      long long dim) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 8,
      [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      float sum = 0.0f;
      for (long long i = 0; i < dim; ++i) {
        sum += x[row * dim + i];
        out[row * dim + i] = sum;
      }
    }
  });
  return Status::kOk;
}

Status reduce_sum_all(const float* x, float* out, long long count) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  double sum = 0.0;
  for (long long i = 0; i < count; ++i) sum += x[i];
  *out = static_cast<float>(sum);
  return Status::kOk;
}

Status reduce_mean(const float* x, float* out, long long rows, long long dim) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 16,
      [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      double sum = 0.0;
      for (long long i = 0; i < dim; ++i) sum += x[row * dim + i];
      out[row] = static_cast<float>(sum / dim);
    }
  });
  return Status::kOk;
}

Status count_equal(const std::int32_t* x, const std::int32_t* y,
                   long long count, long long* out) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, y, out)) return Status::kInvalidArgument;
  long long equal = 0;
  for (long long i = 0; i < count; ++i) equal += x[i] == y[i];
  *out = equal;
  return Status::kOk;
}

Status argsort(const float* x, int* indices, long long rows, long long dim,
               bool descending) {
  if (!detail::valid_product({rows, dim}) || dim > INT32_MAX) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, indices)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 1,
      [&](long long begin, long long end, int) {
    std::vector<int> order(static_cast<std::size_t>(dim));
    for (long long row = begin; row < end; ++row) {
      std::iota(order.begin(), order.end(), 0);
      const float* values = x + row * dim;
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return descending ? values[a] > values[b] : values[a] < values[b];
      });
      std::copy(order.begin(), order.end(), indices + row * dim);
    }
  });
  return Status::kOk;
}

Status concat(const float* a, const float* b, float* out, long long outer,
              long long a_axis, long long b_axis, long long inner) {
  if (!detail::valid_product({outer, a_axis, inner}) ||
      !detail::valid_product({outer, b_axis, inner}) ||
      a_axis > LLONG_MAX - b_axis) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, out)) return Status::kInvalidArgument;
  const long long output_axis = a_axis + b_axis;
  for (long long i = 0; i < outer; ++i) {
    std::copy_n(a + i * a_axis * inner, a_axis * inner,
                out + i * output_axis * inner);
    std::copy_n(b + i * b_axis * inner, b_axis * inner,
                out + (i * output_axis + a_axis) * inner);
  }
  return Status::kOk;
}

Status repeat_2d(const float* x, float* out, long long source_rows,
                 long long source_cols, long long output_rows,
                 long long output_cols) {
  if (!detail::valid_product({source_rows, source_cols}) ||
      !detail::valid_product({output_rows, output_cols}) ||
      output_rows % source_rows != 0 || output_cols % source_cols != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  for (long long row = 0; row < output_rows; ++row) {
    for (long long col = 0; col < output_cols; ++col) {
      out[row * output_cols + col] =
          x[(row % source_rows) * source_cols + col % source_cols];
    }
  }
  return Status::kOk;
}

Status repeat_backward_2d(const float* grad_out, float* grad_in,
                          long long source_rows, long long source_cols,
                          long long output_rows, long long output_cols) {
  if (!detail::valid_product({source_rows, source_cols}) ||
      !detail::valid_product({output_rows, output_cols}) ||
      output_rows % source_rows != 0 || output_cols % source_cols != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_out, grad_in)) return Status::kInvalidArgument;
  std::fill_n(grad_in, source_rows * source_cols, 0.0f);
  for (long long row = 0; row < output_rows; ++row) {
    for (long long col = 0; col < output_cols; ++col) {
      grad_in[(row % source_rows) * source_cols + col % source_cols] +=
          grad_out[row * output_cols + col];
    }
  }
  return Status::kOk;
}

Status diag_embed(const float* diagonal, float* out, long long batch,
                  long long dim) {
  if (!detail::valid_product({batch, dim, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(diagonal, out)) return Status::kInvalidArgument;
  std::fill_n(out, batch * dim * dim, 0.0f);
  for (long long b = 0; b < batch; ++b) {
    for (long long i = 0; i < dim; ++i) {
      out[(b * dim + i) * dim + i] = diagonal[b * dim + i];
    }
  }
  return Status::kOk;
}

Status diag_mask(const float* x, float* out, long long rows, long long cols,
                 long long past, bool use_negative_infinity) {
  if (!detail::valid_product({rows, cols}) || past < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  const float masked = use_negative_infinity
      ? -std::numeric_limits<float>::infinity()
      : 0.0f;
  for (long long row = 0; row < rows; ++row) {
    for (long long col = 0; col < cols; ++col) {
      out[row * cols + col] = col > past + row ? masked : x[row * cols + col];
    }
  }
  return Status::kOk;
}

Status triangular_fill(const float* x, float* out, long long rows,
                       long long cols, long long diagonal, bool upper,
                       float fill_value) {
  if (!detail::valid_product({rows, cols}) || !std::isfinite(fill_value)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    for (long long col = 0; col < cols; ++col) {
      const bool selected = upper ? col - row >= diagonal
                                  : col - row <= diagonal;
      out[row * cols + col] = selected ? x[row * cols + col] : fill_value;
    }
  }
  return Status::kOk;
}

Status roll_2d(const float* x, float* out, long long rows, long long cols,
               long long row_shift, long long col_shift) {
  if (!detail::valid_product({rows, cols})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, out) || x == out) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    for (long long col = 0; col < cols; ++col) {
      const long long output_row = positive_mod(row + row_shift, rows);
      const long long output_col = positive_mod(col + col_shift, cols);
      out[output_row * cols + output_col] = x[row * cols + col];
    }
  }
  return Status::kOk;
}

Status pad_2d(const float* x, float* out, long long rows, long long cols,
              long long top, long long bottom, long long left,
              long long right, float value) {
  if (!detail::valid_product({rows, cols}) || top < 0 || bottom < 0 ||
      left < 0 || right < 0 || rows > LLONG_MAX - top - bottom ||
      cols > LLONG_MAX - left - right || !std::isfinite(value)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  const long long output_rows = rows + top + bottom;
  const long long output_cols = cols + left + right;
  if (!detail::valid_product({output_rows, output_cols})) {
    return Status::kInvalidShape;
  }
  std::fill_n(out, output_rows * output_cols, value);
  for (long long row = 0; row < rows; ++row) {
    std::copy_n(x + row * cols, cols,
                out + (row + top) * output_cols + left);
  }
  return Status::kOk;
}

Status pad_reflect_1d(const float* x, float* out, long long rows,
                      long long length, long long left, long long right) {
  if (!detail::valid_product({rows, length}) || left < 0 || right < 0 ||
      left >= length || right >= length || length > LLONG_MAX - left - right) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  const long long output_length = left + length + right;
  for (long long row = 0; row < rows; ++row) {
    for (long long i = 0; i < output_length; ++i) {
      long long source = i - left;
      if (source < 0) source = -source;
      if (source >= length) source = 2 * length - source - 2;
      out[row * output_length + i] = x[row * length + source];
    }
  }
  return Status::kOk;
}

Status upscale_nearest_2d(const float* x, float* out, long long channels,
                          long long input_height, long long input_width,
                          long long scale_height, long long scale_width) {
  if (!detail::valid_product({channels, input_height, input_width,
                              scale_height, scale_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  const long long output_height = input_height * scale_height;
  const long long output_width = input_width * scale_width;
  for (long long channel = 0; channel < channels; ++channel) {
    for (long long row = 0; row < output_height; ++row) {
      for (long long col = 0; col < output_width; ++col) {
        out[(channel * output_height + row) * output_width + col] =
            x[(channel * input_height + row / scale_height) * input_width +
              col / scale_width];
      }
    }
  }
  return Status::kOk;
}

Status group_norm(const float* x, const float* weight, const float* bias,
                  float* out, long long batch, long long channels,
                  long long spatial, long long groups, float eps) {
  if (!detail::valid_product({batch, channels, spatial}) || groups <= 0 ||
      channels % groups != 0 || !std::isfinite(eps) || eps <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  const long long channels_per_group = channels / groups;
  const long long group_size = channels_per_group * spatial;
  threading::parallel_ranges(batch * groups, 1,
      [&](long long begin, long long end, int) {
    for (long long index = begin; index < end; ++index) {
      const long long b = index / groups;
      const long long group = index % groups;
      const long long offset = (b * channels + group * channels_per_group) *
                               spatial;
      double sum = 0.0;
      double sum_squared = 0.0;
      for (long long i = 0; i < group_size; ++i) {
        sum += x[offset + i];
        sum_squared += static_cast<double>(x[offset + i]) * x[offset + i];
      }
      const double mean = sum / group_size;
      const float inverse = static_cast<float>(
          1.0 / std::sqrt(sum_squared / group_size - mean * mean + eps));
      for (long long channel = 0; channel < channels_per_group; ++channel) {
        const long long global_channel = group * channels_per_group + channel;
        const float gain = weight == nullptr ? 1.0f : weight[global_channel];
        const float shift = bias == nullptr ? 0.0f : bias[global_channel];
        for (long long i = 0; i < spatial; ++i) {
          const long long location = offset + channel * spatial + i;
          out[location] = (x[location] - static_cast<float>(mean)) * inverse *
                          gain + shift;
        }
      }
    }
  });
  return Status::kOk;
}

Status l2_normalize(const float* x, float* out, long long rows, long long dim,
                    float eps) {
  if (!detail::valid_product({rows, dim}) || !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 8,
      [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      double sum = 0.0;
      for (long long i = 0; i < dim; ++i) {
        sum += static_cast<double>(x[row * dim + i]) * x[row * dim + i];
      }
      const float inverse = static_cast<float>(1.0 / std::sqrt(sum + eps));
      for (long long i = 0; i < dim; ++i) {
        out[row * dim + i] = x[row * dim + i] * inverse;
      }
    }
  });
  return Status::kOk;
}

Status softmax_backward(const float* grad_out, const float* softmax_output,
                        float* grad_in, long long rows, long long dim) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(grad_out, softmax_output, grad_in)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 8,
      [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      double dot = 0.0;
      for (long long i = 0; i < dim; ++i) {
        dot += static_cast<double>(grad_out[row * dim + i]) *
               softmax_output[row * dim + i];
      }
      for (long long i = 0; i < dim; ++i) {
        grad_in[row * dim + i] = softmax_output[row * dim + i] *
            (grad_out[row * dim + i] - static_cast<float>(dot));
      }
    }
  });
  return Status::kOk;
}

Status rope_backward(const float* grad_out, float* grad_in, long long tokens,
                     long long heads, long long head_dim, float base,
                     long long pos0) {
  if (!detail::valid_product({tokens, heads, head_dim}) ||
      (head_dim & 1) != 0 || !std::isfinite(base) || base <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_out, grad_in)) return Status::kInvalidArgument;
  const long long half = head_dim / 2;
  for (long long token = 0; token < tokens; ++token) {
    const double position = static_cast<double>(pos0 + token);
    for (long long head = 0; head < heads; ++head) {
      const long long offset = (token * heads + head) * head_dim;
      for (long long i = 0; i < half; ++i) {
        const double theta = position *
            std::pow(static_cast<double>(base), -2.0 * i / head_dim);
        const float cosine_value = static_cast<float>(std::cos(theta));
        const float sine_value = static_cast<float>(std::sin(theta));
        const float first = grad_out[offset + i];
        const float second = grad_out[offset + half + i];
        grad_in[offset + i] = first * cosine_value + second * sine_value;
        grad_in[offset + half + i] =
            -first * sine_value + second * cosine_value;
      }
    }
  }
  return Status::kOk;
}

Status outer_product(const float* x, const float* y, float* out,
                     long long rows, long long cols) {
  if (!detail::valid_product({rows, cols})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, y, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 16,
      [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long col = 0; col < cols; ++col) {
        out[row * cols + col] = x[row] * y[col];
      }
    }
  });
  return Status::kOk;
}

Status set_rows(const float* source, const int* row_ids, float* destination,
                long long source_rows, long long destination_rows,
                long long row_width) {
  if (!detail::valid_product({source_rows, row_width}) ||
      !detail::valid_product({destination_rows, row_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(source, row_ids, destination)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < source_rows; ++row) {
    if (row_ids[row] < 0 || row_ids[row] >= destination_rows) {
      return Status::kInvalidArgument;
    }
  }
  for (long long row = 0; row < source_rows; ++row) {
    std::copy_n(source + row * row_width, row_width,
                destination + static_cast<long long>(row_ids[row]) * row_width);
  }
  return Status::kOk;
}

Status accumulate(float* destination, const float* source, long long count,
                  float alpha) {
  if (!detail::valid_product({count}) || !std::isfinite(alpha)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(destination, source)) return Status::kInvalidArgument;
  for (long long i = 0; i < count; ++i) destination[i] += alpha * source[i];
  return Status::kOk;
}

Status sgd(float* parameters, const float* gradients, long long count,
           float learning_rate, float weight_decay) {
  if (!detail::valid_product({count}) || !std::isfinite(learning_rate) ||
      learning_rate < 0.0f || !std::isfinite(weight_decay) ||
      weight_decay < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(parameters, gradients)) {
    return Status::kInvalidArgument;
  }
  for (long long i = 0; i < count; ++i) {
    parameters[i] -=
        learning_rate * (gradients[i] + weight_decay * parameters[i]);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
