#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool convolution_output(long long input, long long kernel, long long stride,
                        long long padding, long long dilation,
                        long long* output) {
  if (input <= 0 || kernel <= 0 || stride <= 0 || padding < 0 ||
      dilation <= 0 || kernel - 1 > (LLONG_MAX - 1) / dilation ||
      padding > LLONG_MAX / 2) {
    return false;
  }
  const long long extent = dilation * (kernel - 1) + 1;
  const long long padded = 2 * padding;
  if (input > LLONG_MAX - padded || input + padded < extent) {
    return false;
  }
  *output = (input + 2 * padding - extent) / stride + 1;
  return *output > 0;
}

bool valid_pool_mode(PoolMode mode) {
  return mode == PoolMode::kAverage || mode == PoolMode::kMaximum;
}

}  // namespace

Status add_id(const float* x, const float* rows, const int* ids, float* out,
              long long count, long long row_count, long long width) {
  if (!detail::valid_product({count, width}) ||
      !detail::valid_product({row_count, width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, rows, ids, out)) return Status::kInvalidArgument;
  for (long long i = 0; i < count; ++i) {
    if (ids[i] < 0 || ids[i] >= row_count) return Status::kInvalidArgument;
  }
  threading::parallel_ranges(count, 16,
      [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) {
      for (long long j = 0; j < width; ++j) {
        out[i * width + j] = x[i * width + j] + rows[ids[i] * width + j];
      }
    }
  });
  return Status::kOk;
}

Status tensor_copy(const float* source, float* destination, long long count) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(source, destination)) return Status::kInvalidArgument;
  std::copy_n(source, count, destination);
  return Status::kOk;
}

Status tensor_set_4d(const float* base, const float* update, float* output,
                     long long output_count, long long n0, long long n1,
                     long long n2, long long n3, long long stride1,
                     long long stride2, long long stride3,
                     long long offset) {
  if (!detail::valid_product({output_count}) ||
      !detail::valid_product({n0, n1, n2, n3}) || stride1 < n0 ||
      stride2 < 0 || stride3 < 0 || offset < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(base, update, output)) return Status::kInvalidArgument;
  long long maximum = offset;
  const long long terms[][2] = {
      {n3 - 1, stride3}, {n2 - 1, stride2}, {n1 - 1, stride1}, {1, n0}};
  for (const auto& term : terms) {
    if (term[0] != 0 && term[1] > (LLONG_MAX - maximum) / term[0]) {
      return Status::kInvalidShape;
    }
    maximum += term[0] * term[1];
  }
  if (maximum > output_count) return Status::kInvalidShape;
  if (output != base) std::copy_n(base, output_count, output);
  threading::parallel_ranges(n3 * n2, 1,
      [&](long long begin, long long end, int) {
    for (long long plane = begin; plane < end; ++plane) {
      const long long i3 = plane / n2;
      const long long i2 = plane % n2;
      for (long long i1 = 0; i1 < n1; ++i1) {
        const float* source = update + ((i3 * n2 + i2) * n1 + i1) * n0;
        float* destination = output + offset + i3 * stride3 + i2 * stride2 +
                             i1 * stride1;
        std::copy_n(source, n0, destination);
      }
    }
  });
  return Status::kOk;
}

Status im2col_2d(const float* image, float* columns, long long batch,
                 long long channels, long long input_height,
                 long long input_width, long long kernel_height,
                 long long kernel_width, long long stride_height,
                 long long stride_width, long long pad_height,
                 long long pad_width, long long dilation_height,
                 long long dilation_width) {
  long long output_height = 0, output_width = 0;
  if (!detail::valid_product({batch, channels, input_height, input_width}) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, dilation_height, &output_height) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          dilation_width, &output_width) ||
      !detail::valid_product({batch, output_height, output_width, channels,
                              kernel_height, kernel_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(image, columns)) return Status::kInvalidArgument;
  const long long patch = channels * kernel_height * kernel_width;
  threading::parallel_ranges(batch * output_height, 1,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long n = item / output_height;
      const long long oh = item % output_height;
      for (long long ow = 0; ow < output_width; ++ow) {
        float* destination = columns +
            ((n * output_height + oh) * output_width + ow) * patch;
        for (long long c = 0; c < channels; ++c) {
          for (long long kh = 0; kh < kernel_height; ++kh) {
            const long long ih = oh * stride_height + kh * dilation_height -
                                 pad_height;
            for (long long kw = 0; kw < kernel_width; ++kw) {
              const long long iw = ow * stride_width + kw * dilation_width -
                                   pad_width;
              const long long index = (c * kernel_height + kh) * kernel_width + kw;
              destination[index] = ih >= 0 && ih < input_height && iw >= 0 &&
                                           iw < input_width
                  ? image[((n * channels + c) * input_height + ih) *
                          input_width + iw]
                  : 0.0f;
            }
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status col2im_2d(const float* columns, float* image, long long batch,
                 long long channels, long long input_height,
                 long long input_width, long long kernel_height,
                 long long kernel_width, long long stride_height,
                 long long stride_width, long long pad_height,
                 long long pad_width, long long dilation_height,
                 long long dilation_width) {
  long long output_height = 0, output_width = 0;
  if (!detail::valid_product({batch, channels, input_height, input_width}) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, dilation_height, &output_height) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          dilation_width, &output_width) ||
      !detail::valid_product({batch, output_height, output_width, channels,
                              kernel_height, kernel_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(columns, image)) return Status::kInvalidArgument;
  std::fill_n(image, batch * channels * input_height * input_width, 0.0f);
  const long long patch = channels * kernel_height * kernel_width;
  for (long long n = 0; n < batch; ++n) {
    for (long long oh = 0; oh < output_height; ++oh) {
      for (long long ow = 0; ow < output_width; ++ow) {
        const float* source = columns +
            ((n * output_height + oh) * output_width + ow) * patch;
        for (long long c = 0; c < channels; ++c) {
          for (long long kh = 0; kh < kernel_height; ++kh) {
            const long long ih = oh * stride_height + kh * dilation_height -
                                 pad_height;
            for (long long kw = 0; kw < kernel_width; ++kw) {
              const long long iw = ow * stride_width + kw * dilation_width -
                                   pad_width;
              if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                image[((n * channels + c) * input_height + ih) * input_width + iw] +=
                    source[(c * kernel_height + kh) * kernel_width + kw];
              }
            }
          }
        }
      }
    }
  }
  return Status::kOk;
}

Status col2im_1d(const float* columns, float* signal, long long time_in,
                 long long channels, long long kernel, long long stride,
                 long long padding) {
  if (!detail::valid_product({time_in, channels, kernel, stride}) ||
      padding < 0 || padding > LLONG_MAX / 2 ||
      kernel > LLONG_MAX - 2 * padding ||
      time_in - 1 > (LLONG_MAX - kernel) / stride) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(columns, signal)) return Status::kInvalidArgument;
  const long long time_out = (time_in - 1) * stride + kernel - 2 * padding;
  if (time_out <= 0 || !detail::valid_product({channels, time_out})) {
    return Status::kInvalidShape;
  }
  threading::parallel_ranges(channels, 1,
      [&](long long begin, long long end, int) {
    for (long long channel = begin; channel < end; ++channel) {
      for (long long output_index = 0; output_index < time_out; ++output_index) {
        const long long absolute = output_index + padding;
        float sum = 0.0f;
        for (long long input_index = 0; input_index < time_in; ++input_index) {
          const long long tap = absolute - input_index * stride;
          if (tap >= 0 && tap < kernel) {
            sum += columns[(input_index * channels + channel) * kernel + tap];
          }
        }
        signal[channel * time_out + output_index] = sum;
      }
    }
  });
  return Status::kOk;
}

Status im2col_3d(const float* volume, float* columns, long long batch,
                 long long channels, long long input_depth,
                 long long input_height, long long input_width,
                 long long kernel_depth, long long kernel_height,
                 long long kernel_width, long long stride_depth,
                 long long stride_height, long long stride_width,
                 long long pad_depth, long long pad_height,
                 long long pad_width, long long dilation_depth,
                 long long dilation_height, long long dilation_width) {
  long long od = 0, oh = 0, ow = 0;
  if (!detail::valid_product({batch, channels, input_depth, input_height,
                              input_width}) ||
      !convolution_output(input_depth, kernel_depth, stride_depth, pad_depth,
                          dilation_depth, &od) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, dilation_height, &oh) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          dilation_width, &ow) ||
      !detail::valid_product({batch, od, oh, ow, channels, kernel_depth,
                              kernel_height, kernel_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(volume, columns)) return Status::kInvalidArgument;
  const long long patch = channels * kernel_depth * kernel_height * kernel_width;
  for (long long n = 0; n < batch; ++n) {
    for (long long oz = 0; oz < od; ++oz) {
      for (long long oy = 0; oy < oh; ++oy) {
        for (long long ox = 0; ox < ow; ++ox) {
          float* destination = columns +
              (((n * od + oz) * oh + oy) * ow + ox) * patch;
          long long index = 0;
          for (long long c = 0; c < channels; ++c) {
            for (long long kz = 0; kz < kernel_depth; ++kz) {
              const long long iz = oz * stride_depth + kz * dilation_depth - pad_depth;
              for (long long ky = 0; ky < kernel_height; ++ky) {
                const long long iy = oy * stride_height + ky * dilation_height - pad_height;
                for (long long kx = 0; kx < kernel_width; ++kx, ++index) {
                  const long long ix = ox * stride_width + kx * dilation_width - pad_width;
                  destination[index] = iz >= 0 && iz < input_depth &&
                      iy >= 0 && iy < input_height && ix >= 0 && ix < input_width
                      ? volume[(((n * channels + c) * input_depth + iz) *
                                input_height + iy) * input_width + ix]
                      : 0.0f;
                }
              }
            }
          }
        }
      }
    }
  }
  return Status::kOk;
}

Status conv2d(const float* input, const float* weights, const float* bias,
              float* output, long long batch, long long input_channels,
              long long output_channels, long long input_height,
              long long input_width, long long kernel_height,
              long long kernel_width, long long stride_height,
              long long stride_width, long long pad_height,
              long long pad_width, long long dilation_height,
              long long dilation_width) {
  long long oh = 0, ow = 0;
  if (!detail::valid_product({batch, input_channels, output_channels,
                              input_height, input_width}) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, dilation_height, &oh) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          dilation_width, &ow) ||
      !detail::valid_product({output_channels, input_channels, kernel_height,
                              kernel_width}) ||
      !detail::valid_product({batch, output_channels, oh, ow})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) return Status::kInvalidArgument;
  threading::parallel_ranges(batch * output_channels, 1,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long n = item / output_channels;
      const long long oc = item % output_channels;
      for (long long oy = 0; oy < oh; ++oy) {
        for (long long ox = 0; ox < ow; ++ox) {
          float sum = bias == nullptr ? 0.0f : bias[oc];
          for (long long ic = 0; ic < input_channels; ++ic) {
            for (long long ky = 0; ky < kernel_height; ++ky) {
              const long long iy = oy * stride_height + ky * dilation_height - pad_height;
              if (iy < 0 || iy >= input_height) continue;
              for (long long kx = 0; kx < kernel_width; ++kx) {
                const long long ix = ox * stride_width + kx * dilation_width - pad_width;
                if (ix < 0 || ix >= input_width) continue;
                sum += input[((n * input_channels + ic) * input_height + iy) *
                             input_width + ix] *
                       weights[((oc * input_channels + ic) * kernel_height + ky) *
                               kernel_width + kx];
              }
            }
          }
          output[((n * output_channels + oc) * oh + oy) * ow + ox] =
              sum;
        }
      }
    }
  });
  return Status::kOk;
}

Status depthwise_conv2d(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long channels, long long multiplier,
    long long input_height, long long input_width, long long kernel_height,
    long long kernel_width, long long stride_height, long long stride_width,
    long long pad_height, long long pad_width, long long dilation_height,
    long long dilation_width) {
  long long oh = 0, ow = 0;
  if (!detail::valid_product({batch, channels, multiplier, input_height,
                              input_width}) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, dilation_height, &oh) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          dilation_width, &ow) ||
      !detail::valid_product({channels, multiplier, kernel_height,
                              kernel_width}) ||
      !detail::valid_product({batch, channels, multiplier, oh, ow})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) return Status::kInvalidArgument;
  const long long output_channels = channels * multiplier;
  threading::parallel_ranges(batch * output_channels, 1,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long n = item / output_channels;
      const long long oc = item % output_channels;
      const long long channel = oc / multiplier;
      const long long multiple = oc % multiplier;
      for (long long oy = 0; oy < oh; ++oy) {
        for (long long ox = 0; ox < ow; ++ox) {
          float sum = bias == nullptr ? 0.0f : bias[oc];
          for (long long ky = 0; ky < kernel_height; ++ky) {
            const long long iy = oy * stride_height + ky * dilation_height - pad_height;
            if (iy < 0 || iy >= input_height) continue;
            for (long long kx = 0; kx < kernel_width; ++kx) {
              const long long ix = ox * stride_width + kx * dilation_width - pad_width;
              if (ix < 0 || ix >= input_width) continue;
              sum += input[((n * channels + channel) * input_height + iy) *
                           input_width + ix] *
                     weights[((channel * multiplier + multiple) * kernel_height + ky) *
                             kernel_width + kx];
            }
          }
          output[((n * output_channels + oc) * oh + oy) * ow + ox] =
              sum;
        }
      }
    }
  });
  return Status::kOk;
}

Status conv3d(const float* input, const float* weights, const float* bias,
              float* output, long long batch, long long input_channels,
              long long output_channels, long long input_depth,
              long long input_height, long long input_width,
              long long kernel_depth, long long kernel_height,
              long long kernel_width, long long stride_depth,
              long long stride_height, long long stride_width,
              long long pad_depth, long long pad_height, long long pad_width,
              long long dilation_depth, long long dilation_height,
              long long dilation_width) {
  long long od = 0, oh = 0, ow = 0;
  if (!detail::valid_product({batch, input_channels, output_channels,
                              input_depth, input_height, input_width}) ||
      !convolution_output(input_depth, kernel_depth, stride_depth, pad_depth,
                          dilation_depth, &od) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, dilation_height, &oh) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          dilation_width, &ow) ||
      !detail::valid_product({output_channels, input_channels, kernel_depth,
                              kernel_height, kernel_width}) ||
      !detail::valid_product({batch, output_channels, od, oh, ow})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) return Status::kInvalidArgument;
  threading::parallel_ranges(batch * output_channels, 1,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long n = item / output_channels, oc = item % output_channels;
      for (long long oz = 0; oz < od; ++oz) for (long long oy = 0; oy < oh; ++oy)
      for (long long ox = 0; ox < ow; ++ox) {
        float sum = bias == nullptr ? 0.0f : bias[oc];
        for (long long ic = 0; ic < input_channels; ++ic)
        for (long long kz = 0; kz < kernel_depth; ++kz) {
          const long long iz = oz * stride_depth + kz * dilation_depth - pad_depth;
          if (iz < 0 || iz >= input_depth) continue;
          for (long long ky = 0; ky < kernel_height; ++ky) {
            const long long iy = oy * stride_height + ky * dilation_height - pad_height;
            if (iy < 0 || iy >= input_height) continue;
            for (long long kx = 0; kx < kernel_width; ++kx) {
              const long long ix = ox * stride_width + kx * dilation_width - pad_width;
              if (ix < 0 || ix >= input_width) continue;
              sum += input[(((n * input_channels + ic) * input_depth + iz) *
                            input_height + iy) * input_width + ix] *
                     weights[(((oc * input_channels + ic) * kernel_depth + kz) *
                              kernel_height + ky) * kernel_width + kx];
            }
          }
        }
        output[(((n * output_channels + oc) * od + oz) * oh + oy) * ow + ox] =
            sum;
      }
    }
  });
  return Status::kOk;
}

Status conv_transpose_1d(const float* input, const float* weights,
                         const float* bias, float* output, long long batch,
                         long long input_channels, long long output_channels,
                         long long input_length, long long kernel,
                         long long stride, long long padding) {
  if (!detail::valid_product({batch, input_channels, output_channels,
                              input_length, kernel, stride}) || padding < 0 ||
      padding > LLONG_MAX / 2 ||
      kernel > LLONG_MAX - 2 * padding ||
      (input_length - 1) > (LLONG_MAX - kernel) / stride) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) return Status::kInvalidArgument;
  const long long output_length = (input_length - 1) * stride - 2 * padding + kernel;
  if (output_length <= 0 ||
      !detail::valid_product({batch, output_channels, output_length}) ||
      !detail::valid_product({input_channels, output_channels, kernel})) {
    return Status::kInvalidShape;
  }
  for (long long n = 0; n < batch; ++n) for (long long oc = 0; oc < output_channels; ++oc)
    std::fill_n(output + (n * output_channels + oc) * output_length,
                output_length, bias == nullptr ? 0.0f : bias[oc]);
  for (long long n = 0; n < batch; ++n) for (long long ic = 0; ic < input_channels; ++ic)
  for (long long i = 0; i < input_length; ++i) for (long long kx = 0; kx < kernel; ++kx) {
    const long long ox = i * stride + kx - padding;
    if (ox < 0 || ox >= output_length) continue;
    for (long long oc = 0; oc < output_channels; ++oc) {
      output[(n * output_channels + oc) * output_length + ox] +=
          input[(n * input_channels + ic) * input_length + i] *
          weights[(ic * output_channels + oc) * kernel + kx];
    }
  }
  return Status::kOk;
}

Status conv_transpose_2d(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_channels, long long output_channels,
    long long input_height, long long input_width, long long kernel_height,
    long long kernel_width, long long stride_height, long long stride_width,
    long long pad_height, long long pad_width) {
  if (!detail::valid_product({batch, input_channels, output_channels,
                              input_height, input_width, kernel_height,
                              kernel_width, stride_height, stride_width}) ||
      pad_height < 0 || pad_width < 0 || pad_height > LLONG_MAX / 2 ||
      pad_width > LLONG_MAX / 2 || kernel_height > LLONG_MAX - 2 * pad_height ||
      kernel_width > LLONG_MAX - 2 * pad_width ||
      (input_height - 1) >
          (LLONG_MAX - kernel_height) / stride_height ||
      (input_width - 1) >
          (LLONG_MAX - kernel_width) / stride_width) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) return Status::kInvalidArgument;
  const long long oh = (input_height - 1) * stride_height - 2 * pad_height + kernel_height;
  const long long ow = (input_width - 1) * stride_width - 2 * pad_width + kernel_width;
  if (oh <= 0 || ow <= 0 ||
      !detail::valid_product({batch, output_channels, oh, ow}) ||
      !detail::valid_product({input_channels, output_channels, kernel_height,
                              kernel_width})) return Status::kInvalidShape;
  for (long long n = 0; n < batch; ++n) for (long long oc = 0; oc < output_channels; ++oc)
    std::fill_n(output + (n * output_channels + oc) * oh * ow, oh * ow,
                bias == nullptr ? 0.0f : bias[oc]);
  for (long long n = 0; n < batch; ++n) for (long long ic = 0; ic < input_channels; ++ic)
  for (long long iy = 0; iy < input_height; ++iy) for (long long ix = 0; ix < input_width; ++ix)
  for (long long ky = 0; ky < kernel_height; ++ky) for (long long kx = 0; kx < kernel_width; ++kx) {
    const long long oy = iy * stride_height + ky - pad_height;
    const long long ox = ix * stride_width + kx - pad_width;
    if (oy < 0 || oy >= oh || ox < 0 || ox >= ow) continue;
    for (long long oc = 0; oc < output_channels; ++oc) {
      output[((n * output_channels + oc) * oh + oy) * ow + ox] +=
          input[((n * input_channels + ic) * input_height + iy) * input_width + ix] *
          weights[((ic * output_channels + oc) * kernel_height + ky) * kernel_width + kx];
    }
  }
  return Status::kOk;
}

Status pool1d(const float* input, float* output, long long batch,
              long long channels, long long input_length, long long kernel,
              long long stride, long long padding, PoolMode mode) {
  long long length = 0;
  if (!detail::valid_product({batch, channels, input_length}) ||
      !valid_pool_mode(mode) ||
      !convolution_output(input_length, kernel, stride, padding, 1, &length)) {
    return Status::kInvalidShape;
  }
  if (!detail::valid_product({batch, channels, length}))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  for (long long n = 0; n < batch; ++n) for (long long c = 0; c < channels; ++c)
  for (long long o = 0; o < length; ++o) {
    float value = mode == PoolMode::kMaximum
        ? -std::numeric_limits<float>::infinity() : 0.0f;
    long long count = 0;
    for (long long kx = 0; kx < kernel; ++kx) {
      const long long i = o * stride + kx - padding;
      if (i < 0 || i >= input_length) continue;
      const float sample = input[(n * channels + c) * input_length + i];
      value = mode == PoolMode::kMaximum ? std::max(value, sample) : value + sample;
      ++count;
    }
    if (count == 0) return Status::kInvalidShape;
    output[(n * channels + c) * length + o] =
        mode == PoolMode::kAverage ? value / count : value;
  }
  return Status::kOk;
}

Status pool2d(const float* input, float* output, long long batch,
              long long channels, long long input_height,
              long long input_width, long long kernel_height,
              long long kernel_width, long long stride_height,
              long long stride_width, long long pad_height,
              long long pad_width, PoolMode mode) {
  long long oh = 0, ow = 0;
  if (!detail::valid_product({batch, channels, input_height, input_width}) ||
      !valid_pool_mode(mode) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, 1, &oh) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          1, &ow) ||
      !detail::valid_product({batch, channels, oh, ow})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  for (long long n = 0; n < batch; ++n) for (long long c = 0; c < channels; ++c)
  for (long long oy = 0; oy < oh; ++oy) for (long long ox = 0; ox < ow; ++ox) {
    float value = mode == PoolMode::kMaximum
        ? -std::numeric_limits<float>::infinity() : 0.0f;
    long long count = 0;
    for (long long ky = 0; ky < kernel_height; ++ky) for (long long kx = 0; kx < kernel_width; ++kx) {
      const long long iy = oy * stride_height + ky - pad_height;
      const long long ix = ox * stride_width + kx - pad_width;
      if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) continue;
      const float sample = input[((n * channels + c) * input_height + iy) * input_width + ix];
      value = mode == PoolMode::kMaximum ? std::max(value, sample) : value + sample;
      ++count;
    }
    if (count == 0) return Status::kInvalidShape;
    output[((n * channels + c) * oh + oy) * ow + ox] =
        mode == PoolMode::kAverage ? value / count : value;
  }
  return Status::kOk;
}

Status pool2d_backward(
    const float* input, const float* grad_out, float* grad_in, long long batch,
    long long channels, long long input_height, long long input_width,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, long long pad_height, long long pad_width,
    PoolMode mode) {
  long long oh = 0, ow = 0;
  if (!detail::valid_product({batch, channels, input_height, input_width}) ||
      !valid_pool_mode(mode) ||
      !convolution_output(input_height, kernel_height, stride_height,
                          pad_height, 1, &oh) ||
      !convolution_output(input_width, kernel_width, stride_width, pad_width,
                          1, &ow) ||
      !detail::valid_product({batch, channels, oh, ow})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, grad_out, grad_in)) return Status::kInvalidArgument;
  std::fill_n(grad_in, batch * channels * input_height * input_width, 0.0f);
  for (long long n = 0; n < batch; ++n) for (long long c = 0; c < channels; ++c)
  for (long long oy = 0; oy < oh; ++oy) for (long long ox = 0; ox < ow; ++ox) {
    long long count = 0, best_y = -1, best_x = -1;
    float best = -std::numeric_limits<float>::infinity();
    for (long long ky = 0; ky < kernel_height; ++ky) for (long long kx = 0; kx < kernel_width; ++kx) {
      const long long iy = oy * stride_height + ky - pad_height;
      const long long ix = ox * stride_width + kx - pad_width;
      if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) continue;
      ++count;
      const float sample = input[((n * channels + c) * input_height + iy) * input_width + ix];
      if (sample > best) { best = sample; best_y = iy; best_x = ix; }
    }
    if (count == 0) return Status::kInvalidShape;
    const float gradient = grad_out[((n * channels + c) * oh + oy) * ow + ox];
    if (mode == PoolMode::kMaximum) {
      grad_in[((n * channels + c) * input_height + best_y) * input_width + best_x] += gradient;
    } else {
      for (long long ky = 0; ky < kernel_height; ++ky) for (long long kx = 0; kx < kernel_width; ++kx) {
        const long long iy = oy * stride_height + ky - pad_height;
        const long long ix = ox * stride_width + kx - pad_width;
        if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width)
          grad_in[((n * channels + c) * input_height + iy) * input_width + ix] += gradient / count;
      }
    }
  }
  return Status::kOk;
}

Status timestep_embedding(const float* timesteps, float* output,
                          long long count, long long dim, float max_period) {
  if (!detail::valid_product({count, dim}) || !std::isfinite(max_period) ||
      max_period <= 0.0f) return Status::kInvalidShape;
  if (!detail::all_nonnull(timesteps, output)) return Status::kInvalidArgument;
  const long long half = dim / 2;
  for (long long i = 0; i < count; ++i) {
    for (long long j = 0; j < half; ++j) {
      const float frequency = std::exp(-std::log(max_period) * j / half);
      const float argument = timesteps[i] * frequency;
      output[i * dim + j] = std::cos(argument);
      output[i * dim + half + j] = std::sin(argument);
    }
    if ((dim & 1) != 0) output[i * dim + dim - 1] = 0.0f;
  }
  return Status::kOk;
}

Status solve_lower_triangular(const float* a, const float* b, float* x,
                              long long batch, long long n,
                              long long right_hand_sides) {
  if (!detail::valid_product({batch, n, n}) ||
      !detail::valid_product({batch, n, right_hand_sides})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, x)) return Status::kInvalidArgument;
  for (long long item = 0; item < batch; ++item) {
    const float* matrix = a + item * n * n;
    for (long long row = 0; row < n; ++row) {
      if (matrix[row * n + row] == 0.0f) return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(batch * right_hand_sides, 1,
      [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long batch_index = item / right_hand_sides;
      const long long rhs = item % right_hand_sides;
      const float* matrix = a + batch_index * n * n;
      const float* source = b + batch_index * n * right_hand_sides;
      float* destination = x + batch_index * n * right_hand_sides;
      for (long long row = 0; row < n; ++row) {
        float sum = source[row * right_hand_sides + rhs];
        for (long long col = 0; col < row; ++col) {
          sum -= matrix[row * n + col] * destination[col * right_hand_sides + rhs];
        }
        destination[row * right_hand_sides + rhs] =
            sum / matrix[row * n + row];
      }
    }
  });
  return Status::kOk;
}

Status get_relative_position(const float* table, float* output,
                             long long width, long long dim) {
  if (!detail::valid_product({width, width, dim}) || width > LLONG_MAX / 2 ||
      !detail::valid_product({2 * width - 1, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(table, output)) return Status::kInvalidArgument;
  for (long long query = 0; query < width; ++query) for (long long key = 0; key < width; ++key)
    std::copy_n(table + (width - query - 1 + key) * dim, dim,
                output + (query * width + key) * dim);
  return Status::kOk;
}

Status add_relative_position_2d(
    const float* attention, const float* relative_height,
    const float* relative_width, float* output, long long batches,
    long long query_height, long long query_width, long long key_height,
    long long key_width) {
  if (!detail::valid_product({batches, query_height, query_width, key_height,
                              key_width})) return Status::kInvalidShape;
  if (!detail::all_nonnull(attention, relative_height, relative_width, output))
    return Status::kInvalidArgument;
  for (long long b = 0; b < batches; ++b) for (long long qh = 0; qh < query_height; ++qh)
  for (long long qw = 0; qw < query_width; ++qw) for (long long kh = 0; kh < key_height; ++kh)
  for (long long kw = 0; kw < key_width; ++kw) {
    const long long index = ((((b * query_height + qh) * query_width + qw) * key_height + kh) * key_width + kw);
    output[index] = attention[index] +
        relative_height[((b * query_height + qh) * query_width + qw) * key_height + kh] +
        relative_width[((b * query_height + qh) * query_width + qw) * key_width + kw];
  }
  return Status::kOk;
}

Status window_partition(const float* image, float* windows, long long height,
                        long long width, long long channels,
                        long long window_size) {
  if (!detail::valid_product({height, width, channels, window_size}) ||
      height > LLONG_MAX - (window_size - 1) ||
      width > LLONG_MAX - (window_size - 1))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(image, windows)) return Status::kInvalidArgument;
  const long long windows_x = (width + window_size - 1) / window_size;
  const long long windows_y = (height + window_size - 1) / window_size;
  if (!detail::valid_product({windows_y, windows_x, window_size, window_size,
                              channels})) return Status::kInvalidShape;
  for (long long wy = 0; wy < windows_y; ++wy) for (long long wx = 0; wx < windows_x; ++wx)
  for (long long y = 0; y < window_size; ++y) for (long long x = 0; x < window_size; ++x)
  for (long long c = 0; c < channels; ++c) {
    const long long iy = wy * window_size + y, ix = wx * window_size + x;
    const long long destination = ((((wy * windows_x + wx) * window_size + y) * window_size + x) * channels + c);
    windows[destination] = iy < height && ix < width
        ? image[(iy * width + ix) * channels + c] : 0.0f;
  }
  return Status::kOk;
}

Status window_unpartition(const float* windows, float* image, long long height,
                          long long width, long long channels,
                          long long window_size) {
  if (!detail::valid_product({height, width, channels, window_size}) ||
      height > LLONG_MAX - (window_size - 1) ||
      width > LLONG_MAX - (window_size - 1))
    return Status::kInvalidShape;
  if (!detail::all_nonnull(windows, image)) return Status::kInvalidArgument;
  const long long windows_x = (width + window_size - 1) / window_size;
  const long long windows_y = (height + window_size - 1) / window_size;
  if (!detail::valid_product({windows_y, windows_x, window_size, window_size,
                              channels})) return Status::kInvalidShape;
  for (long long y = 0; y < height; ++y) for (long long x = 0; x < width; ++x)
  for (long long c = 0; c < channels; ++c) {
    const long long wy = y / window_size, wx = x / window_size;
    const long long local_y = y % window_size, local_x = x % window_size;
    image[(y * width + x) * channels + c] =
        windows[((((wy * windows_x + wx) * window_size + local_y) *
                   window_size + local_x) * channels + c)];
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
