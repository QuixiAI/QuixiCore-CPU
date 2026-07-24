#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "kernels/common/float_storage_access.h"
#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

long long patch_output(long long input, long long kernel, long long stride,
                       long long padding) {
  return (input + 2 * padding - kernel) / stride + 1;
}

long long pool_output(long long input, long long kernel, long long stride,
                      bool ceil_mode) {
  return ceil_mode ? (input - kernel + stride - 1) / stride + 1
                   : (input - kernel) / stride + 1;
}

bool valid_spatial(long long input_height, long long input_width,
                   long long channels, long long kernel_height,
                   long long kernel_width, long long stride_height,
                   long long stride_width) {
  return detail::valid_product(
             {input_height, input_width, channels, kernel_height,
              kernel_width, stride_height, stride_width}) &&
         stride_height > 0 && stride_width > 0;
}

template <FloatStorageType Type>
Status extract_patches_typed(const void* input, void* output, long long batch,
                             long long input_height, long long input_width,
                             long long channels, long long kernel_height,
                             long long kernel_width, long long stride_height,
                             long long stride_width, long long pad_height,
                             long long pad_width, long long output_height,
                             long long output_width) {
  if (input == nullptr || output == nullptr) return Status::kInvalidArgument;
  using Element = std::conditional_t<Type == FloatStorageType::kF32, float,
                                     std::uint16_t>;
  const auto* input_values = static_cast<const Element*>(input);
  auto* output_values = static_cast<Element*>(output);
  const long long spatial_count = output_height * output_width;
  const long long patch_dim = kernel_height * kernel_width * channels;
  threading::parallel_ranges(
      batch * spatial_count, 1, [&](long long begin, long long end, int) {
        for (long long patch = begin; patch < end; ++patch) {
          const long long item = patch / spatial_count;
          const long long spatial = patch - item * spatial_count;
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          long long destination = patch * patch_dim;
          for (long long kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            const long long input_y =
                output_y * stride_height + kernel_y - pad_height;
            for (long long kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
              const long long input_x =
                  output_x * stride_width + kernel_x - pad_width;
              if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                  input_x < input_width) {
                const long long source =
                    ((item * input_height + input_y) * input_width + input_x) *
                    channels;
                std::copy_n(input_values + source, channels,
                            output_values + destination);
              } else {
                std::fill_n(output_values + destination, channels, Element{});
              }
              destination += channels;
            }
          }
        }
      });
  return Status::kOk;
}

template <FloatStorageType Type>
Status extract_patches_3d_typed(
    const void* input, void* output, long long batch, long long input_frames,
    long long input_height, long long input_width, long long channels,
    long long kernel_frames, long long kernel_height, long long kernel_width,
    long long stride_frames, long long stride_height, long long stride_width,
    long long pad_frames, long long pad_height, long long pad_width,
    long long output_frames, long long output_height,
    long long output_width) {
  if (input == nullptr || output == nullptr) return Status::kInvalidArgument;
  using Element = std::conditional_t<Type == FloatStorageType::kF32, float,
                                     std::uint16_t>;
  const auto* input_values = static_cast<const Element*>(input);
  auto* output_values = static_cast<Element*>(output);
  const long long spatial_count =
      output_frames * output_height * output_width;
  const long long patch_dim =
      kernel_frames * kernel_height * kernel_width * channels;
  threading::parallel_ranges(
      batch * spatial_count, 1,
      [&](long long begin, long long end, int) {
        for (long long patch = begin; patch < end; ++patch) {
          const long long item = patch / spatial_count;
          long long spatial = patch - item * spatial_count;
          const long long output_t = spatial / (output_height * output_width);
          spatial -= output_t * output_height * output_width;
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          long long destination = patch * patch_dim;
          const long long input_t0 =
              output_t * stride_frames - pad_frames;
          const long long input_y0 =
              output_y * stride_height - pad_height;
          const long long input_x0 = output_x * stride_width - pad_width;
          if (input_t0 >= 0 && input_y0 >= 0 && input_x0 >= 0 &&
              input_t0 + kernel_frames <= input_frames &&
              input_y0 + kernel_height <= input_height &&
              input_x0 + kernel_width <= input_width) {
            const long long row_values = kernel_width * channels;
            for (long long kernel_t = 0; kernel_t < kernel_frames;
                 ++kernel_t) {
              for (long long kernel_y = 0; kernel_y < kernel_height;
                   ++kernel_y) {
                const long long source =
                    ((((item * input_frames + input_t0 + kernel_t) *
                            input_height +
                        input_y0 + kernel_y) *
                           input_width +
                       input_x0) *
                      channels);
                std::copy_n(input_values + source, row_values,
                            output_values + destination);
                destination += row_values;
              }
            }
            continue;
          }
          for (long long kernel_t = 0; kernel_t < kernel_frames; ++kernel_t) {
            const long long input_t =
                output_t * stride_frames + kernel_t - pad_frames;
            for (long long kernel_y = 0; kernel_y < kernel_height;
                 ++kernel_y) {
              const long long input_y =
                  output_y * stride_height + kernel_y - pad_height;
              for (long long kernel_x = 0; kernel_x < kernel_width;
                   ++kernel_x) {
                const long long input_x =
                    output_x * stride_width + kernel_x - pad_width;
                if (input_t >= 0 && input_t < input_frames && input_y >= 0 &&
                    input_y < input_height && input_x >= 0 &&
                    input_x < input_width) {
                  const long long source =
                      ((((item * input_frames + input_t) * input_height +
                         input_y) *
                            input_width +
                        input_x) *
                       channels);
                  std::copy_n(input_values + source, channels,
                              output_values + destination);
                } else {
                  std::fill_n(output_values + destination, channels,
                              Element{});
                }
                destination += channels;
              }
            }
          }
        }
      });
  return Status::kOk;
}

template <FloatStorageType Type>
Status interpolate_position_typed(const void* table, void* output,
                                  long long input_height,
                                  long long input_width,
                                  long long output_height,
                                  long long output_width, long long channels,
                                  bool align_corners) {
  if (table == nullptr || output == nullptr) return Status::kInvalidArgument;
  threading::parallel_ranges(
      output_height * output_width, 1,
      [&](long long begin, long long end, int) {
        for (long long spatial = begin; spatial < end; ++spatial) {
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          const double source_y = align_corners && output_height > 1
                                      ? static_cast<double>(output_y) *
                                            (input_height - 1) /
                                            (output_height - 1)
                                      : (output_y + 0.5) * input_height /
                                                output_height -
                                            0.5;
          const double source_x = align_corners && output_width > 1
                                      ? static_cast<double>(output_x) *
                                            (input_width - 1) /
                                            (output_width - 1)
                                      : (output_x + 0.5) * input_width /
                                                output_width -
                                            0.5;
          const double cy =
              std::clamp(source_y, 0.0, static_cast<double>(input_height - 1));
          const double cx =
              std::clamp(source_x, 0.0, static_cast<double>(input_width - 1));
          const long long y0 = static_cast<long long>(std::floor(cy));
          const long long x0 = static_cast<long long>(std::floor(cx));
          const long long y1 = std::min(y0 + 1, input_height - 1);
          const long long x1 = std::min(x0 + 1, input_width - 1);
          const float wy = static_cast<float>(cy - y0);
          const float wx = static_cast<float>(cx - x0);
          for (long long channel = 0; channel < channels; ++channel) {
            const float a = detail::load_storage<Type>(
                table, (y0 * input_width + x0) * channels + channel);
            const float b = detail::load_storage<Type>(
                table, (y0 * input_width + x1) * channels + channel);
            const float c = detail::load_storage<Type>(
                table, (y1 * input_width + x0) * channels + channel);
            const float d = detail::load_storage<Type>(
                table, (y1 * input_width + x1) * channels + channel);
            const float top = a + wx * (b - a);
            const float bottom = c + wx * (d - c);
            detail::store_storage<Type>(output, spatial * channels + channel,
                                        top + wy * (bottom - top));
          }
        }
      });
  return Status::kOk;
}

template <FloatStorageType Type>
Status avg_pool_typed(const void* input, void* output, long long batch,
                      long long input_height, long long input_width,
                      long long channels, long long kernel_height,
                      long long kernel_width, long long stride_height,
                      long long stride_width, long long output_height,
                      long long output_width) {
  if (input == nullptr || output == nullptr) return Status::kInvalidArgument;
  const long long spatial_count = output_height * output_width;
  threading::parallel_ranges(
      batch * spatial_count, 1,
      [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long item = index / spatial_count;
          const long long spatial = index - item * spatial_count;
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          const long long y0 = output_y * stride_height;
          const long long x0 = output_x * stride_width;
          const long long y1 = std::min(y0 + kernel_height, input_height);
          const long long x1 = std::min(x0 + kernel_width, input_width);
          const float inverse_count =
              1.0f / static_cast<float>((y1 - y0) * (x1 - x0));
          for (long long channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (long long y = y0; y < y1; ++y) {
              for (long long x = x0; x < x1; ++x) {
                sum += detail::load_storage<Type>(
                    input,
                    ((item * input_height + y) * input_width + x) * channels +
                        channel);
              }
            }
            detail::store_storage<Type>(output, index * channels + channel,
                                        sum * inverse_count);
          }
        }
      });
  return Status::kOk;
}

}  // namespace

Status extract_patches_2d(const float* input, float* output, long long batch,
                          long long input_height, long long input_width,
                          long long channels, long long kernel_height,
                          long long kernel_width, long long stride_height,
                          long long stride_width, long long pad_height,
                          long long pad_width) {
  if (!detail::valid_product({batch}) ||
      !valid_spatial(input_height, input_width, channels, kernel_height,
                     kernel_width, stride_height, stride_width) ||
      pad_height < 0 || pad_width < 0) {
    return Status::kInvalidShape;
  }
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  if (!detail::valid_product(
          {batch, output_height, output_width, kernel_height, kernel_width,
           channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  const long long spatial_count = output_height * output_width;
  const long long patch_dim = kernel_height * kernel_width * channels;
  threading::parallel_ranges(
      batch * spatial_count, 1, [&](long long begin, long long end, int) {
        for (long long patch = begin; patch < end; ++patch) {
          const long long item = patch / spatial_count;
          const long long spatial = patch - item * spatial_count;
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          float* destination = output + patch * patch_dim;
          for (long long kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            const long long input_y =
                output_y * stride_height + kernel_y - pad_height;
            for (long long kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
              const long long input_x =
                  output_x * stride_width + kernel_x - pad_width;
              if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                  input_x < input_width) {
                const float* source =
                    input +
                    ((item * input_height + input_y) * input_width + input_x) *
                        channels;
                std::copy_n(source, channels, destination);
              } else {
                std::fill_n(destination, channels, 0.0f);
              }
              destination += channels;
            }
          }
        }
      });
  return Status::kOk;
}

Status extract_patches_3d(
    const float* input, float* output, long long batch, long long input_frames,
    long long input_height, long long input_width, long long channels,
    long long kernel_frames, long long kernel_height, long long kernel_width,
    long long stride_frames, long long stride_height, long long stride_width,
    long long pad_frames, long long pad_height, long long pad_width) {
  if (!detail::valid_product({batch, input_frames, input_height, input_width,
                              channels, kernel_frames, kernel_height,
                              kernel_width, stride_frames, stride_height,
                              stride_width}) ||
      pad_frames < 0 || pad_height < 0 || pad_width < 0) {
    return Status::kInvalidShape;
  }
  const long long output_frames =
      patch_output(input_frames, kernel_frames, stride_frames, pad_frames);
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  if (!detail::valid_product({batch, output_frames, output_height, output_width,
                              kernel_frames, kernel_height, kernel_width,
                              channels})) {
    return Status::kInvalidShape;
  }
  return extract_patches_3d_typed<FloatStorageType::kF32>(
      input, output, batch, input_frames, input_height, input_width, channels,
      kernel_frames, kernel_height, kernel_width, stride_frames, stride_height,
      stride_width, pad_frames, pad_height, pad_width, output_frames,
      output_height, output_width);
}

Status interpolate_position_2d(const float* table, float* output,
                               long long input_height, long long input_width,
                               long long output_height, long long output_width,
                               long long channels, bool align_corners) {
  if (!detail::valid_product({input_height, input_width, output_height,
                              output_width, channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(table, output)) return Status::kInvalidArgument;
  threading::parallel_ranges(
      output_height * output_width, 1,
      [&](long long begin, long long end, int) {
        for (long long spatial = begin; spatial < end; ++spatial) {
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          const double source_y = align_corners && output_height > 1
                                      ? static_cast<double>(output_y) *
                                            (input_height - 1) /
                                            (output_height - 1)
                                      : (output_y + 0.5) * input_height /
                                                output_height -
                                            0.5;
          const double source_x = align_corners && output_width > 1
                                      ? static_cast<double>(output_x) *
                                            (input_width - 1) /
                                            (output_width - 1)
                                      : (output_x + 0.5) * input_width /
                                                output_width -
                                            0.5;
          const double clamped_y =
              std::clamp(source_y, 0.0, static_cast<double>(input_height - 1));
          const double clamped_x =
              std::clamp(source_x, 0.0, static_cast<double>(input_width - 1));
          const long long y0 = static_cast<long long>(std::floor(clamped_y));
          const long long x0 = static_cast<long long>(std::floor(clamped_x));
          const long long y1 = std::min(y0 + 1, input_height - 1);
          const long long x1 = std::min(x0 + 1, input_width - 1);
          const float weight_y = static_cast<float>(clamped_y - y0);
          const float weight_x = static_cast<float>(clamped_x - x0);
          const float* row00 = table + (y0 * input_width + x0) * channels;
          const float* row01 = table + (y0 * input_width + x1) * channels;
          const float* row10 = table + (y1 * input_width + x0) * channels;
          const float* row11 = table + (y1 * input_width + x1) * channels;
          float* destination = output + spatial * channels;
          for (long long channel = 0; channel < channels; ++channel) {
            const float top = row00[channel] +
                              weight_x * (row01[channel] - row00[channel]);
            const float bottom = row10[channel] +
                                 weight_x * (row11[channel] - row10[channel]);
            destination[channel] = top + weight_y * (bottom - top);
          }
        }
      });
  return Status::kOk;
}

Status avg_pool2d_tokens(const float* input, float* output, long long batch,
                         long long input_height, long long input_width,
                         long long channels, long long kernel_height,
                         long long kernel_width, long long stride_height,
                         long long stride_width, bool ceil_mode) {
  if (!detail::valid_product({batch}) ||
      !valid_spatial(input_height, input_width, channels, kernel_height,
                     kernel_width, stride_height, stride_width)) {
    return Status::kInvalidShape;
  }
  const long long output_height =
      pool_output(input_height, kernel_height, stride_height, ceil_mode);
  const long long output_width =
      pool_output(input_width, kernel_width, stride_width, ceil_mode);
  if (!detail::valid_product({batch, output_height, output_width, channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  const long long spatial_count = output_height * output_width;
  threading::parallel_ranges(
      batch * spatial_count, 1,
      [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long item = index / spatial_count;
          const long long spatial = index - item * spatial_count;
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          const long long y0 = output_y * stride_height;
          const long long x0 = output_x * stride_width;
          const long long y1 = std::min(y0 + kernel_height, input_height);
          const long long x1 = std::min(x0 + kernel_width, input_width);
          const float inverse_count =
              1.0f / static_cast<float>((y1 - y0) * (x1 - x0));
          float* destination = output + index * channels;
          std::fill_n(destination, channels, 0.0f);
          for (long long y = y0; y < y1; ++y) {
            const float* source =
                input + ((item * input_height + y) * input_width + x0) *
                            channels;
            for (long long x = x0; x < x1; ++x) {
              for (long long channel = 0; channel < channels; ++channel) {
                destination[channel] +=
                    source[(x - x0) * channels + channel];
              }
            }
          }
          for (long long channel = 0; channel < channels; ++channel) {
            destination[channel] *= inverse_count;
          }
        }
      });
  return Status::kOk;
}

Status vision_patch_projection(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_height, long long input_width,
    long long input_channels, long long output_channels,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, long long pad_height, long long pad_width) {
  if (!detail::valid_product({batch, input_height, input_width, input_channels,
                              output_channels, kernel_height, kernel_width,
                              stride_height, stride_width}) ||
      pad_height < 0 || pad_width < 0) {
    return Status::kInvalidShape;
  }
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  const long long patch_dim = kernel_height * kernel_width * input_channels;
  if (!detail::valid_product(
          {batch, output_height, output_width, patch_dim, output_channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  const long long spatial_count = output_height * output_width;
  const long long rows = batch * output_height * output_width;
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const long long item = row / spatial_count;
      const long long spatial = row - item * spatial_count;
      const long long output_y = spatial / output_width;
      const long long output_x = spatial - output_y * output_width;
      const long long input_y0 = output_y * stride_height - pad_height;
      const long long input_x0 = output_x * stride_width - pad_width;
      const bool interior = input_y0 >= 0 && input_x0 >= 0 &&
                            input_y0 + kernel_height <= input_height &&
                            input_x0 + kernel_width <= input_width;
      for (long long output_channel = 0; output_channel < output_channels;
           ++output_channel) {
        const float* weight = weights + output_channel * patch_dim;
        float sum = bias == nullptr ? 0.0f : bias[output_channel];
        long long feature = 0;
        if (interior) {
          for (long long kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            const float* source =
                input +
                ((item * input_height + input_y0 + kernel_y) * input_width +
                 input_x0) *
                    input_channels;
            for (long long kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
              const float* pixel = source + kernel_x * input_channels;
              for (long long channel = 0; channel < input_channels;
                   ++channel) {
                sum += pixel[channel] * weight[feature + channel];
              }
              feature += input_channels;
            }
          }
        } else {
          for (long long kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            const long long input_y = input_y0 + kernel_y;
            for (long long kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
              const long long input_x = input_x0 + kernel_x;
              if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                  input_x < input_width) {
                const float* pixel =
                    input +
                    ((item * input_height + input_y) * input_width + input_x) *
                        input_channels;
                for (long long channel = 0; channel < input_channels;
                     ++channel) {
                  sum += pixel[channel] * weight[feature + channel];
                }
              }
              feature += input_channels;
            }
          }
        }
        output[row * output_channels + output_channel] = sum;
      }
    }
  });
  return Status::kOk;
}

Status vision_patch_projection_3d(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_frames, long long input_height,
    long long input_width, long long input_channels,
    long long output_channels, long long kernel_frames,
    long long kernel_height, long long kernel_width, long long stride_frames,
    long long stride_height, long long stride_width, long long pad_frames,
    long long pad_height, long long pad_width) {
  if (!detail::valid_product({batch, input_frames, input_height, input_width,
                              input_channels, output_channels, kernel_frames,
                              kernel_height, kernel_width, stride_frames,
                              stride_height, stride_width}) ||
      pad_frames < 0 || pad_height < 0 || pad_width < 0) {
    return Status::kInvalidShape;
  }
  const long long output_frames =
      patch_output(input_frames, kernel_frames, stride_frames, pad_frames);
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  const long long patch_dim =
      kernel_frames * kernel_height * kernel_width * input_channels;
  if (!detail::valid_product({batch, output_frames, output_height, output_width,
                              patch_dim, output_channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  const long long spatial_count =
      output_frames * output_height * output_width;
  threading::parallel_ranges(
      batch * spatial_count, 1,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long item = row / spatial_count;
          long long spatial = row - item * spatial_count;
          const long long output_t =
              spatial / (output_height * output_width);
          spatial -= output_t * output_height * output_width;
          const long long output_y = spatial / output_width;
          const long long output_x = spatial - output_y * output_width;
          const long long input_t0 =
              output_t * stride_frames - pad_frames;
          const long long input_y0 =
              output_y * stride_height - pad_height;
          const long long input_x0 = output_x * stride_width - pad_width;
          const bool interior =
              input_t0 >= 0 && input_y0 >= 0 && input_x0 >= 0 &&
              input_t0 + kernel_frames <= input_frames &&
              input_y0 + kernel_height <= input_height &&
              input_x0 + kernel_width <= input_width;
          for (long long output_channel = 0;
               output_channel < output_channels; ++output_channel) {
            const float* weight = weights + output_channel * patch_dim;
            float sum = bias == nullptr ? 0.0f : bias[output_channel];
            long long feature = 0;
            if (interior) {
              float sum1 = 0.0f;
              float sum2 = 0.0f;
              float sum3 = 0.0f;
              for (long long kernel_t = 0; kernel_t < kernel_frames;
                   ++kernel_t) {
                const long long input_t = input_t0 + kernel_t;
                for (long long kernel_y = 0; kernel_y < kernel_height;
                     ++kernel_y) {
                  const long long input_y = input_y0 + kernel_y;
                  const float* source =
                      input +
                      ((((item * input_frames + input_t) * input_height +
                         input_y) *
                            input_width +
                        input_x0) *
                       input_channels);
                  const long long row_values =
                      kernel_width * input_channels;
                  long long value = 0;
                  for (; value + 4 <= row_values; value += 4) {
                    sum += source[value] * weight[feature + value];
                    sum1 += source[value + 1] * weight[feature + value + 1];
                    sum2 += source[value + 2] * weight[feature + value + 2];
                    sum3 += source[value + 3] * weight[feature + value + 3];
                  }
                  for (; value < row_values; ++value) {
                    sum += source[value] * weight[feature + value];
                  }
                  feature += row_values;
                }
              }
              sum = (sum + sum1) + (sum2 + sum3);
            } else {
              for (long long kernel_t = 0; kernel_t < kernel_frames;
                   ++kernel_t) {
                const long long input_t = input_t0 + kernel_t;
                for (long long kernel_y = 0; kernel_y < kernel_height;
                     ++kernel_y) {
                  const long long input_y = input_y0 + kernel_y;
                  for (long long kernel_x = 0; kernel_x < kernel_width;
                       ++kernel_x) {
                    const long long input_x = input_x0 + kernel_x;
                    if (input_t >= 0 && input_t < input_frames &&
                        input_y >= 0 && input_y < input_height &&
                        input_x >= 0 && input_x < input_width) {
                    const float* voxel =
                        input +
                        ((((item * input_frames + input_t) * input_height +
                           input_y) *
                              input_width +
                          input_x) *
                         input_channels);
                    for (long long channel = 0; channel < input_channels;
                         ++channel) {
                      sum += voxel[channel] * weight[feature + channel];
                    }
                    }
                    feature += input_channels;
                  }
                }
              }
            }
            output[row * output_channels + output_channel] = sum;
          }
        }
      });
  return Status::kOk;
}

Status extract_patches_2d_storage(
    FloatStorageInput input, FloatStorageOutput output, long long batch,
    long long input_height, long long input_width, long long channels,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, long long pad_height, long long pad_width,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, input_height, input_width, channels,
                              kernel_height, kernel_width, stride_height,
                              stride_width}) ||
      pad_height < 0 || pad_width < 0 ||
      input.count != batch * input_height * input_width * channels) {
    return Status::kInvalidShape;
  }
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  if (!detail::valid_product({batch, output_height, output_width, kernel_height,
                              kernel_width, channels}) ||
      output.count != batch * output_height * output_width * kernel_height *
                          kernel_width * channels) {
    return Status::kInvalidShape;
  }
  if (input.type == output.type) {
    switch (input.type) {
      case FloatStorageType::kF32:
        return extract_patches_typed<FloatStorageType::kF32>(
            input.data, output.data, batch, input_height, input_width, channels,
            kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width, output_height, output_width);
      case FloatStorageType::kF16:
        return extract_patches_typed<FloatStorageType::kF16>(
            input.data, output.data, batch, input_height, input_width, channels,
            kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width, output_height, output_width);
      case FloatStorageType::kBF16:
        return extract_patches_typed<FloatStorageType::kBF16>(
            input.data, output.data, batch, input_height, input_width, channels,
            kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width, output_height, output_width);
    }
  }
  return with_float_storage(
      &input, 1, &output, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return extract_patches_2d(
            inputs[0], outputs[0], batch, input_height, input_width, channels,
            kernel_height, kernel_width, stride_height, stride_width,
            pad_height, pad_width);
      },
      workspace);
}

Status extract_patches_3d_storage(
    FloatStorageInput input, FloatStorageOutput output, long long batch,
    long long input_frames, long long input_height, long long input_width,
    long long channels, long long kernel_frames, long long kernel_height,
    long long kernel_width, long long stride_frames, long long stride_height,
    long long stride_width, long long pad_frames, long long pad_height,
    long long pad_width, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, input_frames, input_height, input_width,
                              channels, kernel_frames, kernel_height,
                              kernel_width, stride_frames, stride_height,
                              stride_width}) ||
      pad_frames < 0 || pad_height < 0 || pad_width < 0 ||
      input.count != batch * input_frames * input_height * input_width *
                         channels) {
    return Status::kInvalidShape;
  }
  const long long output_frames =
      patch_output(input_frames, kernel_frames, stride_frames, pad_frames);
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  if (!detail::valid_product({batch, output_frames, output_height, output_width,
                              kernel_frames, kernel_height, kernel_width,
                              channels}) ||
      output.count != batch * output_frames * output_height * output_width *
                          kernel_frames * kernel_height * kernel_width *
                          channels) {
    return Status::kInvalidShape;
  }
  if (input.type == output.type) {
    switch (input.type) {
      case FloatStorageType::kF32:
        return extract_patches_3d_typed<FloatStorageType::kF32>(
            input.data, output.data, batch, input_frames, input_height,
            input_width, channels, kernel_frames, kernel_height, kernel_width,
            stride_frames, stride_height, stride_width, pad_frames, pad_height,
            pad_width, output_frames, output_height, output_width);
      case FloatStorageType::kF16:
        return extract_patches_3d_typed<FloatStorageType::kF16>(
            input.data, output.data, batch, input_frames, input_height,
            input_width, channels, kernel_frames, kernel_height, kernel_width,
            stride_frames, stride_height, stride_width, pad_frames, pad_height,
            pad_width, output_frames, output_height, output_width);
      case FloatStorageType::kBF16:
        return extract_patches_3d_typed<FloatStorageType::kBF16>(
            input.data, output.data, batch, input_frames, input_height,
            input_width, channels, kernel_frames, kernel_height, kernel_width,
            stride_frames, stride_height, stride_width, pad_frames, pad_height,
            pad_width, output_frames, output_height, output_width);
    }
  }
  return with_float_storage(
      &input, 1, &output, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return extract_patches_3d(
            inputs[0], outputs[0], batch, input_frames, input_height,
            input_width, channels, kernel_frames, kernel_height, kernel_width,
            stride_frames, stride_height, stride_width, pad_frames, pad_height,
            pad_width);
      },
      workspace);
}

Status interpolate_position_2d_storage(
    FloatStorageInput table, FloatStorageOutput output, long long input_height,
    long long input_width, long long output_height, long long output_width,
    long long channels, bool align_corners,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({input_height, input_width, output_height,
                              output_width, channels}) ||
      table.count != input_height * input_width * channels ||
      output.count != output_height * output_width * channels) {
    return Status::kInvalidShape;
  }
  return with_float_storage(
      &table, 1, &output, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return interpolate_position_2d(inputs[0], outputs[0], input_height,
                                       input_width, output_height, output_width,
                                       channels, align_corners);
      },
      workspace);
}

Status avg_pool2d_tokens_storage(
    FloatStorageInput input, FloatStorageOutput output, long long batch,
    long long input_height, long long input_width, long long channels,
    long long kernel_height, long long kernel_width, long long stride_height,
    long long stride_width, bool ceil_mode,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, input_height, input_width, channels,
                              kernel_height, kernel_width, stride_height,
                              stride_width}) ||
      input.count != batch * input_height * input_width * channels) {
    return Status::kInvalidShape;
  }
  const long long output_height =
      pool_output(input_height, kernel_height, stride_height, ceil_mode);
  const long long output_width =
      pool_output(input_width, kernel_width, stride_width, ceil_mode);
  if (!detail::valid_product({batch, output_height, output_width, channels}) ||
      output.count != batch * output_height * output_width * channels) {
    return Status::kInvalidShape;
  }
  return with_float_storage(
      &input, 1, &output, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return avg_pool2d_tokens(inputs[0], outputs[0], batch, input_height,
                                 input_width, channels, kernel_height,
                                 kernel_width, stride_height, stride_width,
                                 ceil_mode);
      },
      workspace);
}

Status vision_patch_projection_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_height,
    long long input_width, long long input_channels,
    long long output_channels, long long kernel_height,
    long long kernel_width, long long stride_height, long long stride_width,
    long long pad_height, long long pad_width,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, input_height, input_width, input_channels,
                              output_channels, kernel_height, kernel_width,
                              stride_height, stride_width}) ||
      pad_height < 0 || pad_width < 0 ||
      input.count != batch * input_height * input_width * input_channels ||
      weights.count !=
          output_channels * kernel_height * kernel_width * input_channels ||
      (bias.data == nullptr ? bias.count != 0
                            : bias.count != output_channels)) {
    return Status::kInvalidShape;
  }
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  if (!detail::valid_product(
          {batch, output_height, output_width, output_channels}) ||
      output.count != batch * output_height * output_width * output_channels) {
    return Status::kInvalidShape;
  }
  if (bias.data == nullptr) {
    const FloatStorageInput inputs[] = {input, weights};
    return with_float_storage(
        inputs, 2, &output, 1,
        [&](const float* const* values, float* const* outputs) {
          return vision_patch_projection(
              values[0], values[1], nullptr, outputs[0], batch, input_height,
              input_width, input_channels, output_channels, kernel_height,
              kernel_width, stride_height, stride_width, pad_height,
              pad_width);
        },
        workspace);
  }
  const FloatStorageInput inputs[] = {input, weights, bias};
  return with_float_storage(
      inputs, 3, &output, 1,
      [&](const float* const* values, float* const* outputs) {
        return vision_patch_projection(
            values[0], values[1], values[2], outputs[0], batch, input_height,
            input_width, input_channels, output_channels, kernel_height,
            kernel_width, stride_height, stride_width, pad_height, pad_width);
      },
      workspace);
}

Status vision_patch_projection_3d_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_frames,
    long long input_height, long long input_width, long long input_channels,
    long long output_channels, long long kernel_frames,
    long long kernel_height, long long kernel_width, long long stride_frames,
    long long stride_height, long long stride_width, long long pad_frames,
    long long pad_height, long long pad_width,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, input_frames, input_height, input_width,
                              input_channels, output_channels, kernel_frames,
                              kernel_height, kernel_width, stride_frames,
                              stride_height, stride_width}) ||
      pad_frames < 0 || pad_height < 0 || pad_width < 0 ||
      input.count != batch * input_frames * input_height * input_width *
                         input_channels ||
      weights.count != output_channels * kernel_frames * kernel_height *
                           kernel_width * input_channels ||
      (bias.data == nullptr ? bias.count != 0
                            : bias.count != output_channels)) {
    return Status::kInvalidShape;
  }
  const long long output_frames =
      patch_output(input_frames, kernel_frames, stride_frames, pad_frames);
  const long long output_height =
      patch_output(input_height, kernel_height, stride_height, pad_height);
  const long long output_width =
      patch_output(input_width, kernel_width, stride_width, pad_width);
  if (!detail::valid_product({batch, output_frames, output_height, output_width,
                              output_channels}) ||
      output.count != batch * output_frames * output_height * output_width *
                          output_channels) {
    return Status::kInvalidShape;
  }
  if (bias.data == nullptr) {
    const FloatStorageInput inputs[] = {input, weights};
    return with_float_storage(
        inputs, 2, &output, 1,
        [&](const float* const* values, float* const* outputs) {
          return vision_patch_projection_3d(
              values[0], values[1], nullptr, outputs[0], batch, input_frames,
              input_height, input_width, input_channels, output_channels,
              kernel_frames, kernel_height, kernel_width, stride_frames,
              stride_height, stride_width, pad_frames, pad_height, pad_width);
        },
        workspace);
  }
  const FloatStorageInput inputs[] = {input, weights, bias};
  return with_float_storage(
      inputs, 3, &output, 1,
      [&](const float* const* values, float* const* outputs) {
        return vision_patch_projection_3d(
            values[0], values[1], values[2], outputs[0], batch, input_frames,
            input_height, input_width, input_channels, output_channels,
            kernel_frames, kernel_height, kernel_width, stride_frames,
            stride_height, stride_width, pad_frames, pad_height, pad_width);
      },
      workspace);
}

Status factorized_position_2d(const int* position_ids, const float* table,
                              const int* valid_mask, float* output,
                              long long batch, long long tokens,
                              long long max_position, long long channels) {
  if (!detail::valid_product({batch, tokens, max_position, channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(position_ids, table, valid_mask, output)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(
      batch * tokens, 1, [&](long long begin, long long end, int) {
        for (long long token = begin; token < end; ++token) {
          float* destination = output + token * channels;
          const int position_x = position_ids[token * 2];
          const int position_y = position_ids[token * 2 + 1];
          if (valid_mask[token] == 0 || position_x < 0 ||
              position_x >= max_position || position_y < 0 ||
              position_y >= max_position) {
            std::fill_n(destination, channels, 0.0f);
            continue;
          }
          const float* table_x = table + position_x * channels;
          const float* table_y =
              table + (max_position + position_y) * channels;
          for (long long channel = 0; channel < channels; ++channel) {
            destination[channel] = table_x[channel] + table_y[channel];
          }
        }
      });
  return Status::kOk;
}

Status factorized_position_2d_storage(
    const int* position_ids, FloatStorageInput table, const int* valid_mask,
    FloatStorageOutput output, long long batch, long long tokens,
    long long max_position, long long channels,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, tokens, max_position, channels}) ||
      table.count != 2 * max_position * channels ||
      output.count != batch * tokens * channels) {
    return Status::kInvalidShape;
  }
  return with_float_storage(
      &table, 1, &output, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return factorized_position_2d(position_ids, inputs[0], valid_mask,
                                      outputs[0], batch, tokens, max_position,
                                      channels);
      },
      workspace);
}

Status pool_tokens_by_position(
    const float* input, const int* position_ids, const int* valid_mask,
    float* output, int* output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width) {
  if (!detail::valid_product(
          {batch, tokens, channels, output_length, kernel_size, source_width}) ||
      source_width % kernel_size != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, position_ids, valid_mask, output,
                           output_mask)) {
    return Status::kInvalidArgument;
  }
  const long long pooled_width = source_width / kernel_size;
  const float scale = std::sqrt(static_cast<float>(channels)) /
                      static_cast<float>(kernel_size * kernel_size);
  std::fill_n(output, batch * output_length * channels, 0.0f);
  std::fill_n(output_mask, batch * output_length, 0);

  threading::parallel_ranges(
      batch, 1, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          for (long long token = 0; token < tokens; ++token) {
            const long long input_token = item * tokens + token;
            if (valid_mask[input_token] == 0) continue;
            const int position_x = position_ids[input_token * 2];
            const int position_y = position_ids[input_token * 2 + 1];
            if (position_x < 0 || position_y < 0) continue;
            const long long bucket = position_x / kernel_size +
                                     pooled_width *
                                         (position_y / kernel_size);
            if (bucket < 0 || bucket >= output_length) continue;
            const long long output_token = item * output_length + bucket;
            output_mask[output_token] = 1;
            const float* source = input + input_token * channels;
            float* destination = output + output_token * channels;
            for (long long channel = 0; channel < channels; ++channel) {
              destination[channel] += source[channel] * scale;
            }
          }
        }
      });
  return Status::kOk;
}

Status pool_tokens_by_position_storage(
    FloatStorageInput input, const int* position_ids, const int* valid_mask,
    float* output, int* output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product(
          {batch, tokens, channels, output_length, kernel_size, source_width}) ||
      input.count != batch * tokens * channels) {
    return Status::kInvalidShape;
  }
  FloatStorageOutput output_view{
      output, FloatStorageType::kF32, batch * output_length * channels};
  return with_float_storage(
      &input, 1, &output_view, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return pool_tokens_by_position(
            inputs[0], position_ids, valid_mask, outputs[0], output_mask,
            batch, tokens, channels, output_length, kernel_size, source_width);
      },
      workspace);
}

}  // namespace quixicore_cpu
