#include <algorithm>
#include <cmath>

#include "kernels/common/float_storage_access.h"
#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

long long conv_output(long long length, long long kernel, long long stride,
                      long long padding, long long dilation) {
  return (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

bool valid_audio_shape(long long batch, long long length, long long channels,
                       long long kernel, long long stride, long long padding,
                       long long dilation) {
  return detail::valid_product({batch, length, channels, kernel, stride,
                                dilation}) &&
         padding >= 0;
}

template <FloatStorageType Type>
Status audio_conv_typed(const void* input, const void* weights,
                        const void* bias, void* output, long long batch,
                        long long input_length, long long input_channels,
                        long long output_channels, long long kernel,
                        long long stride, long long padding,
                        long long dilation, long long output_length) {
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  const long long outputs_per_batch = output_length * output_channels;
  threading::parallel_ranges(
      batch * outputs_per_batch, 64,
      [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long item = index / outputs_per_batch;
          const long long remainder = index - item * outputs_per_batch;
          const long long output_t = remainder / output_channels;
          const long long output_channel =
              remainder - output_t * output_channels;
          float sum = bias == nullptr
                          ? 0.0f
                          : detail::load_storage<Type>(bias, output_channel);
          for (long long kernel_index = 0; kernel_index < kernel;
               ++kernel_index) {
            const long long input_t = output_t * stride +
                                      kernel_index * dilation - padding;
            if (input_t < 0 || input_t >= input_length) continue;
            const long long input_base =
                (item * input_length + input_t) * input_channels;
            const long long weight_base =
                (output_channel * kernel + kernel_index) * input_channels;
            for (long long channel = 0; channel < input_channels; ++channel) {
              sum += detail::load_storage<Type>(input, input_base + channel) *
                     detail::load_storage<Type>(weights,
                                                weight_base + channel);
            }
          }
          detail::store_storage<Type>(output, index, sum);
        }
      });
  return Status::kOk;
}

template <FloatStorageType Type>
Status audio_depthwise_typed(
    const void* input, const void* weights, const void* bias, void* output,
    long long batch, long long input_length, long long channels,
    long long kernel, long long stride, long long padding, long long dilation,
    bool apply_silu, long long output_length) {
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(
      batch * output_length, 1,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long item = row / output_length;
          const long long output_t = row - item * output_length;
          for (long long channel = 0; channel < channels; ++channel) {
            float sum = bias == nullptr
                            ? 0.0f
                            : detail::load_storage<Type>(bias, channel);
            for (long long kernel_index = 0; kernel_index < kernel;
                 ++kernel_index) {
              const long long input_t = output_t * stride +
                                        kernel_index * dilation - padding;
              if (input_t >= 0 && input_t < input_length) {
                sum += detail::load_storage<Type>(
                           input,
                           (item * input_length + input_t) * channels +
                               channel) *
                       detail::load_storage<Type>(
                           weights, channel * kernel + kernel_index);
              }
            }
            if (apply_silu) sum /= 1.0f + std::exp(-sum);
            detail::store_storage<Type>(
                output, (item * output_length + output_t) * channels + channel,
                sum);
          }
        }
      });
  return Status::kOk;
}

}  // namespace

Status audio_conv1d_direct(const float* input, const float* weights,
                           const float* bias, float* output, long long batch,
                           long long input_length, long long input_channels,
                           long long output_channels, long long kernel,
                           long long stride, long long padding,
                           long long dilation) {
  if (!valid_audio_shape(batch, input_length, input_channels, kernel, stride,
                         padding, dilation) ||
      output_channels <= 0) {
    return Status::kInvalidShape;
  }
  const long long output_length =
      conv_output(input_length, kernel, stride, padding, dilation);
  if (!detail::valid_product(
          {batch, output_length, output_channels, kernel, input_channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  const long long outputs_per_batch = output_length * output_channels;
  threading::parallel_ranges(
      batch * outputs_per_batch, 64,
      [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long item = index / outputs_per_batch;
          const long long remainder = index - item * outputs_per_batch;
          const long long output_t = remainder / output_channels;
          const long long output_channel =
              remainder - output_t * output_channels;
          float sum = bias == nullptr ? 0.0f : bias[output_channel];
          const float* weight =
              weights + output_channel * kernel * input_channels;
          for (long long kernel_index = 0; kernel_index < kernel;
               ++kernel_index) {
            const long long input_t = output_t * stride +
                                      kernel_index * dilation - padding;
            if (input_t < 0 || input_t >= input_length) continue;
            const float* source =
                input + (item * input_length + input_t) * input_channels;
            const float* weight_row =
                weight + kernel_index * input_channels;
            for (long long channel = 0; channel < input_channels; ++channel) {
              sum += source[channel] * weight_row[channel];
            }
          }
          output[index] = sum;
        }
      });
  return Status::kOk;
}

Status audio_depthwise_conv1d(const float* input, const float* weights,
                              const float* bias, float* output,
                              long long batch, long long input_length,
                              long long channels, long long kernel,
                              long long stride, long long padding,
                              long long dilation, bool apply_silu) {
  if (!valid_audio_shape(batch, input_length, channels, kernel, stride,
                         padding, dilation)) {
    return Status::kInvalidShape;
  }
  const long long output_length =
      conv_output(input_length, kernel, stride, padding, dilation);
  if (!detail::valid_product({batch, output_length, channels, kernel})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  const long long outputs_per_batch = output_length * channels;
  threading::parallel_ranges(
      batch * output_length, 1,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long item = row / output_length;
          const long long output_t = row - item * output_length;
          float* destination = output + item * outputs_per_batch +
                               output_t * channels;
          if (bias == nullptr) {
            std::fill_n(destination, channels, 0.0f);
          } else {
            std::copy_n(bias, channels, destination);
          }
          for (long long kernel_index = 0; kernel_index < kernel;
               ++kernel_index) {
            const long long input_t = output_t * stride +
                                      kernel_index * dilation - padding;
            if (input_t < 0 || input_t >= input_length) continue;
            const float* source =
                input + (item * input_length + input_t) * channels;
            for (long long channel = 0; channel < channels; ++channel) {
              destination[channel] +=
                  source[channel] * weights[channel * kernel + kernel_index];
            }
          }
          for (long long channel = 0; channel < channels; ++channel) {
            if (apply_silu) {
              destination[channel] /=
                  1.0f + std::exp(-destination[channel]);
            }
          }
        }
      });
  return Status::kOk;
}

Status audio_conv1d_direct_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_length,
    long long input_channels, long long output_channels, long long kernel,
    long long stride, long long padding, long long dilation,
    FloatStorageWorkspace* workspace) {
  if (!valid_audio_shape(batch, input_length, input_channels, kernel, stride,
                         padding, dilation) ||
      output_channels <= 0 ||
      input.count != batch * input_length * input_channels ||
      weights.count != output_channels * kernel * input_channels ||
      (bias.data == nullptr ? bias.count != 0 : bias.count != output_channels)) {
    return Status::kInvalidShape;
  }
  const long long output_length =
      conv_output(input_length, kernel, stride, padding, dilation);
  if (!detail::valid_product({batch, output_length, output_channels}) ||
      output.count != batch * output_length * output_channels) {
    return Status::kInvalidShape;
  }
  if (bias.data == nullptr) {
    const FloatStorageInput inputs[] = {input, weights};
    return with_float_storage(
        inputs, 2, &output, 1,
        [&](const float* const* values, float* const* outputs) {
          return audio_conv1d_direct(
              values[0], values[1], nullptr, outputs[0], batch, input_length,
              input_channels, output_channels, kernel, stride, padding,
              dilation);
        },
        workspace);
  }
  const FloatStorageInput inputs[] = {input, weights, bias};
  return with_float_storage(
      inputs, 3, &output, 1,
      [&](const float* const* values, float* const* outputs) {
        return audio_conv1d_direct(
            values[0], values[1], values[2], outputs[0], batch, input_length,
            input_channels, output_channels, kernel, stride, padding,
            dilation);
      },
      workspace);
}

Status audio_depthwise_conv1d_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_length,
    long long channels, long long kernel, long long stride, long long padding,
    long long dilation, bool apply_silu, FloatStorageWorkspace* workspace) {
  if (!valid_audio_shape(batch, input_length, channels, kernel, stride, padding,
                         dilation) ||
      input.count != batch * input_length * channels ||
      weights.count != channels * kernel ||
      (bias.data == nullptr ? bias.count != 0 : bias.count != channels)) {
    return Status::kInvalidShape;
  }
  const long long output_length =
      conv_output(input_length, kernel, stride, padding, dilation);
  if (!detail::valid_product({batch, output_length, channels}) ||
      output.count != batch * output_length * channels) {
    return Status::kInvalidShape;
  }
  if (bias.data == nullptr) {
    const FloatStorageInput inputs[] = {input, weights};
    return with_float_storage(
        inputs, 2, &output, 1,
        [&](const float* const* values, float* const* outputs) {
          return audio_depthwise_conv1d(
              values[0], values[1], nullptr, outputs[0], batch, input_length,
              channels, kernel, stride, padding, dilation, apply_silu);
        },
        workspace);
  }
  const FloatStorageInput inputs[] = {input, weights, bias};
  return with_float_storage(
      inputs, 3, &output, 1,
      [&](const float* const* values, float* const* outputs) {
        return audio_depthwise_conv1d(
            values[0], values[1], values[2], outputs[0], batch, input_length,
            channels, kernel, stride, padding, dilation, apply_silu);
      },
      workspace);
}

Status audio_causal_depthwise_conv1d(
    const float* input, const float* weights, const float* bias, float* output,
    long long batch, long long input_length, long long channels,
    long long kernel, long long dilation) {
  if (!detail::valid_product(
          {batch, input_length, channels, kernel, dilation})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weights, output)) {
    return Status::kInvalidArgument;
  }
  const long long pad_left = dilation * (kernel - 1);
  threading::parallel_ranges(
      batch * input_length, 1,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long item = row / input_length;
          const long long output_t = row - item * input_length;
          float* destination = output + row * channels;
          if (kernel == 5 && dilation == 1 && output_t >= 4) {
            const float* source0 =
                input + (item * input_length + output_t - 4) * channels;
            const float* source1 = source0 + channels;
            const float* source2 = source1 + channels;
            const float* source3 = source2 + channels;
            const float* source4 = source3 + channels;
            for (long long channel = 0; channel < channels; ++channel) {
              const float* weight = weights + channel * 5;
              destination[channel] =
                  (bias == nullptr ? 0.0f : bias[channel]) +
                  source0[channel] * weight[0] +
                  source1[channel] * weight[1] +
                  source2[channel] * weight[2] +
                  source3[channel] * weight[3] +
                  source4[channel] * weight[4];
            }
            continue;
          }
          if (bias == nullptr) {
            std::fill_n(destination, channels, 0.0f);
          } else {
            std::copy_n(bias, channels, destination);
          }
          for (long long kernel_index = 0; kernel_index < kernel;
               ++kernel_index) {
            const long long input_t =
                output_t + kernel_index * dilation - pad_left;
            if (input_t < 0 || input_t >= input_length) continue;
            const float* source =
                input + (item * input_length + input_t) * channels;
            for (long long channel = 0; channel < channels; ++channel) {
              destination[channel] +=
                  source[channel] * weights[channel * kernel + kernel_index];
            }
          }
        }
      });
  return Status::kOk;
}

Status audio_causal_depthwise_conv1d_storage(
    FloatStorageInput input, FloatStorageInput weights, FloatStorageInput bias,
    FloatStorageOutput output, long long batch, long long input_length,
    long long channels, long long kernel, long long dilation,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product(
          {batch, input_length, channels, kernel, dilation}) ||
      input.count != batch * input_length * channels ||
      weights.count != channels * kernel || output.count != input.count ||
      (bias.data == nullptr ? bias.count != 0 : bias.count != channels)) {
    return Status::kInvalidShape;
  }
  if (bias.data == nullptr) {
    const FloatStorageInput inputs[] = {input, weights};
    return with_float_storage(
        inputs, 2, &output, 1,
        [&](const float* const* values, float* const* outputs) {
          return audio_causal_depthwise_conv1d(
              values[0], values[1], nullptr, outputs[0], batch, input_length,
              channels, kernel, dilation);
        },
        workspace);
  }
  const FloatStorageInput inputs[] = {input, weights, bias};
  return with_float_storage(
      inputs, 3, &output, 1,
      [&](const float* const* values, float* const* outputs) {
        return audio_causal_depthwise_conv1d(
            values[0], values[1], values[2], outputs[0], batch, input_length,
            channels, kernel, dilation);
      },
      workspace);
}

}  // namespace quixicore_cpu
