#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/threading.h"

namespace {

using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

struct Buffer {
  FloatStorageType type = FloatStorageType::kF32;
  std::vector<float> f32;
  std::vector<std::uint16_t> bits;

  static Buffer input(const std::vector<float>& values, FloatStorageType type) {
    Buffer result;
    result.type = type;
    result.f32 = values;
    if (type != FloatStorageType::kF32) {
      result.bits.resize(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        result.bits[i] = type == FloatStorageType::kF16
                             ? quixicore_cpu::float_to_f16(values[i])
                             : quixicore_cpu::float_to_bf16(values[i]);
        result.f32[i] = type == FloatStorageType::kF16
                            ? quixicore_cpu::f16_to_float(result.bits[i])
                            : quixicore_cpu::bf16_to_float(result.bits[i]);
      }
    }
    return result;
  }

  static Buffer output(std::size_t count, FloatStorageType type) {
    Buffer result;
    result.type = type;
    result.f32.resize(count);
    if (type != FloatStorageType::kF32) result.bits.resize(count);
    return result;
  }

  const void* data() const {
    return type == FloatStorageType::kF32
               ? static_cast<const void*>(f32.data())
               : static_cast<const void*>(bits.data());
  }
  void* mutable_data() {
    return type == FloatStorageType::kF32 ? static_cast<void*>(f32.data())
                                          : static_cast<void*>(bits.data());
  }
  std::vector<float> decoded() const {
    if (type == FloatStorageType::kF32) return f32;
    std::vector<float> result(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
      result[i] = type == FloatStorageType::kF16
                      ? quixicore_cpu::f16_to_float(bits[i])
                      : quixicore_cpu::bf16_to_float(bits[i]);
    }
    return result;
  }
};

FloatStorageInput input_view(const Buffer& buffer) {
  return {buffer.data(), buffer.type,
          static_cast<long long>(buffer.f32.size())};
}

FloatStorageOutput output_view(Buffer& buffer) {
  return {buffer.mutable_data(), buffer.type,
          static_cast<long long>(buffer.f32.size())};
}

float tolerance(FloatStorageType type) {
  return type == FloatStorageType::kF32 ? 2.0e-5f : 3.0e-2f;
}

bool close(const std::vector<float>& actual, const std::vector<float>& expected,
           FloatStorageType type, const char* message) {
  bool ok = require(actual.size() == expected.size(), message);
  const float tol = tolerance(type);
  for (std::size_t i = 0; ok && i < actual.size(); ++i) {
    const float delta = std::fabs(actual[i] - expected[i]);
    ok &= require(delta <= tol + tol * std::fabs(expected[i]), message);
  }
  return ok;
}

bool test_aux(FloatStorageType type) {
  bool ok = true;
  const Buffer x = Buffer::input(
      {1.0f, -2.0f, 3.0f, -4.0f, -5.0f, 1.0f, -8.0f, 2.0f,
       4.0f, -7.0f, 6.0f, -3.0f},
      type);
  const float running[] = {6.0f, 0.5f, 2.0f, 9.0f};
  float maximum[4] = {};
  ok &= require(quixicore_cpu::calibration_absmax_storage(
                    input_view(x), running, maximum, 3, 4) == Status::kOk,
                "calibration status");
  ok &= close({maximum, maximum + 4}, {6.0f, 7.0f, 8.0f, 9.0f},
              FloatStorageType::kF32, "calibration oracle");
  const Buffer nonfinite = Buffer::input(
      {std::numeric_limits<float>::quiet_NaN(), 1.0f,
       std::numeric_limits<float>::infinity(), 2.0f, 3.0f,
       std::numeric_limits<float>::quiet_NaN(), -1.0f, 4.0f},
      type);
  float nonfinite_out[4] = {};
  ok &= require(quixicore_cpu::calibration_absmax_storage(
                    input_view(nonfinite), nullptr, nonfinite_out, 2, 4) ==
                    Status::kOk,
                "calibration nonfinite status");
  ok &= require(std::isnan(nonfinite_out[0]) &&
                    std::isnan(nonfinite_out[1]) &&
                    std::isinf(nonfinite_out[2]) && nonfinite_out[2] > 0.0f &&
                    nonfinite_out[3] == 4.0f,
                "calibration nonfinite propagation");

  Buffer out = Buffer::output(x.f32.size(), type);
  ok &= require(quixicore_cpu::logits_softcap_storage(
                    input_view(x), output_view(out), 3.0f) == Status::kOk,
                "softcap status");
  std::vector<float> expected(x.f32.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = 3.0f * std::tanh(x.f32[i] / 3.0f);
  }
  ok &= close(out.decoded(), expected, type, "softcap oracle");

  Buffer clipped = Buffer::output(x.f32.size(), type);
  ok &= require(quixicore_cpu::value_clip_storage(
                    input_view(x), output_view(clipped), -2.5f, 3.5f) ==
                    Status::kOk,
                "value clip status");
  std::vector<float> clip_expected(x.f32.size());
  for (std::size_t i = 0; i < clip_expected.size(); ++i) {
    clip_expected[i] = std::clamp(x.f32[i], -2.5f, 3.5f);
  }
  ok &= close(clipped.decoded(), clip_expected, type, "value clip oracle");
  Buffer unbounded = Buffer::output(x.f32.size(), type);
  ok &= require(
      quixicore_cpu::value_clip_storage(
          input_view(x), output_view(unbounded),
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity()) == Status::kOk,
      "value clip infinite bounds");
  ok &= close(unbounded.decoded(), x.f32, type, "value clip unbounded oracle");
  return ok;
}

bool test_embedding(FloatStorageType type) {
  bool ok = true;
  const long long token_vocab = 3, type_vocab = 2, count = 4, dim = 5;
  std::vector<float> tokens(token_vocab * dim), types(type_vocab * dim);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = 0.1f * (i + 1);
  for (std::size_t i = 0; i < types.size(); ++i) types[i] = -0.03f * (i + 1);
  const Buffer token_table = Buffer::input(tokens, type);
  const Buffer type_table = Buffer::input(types, type);
  const int token_ids[] = {2, -1, 0, 9};
  const int type_ids[] = {1, 0, 8, -1};
  Buffer out = Buffer::output(count * dim, type);
  ok &= require(quixicore_cpu::embedding_lookup_types_storage(
                    token_ids, type_ids, input_view(token_table),
                    input_view(type_table), output_view(out), token_vocab,
                    type_vocab, count, dim, 1.25f) == Status::kOk,
                "embedding types status");
  std::vector<float> expected(count * dim);
  for (long long token = 0; token < count; ++token) {
    for (long long feature = 0; feature < dim; ++feature) {
      expected[token * dim + feature] =
          (token_ids[token] >= 0 && token_ids[token] < token_vocab
               ? 1.25f * token_table.f32[token_ids[token] * dim + feature]
               : 0.0f) +
          (type_ids[token] >= 0 && type_ids[token] < type_vocab
               ? type_table.f32[type_ids[token] * dim + feature]
               : 0.0f);
    }
  }
  ok &= close(out.decoded(), expected, type, "embedding types oracle");

  const long long batch = 2, sequence = 4, hidden = 5;
  std::vector<float> values(batch * sequence * hidden), weights(hidden);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 0.2f * std::sin(0.3f * static_cast<float>(i + 1));
  }
  for (long long i = 0; i < hidden; ++i) weights[i] = 0.8f + 0.05f * i;
  const Buffer x = Buffer::input(values, type);
  const Buffer weight = Buffer::input(weights, type);
  const int mask[] = {1, 0, 1, 0, 0, 0, 0, 0};
  Buffer pooled = Buffer::output(batch * hidden, type);
  ok &= require(quixicore_cpu::masked_mean_pool_rms_l2_storage(
                    input_view(x), mask, input_view(weight),
                    output_view(pooled), batch, sequence, hidden, 1.0e-6f) ==
                    Status::kOk,
                "masked pool status");
  std::vector<float> pooled_expected(batch * hidden, 0.0f);
  double square = 0.0;
  for (long long feature = 0; feature < hidden; ++feature) {
    const float value =
        0.5f * (x.f32[feature] + x.f32[2 * hidden + feature]);
    pooled_expected[feature] = value;
    square += static_cast<double>(value) * value;
  }
  const double inverse_rms = 1.0 / std::sqrt(square / hidden + 1.0e-6);
  double l2 = 0.0;
  for (long long feature = 0; feature < hidden; ++feature) {
    pooled_expected[feature] = static_cast<float>(
        pooled_expected[feature] * inverse_rms * weight.f32[feature]);
    l2 += static_cast<double>(pooled_expected[feature]) *
          pooled_expected[feature];
  }
  const double inverse_l2 = 1.0 / std::sqrt(l2 + 1.0e-12);
  for (long long feature = 0; feature < hidden; ++feature) {
    pooled_expected[feature] =
        static_cast<float>(pooled_expected[feature] * inverse_l2);
  }
  ok &= close(pooled.decoded(), pooled_expected, type, "masked pool oracle");
  return ok;
}

bool test_vision(FloatStorageType type) {
  bool ok = true;
  const long long batch = 1, height = 3, width = 4, channels = 2;
  std::vector<float> values(batch * height * width * channels);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i + 1);
  const Buffer input = Buffer::input(values, type);
  const long long oh = 2, ow = 3, patch_dim = 2 * 2 * channels;
  Buffer patches = Buffer::output(batch * oh * ow * patch_dim, type);
  ok &= require(quixicore_cpu::extract_patches_2d_storage(
                    input_view(input), output_view(patches), batch, height,
                    width, channels, 2, 2, 1, 1) == Status::kOk,
                "patch status");
  std::vector<float> patch_expected(patches.f32.size());
  for (long long y = 0; y < oh; ++y) {
    for (long long x = 0; x < ow; ++x) {
      for (long long ky = 0; ky < 2; ++ky) {
        for (long long kx = 0; kx < 2; ++kx) {
          for (long long c = 0; c < channels; ++c) {
            patch_expected[((y * ow + x) * patch_dim +
                            (ky * 2 + kx) * channels + c)] =
                input.f32[((y + ky) * width + x + kx) * channels + c];
          }
        }
      }
    }
  }
  ok &= close(patches.decoded(), patch_expected, type, "patch oracle");

  const long long output_channels = 3;
  std::vector<float> projection_weights(output_channels * patch_dim);
  std::vector<float> projection_bias(output_channels);
  for (std::size_t i = 0; i < projection_weights.size(); ++i) {
    projection_weights[i] = 0.01f * static_cast<float>(i + 1);
  }
  for (long long i = 0; i < output_channels; ++i) {
    projection_bias[i] = -0.05f * static_cast<float>(i + 1);
  }
  const Buffer projection_weight = Buffer::input(projection_weights, type);
  const Buffer projection_bias_buffer = Buffer::input(projection_bias, type);
  Buffer projection =
      Buffer::output(batch * oh * ow * output_channels, type);
  ok &= require(quixicore_cpu::vision_patch_projection_storage(
                    input_view(input), input_view(projection_weight),
                    input_view(projection_bias_buffer), output_view(projection),
                    batch, height, width, channels, output_channels, 2, 2, 1,
                    1) == Status::kOk,
                "patch projection status");
  std::vector<float> projection_expected(projection.f32.size());
  for (long long row = 0; row < batch * oh * ow; ++row) {
    for (long long output_channel = 0; output_channel < output_channels;
         ++output_channel) {
      float sum = projection_bias_buffer.f32[output_channel];
      for (long long feature = 0; feature < patch_dim; ++feature) {
        sum += patch_expected[row * patch_dim + feature] *
               projection_weight.f32[output_channel * patch_dim + feature];
      }
      projection_expected[row * output_channels + output_channel] = sum;
    }
  }
  ok &= close(projection.decoded(), projection_expected, type,
              "patch projection oracle");

  constexpr long long volume_frames = 3;
  constexpr long long volume_height = 3;
  constexpr long long volume_width = 4;
  constexpr long long volume_channels = 2;
  constexpr long long volume_kernel_t = 2;
  constexpr long long volume_kernel_h = 2;
  constexpr long long volume_kernel_w = 2;
  constexpr long long volume_output_t = 4;
  constexpr long long volume_output_h = 2;
  constexpr long long volume_output_w = 2;
  constexpr long long volume_patch_dim =
      volume_kernel_t * volume_kernel_h * volume_kernel_w * volume_channels;
  std::vector<float> volume_values(volume_frames * volume_height *
                                   volume_width * volume_channels);
  for (std::size_t i = 0; i < volume_values.size(); ++i) {
    volume_values[i] = 0.05f * static_cast<float>(i + 1);
  }
  const Buffer volume = Buffer::input(volume_values, type);
  Buffer volume_patches = Buffer::output(
      volume_output_t * volume_output_h * volume_output_w * volume_patch_dim,
      type);
  ok &= require(quixicore_cpu::extract_patches_3d_storage(
                    input_view(volume), output_view(volume_patches), 1,
                    volume_frames, volume_height, volume_width,
                    volume_channels, volume_kernel_t, volume_kernel_h,
                    volume_kernel_w, 1, 1, 2, 1, 0, 0) == Status::kOk,
                "3d patch status");
  std::vector<float> volume_patch_expected(volume_patches.f32.size(), 0.0f);
  for (long long ot = 0; ot < volume_output_t; ++ot) {
    for (long long oy = 0; oy < volume_output_h; ++oy) {
      for (long long ox = 0; ox < volume_output_w; ++ox) {
        const long long row =
            (ot * volume_output_h + oy) * volume_output_w + ox;
        long long feature = 0;
        for (long long kt = 0; kt < volume_kernel_t; ++kt) {
          const long long it = ot + kt - 1;
          for (long long ky = 0; ky < volume_kernel_h; ++ky) {
            for (long long kx = 0; kx < volume_kernel_w; ++kx) {
              for (long long c = 0; c < volume_channels; ++c, ++feature) {
                if (it >= 0 && it < volume_frames) {
                  volume_patch_expected[row * volume_patch_dim + feature] =
                      volume.f32[(((it * volume_height + oy + ky) *
                                       volume_width +
                                   ox * 2 + kx) *
                                  volume_channels) +
                                 c];
                }
              }
            }
          }
        }
      }
    }
  }
  ok &= close(volume_patches.decoded(), volume_patch_expected, type,
              "3d patch oracle");

  constexpr long long volume_output_channels = 3;
  std::vector<float> volume_weights(volume_output_channels *
                                    volume_patch_dim);
  std::vector<float> volume_bias(volume_output_channels);
  for (std::size_t i = 0; i < volume_weights.size(); ++i) {
    volume_weights[i] = 0.02f * std::cos(0.07f * static_cast<float>(i + 1));
  }
  for (long long i = 0; i < volume_output_channels; ++i) {
    volume_bias[i] = -0.03f * static_cast<float>(i + 1);
  }
  const Buffer volume_weight = Buffer::input(volume_weights, type);
  const Buffer volume_bias_buffer = Buffer::input(volume_bias, type);
  Buffer volume_projection = Buffer::output(
      volume_output_t * volume_output_h * volume_output_w *
          volume_output_channels,
      type);
  ok &= require(quixicore_cpu::vision_patch_projection_3d_storage(
                    input_view(volume), input_view(volume_weight),
                    input_view(volume_bias_buffer),
                    output_view(volume_projection), 1, volume_frames,
                    volume_height, volume_width, volume_channels,
                    volume_output_channels, volume_kernel_t, volume_kernel_h,
                    volume_kernel_w, 1, 1, 2, 1, 0, 0) == Status::kOk,
                "3d patch projection status");
  std::vector<float> volume_projection_expected(
      volume_projection.f32.size());
  for (long long row = 0;
       row < volume_output_t * volume_output_h * volume_output_w; ++row) {
    for (long long output_channel = 0;
         output_channel < volume_output_channels; ++output_channel) {
      float sum = volume_bias_buffer.f32[output_channel];
      for (long long feature = 0; feature < volume_patch_dim; ++feature) {
        sum += volume_patch_expected[row * volume_patch_dim + feature] *
               volume_weight
                   .f32[output_channel * volume_patch_dim + feature];
      }
      volume_projection_expected[row * volume_output_channels +
                                 output_channel] = sum;
    }
  }
  ok &= close(volume_projection.decoded(), volume_projection_expected, type,
              "3d patch projection oracle");

  Buffer resized = Buffer::output(5 * 2 * channels, type);
  ok &= require(quixicore_cpu::interpolate_position_2d_storage(
                    input_view(input), output_view(resized), height, width, 5,
                    2, channels, false) == Status::kOk,
                "position interpolation status");
  std::vector<float> resize_expected(resized.f32.size());
  for (long long y = 0; y < 5; ++y) {
    const double fy = std::clamp((y + 0.5) * height / 5.0 - 0.5, 0.0, 2.0);
    const long long y0 = static_cast<long long>(std::floor(fy));
    const long long y1 = std::min(y0 + 1, height - 1);
    for (long long x = 0; x < 2; ++x) {
      const double fx = std::clamp((x + 0.5) * width / 2.0 - 0.5, 0.0, 3.0);
      const long long x0 = static_cast<long long>(std::floor(fx));
      const long long x1 = std::min(x0 + 1, width - 1);
      for (long long c = 0; c < channels; ++c) {
        const float a = input.f32[(y0 * width + x0) * channels + c];
        const float b = input.f32[(y0 * width + x1) * channels + c];
        const float d = input.f32[(y1 * width + x0) * channels + c];
        const float e = input.f32[(y1 * width + x1) * channels + c];
        const float top = a + static_cast<float>(fx - x0) * (b - a);
        const float bottom = d + static_cast<float>(fx - x0) * (e - d);
        resize_expected[(y * 2 + x) * channels + c] =
            top + static_cast<float>(fy - y0) * (bottom - top);
      }
    }
  }
  ok &= close(resized.decoded(), resize_expected, type, "resize oracle");

  Buffer aligned = Buffer::output(5 * 2 * channels, type);
  ok &= require(quixicore_cpu::interpolate_position_2d_storage(
                    input_view(input), output_view(aligned), height, width, 5,
                    2, channels, true) == Status::kOk,
                "aligned position interpolation status");
  std::vector<float> aligned_expected(aligned.f32.size());
  for (long long y = 0; y < 5; ++y) {
    const double fy = y * (height - 1) / 4.0;
    const long long y0 = static_cast<long long>(std::floor(fy));
    const long long y1 = std::min(y0 + 1, height - 1);
    for (long long x = 0; x < 2; ++x) {
      const double fx = x * (width - 1);
      const long long x0 = static_cast<long long>(std::floor(fx));
      const long long x1 = std::min(x0 + 1, width - 1);
      for (long long c = 0; c < channels; ++c) {
        const float a = input.f32[(y0 * width + x0) * channels + c];
        const float b = input.f32[(y0 * width + x1) * channels + c];
        const float d = input.f32[(y1 * width + x0) * channels + c];
        const float e = input.f32[(y1 * width + x1) * channels + c];
        const float top = a + static_cast<float>(fx - x0) * (b - a);
        const float bottom = d + static_cast<float>(fx - x0) * (e - d);
        aligned_expected[(y * 2 + x) * channels + c] =
            top + static_cast<float>(fy - y0) * (bottom - top);
      }
    }
  }
  ok &= close(aligned.decoded(), aligned_expected, type,
              "aligned resize oracle");

  Buffer pooled = Buffer::output(1 * 2 * 2 * channels, type);
  ok &= require(quixicore_cpu::avg_pool2d_tokens_storage(
                    input_view(input), output_view(pooled), batch, height,
                    width, channels, 2, 2, 2, 2, true) == Status::kOk,
                "average pool status");
  std::vector<float> pool_expected(pooled.f32.size());
  for (long long y = 0; y < 2; ++y) {
    for (long long x = 0; x < 2; ++x) {
      const long long y1 = std::min(y * 2 + 2, height);
      const long long x1 = std::min(x * 2 + 2, width);
      for (long long c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (long long iy = y * 2; iy < y1; ++iy) {
          for (long long ix = x * 2; ix < x1; ++ix) {
            sum += input.f32[(iy * width + ix) * channels + c];
          }
        }
        pool_expected[(y * 2 + x) * channels + c] =
            sum / static_cast<float>((y1 - y * 2) * (x1 - x * 2));
      }
    }
  }
  ok &= close(pooled.decoded(), pool_expected, type, "average pool oracle");

  constexpr long long position_batch = 2;
  constexpr long long position_tokens = 4;
  constexpr long long max_position = 7;
  constexpr long long position_dim = 5;
  std::vector<float> position_table(2 * max_position * position_dim);
  for (std::size_t i = 0; i < position_table.size(); ++i) {
    position_table[i] = 0.03f * std::sin(0.11f * static_cast<float>(i + 1));
  }
  const int position_ids[] = {0, 1, 4, 2, 6, 5, 3, 0,
                              2, 6, 1, 3, 5, 4, 8, -1};
  const int position_valid[] = {1, 1, 0, 1, 1, 0, 1, 1};
  const Buffer position_table_buffer = Buffer::input(position_table, type);
  Buffer factorized =
      Buffer::output(position_batch * position_tokens * position_dim, type);
  ok &= require(quixicore_cpu::factorized_position_2d_storage(
                    position_ids, input_view(position_table_buffer),
                    position_valid, output_view(factorized), position_batch,
                    position_tokens, max_position, position_dim) ==
                    Status::kOk,
                "factorized position status");
  std::vector<float> factorized_expected(factorized.f32.size(), 0.0f);
  for (long long token = 0; token < position_batch * position_tokens;
       ++token) {
    const int px = position_ids[token * 2];
    const int py = position_ids[token * 2 + 1];
    if (position_valid[token] == 0 || px < 0 || px >= max_position || py < 0 ||
        py >= max_position) {
      continue;
    }
    for (long long d = 0; d < position_dim; ++d) {
      factorized_expected[token * position_dim + d] =
          position_table_buffer.f32[px * position_dim + d] +
          position_table_buffer
              .f32[(max_position + py) * position_dim + d];
    }
  }
  ok &= close(factorized.decoded(), factorized_expected, type,
              "factorized position oracle");

  constexpr long long pool_tokens = 6;
  constexpr long long pool_dim = 5;
  const int pool_positions[] = {0, 0, 1, 0, 3, 0, 0, 3, 2, 2, -1, 1};
  const int pool_valid[] = {1, 1, 1, 1, 0, 1};
  std::vector<float> pool_values(pool_tokens * pool_dim);
  for (std::size_t i = 0; i < pool_values.size(); ++i) {
    pool_values[i] = 0.07f * std::cos(0.09f * static_cast<float>(i + 1));
  }
  const Buffer pool_input = Buffer::input(pool_values, type);
  std::vector<float> position_pool(4 * pool_dim);
  int position_pool_mask[4] = {};
  ok &= require(quixicore_cpu::pool_tokens_by_position_storage(
                    input_view(pool_input), pool_positions, pool_valid,
                    position_pool.data(), position_pool_mask, 1, pool_tokens,
                    pool_dim, 4, 2, 4) == Status::kOk,
                "position pool status");
  std::vector<float> position_pool_expected(4 * pool_dim, 0.0f);
  int position_pool_mask_expected[4] = {};
  const float position_scale = std::sqrt(static_cast<float>(pool_dim)) / 4.0f;
  for (long long token = 0; token < pool_tokens; ++token) {
    if (pool_valid[token] == 0 || pool_positions[token * 2] < 0 ||
        pool_positions[token * 2 + 1] < 0) {
      continue;
    }
    const long long bucket = pool_positions[token * 2] / 2 +
                             2 * (pool_positions[token * 2 + 1] / 2);
    if (bucket < 0 || bucket >= 4) continue;
    position_pool_mask_expected[bucket] = 1;
    for (long long d = 0; d < pool_dim; ++d) {
      position_pool_expected[bucket * pool_dim + d] +=
          pool_input.f32[token * pool_dim + d] * position_scale;
    }
  }
  ok &= close(position_pool, position_pool_expected, FloatStorageType::kF32,
              "position pool oracle");
  ok &= require(std::equal(position_pool_mask, position_pool_mask + 4,
                           position_pool_mask_expected),
                "position pool mask oracle");

  constexpr long long rope_heads = 2;
  constexpr long long rope_tokens = 3;
  constexpr long long rope_dim = 64;
  constexpr long long rope_positions = 4;
  std::vector<float> rope_x(rope_heads * rope_tokens * rope_dim);
  std::vector<float> rope_cos(rope_positions * rope_dim / 4);
  std::vector<float> rope_sin(rope_cos.size());
  for (std::size_t i = 0; i < rope_x.size(); ++i) {
    rope_x[i] = 0.2f * std::sin(0.013f * static_cast<float>(i + 1));
  }
  for (std::size_t i = 0; i < rope_cos.size(); ++i) {
    rope_cos[i] = std::cos(0.017f * static_cast<float>(i + 1));
    rope_sin[i] = std::sin(0.017f * static_cast<float>(i + 1));
  }
  const Buffer rope_input = Buffer::input(rope_x, type);
  const Buffer rope_cosine = Buffer::input(rope_cos, type);
  const Buffer rope_sine = Buffer::input(rope_sin, type);
  const int rope_ids[] = {-2, 1, 2, 8, 3, 0};
  Buffer rope_output = Buffer::output(rope_x.size(), type);
  ok &= require(quixicore_cpu::vision_rope_2d_storage(
                    input_view(rope_input), input_view(rope_cosine),
                    input_view(rope_sine), rope_ids, output_view(rope_output),
                    1, rope_heads, rope_tokens, rope_dim, rope_positions) ==
                    Status::kOk,
                "vision rope status");
  std::vector<float> rope_expected(rope_x.size());
  const long long rope_pairs = rope_dim / 4;
  for (long long h = 0; h < rope_heads; ++h) {
    for (long long token = 0; token < rope_tokens; ++token) {
      const long long row = h * rope_tokens + token;
      const int px = std::clamp(rope_ids[token * 2], 0,
                                static_cast<int>(rope_positions - 1));
      const int py = std::clamp(rope_ids[token * 2 + 1], 0,
                                static_cast<int>(rope_positions - 1));
      for (long long pair = 0; pair < rope_pairs; ++pair) {
        const float x0 = rope_input.f32[row * rope_dim + pair];
        const float x1 = rope_input.f32[row * rope_dim + rope_pairs + pair];
        const float y0 = rope_input.f32[row * rope_dim + 2 * rope_pairs + pair];
        const float y1 = rope_input.f32[row * rope_dim + 3 * rope_pairs + pair];
        const float cx = rope_cosine.f32[px * rope_pairs + pair];
        const float sx = rope_sine.f32[px * rope_pairs + pair];
        const float cy = rope_cosine.f32[py * rope_pairs + pair];
        const float sy = rope_sine.f32[py * rope_pairs + pair];
        rope_expected[row * rope_dim + pair] = x0 * cx - x1 * sx;
        rope_expected[row * rope_dim + rope_pairs + pair] = x0 * sx + x1 * cx;
        rope_expected[row * rope_dim + 2 * rope_pairs + pair] =
            y0 * cy - y1 * sy;
        rope_expected[row * rope_dim + 3 * rope_pairs + pair] =
            y0 * sy + y1 * cy;
      }
    }
  }
  ok &= close(rope_output.decoded(), rope_expected, type,
              "vision rope oracle");

  Buffer qwen_rope_output = Buffer::output(rope_x.size(), type);
  ok &= require(quixicore_cpu::qwen_vision_rope_2d_storage(
                    input_view(rope_input), input_view(rope_cosine),
                    input_view(rope_sine), rope_ids,
                    output_view(qwen_rope_output), 1, rope_heads, rope_tokens,
                    rope_dim, rope_positions) == Status::kOk,
                "qwen vision rope status");
  std::vector<float> qwen_rope_expected(rope_x.size());
  for (long long h = 0; h < rope_heads; ++h) {
    for (long long token = 0; token < rope_tokens; ++token) {
      const long long row = h * rope_tokens + token;
      const int px = std::clamp(rope_ids[token * 2], 0,
                                static_cast<int>(rope_positions - 1));
      const int py = std::clamp(rope_ids[token * 2 + 1], 0,
                                static_cast<int>(rope_positions - 1));
      for (long long pair = 0; pair < rope_pairs; ++pair) {
        const float x0 = rope_input.f32[row * rope_dim + pair];
        const float y0 =
            rope_input.f32[row * rope_dim + rope_pairs + pair];
        const float x1 =
            rope_input.f32[row * rope_dim + 2 * rope_pairs + pair];
        const float y1 =
            rope_input.f32[row * rope_dim + 3 * rope_pairs + pair];
        const float cx = rope_cosine.f32[px * rope_pairs + pair];
        const float sx = rope_sine.f32[px * rope_pairs + pair];
        const float cy = rope_cosine.f32[py * rope_pairs + pair];
        const float sy = rope_sine.f32[py * rope_pairs + pair];
        qwen_rope_expected[row * rope_dim + pair] = x0 * cx - x1 * sx;
        qwen_rope_expected[row * rope_dim + rope_pairs + pair] =
            y0 * cy - y1 * sy;
        qwen_rope_expected[row * rope_dim + 2 * rope_pairs + pair] =
            x0 * sx + x1 * cx;
        qwen_rope_expected[row * rope_dim + 3 * rope_pairs + pair] =
            y0 * sy + y1 * cy;
      }
    }
  }
  ok &= close(qwen_rope_output.decoded(), qwen_rope_expected, type,
              "qwen vision rope oracle");
  return ok;
}

bool test_audio(FloatStorageType type) {
  bool ok = true;
  const long long batch = 2, length = 7, input_channels = 3;
  const long long output_channels = 4, kernel = 3;
  std::vector<float> values(batch * length * input_channels);
  std::vector<float> weights(output_channels * kernel * input_channels);
  std::vector<float> bias(output_channels);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 0.1f * std::sin(0.2f * static_cast<float>(i + 1));
  }
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = 0.2f * std::cos(0.17f * static_cast<float>(i + 1));
  }
  for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = 0.03f * (i + 1);
  const Buffer input = Buffer::input(values, type);
  const Buffer weight = Buffer::input(weights, type);
  const Buffer bias_buffer = Buffer::input(bias, type);
  const long long output_length = 4;
  Buffer output = Buffer::output(batch * output_length * output_channels, type);
  ok &= require(quixicore_cpu::audio_conv1d_direct_storage(
                    input_view(input), input_view(weight),
                    input_view(bias_buffer), output_view(output), batch, length,
                    input_channels, output_channels, kernel, 2, 1, 1) ==
                    Status::kOk,
                "audio conv status");
  std::vector<float> expected(output.f32.size());
  for (long long b = 0; b < batch; ++b) {
    for (long long t = 0; t < output_length; ++t) {
      for (long long o = 0; o < output_channels; ++o) {
        float sum = bias_buffer.f32[o];
        for (long long k = 0; k < kernel; ++k) {
          const long long source_t = t * 2 + k - 1;
          if (source_t < 0 || source_t >= length) continue;
          for (long long c = 0; c < input_channels; ++c) {
            sum += input.f32[(b * length + source_t) * input_channels + c] *
                   weight.f32[(o * kernel + k) * input_channels + c];
          }
        }
        expected[(b * output_length + t) * output_channels + o] = sum;
      }
    }
  }
  ok &= close(output.decoded(), expected, type, "audio conv oracle");

  const long long channels = 5;
  std::vector<float> dw_input(batch * length * channels);
  std::vector<float> dw_weight(channels * kernel), dw_bias(channels);
  for (std::size_t i = 0; i < dw_input.size(); ++i) {
    dw_input[i] = 0.15f * std::sin(0.1f * static_cast<float>(i + 2));
  }
  for (std::size_t i = 0; i < dw_weight.size(); ++i) {
    dw_weight[i] = 0.12f * std::cos(0.13f * static_cast<float>(i + 4));
  }
  for (std::size_t i = 0; i < dw_bias.size(); ++i) dw_bias[i] = -0.01f * i;
  const Buffer dw_x = Buffer::input(dw_input, type);
  const Buffer dw_w = Buffer::input(dw_weight, type);
  const Buffer dw_b = Buffer::input(dw_bias, type);
  Buffer dw_out = Buffer::output(batch * length * channels, type);
  ok &= require(quixicore_cpu::audio_depthwise_conv1d_storage(
                    input_view(dw_x), input_view(dw_w), input_view(dw_b),
                    output_view(dw_out), batch, length, channels, kernel, 1, 1,
                    1, true) == Status::kOk,
                "audio depthwise status");
  std::vector<float> dw_expected(dw_out.f32.size());
  for (long long b = 0; b < batch; ++b) {
    for (long long t = 0; t < length; ++t) {
      for (long long c = 0; c < channels; ++c) {
        float sum = dw_b.f32[c];
        for (long long k = 0; k < kernel; ++k) {
          const long long source_t = t + k - 1;
          if (source_t >= 0 && source_t < length) {
            sum += dw_x.f32[(b * length + source_t) * channels + c] *
                   dw_w.f32[c * kernel + k];
          }
        }
        dw_expected[(b * length + t) * channels + c] =
            sum / (1.0f + std::exp(-sum));
      }
    }
  }
  ok &= close(dw_out.decoded(), dw_expected, type, "audio depthwise oracle");
  Buffer dw_plain = Buffer::output(batch * length * channels, type);
  ok &= require(quixicore_cpu::audio_depthwise_conv1d_storage(
                    input_view(dw_x), input_view(dw_w), input_view(dw_b),
                    output_view(dw_plain), batch, length, channels, kernel, 1,
                    1, 1, false) == Status::kOk,
                "audio depthwise plain status");
  for (long long b = 0; b < batch; ++b) {
    for (long long t = 0; t < length; ++t) {
      for (long long c = 0; c < channels; ++c) {
        float sum = dw_b.f32[c];
        for (long long k = 0; k < kernel; ++k) {
          const long long source_t = t + k - 1;
          if (source_t >= 0 && source_t < length) {
            sum += dw_x.f32[(b * length + source_t) * channels + c] *
                   dw_w.f32[c * kernel + k];
          }
        }
        dw_expected[(b * length + t) * channels + c] = sum;
      }
    }
  }
  ok &= close(dw_plain.decoded(), dw_expected, type,
              "audio depthwise plain oracle");

  Buffer causal = Buffer::output(batch * length * channels, type);
  ok &= require(quixicore_cpu::audio_causal_depthwise_conv1d_storage(
                    input_view(dw_x), input_view(dw_w), input_view(dw_b),
                    output_view(causal), batch, length, channels, kernel, 2) ==
                    Status::kOk,
                "audio causal depthwise status");
  std::vector<float> causal_expected(causal.f32.size());
  const long long pad_left = 2 * (kernel - 1);
  for (long long b = 0; b < batch; ++b) {
    for (long long t = 0; t < length; ++t) {
      for (long long c = 0; c < channels; ++c) {
        float sum = dw_b.f32[c];
        for (long long k = 0; k < kernel; ++k) {
          const long long source_t = t + 2 * k - pad_left;
          if (source_t >= 0 && source_t < length) {
            sum += dw_x.f32[(b * length + source_t) * channels + c] *
                   dw_w.f32[c * kernel + k];
          }
        }
        causal_expected[(b * length + t) * channels + c] = sum;
      }
    }
  }
  ok &= close(causal.decoded(), causal_expected, type,
              "audio causal depthwise oracle");
  return ok;
}

bool test_cross_attention(FloatStorageType type) {
  constexpr long long batch = 2;
  constexpr long long query_heads = 4;
  constexpr long long kv_heads = 2;
  constexpr long long query_length = 3;
  constexpr long long key_length = 5;
  constexpr long long head_dim = 64;
  std::vector<float> q(batch * query_heads * query_length * head_dim);
  std::vector<float> k(batch * kv_heads * key_length * head_dim);
  std::vector<float> v(k.size());
  std::vector<float> bias(batch * query_heads * query_length * key_length);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.15f * std::sin(0.017f * static_cast<float>(i + 1));
  }
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.15f * std::cos(0.013f * static_cast<float>(i + 3));
    v[i] = 0.20f * std::sin(0.011f * static_cast<float>(i + 5));
  }
  for (std::size_t i = 0; i < bias.size(); ++i) {
    bias[i] = 0.03f * std::cos(0.019f * static_cast<float>(i + 7));
  }
  const Buffer tq = Buffer::input(q, type);
  const Buffer tk = Buffer::input(k, type);
  const Buffer tv = Buffer::input(v, type);
  Buffer output = Buffer::output(q.size(), type);
  const int lengths[] = {5, 0};
  bool ok = require(quixicore_cpu::cross_attention_storage(
                        input_view(tq), input_view(tk), input_view(tv), lengths,
                        bias.data(), output_view(output), batch, query_heads,
                        kv_heads, query_length, key_length, head_dim, 0.0f,
                        3.0f) == Status::kOk,
                    "cross attention status");
  std::vector<float> expected(q.size(), 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (long long b = 0; b < batch; ++b) {
    for (long long h = 0; h < query_heads; ++h) {
      const long long kh = h / (query_heads / kv_heads);
      for (long long t = 0; t < query_length; ++t) {
        if (lengths[b] == 0) continue;
        const long long query_row =
            ((b * query_heads + h) * query_length + t);
        std::vector<double> scores(lengths[b]);
        double maximum = -std::numeric_limits<double>::infinity();
        for (long long key = 0; key < lengths[b]; ++key) {
          const long long kv_row =
              ((b * kv_heads + kh) * key_length + key);
          double score = 0.0;
          for (long long d = 0; d < head_dim; ++d) {
            score += static_cast<double>(tq.f32[query_row * head_dim + d]) *
                     tk.f32[kv_row * head_dim + d];
          }
          score = score * scale + bias[query_row * key_length + key];
          score = 3.0 * std::tanh(score / 3.0);
          scores[key] = score;
          maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (double& score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (long long key = 0; key < lengths[b]; ++key) {
          const long long kv_row =
              ((b * kv_heads + kh) * key_length + key);
          const float probability =
              static_cast<float>(scores[key] / denominator);
          for (long long d = 0; d < head_dim; ++d) {
            expected[query_row * head_dim + d] +=
                probability * tv.f32[kv_row * head_dim + d];
          }
        }
      }
    }
  }
  ok &= close(output.decoded(), expected, type, "cross attention oracle");
  return ok;
}

bool test_audio_relative_attention(FloatStorageType type) {
  constexpr long long batch = 2;
  constexpr long long length = 7;
  constexpr long long heads = 2;
  constexpr long long head_dim = 64;
  constexpr long long relative_positions = 5;
  constexpr long long chunk = 4;
  constexpr long long left = 3;
  constexpr long long right = 1;
  const long long count = batch * length * heads * head_dim;
  std::vector<float> q(count), k(count), v(count);
  std::vector<float> relative(relative_positions * heads * head_dim);
  std::vector<float> per_dim(head_dim);
  for (long long i = 0; i < count; ++i) {
    q[i] = 0.08f * std::sin(0.013f * static_cast<float>(i + 1));
    k[i] = 0.08f * std::cos(0.017f * static_cast<float>(i + 2));
    v[i] = 0.20f * std::sin(0.011f * static_cast<float>(i + 3));
  }
  for (std::size_t i = 0; i < relative.size(); ++i) {
    relative[i] = 0.08f * std::cos(0.019f * static_cast<float>(i + 4));
  }
  for (long long i = 0; i < head_dim; ++i) {
    per_dim[i] = 0.2f * std::sin(0.07f * static_cast<float>(i + 1));
  }
  const Buffer tq = Buffer::input(q, type);
  const Buffer tk = Buffer::input(k, type);
  const Buffer tv = Buffer::input(v, type);
  const Buffer tr = Buffer::input(relative, type);
  Buffer output = Buffer::output(count, type);
  const int lengths[] = {7, 4};
  bool ok = require(quixicore_cpu::audio_relative_attention_storage(
                        input_view(tq), input_view(tk), input_view(tv),
                        input_view(tr), per_dim.data(), lengths,
                        output_view(output), batch, length, heads, head_dim,
                        relative_positions, chunk, left, right, 0.0f, 0.0f,
                        5.0f) == Status::kOk,
                    "audio relative attention status");
  std::vector<float> expected(count, 0.0f);
  const float log_two = std::log(2.0f);
  const float query_scale =
      1.0f / (std::sqrt(static_cast<float>(head_dim)) * log_two);
  const float key_scale = 1.0f / log_two;
  for (long long b = 0; b < batch; ++b) {
    for (long long t = 0; t < lengths[b]; ++t) {
      const long long query_in_chunk = t % chunk;
      const long long context_start = (t / chunk) * chunk - (left - 1);
      const long long context_length = chunk + left - 1 + right;
      for (long long h = 0; h < heads; ++h) {
        std::vector<float> scaled_query(head_dim);
        for (long long d = 0; d < head_dim; ++d) {
          const float learned =
              std::max(per_dim[d], 0.0f) +
              std::log1p(std::exp(-std::fabs(per_dim[d])));
          scaled_query[d] =
              tq.f32[((b * length + t) * heads + h) * head_dim + d] *
              query_scale * learned;
        }
        std::vector<float> scores;
        std::vector<long long> rows;
        float maximum = -std::numeric_limits<float>::infinity();
        for (long long ci = 0; ci < context_length; ++ci) {
          const long long key_t = context_start + ci;
          if (key_t < 0 || key_t >= lengths[b]) continue;
          const long long kv_row = (b * length + key_t) * heads + h;
          const long long ri = ci - query_in_chunk;
          float score = 0.0f;
          for (long long d = 0; d < head_dim; ++d) {
            score += scaled_query[d] * tk.f32[kv_row * head_dim + d] *
                     key_scale;
            if (ri >= 0 && ri < relative_positions) {
              score += scaled_query[d] *
                       tr.f32[(ri * heads + h) * head_dim + d];
            }
          }
          score = 5.0f * std::tanh(score / 5.0f);
          scores.push_back(score);
          rows.push_back(kv_row);
          maximum = std::max(maximum, score);
        }
        float denominator = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        const long long output_row = (b * length + t) * heads + h;
        for (std::size_t index = 0; index < scores.size(); ++index) {
          const float probability = scores[index] / denominator;
          for (long long d = 0; d < head_dim; ++d) {
            expected[output_row * head_dim + d] +=
                probability * tv.f32[rows[index] * head_dim + d];
          }
        }
      }
    }
  }
  ok &= close(output.decoded(), expected, type,
              "audio relative attention oracle");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  quixicore_cpu::set_num_threads(4);
  for (FloatStorageType type : {FloatStorageType::kF32,
                                FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    ok &= test_aux(type);
    ok &= test_embedding(type);
    ok &= test_vision(type);
    ok &= test_audio(type);
    ok &= test_cross_attention(type);
    ok &= test_audio_relative_attention(type);
  }
  float value = 0.0f;
  ok &= require(quixicore_cpu::logits_softcap(&value, &value, 1, 0.0f) ==
                    Status::kInvalidArgument,
                "softcap rejects zero cap");
  ok &= require(quixicore_cpu::extract_patches_2d(
                    &value, &value, 1, 1, 1, 1, 2, 2, 1, 1) ==
                    Status::kInvalidShape,
                "patch rejects empty output");
  ok &= require(quixicore_cpu::extract_patches_3d(
                    &value, &value, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1) ==
                    Status::kInvalidShape,
                "3d patch rejects empty output");
  ok &= require(quixicore_cpu::value_clip(
                    &value, std::numeric_limits<float>::quiet_NaN(), 1.0f,
                    &value, 1) == Status::kInvalidShape,
                "value clip rejects NaN bound");
  ok &= require(quixicore_cpu::value_clip(&value, 2.0f, -2.0f, &value, 1) ==
                    Status::kInvalidShape,
                "value clip rejects reversed bounds");
  ok &= require(quixicore_cpu::audio_conv1d_direct(
                    &value, &value, nullptr, &value, 1, 1, 1, 1, 3, 1, 0,
                    1) == Status::kInvalidShape,
                "audio rejects empty output");
  quixicore_cpu::set_num_threads(1);
  if (!ok) return 1;
  std::cout << "BaseRT kernel tests passed\n";
  return 0;
}
