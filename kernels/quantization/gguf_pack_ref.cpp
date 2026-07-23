#include "kernels/quantization/gguf_pack_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "kernels/common/fp16.h"
#include "kernels/quantization/gguf_ref.h"
#include "kernels/quantization/iq_pack_ref.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

constexpr std::int8_t kFp4Values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constexpr std::int8_t kIq4Values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};
constexpr float kGroupEpsilon = 1.0e-15f;

void put_half(std::uint8_t* destination, float value) {
  const std::uint16_t bits = fp32_to_fp16(value);
  std::memcpy(destination, &bits, sizeof(bits));
}

void put_u32(std::uint8_t* destination, std::uint32_t value) {
  std::memcpy(destination, &value, sizeof(value));
}

float half_round(float value) {
  return fp16_to_fp32(fp32_to_fp16(value));
}

int nearest_int(float value) {
  // Same round-to-nearest-even construction as llama.cpp's canonical GGUF
  // authoring path. All callers feed small normalized values.
  value += 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

float make_qx_quants(int count, int maximum, const float* values,
                     std::int8_t* levels) {
  float signed_maximum = 0.0f;
  float absolute_maximum = 0.0f;
  for (int index = 0; index < count; ++index) {
    const float magnitude = std::fabs(values[index]);
    if (magnitude > absolute_maximum) {
      absolute_maximum = magnitude;
      signed_maximum = values[index];
    }
  }
  if (absolute_maximum < kGroupEpsilon) {
    std::fill_n(levels, count, 0);
    return 0.0f;
  }
  float inverse = -static_cast<float>(maximum) / signed_maximum;
  float sum_lx = 0.0f;
  float sum_l2 = 0.0f;
  for (int index = 0; index < count; ++index) {
    int level = nearest_int(inverse * values[index]);
    level = std::clamp(level, -maximum, maximum - 1);
    levels[index] = static_cast<std::int8_t>(level + maximum);
    const float weight = values[index] * values[index];
    sum_lx += weight * values[index] * level;
    sum_l2 += weight * level * level;
  }
  float scale = sum_l2 != 0.0f ? sum_lx / sum_l2 : 0.0f;
  float best = scale * sum_lx;
  for (int attempt = -9; attempt <= 9; ++attempt) {
    if (attempt == 0) continue;
    inverse = -(maximum + 0.1f * attempt) / signed_maximum;
    sum_lx = 0.0f;
    sum_l2 = 0.0f;
    for (int index = 0; index < count; ++index) {
      int level = nearest_int(inverse * values[index]);
      level = std::clamp(level, -maximum, maximum - 1);
      const float weight = values[index] * values[index];
      sum_lx += weight * values[index] * level;
      sum_l2 += weight * level * level;
    }
    if (sum_l2 > 0.0f && sum_lx * sum_lx > best * sum_l2) {
      for (int index = 0; index < count; ++index) {
        int level = nearest_int(inverse * values[index]);
        levels[index] = static_cast<std::int8_t>(
            maximum + std::clamp(level, -maximum, maximum - 1));
      }
      scale = sum_lx / sum_l2;
      best = scale * sum_lx;
    }
  }
  return scale;
}

float make_q3_quants(int count, int maximum, const float* values,
                     std::int8_t* levels) {
  float signed_maximum = 0.0f;
  float absolute_maximum = 0.0f;
  for (int index = 0; index < count; ++index) {
    const float magnitude = std::fabs(values[index]);
    if (magnitude > absolute_maximum) {
      absolute_maximum = magnitude;
      signed_maximum = values[index];
    }
  }
  if (absolute_maximum < kGroupEpsilon) {
    std::fill_n(levels, count, 0);
    return 0.0f;
  }
  const float inverse = -static_cast<float>(maximum) / signed_maximum;
  float sum_lx = 0.0f;
  float sum_l2 = 0.0f;
  for (int index = 0; index < count; ++index) {
    int level = nearest_int(inverse * values[index]);
    level = std::clamp(level, -maximum, maximum - 1);
    levels[index] = static_cast<std::int8_t>(level);
    const float weight = values[index] * values[index];
    sum_lx += weight * values[index] * level;
    sum_l2 += weight * level * level;
  }
  for (int attempt = 0; attempt < 5; ++attempt) {
    int changed = 0;
    for (int index = 0; index < count; ++index) {
      const float weight = values[index] * values[index];
      float candidate_lx =
          sum_lx - weight * values[index] * levels[index];
      if (candidate_lx <= 0.0f) continue;
      float candidate_l2 =
          sum_l2 - weight * levels[index] * levels[index];
      int level = nearest_int(values[index] * candidate_l2 / candidate_lx);
      level = std::clamp(level, -maximum, maximum - 1);
      if (level == levels[index]) continue;
      candidate_lx += weight * values[index] * level;
      candidate_l2 += weight * level * level;
      if (candidate_l2 > 0.0f &&
          candidate_lx * candidate_lx * sum_l2 >
              sum_lx * sum_lx * candidate_l2) {
        levels[index] = static_cast<std::int8_t>(level);
        sum_lx = candidate_lx;
        sum_l2 = candidate_l2;
        ++changed;
      }
    }
    if (changed == 0) break;
  }
  for (int index = 0; index < count; ++index) levels[index] += maximum;
  return sum_l2 > 0.0f ? sum_lx / sum_l2 : 0.0f;
}

float make_affine_quants(int count, int maximum, const float* values,
                         const float* weights, std::uint8_t* levels,
                         float* minimum_out, std::uint8_t* scratch,
                         float range_min, float range_step, int steps,
                         bool absolute_error) {
  float minimum = values[0];
  float maximum_value = values[0];
  float weight_sum = weights[0];
  float weighted_values = weights[0] * values[0];
  for (int index = 1; index < count; ++index) {
    minimum = std::min(minimum, values[index]);
    maximum_value = std::max(maximum_value, values[index]);
    weight_sum += weights[index];
    weighted_values += weights[index] * values[index];
  }
  minimum = std::min(minimum, 0.0f);
  if (maximum_value == minimum) {
    std::fill_n(levels, count, 0);
    *minimum_out = -minimum;
    return 0.0f;
  }
  float inverse = maximum / (maximum_value - minimum);
  float scale = 1.0f / inverse;
  float best_error = 0.0f;
  for (int index = 0; index < count; ++index) {
    levels[index] = static_cast<std::uint8_t>(std::clamp(
        nearest_int(inverse * (values[index] - minimum)), 0, maximum));
    float error = scale * levels[index] + minimum - values[index];
    error = absolute_error ? std::fabs(error) : error * error;
    best_error += weights[index] * error;
  }
  for (int attempt = 0; attempt <= steps; ++attempt) {
    inverse = (range_min + range_step * attempt + maximum) /
              (maximum_value - minimum);
    float sum_l = 0.0f;
    float sum_l2 = 0.0f;
    float sum_xl = 0.0f;
    for (int index = 0; index < count; ++index) {
      const int level = std::clamp(
          nearest_int(inverse * (values[index] - minimum)), 0, maximum);
      scratch[index] = static_cast<std::uint8_t>(level);
      sum_l += weights[index] * level;
      sum_l2 += weights[index] * level * level;
      sum_xl += weights[index] * level * values[index];
    }
    const float determinant = weight_sum * sum_l2 - sum_l * sum_l;
    if (determinant <= 0.0f) continue;
    float candidate_scale =
        (weight_sum * sum_xl - weighted_values * sum_l) / determinant;
    float candidate_minimum =
        (sum_l2 * weighted_values - sum_l * sum_xl) / determinant;
    if (candidate_minimum > 0.0f) {
      candidate_minimum = 0.0f;
      candidate_scale = sum_xl / sum_l2;
    }
    float error_sum = 0.0f;
    for (int index = 0; index < count; ++index) {
      float error = candidate_scale * scratch[index] + candidate_minimum -
                    values[index];
      error = absolute_error ? std::fabs(error) : error * error;
      error_sum += weights[index] * error;
    }
    if (error_sum < best_error) {
      std::copy_n(scratch, count, levels);
      best_error = error_sum;
      scale = candidate_scale;
      minimum = candidate_minimum;
    }
  }
  *minimum_out = -minimum;
  return scale;
}

void scale_min_k4(int index, const std::uint8_t* scales,
                  std::uint8_t* scale, std::uint8_t* minimum) {
  if (index < 4) {
    *scale = scales[index] & 63;
    *minimum = scales[index + 4] & 63;
  } else {
    *scale = static_cast<std::uint8_t>(
        (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4));
    *minimum = static_cast<std::uint8_t>(
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
  }
}

int best_iq4(float value) {
  if (value <= kIq4Values[0]) return 0;
  if (value >= kIq4Values[15]) return 15;
  int low = 0;
  int high = 15;
  while (high - low > 1) {
    const int middle = (low + high) / 2;
    if (value < kIq4Values[middle]) high = middle;
    else low = middle;
  }
  return value - kIq4Values[high - 1] < kIq4Values[high] - value
             ? high - 1
             : high;
}

float e8m0_half(std::uint8_t exponent) {
  return std::ldexp(1.0f, static_cast<int>(exponent) - 128);
}

float ue4m3_half(std::uint8_t value) {
  if (value == 0 || value == 0x7f) return 0.0f;
  const int exponent = (value >> 3) & 15;
  const int mantissa = value & 7;
  if (exponent == 0) {
    return std::ldexp(static_cast<float>(mantissa), -10);
  }
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                    exponent - 8);
}

std::uint8_t fp32_to_ue4m3(float value) {
  if (!(value > 0.0f)) return 0;
  value = std::min(value, 448.0f);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  int exponent = static_cast<int>((bits >> 23) & 255) - 127 + 7;
  int mantissa = static_cast<int>((bits >> 20) & 7);
  if (exponent <= 0) {
    mantissa = static_cast<int>(value * 512.0f + 0.5f);
    if (mantissa > 7) mantissa = 7;
    return mantissa < 1 ? 0 : static_cast<std::uint8_t>(mantissa);
  }
  if (exponent >= 15) return 0x7e;
  mantissa += static_cast<int>((bits >> 19) & 1);
  if (mantissa > 7) {
    mantissa = 0;
    if (++exponent >= 15) return 0x7e;
  }
  return static_cast<std::uint8_t>((exponent << 3) | mantissa);
}

std::uint8_t best_fp4(float value, float half_scale) {
  int best = 0;
  float error = std::fabs(static_cast<float>(kFp4Values[0]) * half_scale -
                          value);
  for (int candidate = 1; candidate < 16; ++candidate) {
    const float candidate_error =
        std::fabs(static_cast<float>(kFp4Values[candidate]) * half_scale -
                  value);
    if (candidate_error < error) {
      best = candidate;
      error = candidate_error;
    }
  }
  return static_cast<std::uint8_t>(best);
}

bool finite_block(const float* values, long long count) {
  for (long long index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) return false;
  }
  return true;
}

void pack_q1_0(const float* values, std::uint8_t* block) {
  float sum_abs = 0.0f;
  for (int index = 0; index < 128; ++index) {
    sum_abs += std::fabs(values[index]);
  }
  put_half(block, sum_abs / 128.0f);
  std::memset(block + 2, 0, 16);
  for (int index = 0; index < 128; ++index) {
    if (values[index] >= 0.0f) {
      block[2 + index / 8] |= static_cast<std::uint8_t>(1u << (index & 7));
    }
  }
}

void pack_q2_0(const float* values, std::uint8_t* block) {
  float maximum = 0.0f;
  for (int index = 0; index < 64; ++index) {
    maximum = std::max(maximum, std::fabs(values[index]));
  }
  put_half(block, maximum);
  std::memset(block + 2, 0, 16);
  for (int index = 0; index < 64; ++index) {
    const float normalized =
        maximum > 0.0f ? values[index] / maximum : 0.0f;
    const int quant = std::clamp(
        static_cast<int>(std::round(normalized)) + 1, 0, 3);
    block[2 + index / 4] |=
        static_cast<std::uint8_t>(quant << (2 * (index & 3)));
  }
}

void pack_q4_1(const float* values, std::uint8_t* block) {
  float minimum = std::numeric_limits<float>::max();
  float maximum = -std::numeric_limits<float>::max();
  for (int index = 0; index < 32; ++index) {
    minimum = std::min(minimum, values[index]);
    maximum = std::max(maximum, values[index]);
  }
  const float scale = (maximum - minimum) / 15.0f;
  const double inverse =
      scale != 0.0f ? 1.0 / static_cast<double>(scale) : 0.0;
  put_half(block, scale);
  put_half(block + 2, minimum);
  for (int index = 0; index < 16; ++index) {
    const int low = std::min(
        15, static_cast<int>((static_cast<double>(values[index]) - minimum) *
                             inverse + 0.5));
    const int high = std::min(
        15,
        static_cast<int>((static_cast<double>(values[16 + index]) - minimum) *
                         inverse + 0.5));
    block[4 + index] = static_cast<std::uint8_t>(low | (high << 4));
  }
}

void pack_q5_0(const float* values, std::uint8_t* block) {
  float absolute_maximum = 0.0f;
  float signed_maximum = 0.0f;
  for (int index = 0; index < 32; ++index) {
    if (absolute_maximum < std::fabs(values[index])) {
      absolute_maximum = std::fabs(values[index]);
      signed_maximum = values[index];
    }
  }
  const float scale = signed_maximum / -16.0f;
  const double inverse =
      scale != 0.0f ? 1.0 / static_cast<double>(scale) : 0.0;
  put_half(block, scale);
  std::uint32_t high_bits = 0;
  for (int index = 0; index < 16; ++index) {
    const int low =
        std::min(31, static_cast<int>(
                         static_cast<double>(values[index]) * inverse + 16.5));
    const int high = std::min(
        31, static_cast<int>(static_cast<double>(values[16 + index]) *
                             inverse + 16.5));
    block[6 + index] =
        static_cast<std::uint8_t>((low & 15) | ((high & 15) << 4));
    high_bits |= static_cast<std::uint32_t>((low >> 4) & 1) << index;
    high_bits |= static_cast<std::uint32_t>((high >> 4) & 1) << (16 + index);
  }
  put_u32(block + 2, high_bits);
}

void pack_q5_1(const float* values, std::uint8_t* block) {
  float minimum = std::numeric_limits<float>::max();
  float maximum = -std::numeric_limits<float>::max();
  for (int index = 0; index < 32; ++index) {
    minimum = std::min(minimum, values[index]);
    maximum = std::max(maximum, values[index]);
  }
  const float scale = (maximum - minimum) / 31.0f;
  const double inverse =
      scale != 0.0f ? 1.0 / static_cast<double>(scale) : 0.0;
  put_half(block, scale);
  put_half(block + 2, minimum);
  std::uint32_t high_bits = 0;
  for (int index = 0; index < 16; ++index) {
    const int low = static_cast<int>(
        (static_cast<double>(values[index]) - minimum) * inverse + 0.5);
    const int high = static_cast<int>(
        (static_cast<double>(values[16 + index]) - minimum) * inverse + 0.5);
    block[8 + index] =
        static_cast<std::uint8_t>((low & 15) | ((high & 15) << 4));
    high_bits |= static_cast<std::uint32_t>((low >> 4) & 1) << index;
    high_bits |= static_cast<std::uint32_t>((high >> 4) & 1) << (16 + index);
  }
  put_u32(block + 4, high_bits);
}

void pack_q2_k(const float* values, std::uint8_t* block) {
  std::uint8_t levels[256] = {};
  std::uint8_t scratch[16] = {};
  float weights[16];
  float minimums[16];
  float scales[16];
  float maximum_scale = 0.0f;
  float maximum_minimum = 0.0f;
  for (int group = 0; group < 16; ++group) {
    for (int index = 0; index < 16; ++index) {
      weights[index] = std::fabs(values[16 * group + index]);
    }
    scales[group] = make_affine_quants(
        16, 3, values + 16 * group, weights, levels + 16 * group,
        &minimums[group], scratch, -0.5f, 0.1f, 15, true);
    maximum_scale = std::max(maximum_scale, scales[group]);
    maximum_minimum = std::max(maximum_minimum, minimums[group]);
  }
  const float inverse_scale =
      maximum_scale > 0.0f ? 15.0f / maximum_scale : 0.0f;
  const float inverse_minimum =
      maximum_minimum > 0.0f ? 15.0f / maximum_minimum : 0.0f;
  for (int group = 0; group < 16; ++group) {
    block[group] = static_cast<std::uint8_t>(
        std::clamp(nearest_int(inverse_scale * scales[group]), 0, 15));
    block[group] |= static_cast<std::uint8_t>(
        std::clamp(nearest_int(inverse_minimum * minimums[group]), 0, 15)
        << 4);
  }
  put_half(block + 80, maximum_scale / 15.0f);
  put_half(block + 82, maximum_minimum / 15.0f);
  const float scale_base = half_round(maximum_scale / 15.0f);
  const float minimum_base = half_round(maximum_minimum / 15.0f);
  for (int group = 0; group < 16; ++group) {
    const float scale = scale_base * (block[group] & 15);
    if (scale == 0.0f) continue;
    const float minimum = minimum_base * (block[group] >> 4);
    for (int index = 0; index < 16; ++index) {
      levels[16 * group + index] = static_cast<std::uint8_t>(std::clamp(
          nearest_int((values[16 * group + index] + minimum) / scale),
          0, 3));
    }
  }
  for (int base = 0; base < 256; base += 128) {
    for (int lane = 0; lane < 32; ++lane) {
      block[16 + base / 4 + lane] = static_cast<std::uint8_t>(
          levels[base + lane] | (levels[base + lane + 32] << 2) |
          (levels[base + lane + 64] << 4) |
          (levels[base + lane + 96] << 6));
    }
  }
}

void pack_q3_k(const float* values, std::uint8_t* block) {
  std::int8_t levels[256] = {};
  float scales[16];
  float maximum_scale = 0.0f;
  float maximum_absolute_scale = 0.0f;
  for (int group = 0; group < 16; ++group) {
    scales[group] = make_q3_quants(16, 4, values + 16 * group,
                                    levels + 16 * group);
    const float magnitude = std::fabs(scales[group]);
    if (magnitude > maximum_absolute_scale) {
      maximum_absolute_scale = magnitude;
      maximum_scale = scales[group];
    }
  }
  std::memset(block, 0, 110);
  if (maximum_scale != 0.0f) {
    const float inverse = -32.0f / maximum_scale;
    for (int group = 0; group < 16; ++group) {
      int level = std::clamp(nearest_int(inverse * scales[group]), -32, 31) +
                  32;
      if (group < 8) block[96 + group] = level & 15;
      else block[96 + group - 8] |= static_cast<std::uint8_t>((level & 15) << 4);
      level >>= 4;
      block[104 + group % 4] |=
          static_cast<std::uint8_t>(level << (2 * (group / 4)));
    }
    put_half(block + 108, 1.0f / inverse);
  }
  const float base_scale = half_round(
      maximum_scale != 0.0f ? -maximum_scale / 32.0f : 0.0f);
  for (int group = 0; group < 16; ++group) {
    int scale = group < 8 ? block[96 + group] & 15
                          : block[96 + group - 8] >> 4;
    scale = (scale |
             (((block[104 + group % 4] >> (2 * (group / 4))) & 3) << 4)) -
            32;
    const float local_scale = base_scale * scale;
    if (local_scale == 0.0f) continue;
    for (int index = 0; index < 16; ++index) {
      const int level = std::clamp(
          nearest_int(values[16 * group + index] / local_scale), -4, 3);
      levels[16 * group + index] = static_cast<std::int8_t>(level + 4);
    }
  }
  int high_index = 0;
  std::uint8_t high_mask = 1;
  for (int index = 0; index < 256; ++index) {
    if (levels[index] > 3) {
      block[high_index] |= high_mask;
      levels[index] -= 4;
    }
    if (++high_index == 32) {
      high_index = 0;
      high_mask <<= 1;
    }
  }
  for (int base = 0; base < 256; base += 128) {
    for (int lane = 0; lane < 32; ++lane) {
      block[32 + base / 4 + lane] = static_cast<std::uint8_t>(
          levels[base + lane] | (levels[base + lane + 32] << 2) |
          (levels[base + lane + 64] << 4) |
          (levels[base + lane + 96] << 6));
    }
  }
}

void pack_affine_k(const float* values, std::uint8_t* block,
                   bool five_bit) {
  const int maximum = five_bit ? 31 : 15;
  std::uint8_t levels[256] = {};
  std::uint8_t scratch[32] = {};
  float weights[32];
  float minimums[8];
  float scales[8];
  float maximum_scale = 0.0f;
  float maximum_minimum = 0.0f;
  for (int group = 0; group < 8; ++group) {
    float square_sum = 0.0f;
    for (int index = 0; index < 32; ++index) {
      const float value = values[32 * group + index];
      square_sum += value * value;
    }
    const float average = std::sqrt(square_sum / 32.0f);
    for (int index = 0; index < 32; ++index) {
      weights[index] = average + std::fabs(values[32 * group + index]);
    }
    scales[group] = make_affine_quants(
        32, maximum, values + 32 * group, weights, levels + 32 * group,
        &minimums[group], scratch, five_bit ? -0.5f : -1.0f, 0.1f,
        five_bit ? 15 : 20, false);
    maximum_scale = std::max(maximum_scale, scales[group]);
    maximum_minimum = std::max(maximum_minimum, minimums[group]);
  }
  std::memset(block, 0, five_bit ? 176 : 144);
  const float inverse_scale =
      maximum_scale > 0.0f ? 63.0f / maximum_scale : 0.0f;
  const float inverse_minimum =
      maximum_minimum > 0.0f ? 63.0f / maximum_minimum : 0.0f;
  std::uint8_t* packed_scales = block + 4;
  for (int group = 0; group < 8; ++group) {
    const int scale = std::clamp(
        nearest_int(inverse_scale * scales[group]), 0, 63);
    const int minimum = std::clamp(
        nearest_int(inverse_minimum * minimums[group]), 0, 63);
    if (group < 4) {
      packed_scales[group] = static_cast<std::uint8_t>(scale);
      packed_scales[group + 4] = static_cast<std::uint8_t>(minimum);
    } else {
      packed_scales[group + 4] = static_cast<std::uint8_t>(
          (scale & 15) | ((minimum & 15) << 4));
      packed_scales[group - 4] |= static_cast<std::uint8_t>((scale >> 4) << 6);
      packed_scales[group] |= static_cast<std::uint8_t>((minimum >> 4) << 6);
    }
  }
  put_half(block, maximum_scale / 63.0f);
  put_half(block + 2, maximum_minimum / 63.0f);
  const float scale_base = half_round(maximum_scale / 63.0f);
  const float minimum_base = half_round(maximum_minimum / 63.0f);
  for (int group = 0; group < 8; ++group) {
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    scale_min_k4(group, packed_scales, &scale, &minimum);
    const float local_scale = scale_base * scale;
    if (local_scale == 0.0f) continue;
    const float local_minimum = minimum_base * minimum;
    for (int index = 0; index < 32; ++index) {
      levels[32 * group + index] = static_cast<std::uint8_t>(std::clamp(
          nearest_int((values[32 * group + index] + local_minimum) /
                      local_scale),
          0, maximum));
    }
  }
  if (!five_bit) {
    std::uint8_t* quants = block + 16;
    for (int base = 0; base < 256; base += 64) {
      for (int lane = 0; lane < 32; ++lane) {
        *quants++ = static_cast<std::uint8_t>(
            levels[base + lane] | (levels[base + lane + 32] << 4));
      }
    }
    return;
  }
  std::uint8_t* high = block + 16;
  std::uint8_t* low = block + 48;
  std::uint8_t mask1 = 1;
  std::uint8_t mask2 = 2;
  for (int base = 0; base < 256; base += 64) {
    for (int lane = 0; lane < 32; ++lane) {
      int first = levels[base + lane];
      int second = levels[base + lane + 32];
      if (first > 15) {
        first -= 16;
        high[lane] |= mask1;
      }
      if (second > 15) {
        second -= 16;
        high[lane] |= mask2;
      }
      *low++ = static_cast<std::uint8_t>(first | (second << 4));
    }
    mask1 <<= 2;
    mask2 <<= 2;
  }
}

void pack_q6_k(const float* values, std::uint8_t* block) {
  std::int8_t levels[256] = {};
  float scales[16];
  float maximum_scale = 0.0f;
  float maximum_absolute_scale = 0.0f;
  for (int group = 0; group < 16; ++group) {
    scales[group] =
        make_qx_quants(16, 32, values + 16 * group, levels + 16 * group);
    const float magnitude = std::fabs(scales[group]);
    if (magnitude > maximum_absolute_scale) {
      maximum_absolute_scale = magnitude;
      maximum_scale = scales[group];
    }
  }
  std::memset(block, 0, 210);
  if (maximum_absolute_scale < kGroupEpsilon) return;
  const float inverse = -128.0f / maximum_scale;
  put_half(block + 208, 1.0f / inverse);
  const float base_scale = half_round(1.0f / inverse);
  for (int group = 0; group < 16; ++group) {
    const int scale = std::min(127, nearest_int(inverse * scales[group]));
    block[192 + group] = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(scale));
    const float local_scale =
        base_scale * static_cast<std::int8_t>(block[192 + group]);
    if (local_scale == 0.0f) continue;
    for (int index = 0; index < 16; ++index) {
      const int level = std::clamp(
          nearest_int(values[16 * group + index] / local_scale), -32, 31);
      levels[16 * group + index] = static_cast<std::int8_t>(level + 32);
    }
  }
  std::uint8_t* low = block;
  std::uint8_t* high = block + 128;
  for (int base = 0; base < 256; base += 128) {
    for (int lane = 0; lane < 32; ++lane) {
      const std::uint8_t first = levels[base + lane];
      const std::uint8_t second = levels[base + lane + 32];
      const std::uint8_t third = levels[base + lane + 64];
      const std::uint8_t fourth = levels[base + lane + 96];
      low[lane] = static_cast<std::uint8_t>((first & 15) | ((third & 15) << 4));
      low[lane + 32] =
          static_cast<std::uint8_t>((second & 15) | ((fourth & 15) << 4));
      high[lane] = static_cast<std::uint8_t>(
          (first >> 4) | ((second >> 4) << 2) | ((third >> 4) << 4) |
          ((fourth >> 4) << 6));
    }
    low += 64;
    high += 32;
  }
}

void pack_iq4(const float* values, std::uint8_t* block, bool superblock) {
  const int count = superblock ? 256 : 32;
  const int groups = count / 32;
  std::uint8_t levels[256] = {};
  float scales[8] = {};
  float weights[32];
  float maximum_scale = 0.0f;
  float maximum_absolute_scale = 0.0f;
  for (int group = 0; group < groups; ++group) {
    const float* source = values + 32 * group;
    for (int index = 0; index < 32; ++index) {
      weights[index] = source[index] * source[index];
    }
    float signed_maximum = 0.0f;
    float absolute_maximum = 0.0f;
    for (int index = 0; index < 32; ++index) {
      if (std::fabs(source[index]) > absolute_maximum) {
        absolute_maximum = std::fabs(source[index]);
        signed_maximum = source[index];
      }
    }
    if (absolute_maximum < kGroupEpsilon) continue;
    float scale = (superblock ? -signed_maximum : signed_maximum) /
                  kIq4Values[0];
    float inverse = 1.0f / scale;
    float sum_qx = 0.0f;
    float sum_q2 = 0.0f;
    for (int index = 0; index < 32; ++index) {
      const int level = best_iq4(inverse * source[index]);
      levels[32 * group + index] = static_cast<std::uint8_t>(level);
      const float quant = kIq4Values[level];
      sum_qx += weights[index] * quant * source[index];
      sum_q2 += weights[index] * quant * quant;
    }
    scale = sum_q2 > 0.0f ? sum_qx / sum_q2 : 0.0f;
    float best = scale * sum_qx;
    if (superblock) {
      for (int attempt = -7; attempt <= 7; ++attempt) {
        inverse = (attempt + kIq4Values[0]) / signed_maximum;
        sum_qx = 0.0f;
        sum_q2 = 0.0f;
        for (int index = 0; index < 32; ++index) {
          const int level = best_iq4(inverse * source[index]);
          const float quant = kIq4Values[level];
          sum_qx += weights[index] * quant * source[index];
          sum_q2 += weights[index] * quant * quant;
        }
        if (sum_q2 > 0.0f && sum_qx * sum_qx > best * sum_q2) {
          scale = sum_qx / sum_q2;
          best = scale * sum_qx;
        }
      }
    }
    scales[group] = scale;
    const float magnitude = std::fabs(scale);
    if (magnitude > maximum_absolute_scale) {
      maximum_absolute_scale = magnitude;
      maximum_scale = scale;
    }
  }
  std::memset(block, 0, superblock ? 136 : 18);
  std::uint8_t* quants = block + (superblock ? 8 : 2);
  if (superblock) {
    const float base_scale = -maximum_scale / 32.0f;
    put_half(block, base_scale);
    const float rounded_base = half_round(base_scale);
    std::uint16_t high_scales = 0;
    for (int group = 0; group < 8; ++group) {
      int local = rounded_base != 0.0f
          ? nearest_int(scales[group] / rounded_base)
          : 0;
      local = std::clamp(local, -32, 31);
      const float local_scale = rounded_base * local;
      const float inverse = local_scale != 0.0f ? 1.0f / local_scale : 0.0f;
      for (int index = 0; index < 32; ++index) {
        levels[32 * group + index] = static_cast<std::uint8_t>(
            best_iq4(inverse * values[32 * group + index]));
      }
      const int encoded = local + 32;
      block[4 + group / 2] |= static_cast<std::uint8_t>(
          (encoded & 15) << (4 * (group & 1)));
      high_scales |= static_cast<std::uint16_t>((encoded >> 4) << (2 * group));
    }
    std::memcpy(block + 2, &high_scales, sizeof(high_scales));
  } else {
    put_half(block, scales[0]);
  }
  for (int group = 0; group < groups; ++group) {
    for (int lane = 0; lane < 16; ++lane) {
      quants[16 * group + lane] = static_cast<std::uint8_t>(
          levels[32 * group + lane] |
          (levels[32 * group + lane + 16] << 4));
    }
  }
}

void pack_mxfp4(const float* values, std::uint8_t* block) {
  float maximum = 0.0f;
  for (int index = 0; index < 32; ++index) {
    maximum = std::max(maximum, std::fabs(values[index]));
  }
  const int raw_exponent = maximum > 0.0f
      ? static_cast<int>(std::floor(std::log2(maximum))) - 2 + 127
      : 0;
  const std::uint8_t exponent =
      static_cast<std::uint8_t>(std::clamp(raw_exponent, 0, 255));
  const float half_scale = e8m0_half(exponent);
  block[0] = exponent;
  for (int index = 0; index < 16; ++index) {
    block[1 + index] = static_cast<std::uint8_t>(
        best_fp4(values[index], half_scale) |
        (best_fp4(values[16 + index], half_scale) << 4));
  }
}

void pack_nvfp4(const float* values, std::uint8_t* block) {
  for (int subblock = 0; subblock < 4; ++subblock) {
    const float* source = values + 16 * subblock;
    float maximum = 0.0f;
    for (int index = 0; index < 16; ++index) {
      maximum = std::max(maximum, std::fabs(source[index]));
    }
    const std::uint8_t scale = fp32_to_ue4m3(maximum / 6.0f);
    const float half_scale = ue4m3_half(scale);
    block[subblock] = scale;
    for (int index = 0; index < 8; ++index) {
      block[4 + subblock * 8 + index] = static_cast<std::uint8_t>(
          best_fp4(source[index], half_scale) |
          (best_fp4(source[8 + index], half_scale) << 4));
    }
  }
}

void pack_tq1_0(const float* values, std::uint8_t* block) {
  float maximum = 0.0f;
  for (int index = 0; index < 256; ++index) {
    maximum = std::max(maximum, std::fabs(values[index]));
  }
  put_half(block + 52, maximum);
  for (int lane = 0; lane < 32; ++lane) {
    int encoded = 0;
    for (int trit = 0; trit < 5; ++trit) {
      encoded = encoded * 3 +
          static_cast<int>(std::lround(
              maximum != 0.0f ? values[lane + trit * 32] / maximum : 0.0f)) +
          1;
    }
    block[lane] = static_cast<std::uint8_t>((encoded * 256 + 242) / 243);
  }
  for (int lane = 0; lane < 16; ++lane) {
    int encoded = 0;
    for (int trit = 0; trit < 5; ++trit) {
      encoded = encoded * 3 + static_cast<int>(
          std::lround(maximum != 0.0f
                          ? values[160 + lane + trit * 16] / maximum
                          : 0.0f)) + 1;
    }
    block[32 + lane] =
        static_cast<std::uint8_t>((encoded * 256 + 242) / 243);
  }
  for (int lane = 0; lane < 4; ++lane) {
    int encoded = 0;
    for (int trit = 0; trit < 4; ++trit) {
      encoded = encoded * 3 + static_cast<int>(
          std::lround(maximum != 0.0f
                          ? values[240 + lane + trit * 4] / maximum
                          : 0.0f)) + 1;
    }
    encoded *= 3;
    block[48 + lane] =
        static_cast<std::uint8_t>((encoded * 256 + 242) / 243);
  }
}

void pack_block(QuantFormat format, const float* source,
                const float* importance, std::uint8_t* destination) {
  switch (format) {
    case QuantFormat::kQ1_0: pack_q1_0(source, destination); break;
    case QuantFormat::kQ2_0: pack_q2_0(source, destination); break;
    case QuantFormat::kQ4_1: pack_q4_1(source, destination); break;
    case QuantFormat::kQ5_0: pack_q5_0(source, destination); break;
    case QuantFormat::kQ5_1: pack_q5_1(source, destination); break;
    case QuantFormat::kQ2_K: pack_q2_k(source, destination); break;
    case QuantFormat::kQ3_K: pack_q3_k(source, destination); break;
    case QuantFormat::kQ4_K: pack_affine_k(source, destination, false); break;
    case QuantFormat::kQ5_K: pack_affine_k(source, destination, true); break;
    case QuantFormat::kQ6_K: pack_q6_k(source, destination); break;
    case QuantFormat::kIQ4_NL: pack_iq4(source, destination, false); break;
    case QuantFormat::kIQ4_XS: pack_iq4(source, destination, true); break;
    case QuantFormat::kIQ2_XXS:
    case QuantFormat::kIQ2_XS:
    case QuantFormat::kIQ3_XXS:
    case QuantFormat::kIQ3_S:
    case QuantFormat::kIQ2_S:
    case QuantFormat::kIQ1_S:
    case QuantFormat::kIQ1_M:
      iq_pack_block_ref(format, source, importance, destination);
      break;
    case QuantFormat::kMXFP4: pack_mxfp4(source, destination); break;
    case QuantFormat::kNVFP4: pack_nvfp4(source, destination); break;
    case QuantFormat::kTQ1_0: pack_tq1_0(source, destination); break;
    default: break;
  }
}

}  // namespace

bool gguf_pack_supported(QuantFormat format) {
  switch (format) {
    case QuantFormat::kQ1_0:
    case QuantFormat::kQ2_0:
    case QuantFormat::kQ4_1:
    case QuantFormat::kQ5_0:
    case QuantFormat::kQ5_1:
    case QuantFormat::kQ2_K:
    case QuantFormat::kQ3_K:
    case QuantFormat::kQ4_K:
    case QuantFormat::kQ5_K:
    case QuantFormat::kQ6_K:
    case QuantFormat::kIQ4_NL:
    case QuantFormat::kIQ4_XS:
    case QuantFormat::kIQ2_XXS:
    case QuantFormat::kIQ2_XS:
    case QuantFormat::kIQ3_XXS:
    case QuantFormat::kIQ3_S:
    case QuantFormat::kIQ2_S:
    case QuantFormat::kIQ1_S:
    case QuantFormat::kIQ1_M:
    case QuantFormat::kMXFP4:
    case QuantFormat::kNVFP4:
    case QuantFormat::kTQ1_0:
      return true;
    default:
      return false;
  }
}

bool gguf_pack_ref(QuantFormat format, const float* weights, long long n,
                   long long k, void* packed, const float* importance) {
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!gguf_pack_supported(format) ||
      !gguf_format_info(format, &block_size, &block_bytes)) {
    return false;
  }
  const long long blocks_per_row = k / block_size;
  const long long total_blocks = n * blocks_per_row;
  auto* bytes = static_cast<std::uint8_t*>(packed);
  if (!finite_block(weights, n * k)) return false;
  // Conversion-time work is independent per canonical block. Keep small
  // matrices inline and distribute sufficiently large authoring jobs through
  // the shared pool without changing their byte order.
  threading::parallel_ranges(total_blocks, 4,
      [&](long long begin, long long end, int) {
    for (long long flat = begin; flat < end; ++flat) {
      const float* source = weights + flat * block_size;
      const float* block_importance =
          importance == nullptr ? nullptr : importance + flat * block_size;
      pack_block(format, source, block_importance,
                 bytes + flat * block_bytes);
    }
  });
  return true;
}

}  // namespace quixicore_cpu::quant
