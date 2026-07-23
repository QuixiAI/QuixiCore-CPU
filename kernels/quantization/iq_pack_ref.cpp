#include "kernels/quantization/iq_pack_ref.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "kernels/common/fp16.h"
#include "kernels/quantization/iq_tables.h"

namespace quixicore_cpu::quant {
namespace {

constexpr float kDelta = 0.125f;

void put_half(std::uint8_t* destination, float value) {
  const std::uint16_t bits = fp32_to_fp16(value);
  std::memcpy(destination, &bits, sizeof(bits));
}

void put_u16(std::uint8_t* destination, std::uint16_t value) {
  std::memcpy(destination, &value, sizeof(value));
}

void put_u32(std::uint8_t* destination, std::uint32_t value) {
  std::memcpy(destination, &value, sizeof(value));
}

template <typename Word>
int code_value(Word word, int element) {
  return static_cast<int>((word >> (8 * element)) & Word{255});
}

int iq1_value(std::uint64_t word, int element) {
  return static_cast<std::int8_t>((word >> (8 * element)) & 255);
}

void block_weights(const float* values, const float* importance,
                   float sigma_factor, std::array<float, 256>* weights) {
  float sum_squared = 0.0f;
  for (int i = 0; i < 256; ++i) sum_squared += values[i] * values[i];
  const float sigma2 = sigma_factor * sum_squared / 256.0f;
  for (int i = 0; i < 256; ++i) {
    const float external = importance == nullptr ? 1.0f : importance[i];
    (*weights)[i] = external * std::sqrt(sigma2 + values[i] * values[i]);
  }
}

std::uint8_t restricted_sign_code(const float* values, const float* weights) {
  std::uint8_t mask = 0;
  for (int i = 0; i < 8; ++i) {
    if (values[i] < 0.0f) mask |= static_cast<std::uint8_t>(1u << i);
  }
  if ((std::popcount(mask) & 1) != 0) {
    int least = 0;
    float cost = weights[0] * values[0] * values[0];
    for (int i = 1; i < 8; ++i) {
      const float candidate = weights[i] * values[i] * values[i];
      if (candidate < cost) {
        cost = candidate;
        least = i;
      }
    }
    mask ^= static_cast<std::uint8_t>(1u << least);
  }
  return mask & 127;
}

template <typename Word>
int best_grid(const Word* table, int table_size, int dimensions,
              const float* values, const float* weights, float scale,
              std::uint8_t signs, int sign_offset = 0) {
  int best_index = 0;
  double best_error = std::numeric_limits<double>::infinity();
  for (int grid = 0; grid < table_size; ++grid) {
    double error = 0.0;
    for (int i = 0; i < dimensions; ++i) {
      const float sign = (signs & (1u << (i + sign_offset))) != 0
                             ? -1.0f
                             : 1.0f;
      const float reconstructed =
          scale * static_cast<float>(code_value(table[grid], i)) * sign;
      const float difference = values[i] - reconstructed;
      error += static_cast<double>(weights[i]) * difference * difference;
    }
    if (error < best_error) {
      best_error = error;
      best_index = grid;
    }
  }
  return best_index;
}

template <typename Word>
float fit_grid_region(const Word* table, int table_size, int dimensions,
                      const float* values, const float* weights, int groups,
                      bool restricted_signs, int smallest_code,
                      int largest_code, int* indices,
                      std::uint8_t* sign_codes) {
  float maximum = 0.0f;
  for (int i = 0; i < groups * dimensions; ++i) {
    maximum = std::max(maximum, std::fabs(values[i]));
  }
  const float seeds[] = {
      maximum / static_cast<float>(smallest_code),
      maximum / static_cast<float>(largest_code),
      maximum / std::sqrt(static_cast<float>(smallest_code * largest_code)),
      2.0f * maximum / static_cast<float>(smallest_code + largest_code),
  };
  float best_scale = 0.0f;
  double best_error = std::numeric_limits<double>::infinity();
  std::array<int, 8> trial_indices{};
  std::array<std::uint8_t, 8> trial_signs{};
  for (float seed : seeds) {
    float scale = seed;
    for (int iteration = 0; iteration < 6; ++iteration) {
      double numerator = 0.0;
      double denominator = 0.0;
      for (int group = 0; group < groups; ++group) {
        const float* source = values + group * dimensions;
        const float* local_weights = weights + group * dimensions;
        std::uint8_t raw_signs = 0;
        for (int i = 0; i < dimensions; ++i) {
          if (source[i] < 0.0f) {
            raw_signs |= static_cast<std::uint8_t>(1u << i);
          }
        }
        const std::uint8_t sign_code =
            restricted_signs ? restricted_sign_code(source, local_weights)
                             : raw_signs;
        const std::uint8_t signs = restricted_signs
            ? iq_tables::ksigns_iq2xs[sign_code]
            : sign_code;
        const int index = best_grid(table, table_size, dimensions, source,
                                    local_weights, scale, signs);
        trial_indices[group] = index;
        trial_signs[group] = sign_code;
        for (int i = 0; i < dimensions; ++i) {
          const float sign = (signs & (1u << i)) != 0 ? -1.0f : 1.0f;
          const float q =
              static_cast<float>(code_value(table[index], i)) * sign;
          numerator +=
              static_cast<double>(local_weights[i]) * source[i] * q;
          denominator +=
              static_cast<double>(local_weights[i]) * q * q;
        }
      }
      scale = denominator > 0.0
                  ? std::max(0.0f,
                             static_cast<float>(numerator / denominator))
                  : 0.0f;
    }
    double error = 0.0;
    for (int group = 0; group < groups; ++group) {
      const std::uint8_t signs = restricted_signs
          ? iq_tables::ksigns_iq2xs[trial_signs[group]]
          : trial_signs[group];
      for (int i = 0; i < dimensions; ++i) {
        const float sign = (signs & (1u << i)) != 0 ? -1.0f : 1.0f;
        const float q = static_cast<float>(
                            code_value(table[trial_indices[group]], i)) * sign;
        const float difference =
            values[group * dimensions + i] - scale * q;
        error += static_cast<double>(weights[group * dimensions + i]) *
                 difference * difference;
      }
    }
    if (error < best_error) {
      best_error = error;
      best_scale = scale;
      std::copy_n(trial_indices.data(), groups, indices);
      std::copy_n(trial_signs.data(), groups, sign_codes);
    }
  }
  return best_scale;
}

template <typename Word>
void choose_grid_region(const Word* table, int table_size, int dimensions,
                        const float* values, const float* weights, int groups,
                        bool restricted_signs, float scale, int* indices,
                        std::uint8_t* sign_codes) {
  for (int group = 0; group < groups; ++group) {
    const float* source = values + group * dimensions;
    const float* local_weights = weights + group * dimensions;
    std::uint8_t raw_signs = 0;
    for (int i = 0; i < dimensions; ++i) {
      if (source[i] < 0.0f) raw_signs |= static_cast<std::uint8_t>(1u << i);
    }
    const std::uint8_t sign_code =
        restricted_signs ? restricted_sign_code(source, local_weights)
                         : raw_signs;
    const std::uint8_t signs = restricted_signs
        ? iq_tables::ksigns_iq2xs[sign_code]
        : sign_code;
    indices[group] = best_grid(table, table_size, dimensions, source,
                               local_weights, scale, signs);
    sign_codes[group] = sign_code;
  }
}

float fit_iq3_xxs_region(const float* values, const float* weights,
                         int* indices, std::uint8_t* sign_codes) {
  float maximum = 0.0f;
  for (int i = 0; i < 32; ++i) maximum = std::max(maximum, std::fabs(values[i]));
  const float seeds[] = {maximum / 4.0f, maximum / 62.0f,
                         maximum / std::sqrt(248.0f), maximum / 33.0f};
  float best_scale = 0.0f;
  double best_error = std::numeric_limits<double>::infinity();
  std::array<int, 8> trial_indices{};
  std::array<std::uint8_t, 4> trial_signs{};
  for (float seed : seeds) {
    float scale = seed;
    for (int iteration = 0; iteration < 6; ++iteration) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int group8 = 0; group8 < 4; ++group8) {
      const float* source = values + 8 * group8;
      const float* local_weights = weights + 8 * group8;
      const std::uint8_t sign_code =
          restricted_sign_code(source, local_weights);
      const std::uint8_t signs = iq_tables::ksigns_iq2xs[sign_code];
      trial_signs[group8] = sign_code;
      for (int half = 0; half < 2; ++half) {
        const int index = best_grid(iq_tables::iq3xxs_grid, 256, 4,
                                    source + 4 * half,
                                    local_weights + 4 * half, scale, signs,
                                    4 * half);
        trial_indices[2 * group8 + half] = index;
        for (int i = 0; i < 4; ++i) {
          const float sign = (signs & (1u << (i + 4 * half))) != 0
                                 ? -1.0f
                                 : 1.0f;
          const float q = static_cast<float>(code_value(
                              iq_tables::iq3xxs_grid[index], i)) * sign;
          numerator += static_cast<double>(local_weights[4 * half + i]) *
                       source[4 * half + i] * q;
          denominator += static_cast<double>(local_weights[4 * half + i]) *
                         q * q;
        }
      }
    }
      scale = denominator > 0.0
                  ? std::max(0.0f,
                             static_cast<float>(numerator / denominator))
                  : 0.0f;
    }
    double error = 0.0;
    for (int group8 = 0; group8 < 4; ++group8) {
      const std::uint8_t actual_signs =
          iq_tables::ksigns_iq2xs[trial_signs[group8]];
      for (int half = 0; half < 2; ++half) {
        const int index = trial_indices[2 * group8 + half];
        for (int i = 0; i < 4; ++i) {
          const int element = 8 * group8 + 4 * half + i;
          const float sign =
              (actual_signs & (1u << (4 * half + i))) != 0 ? -1.0f : 1.0f;
          const float q = static_cast<float>(code_value(
                              iq_tables::iq3xxs_grid[index], i)) * sign;
          const float difference = values[element] - scale * q;
          error += static_cast<double>(weights[element]) * difference *
                   difference;
        }
      }
    }
    if (error < best_error) {
      best_error = error;
      best_scale = scale;
      std::copy(trial_indices.begin(), trial_indices.end(), indices);
      std::copy(trial_signs.begin(), trial_signs.end(), sign_codes);
    }
  }
  return best_scale;
}

void choose_iq3_xxs_region(const float* values, const float* weights,
                           float scale, int* indices,
                           std::uint8_t* sign_codes) {
  for (int group8 = 0; group8 < 4; ++group8) {
    const float* source = values + 8 * group8;
    const float* local_weights = weights + 8 * group8;
    const std::uint8_t sign_code = restricted_sign_code(source, local_weights);
    const std::uint8_t signs = iq_tables::ksigns_iq2xs[sign_code];
    sign_codes[group8] = sign_code;
    for (int half = 0; half < 2; ++half) {
      indices[2 * group8 + half] = best_grid(
          iq_tables::iq3xxs_grid, 256, 4, source + 4 * half,
          local_weights + 4 * half, scale, signs, 4 * half);
    }
  }
}

float iq1_error(const float* values, const float* weights, int groups,
                float scale, float delta, int* indices) {
  double total = 0.0;
  for (int group = 0; group < groups; ++group) {
    int best_index = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (int grid = 0; grid < 2048; ++grid) {
      double error = 0.0;
      for (int i = 0; i < 8; ++i) {
        const float q = static_cast<float>(iq1_value(
                            iq_tables::iq1s_grid[grid], i)) + delta;
        const float difference = values[8 * group + i] - scale * q;
        error += static_cast<double>(weights[8 * group + i]) * difference *
                 difference;
      }
      if (error < best_error) {
        best_error = error;
        best_index = grid;
      }
    }
    indices[group] = best_index;
    total += best_error;
  }
  return static_cast<float>(total);
}

float fit_iq1_region(const float* values, const float* weights, int groups,
                     bool common_delta, int* indices,
                     std::int8_t* delta_signs) {
  float maximum = 0.0f;
  for (int i = 0; i < 8 * groups; ++i) {
    maximum = std::max(maximum, std::fabs(values[i]));
  }
  float best_scale = 0.0f;
  float best_error = std::numeric_limits<float>::infinity();
  const int combinations = common_delta ? 2 : (1 << groups);
  for (int combination = 0; combination < combinations; ++combination) {
    float scale = maximum;
    std::array<int, 4> trial_indices{};
    for (int iteration = 0; iteration < 6; ++iteration) {
      double numerator = 0.0;
      double denominator = 0.0;
      for (int group = 0; group < groups; ++group) {
        const bool negative = common_delta ? combination != 0
                                           : ((combination >> group) & 1) != 0;
        const float delta = negative ? -kDelta : kDelta;
        iq1_error(values + 8 * group, weights + 8 * group, 1, scale, delta,
                  &trial_indices[group]);
        for (int i = 0; i < 8; ++i) {
          const float q = static_cast<float>(iq1_value(
                              iq_tables::iq1s_grid[trial_indices[group]], i)) +
                          delta;
          numerator += static_cast<double>(weights[8 * group + i]) *
                       values[8 * group + i] * q;
          denominator += static_cast<double>(weights[8 * group + i]) * q * q;
        }
      }
      scale = denominator > 0.0
                  ? std::max(0.0f,
                             static_cast<float>(numerator / denominator))
                  : 0.0f;
    }
    float error = 0.0f;
    for (int group = 0; group < groups; ++group) {
      const bool negative = common_delta ? combination != 0
                                         : ((combination >> group) & 1) != 0;
      error += iq1_error(values + 8 * group, weights + 8 * group, 1, scale,
                         negative ? -kDelta : kDelta,
                         &trial_indices[group]);
    }
    if (error < best_error) {
      best_error = error;
      best_scale = scale;
      for (int group = 0; group < groups; ++group) {
        indices[group] = trial_indices[group];
        delta_signs[group] = static_cast<std::int8_t>(
            common_delta ? combination != 0
                         : ((combination >> group) & 1) != 0);
      }
    }
  }
  return best_scale;
}

void choose_iq1_region(const float* values, const float* weights, int groups,
                       float scale, const std::int8_t* delta_signs,
                       int* indices) {
  for (int group = 0; group < groups; ++group) {
    iq1_error(values + 8 * group, weights + 8 * group, 1, scale,
              delta_signs[group] ? -kDelta : kDelta, &indices[group]);
  }
}

int encode_local(float scale, float base, float step, int maximum) {
  if (base == 0.0f) return 0;
  return std::clamp(static_cast<int>(std::lround(scale / base / step - 0.5f)),
                    0, maximum);
}

void pack_iq2_xxs(const float* values, const float* importance,
                  std::uint8_t* block) {
  std::memset(block, 0, 66);
  std::array<float, 256> weights{};
  block_weights(values, importance, 1.0f, &weights);
  std::array<float, 8> scales{};
  std::array<int, 32> indices{};
  std::array<std::uint8_t, 32> signs{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 8; ++region) {
    scales[region] = fit_grid_region(
        iq_tables::iq2xxs_grid, 256, 8, values + 32 * region,
        weights.data() + 32 * region, 4, true, 8, 43,
        indices.data() + 4 * region, signs.data() + 4 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const float raw_base = maximum_scale / 3.875f;
  put_half(block, raw_base);
  const float base = fp16_to_fp32(fp32_to_fp16(raw_base));
  for (int region = 0; region < 8; ++region) {
    const int local = encode_local(scales[region], base, 0.25f, 15);
    const float scale = base * (0.5f + local) * 0.25f;
    choose_grid_region(iq_tables::iq2xxs_grid, 256, 8,
                       values + 32 * region, weights.data() + 32 * region, 4,
                       true, scale, indices.data() + 4 * region,
                       signs.data() + 4 * region);
    std::uint32_t grid_bits = 0;
    std::uint32_t sign_scale = static_cast<std::uint32_t>(local) << 28;
    for (int group = 0; group < 4; ++group) {
      grid_bits |= static_cast<std::uint32_t>(indices[4 * region + group])
                   << (8 * group);
      sign_scale |= static_cast<std::uint32_t>(signs[4 * region + group])
                    << (7 * group);
    }
    put_u32(block + 2 + 8 * region, grid_bits);
    put_u32(block + 6 + 8 * region, sign_scale);
  }
}

void pack_iq2_xs(const float* values, const float* importance,
                 std::uint8_t* block) {
  std::memset(block, 0, 74);
  std::array<float, 256> weights{};
  block_weights(values, importance, 1.0f, &weights);
  std::array<float, 16> scales{};
  std::array<int, 32> indices{};
  std::array<std::uint8_t, 32> signs{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 16; ++region) {
    scales[region] = fit_grid_region(
        iq_tables::iq2xs_grid, 512, 8, values + 16 * region,
        weights.data() + 16 * region, 2, true, 8, 43,
        indices.data() + 2 * region, signs.data() + 2 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const float raw_base = maximum_scale / 3.875f;
  put_half(block, raw_base);
  const float base = fp16_to_fp32(fp32_to_fp16(raw_base));
  for (int region = 0; region < 16; ++region) {
    const int local = encode_local(scales[region], base, 0.25f, 15);
    const float scale = base * (0.5f + local) * 0.25f;
    choose_grid_region(iq_tables::iq2xs_grid, 512, 8,
                       values + 16 * region, weights.data() + 16 * region, 2,
                       true, scale, indices.data() + 2 * region,
                       signs.data() + 2 * region);
    for (int group = 0; group < 2; ++group) {
      const std::uint16_t code = static_cast<std::uint16_t>(
          indices[2 * region + group] | (signs[2 * region + group] << 9));
      put_u16(block + 2 + 2 * (2 * region + group), code);
    }
    block[66 + region / 2] |=
        static_cast<std::uint8_t>(local << (4 * (region & 1)));
  }
}

void pack_iq2_s(const float* values, const float* importance,
                std::uint8_t* block) {
  std::memset(block, 0, 82);
  std::array<float, 256> weights{};
  block_weights(values, importance, 1.0f, &weights);
  std::array<float, 16> scales{};
  std::array<int, 32> indices{};
  std::array<std::uint8_t, 32> signs{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 16; ++region) {
    scales[region] = fit_grid_region(
        iq_tables::iq2s_grid, 1024, 8, values + 16 * region,
        weights.data() + 16 * region, 2, false, 8, 43,
        indices.data() + 2 * region, signs.data() + 2 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const float raw_base = maximum_scale / 3.875f;
  put_half(block, raw_base);
  const float base = fp16_to_fp32(fp32_to_fp16(raw_base));
  for (int region = 0; region < 16; ++region) {
    const int local = encode_local(scales[region], base, 0.25f, 15);
    const float scale = base * (0.5f + local) * 0.25f;
    choose_grid_region(iq_tables::iq2s_grid, 1024, 8,
                       values + 16 * region, weights.data() + 16 * region, 2,
                       false, scale, indices.data() + 2 * region,
                       signs.data() + 2 * region);
    const int block32 = region / 2;
    const int half = region & 1;
    for (int group = 0; group < 2; ++group) {
      const int group8 = 2 * half + group;
      const int code = indices[2 * region + group];
      block[2 + 4 * block32 + group8] = static_cast<std::uint8_t>(code);
      block[34 + 4 * block32 + group8] = signs[2 * region + group];
      block[66 + block32] |=
          static_cast<std::uint8_t>((code >> 8) << (2 * group8));
    }
    block[74 + block32] |=
        static_cast<std::uint8_t>(local << (4 * half));
  }
}

void pack_iq3_xxs(const float* values, const float* importance,
                  std::uint8_t* block) {
  std::memset(block, 0, 98);
  std::array<float, 256> weights{};
  block_weights(values, importance, 1.0f, &weights);
  std::array<float, 8> scales{};
  std::array<int, 64> indices{};
  std::array<std::uint8_t, 32> signs{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 8; ++region) {
    scales[region] = fit_iq3_xxs_region(
        values + 32 * region, weights.data() + 32 * region,
        indices.data() + 8 * region, signs.data() + 4 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const float raw_base = maximum_scale / 7.75f;
  put_half(block, raw_base);
  const float base = fp16_to_fp32(fp32_to_fp16(raw_base));
  for (int region = 0; region < 8; ++region) {
    const int local = encode_local(scales[region], base, 0.5f, 15);
    const float scale = base * (0.5f + local) * 0.5f;
    choose_iq3_xxs_region(values + 32 * region,
                          weights.data() + 32 * region, scale,
                          indices.data() + 8 * region,
                          signs.data() + 4 * region);
    for (int i = 0; i < 8; ++i) {
      block[2 + 8 * region + i] =
          static_cast<std::uint8_t>(indices[8 * region + i]);
    }
    std::uint32_t sign_scale = static_cast<std::uint32_t>(local) << 28;
    for (int group = 0; group < 4; ++group) {
      sign_scale |= static_cast<std::uint32_t>(signs[4 * region + group])
                    << (7 * group);
    }
    put_u32(block + 66 + 4 * region, sign_scale);
  }
}

void pack_iq3_s(const float* values, const float* importance,
                std::uint8_t* block) {
  std::memset(block, 0, 110);
  std::array<float, 256> weights{};
  block_weights(values, importance, 1.0f, &weights);
  std::array<float, 8> scales{};
  std::array<int, 64> indices{};
  std::array<std::uint8_t, 64> signs{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 8; ++region) {
    scales[region] = fit_grid_region(
        iq_tables::iq3s_grid, 512, 4, values + 32 * region,
        weights.data() + 32 * region, 8, false, 1, 15,
        indices.data() + 8 * region, signs.data() + 8 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const float raw_base = maximum_scale / 31.0f;
  put_half(block, raw_base);
  const float base = fp16_to_fp32(fp32_to_fp16(raw_base));
  for (int region = 0; region < 8; ++region) {
    int local = base != 0.0f
        ? std::clamp(static_cast<int>(std::lround(
                         0.5f * (scales[region] / base - 1.0f))), 0, 15)
        : 0;
    const float scale = base * static_cast<float>(1 + 2 * local);
    choose_grid_region(iq_tables::iq3s_grid, 512, 4,
                       values + 32 * region, weights.data() + 32 * region, 8,
                       false, scale, indices.data() + 8 * region,
                       signs.data() + 8 * region);
    const int pair = region / 2;
    const int pair_half = region & 1;
    std::uint8_t high = 0;
    for (int group8 = 0; group8 < 4; ++group8) {
      const std::uint8_t sign_byte = static_cast<std::uint8_t>(
          signs[8 * region + 2 * group8] |
          (signs[8 * region + 2 * group8 + 1] << 4));
      block[74 + 4 * region + group8] = sign_byte;
      for (int half = 0; half < 2; ++half) {
        const int code = indices[8 * region + 2 * group8 + half];
        block[2 + 8 * region + 2 * group8 + half] =
            static_cast<std::uint8_t>(code);
        high |= static_cast<std::uint8_t>(((code >> 8) & 1)
                                         << (2 * group8 + half));
      }
    }
    block[66 + 2 * pair + pair_half] = high;
    block[106 + pair] |=
        static_cast<std::uint8_t>(local << (4 * pair_half));
  }
}

void pack_iq1_s(const float* values, const float* importance,
                std::uint8_t* block) {
  std::memset(block, 0, 50);
  std::array<float, 256> weights{};
  block_weights(values, importance, 2.0f, &weights);
  std::array<float, 8> scales{};
  std::array<int, 32> indices{};
  std::array<std::int8_t, 32> deltas{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 8; ++region) {
    scales[region] = fit_iq1_region(
        values + 32 * region, weights.data() + 32 * region, 4, true,
        indices.data() + 4 * region, deltas.data() + 4 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const float raw_base = maximum_scale / 15.0f;
  put_half(block, raw_base);
  const float base = fp16_to_fp32(fp32_to_fp16(raw_base));
  for (int region = 0; region < 8; ++region) {
    const int local = base != 0.0f
        ? std::clamp(static_cast<int>(std::lround(
                         0.5f * (scales[region] / base - 1.0f))), 0, 7)
        : 0;
    const float scale = base * static_cast<float>(1 + 2 * local);
    choose_iq1_region(values + 32 * region, weights.data() + 32 * region, 4,
                      scale, deltas.data() + 4 * region,
                      indices.data() + 4 * region);
    std::uint16_t high = static_cast<std::uint16_t>(local << 12);
    if (deltas[4 * region] != 0) high |= 0x8000;
    for (int group = 0; group < 4; ++group) {
      const int code = indices[4 * region + group];
      block[2 + 4 * region + group] = static_cast<std::uint8_t>(code);
      high |= static_cast<std::uint16_t>((code >> 8) << (3 * group));
    }
    put_u16(block + 34 + 2 * region, high);
  }
}

void pack_iq1_m(const float* values, const float* importance,
                std::uint8_t* block) {
  std::memset(block, 0, 56);
  std::array<float, 256> weights{};
  block_weights(values, importance, 2.0f, &weights);
  std::array<float, 16> scales{};
  std::array<int, 32> indices{};
  std::array<std::int8_t, 32> deltas{};
  float maximum_scale = 0.0f;
  for (int region = 0; region < 16; ++region) {
    scales[region] = fit_iq1_region(
        values + 16 * region, weights.data() + 16 * region, 2, false,
        indices.data() + 2 * region, deltas.data() + 2 * region);
    maximum_scale = std::max(maximum_scale, scales[region]);
  }
  const std::uint16_t base_bits = fp32_to_fp16(maximum_scale / 15.0f);
  const float base = fp16_to_fp32(base_bits);
  std::uint16_t scale_words[4] = {};
  static constexpr std::uint8_t kDeltaMasks[4] = {0x00, 0x80, 0x08, 0x88};
  for (int region = 0; region < 16; ++region) {
    const int local = base != 0.0f
        ? std::clamp(static_cast<int>(std::lround(
                         0.5f * (scales[region] / base - 1.0f))), 0, 7)
        : 0;
    const float scale = base * static_cast<float>(1 + 2 * local);
    choose_iq1_region(values + 16 * region, weights.data() + 16 * region, 2,
                      scale, deltas.data() + 2 * region,
                      indices.data() + 2 * region);
    const int combination = deltas[2 * region] * 2 + deltas[2 * region + 1];
    std::uint8_t high = kDeltaMasks[combination];
    for (int group = 0; group < 2; ++group) {
      const int code = indices[2 * region + group];
      block[2 * region + group] = static_cast<std::uint8_t>(code);
      high |= static_cast<std::uint8_t>((code >> 8) << (4 * group));
    }
    block[32 + region] = high;
    scale_words[region / 4] |=
        static_cast<std::uint16_t>(local << (3 * (region & 3)));
  }
  scale_words[0] |= static_cast<std::uint16_t>((base_bits & 0x000f) << 12);
  scale_words[1] |= static_cast<std::uint16_t>((base_bits & 0x00f0) << 8);
  scale_words[2] |= static_cast<std::uint16_t>((base_bits & 0x0f00) << 4);
  scale_words[3] |= static_cast<std::uint16_t>(base_bits & 0xf000);
  for (int i = 0; i < 4; ++i) put_u16(block + 48 + 2 * i, scale_words[i]);
}

}  // namespace

bool iq_pack_supported(QuantFormat format) {
  switch (format) {
    case QuantFormat::kIQ2_XXS:
    case QuantFormat::kIQ2_XS:
    case QuantFormat::kIQ3_XXS:
    case QuantFormat::kIQ3_S:
    case QuantFormat::kIQ2_S:
    case QuantFormat::kIQ1_S:
    case QuantFormat::kIQ1_M:
      return true;
    default:
      return false;
  }
}

void iq_pack_block_ref(QuantFormat format, const float* values,
                       const float* importance, std::uint8_t* block) {
  switch (format) {
    case QuantFormat::kIQ2_XXS: pack_iq2_xxs(values, importance, block); break;
    case QuantFormat::kIQ2_XS: pack_iq2_xs(values, importance, block); break;
    case QuantFormat::kIQ3_XXS: pack_iq3_xxs(values, importance, block); break;
    case QuantFormat::kIQ3_S: pack_iq3_s(values, importance, block); break;
    case QuantFormat::kIQ2_S: pack_iq2_s(values, importance, block); break;
    case QuantFormat::kIQ1_S: pack_iq1_s(values, importance, block); break;
    case QuantFormat::kIQ1_M: pack_iq1_m(values, importance, block); break;
    default: break;
  }
}

}  // namespace quixicore_cpu::quant
