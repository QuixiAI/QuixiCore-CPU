#include "kernels/quantization/gguf_ref.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernels/common/fp16.h"
#include "kernels/quantization/iq_tables.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float half_at(const std::uint8_t* bytes) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, bytes, sizeof(bits));
  return fp16_to_fp32(bits);
}

std::uint16_t u16_at(const std::uint8_t* bytes) {
  std::uint16_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

void scale_min_k4(int index, const std::uint8_t* scales, int* scale,
                  int* minimum) {
  if (index < 4) {
    *scale = scales[index] & 63;
    *minimum = scales[index + 4] & 63;
  } else {
    *scale = (scales[index + 4] & 0x0F) |
             ((scales[index - 4] >> 6) << 4);
    *minimum = (scales[index + 4] >> 4) |
               ((scales[index] >> 6) << 4);
  }
}

float dot_i8x16_f32(int8x16_t weights, const float* input);

float sum_f32x16(const float* input) {
  float32x4_t sum = vld1q_f32(input);
  sum = vaddq_f32(sum, vld1q_f32(input + 4));
  sum = vaddq_f32(sum, vld1q_f32(input + 8));
  sum = vaddq_f32(sum, vld1q_f32(input + 12));
  return vaddvq_f32(sum);
}

float e8m0_half(std::uint8_t exponent) {
  return std::ldexp(1.0f, static_cast<int>(exponent) - 128);
}

float ue4m3_half(std::uint8_t value) {
  if (value == 0 || value == 0x7f) return 0.0f;
  const int exponent = (value >> 3) & 15;
  const int mantissa = value & 7;
  return exponent == 0
             ? std::ldexp(static_cast<float>(mantissa), -10)
             : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                          exponent - 8);
}

bool direct_block_dot(QuantFormat format, const std::uint8_t* block,
                      const float* input, float* result) {
  float total = 0.0f;
  if (format == QuantFormat::kQ1_0) {
    alignas(16) std::int8_t quants[16];
    for (int group = 0; group < 8; ++group) {
      for (int lane = 0; lane < 16; ++lane) {
        const int column = group * 16 + lane;
        quants[lane] =
            (block[2 + column / 8] & (1 << (column & 7))) ? 1 : -1;
      }
      total += dot_i8x16_f32(vld1q_s8(quants), input + 16 * group);
    }
    *result = half_at(block) * total;
    return true;
  }
  if (format == QuantFormat::kQ2_0) {
    alignas(16) std::int8_t quants[16];
    for (int group = 0; group < 4; ++group) {
      for (int lane = 0; lane < 16; ++lane) {
        const int column = group * 16 + lane;
        quants[lane] = static_cast<std::int8_t>(
            ((block[2 + column / 4] >> (2 * (column & 3))) & 3) - 1);
      }
      total += dot_i8x16_f32(vld1q_s8(quants), input + 16 * group);
    }
    *result = half_at(block) * total;
    return true;
  }
  if (format == QuantFormat::kQ4_1) {
    const uint8x16_t codes = vld1q_u8(block + 4);
    const uint8x16_t mask = vdupq_n_u8(15);
    const int8x16_t low = vreinterpretq_s8_u8(vandq_u8(codes, mask));
    const int8x16_t high = vreinterpretq_s8_u8(vshrq_n_u8(codes, 4));
    const float quant_dot = dot_i8x16_f32(low, input) +
                            dot_i8x16_f32(high, input + 16);
    const float input_sum = sum_f32x16(input) + sum_f32x16(input + 16);
    *result = half_at(block) * quant_dot + half_at(block + 2) * input_sum;
    return true;
  }
  if (format == QuantFormat::kQ5_0 || format == QuantFormat::kQ5_1) {
    const bool affine = format == QuantFormat::kQ5_1;
    const std::uint32_t high_bits =
        std::uint32_t(u16_at(block + (affine ? 4 : 2))) |
        (std::uint32_t(u16_at(block + (affine ? 6 : 4))) << 16);
    const std::uint8_t* low = block + (affine ? 8 : 6);
    alignas(16) std::int8_t quants0[16];
    alignas(16) std::int8_t quants1[16];
    for (int lane = 0; lane < 16; ++lane) {
      int q0 = (low[lane] & 15) | (((high_bits >> lane) & 1) << 4);
      int q1 = (low[lane] >> 4) |
               (((high_bits >> (16 + lane)) & 1) << 4);
      if (!affine) {
        q0 -= 16;
        q1 -= 16;
      }
      quants0[lane] = static_cast<std::int8_t>(q0);
      quants1[lane] = static_cast<std::int8_t>(q1);
    }
    const float quant_dot = dot_i8x16_f32(vld1q_s8(quants0), input) +
                            dot_i8x16_f32(vld1q_s8(quants1), input + 16);
    total = half_at(block) * quant_dot;
    if (affine) {
      total += half_at(block + 2) *
               (sum_f32x16(input) + sum_f32x16(input + 16));
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kMXFP4 || format == QuantFormat::kNVFP4) {
    static constexpr std::int8_t kFp4Values[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t table = vld1q_s8(kFp4Values);
    const uint8x16_t mask = vdupq_n_u8(15);
    if (format == QuantFormat::kMXFP4) {
      const uint8x16_t codes = vld1q_u8(block + 1);
      const int8x16_t low = vqtbl1q_s8(table, vandq_u8(codes, mask));
      const int8x16_t high = vqtbl1q_s8(table, vshrq_n_u8(codes, 4));
      total = e8m0_half(block[0]) *
              (dot_i8x16_f32(low, input) +
               dot_i8x16_f32(high, input + 16));
    } else {
      for (int sub = 0; sub < 4; ++sub) {
        const uint8x8_t packed = vld1_u8(block + 4 + 8 * sub);
        const uint8x16_t codes =
            vcombine_u8(packed, vshr_n_u8(packed, 4));
        const int8x16_t values =
            vqtbl1q_s8(table, vandq_u8(codes, mask));
        total += ue4m3_half(block[sub]) *
                 dot_i8x16_f32(values, input + 16 * sub);
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kQ2_K) {
    const std::uint8_t* scales = block;
    const std::uint8_t* quants = block + 16;
    const float scale_base = half_at(block + 80);
    const float minimum_base = half_at(block + 82);
    for (int chunk = 0; chunk < 2; ++chunk) {
      for (int scale_index = 0; scale_index < 4; ++scale_index) {
        for (int sub = 0; sub < 2; ++sub) {
          const int index = chunk * 8 + scale_index * 2 + sub;
          const float scale = scale_base * (scales[index] & 15);
          const float minimum = minimum_base * (scales[index] >> 4);
          const int base = chunk * 128 + scale_index * 32 + sub * 16;
          for (int lane = 0; lane < 16; ++lane) {
            const int quant =
                (quants[chunk * 32 + sub * 16 + lane] >>
                 (2 * scale_index)) & 3;
            total += (scale * quant - minimum) * input[base + lane];
          }
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kQ3_K) {
    const std::uint8_t* high_mask = block;
    const std::uint8_t* quants = block + 32;
    const std::uint8_t* scales = block + 96;
    const float scale_base = half_at(block + 108);
    for (int chunk = 0; chunk < 2; ++chunk) {
      for (int scale_index = 0; scale_index < 4; ++scale_index) {
        for (int sub = 0; sub < 2; ++sub) {
          const int index = chunk * 8 + scale_index * 2 + sub;
          const int word = index >> 2;
          const int byte = index & 3;
          int local_scale = 0;
          if (word == 0) {
            local_scale =
                (scales[byte] & 15) | ((scales[8 + byte] & 3) << 4);
          } else if (word == 1) {
            local_scale = (scales[4 + byte] & 15) |
                          (((scales[8 + byte] >> 2) & 3) << 4);
          } else if (word == 2) {
            local_scale = ((scales[byte] >> 4) & 15) |
                          (((scales[8 + byte] >> 4) & 3) << 4);
          } else {
            local_scale = ((scales[4 + byte] >> 4) & 15) |
                          (((scales[8 + byte] >> 6) & 3) << 4);
          }
          const float scale = scale_base * (local_scale - 32);
          const int base = chunk * 128 + scale_index * 32 + sub * 16;
          for (int lane = 0; lane < 16; ++lane) {
            const int low =
                (quants[chunk * 32 + sub * 16 + lane] >>
                 (2 * scale_index)) & 3;
            const int high =
                (high_mask[sub * 16 + lane] &
                 (1 << (chunk * 4 + scale_index))) != 0;
            total += scale * ((low | (high << 2)) - 4) * input[base + lane];
          }
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kQ4_K || format == QuantFormat::kQ5_K) {
    const bool five_bit = format == QuantFormat::kQ5_K;
    const float scale_base = half_at(block);
    const float minimum_base = half_at(block + 2);
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* high = five_bit ? block + 16 : nullptr;
    const std::uint8_t* quants = block + (five_bit ? 48 : 16);
    for (int sub = 0; sub < 8; ++sub) {
      int scale = 0;
      int minimum = 0;
      scale_min_k4(sub, scales, &scale, &minimum);
      const int chunk = sub >> 1;
      const bool upper = (sub & 1) != 0;
      for (int lane = 0; lane < 32; ++lane) {
        int quant = upper ? quants[chunk * 32 + lane] >> 4
                          : quants[chunk * 32 + lane] & 15;
        if (five_bit && (high[lane] & (1 << sub)) != 0) quant += 16;
        total += (scale_base * static_cast<float>(scale * quant) -
                  minimum_base * static_cast<float>(minimum)) *
                 input[sub * 32 + lane];
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kQ6_K) {
    const std::uint8_t* low = block;
    const std::uint8_t* high = block + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
    const float scale_base = half_at(block + 208);
    for (int chunk = 0; chunk < 2; ++chunk) {
      for (int group = 0; group < 4; ++group) {
        for (int lane = 0; lane < 32; ++lane) {
          const int low_byte = low[chunk * 64 + lane + 32 * (group & 1)];
          const int nibble =
              group & 2 ? (low_byte >> 4) : (low_byte & 15);
          const int high_bits =
              (high[chunk * 32 + lane] >> (2 * group)) & 3;
          const int quant = (nibble | (high_bits << 4)) - 32;
          const int scale_index =
              chunk * 8 + (lane >> 4) + group * 2;
          const int column = chunk * 128 + group * 32 + lane;
          total += scale_base * static_cast<float>(scales[scale_index] * quant) *
                   input[column];
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ4_NL) {
    const float scale = half_at(block);
    const std::uint8_t* quants = block + 2;
    static constexpr std::int8_t kIq4Bytes[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1,    13,   25,  38,  53,  69,  89,  113};
    const int8x16_t table = vld1q_s8(kIq4Bytes);
    const uint8x16_t codes = vld1q_u8(quants);
    const uint8x16_t mask = vdupq_n_u8(15);
    const int8x16_t low = vqtbl1q_s8(table, vandq_u8(codes, mask));
    const int8x16_t high = vqtbl1q_s8(table, vshrq_n_u8(codes, 4));
    total = scale * (dot_i8x16_f32(low, input) +
                     dot_i8x16_f32(high, input + 16));
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ4_XS) {
    const float scale_base = half_at(block);
    const std::uint16_t scales_high = u16_at(block + 2);
    const std::uint8_t* scales_low = block + 4;
    const std::uint8_t* quants = block + 8;
    static constexpr std::int8_t kIq4Bytes[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1,    13,   25,  38,  53,  69,  89,  113};
    const int8x16_t table = vld1q_s8(kIq4Bytes);
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int sub = 0; sub < 8; ++sub) {
      const int low_scale =
          (scales_low[sub >> 1] >> (4 * (sub & 1))) & 15;
      const int high_scale = (scales_high >> (2 * sub)) & 3;
      const float scale =
          scale_base * static_cast<float>((low_scale | (high_scale << 4)) - 32);
      const uint8x16_t codes = vld1q_u8(quants + 16 * sub);
      const int8x16_t low = vqtbl1q_s8(table, vandq_u8(codes, mask));
      const int8x16_t high = vqtbl1q_s8(table, vshrq_n_u8(codes, 4));
      total += scale *
               (dot_i8x16_f32(low, input + sub * 32) +
                dot_i8x16_f32(high, input + sub * 32 + 16));
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ2_XXS) {
    const float scale_base = half_at(block);
    for (int block32 = 0; block32 < 8; ++block32) {
      const std::uint8_t* values = block + 2 + 8 * block32;
      const std::uint32_t grids = u16_at(values) |
                                  (std::uint32_t(u16_at(values + 2)) << 16);
      const std::uint32_t signs_scale =
          u16_at(values + 4) |
          (std::uint32_t(u16_at(values + 6)) << 16);
      const float scale =
          scale_base * (0.5f + float((signs_scale >> 28) & 15)) * 0.25f;
      for (int sub = 0; sub < 4; ++sub) {
        const std::uint64_t grid =
            iq_tables::iq2xxs_grid[(grids >> (8 * sub)) & 255];
        const std::uint8_t signs = iq_tables::ksigns_iq2xs[
            (signs_scale >> (7 * sub)) & 127];
        for (int element = 0; element < 8; ++element) {
          const float magnitude = float((grid >> (8 * element)) & 255);
          const float value =
              (signs & iq_tables::kmask_iq2xs[element]) ? -magnitude
                                                        : magnitude;
          total += scale * value * input[block32 * 32 + sub * 8 + element];
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ2_XS) {
    const float scale_base = half_at(block);
    for (int block32 = 0; block32 < 8; ++block32) {
      for (int half = 0; half < 2; ++half) {
        const int local_scale = (block[66 + block32] >> (4 * half)) & 15;
        const float scale = scale_base * (0.5f + local_scale) * 0.25f;
        for (int sub = 0; sub < 2; ++sub) {
          const std::uint16_t index =
              u16_at(block + 2 + 2 * (4 * block32 + 2 * half + sub));
          const std::uint64_t grid = iq_tables::iq2xs_grid[index & 511];
          const std::uint8_t signs = iq_tables::ksigns_iq2xs[index >> 9];
          for (int element = 0; element < 8; ++element) {
            const float magnitude = float((grid >> (8 * element)) & 255);
            const float value =
                (signs & iq_tables::kmask_iq2xs[element]) ? -magnitude
                                                          : magnitude;
            const int column =
                block32 * 32 + half * 16 + sub * 8 + element;
            total += scale * value * input[column];
          }
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ3_XXS) {
    const float scale_base = half_at(block);
    const std::uint8_t* quants = block + 2;
    for (int block32 = 0; block32 < 8; ++block32) {
      const std::uint8_t* local_quants = quants + 8 * block32;
      const std::uint8_t* signs_bytes = quants + 64 + 4 * block32;
      const std::uint32_t signs_scale =
          u16_at(signs_bytes) |
          (std::uint32_t(u16_at(signs_bytes + 2)) << 16);
      const float scale =
          scale_base * (0.5f + float(signs_scale >> 28)) * 0.5f;
      for (int half = 0; half < 2; ++half) {
        for (int group = 0; group < 4; ++group) {
          const std::uint32_t grid =
              iq_tables::iq3xxs_grid[local_quants[4 * half + group]];
          const std::uint8_t signs = iq_tables::ksigns_iq2xs[
              (signs_scale >> (14 * half + 7 * (group >> 1))) & 127];
          for (int element = 0; element < 4; ++element) {
            const float magnitude = float((grid >> (8 * element)) & 255);
            const int sign_element = element + 4 * (group & 1);
            const float value =
                (signs & iq_tables::kmask_iq2xs[sign_element]) ? -magnitude
                                                               : magnitude;
            const int column =
                block32 * 32 + half * 16 + group * 4 + element;
            total += scale * value * input[column];
          }
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ3_S) {
    const float scale_base = half_at(block);
    const std::uint8_t* quants = block + 2;
    const std::uint8_t* high = block + 66;
    const std::uint8_t* signs = block + 74;
    const std::uint8_t* scales = block + 106;
    for (int block32 = 0; block32 < 8; ++block32) {
      const int pair = block32 >> 1;
      const float scale = scale_base * static_cast<float>(
          1 + 2 * ((block32 & 1) == 0 ? (scales[pair] & 15)
                                      : (scales[pair] >> 4)));
      for (int group8 = 0; group8 < 4; ++group8) {
        for (int group4 = 0; group4 < 2; ++group4) {
          const int offset = block32 * 8 + group8 * 2 + group4;
          const int high_bit =
              (high[2 * pair + (block32 & 1)] >> (2 * group8 + group4)) & 1;
          const std::uint32_t grid =
              iq_tables::iq3s_grid[quants[offset] | (high_bit << 8)];
          for (int element = 0; element < 4; ++element) {
            const int within8 = group4 * 4 + element;
            const float magnitude =
                static_cast<float>((grid >> (8 * element)) & 255);
            const float value =
                (signs[block32 * 4 + group8] & (1 << within8))
                    ? -magnitude
                    : magnitude;
            const int column = block32 * 32 + group8 * 8 + within8;
            total += scale * value * input[column];
          }
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ2_S) {
    const float scale_base = half_at(block);
    const std::uint8_t* quants = block + 2;
    const std::uint8_t* signs = block + 34;
    const std::uint8_t* high = block + 66;
    const std::uint8_t* scales = block + 74;
    for (int block32 = 0; block32 < 8; ++block32) {
      for (int group8 = 0; group8 < 4; ++group8) {
        const int grid_index =
            quants[block32 * 4 + group8] |
            (((high[block32] >> (2 * group8)) & 3) << 8);
        const std::uint64_t grid = iq_tables::iq2s_grid[grid_index];
        const int local_scale = group8 < 2 ? (scales[block32] & 15)
                                           : (scales[block32] >> 4);
        const float scale = scale_base * (0.5f + local_scale) * 0.25f;
        for (int element = 0; element < 8; ++element) {
          const float magnitude =
              static_cast<float>((grid >> (8 * element)) & 255);
          const float value =
              (signs[block32 * 4 + group8] & (1 << element))
                  ? -magnitude
                  : magnitude;
          const int column = block32 * 32 + group8 * 8 + element;
          total += scale * value * input[column];
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ1_S) {
    constexpr float kDelta = 0.125f;
    const float scale_base = half_at(block);
    const std::uint8_t* quants = block + 2;
    for (int block32 = 0; block32 < 8; ++block32) {
      const std::uint16_t high = u16_at(block + 34 + 2 * block32);
      const float scale =
          scale_base * float(2 * ((high >> 12) & 7) + 1);
      const float minimum = scale *
          ((high & 0x8000) ? (-1.0f - kDelta) : (-1.0f + kDelta));
      for (int half = 0; half < 2; ++half) {
        const std::uint8_t* local_quants =
            quants + 4 * block32 + 2 * half;
        const std::uint32_t high_index = high >> (6 * half);
        for (int which = 0; which < 4; ++which) {
          const std::uint32_t grid_index =
              (which >> 1) == 0
                  ? (local_quants[0] | ((high_index << 8) & 0x700))
                  : (local_quants[1] | ((high_index << 5) & 0x700));
          for (int element = 0; element < 4; ++element) {
            const std::uint32_t byte =
                (iq_tables::iq1s_grid_gpu[grid_index] >> (8 * element)) & 255;
            const std::uint32_t value =
                which & 1 ? byte >> 4 : byte & 15;
            const int column =
                block32 * 32 + half * 16 + which * 4 + element;
            total += (scale * float(value) + minimum) * input[column];
          }
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ1_M) {
    constexpr float kDelta = 0.125f;
    const std::uint8_t* quants = block;
    const std::uint8_t* high = block + 32;
    const std::uint8_t* scale_bytes = block + 48;
    const std::uint16_t scales[4] = {
        u16_at(scale_bytes), u16_at(scale_bytes + 2),
        u16_at(scale_bytes + 4), u16_at(scale_bytes + 6)};
    const std::uint16_t scale_bits = static_cast<std::uint16_t>(
        (scales[0] >> 12) | ((scales[1] >> 8) & 0x00f0) |
        ((scales[2] >> 4) & 0x0f00) | (scales[3] & 0xf000));
    const float scale_base = fp16_to_fp32(scale_bits);
    for (int block32 = 0; block32 < 8; ++block32) {
      for (int group8 = 0; group8 < 4; ++group8) {
        const std::uint8_t high_byte =
            high[block32 * 2 + group8 / 2];
        const int shift = (group8 & 1) == 0 ? 0 : 4;
        const int grid_index =
            quants[block32 * 4 + group8] |
            (((high_byte >> shift) & 7) << 8);
        const std::uint64_t grid = iq_tables::iq1s_grid[grid_index];
        const bool negative_delta =
            (high_byte & ((group8 & 1) == 0 ? 0x08 : 0x80)) != 0;
        const int scale_shift =
            6 * (block32 & 1) + (group8 < 2 ? 0 : 3);
        const float scale = scale_base * static_cast<float>(
            2 * ((scales[block32 >> 1] >> scale_shift) & 7) + 1);
        const float delta = negative_delta ? -kDelta : kDelta;
        for (int element = 0; element < 8; ++element) {
          const float value = static_cast<float>(static_cast<std::int8_t>(
              (grid >> (8 * element)) & 255));
          const int column = block32 * 32 + group8 * 8 + element;
          total += scale * (value + delta) * input[column];
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kTQ1_0) {
    static constexpr std::uint8_t kPow3[5] = {1, 3, 9, 27, 81};
    const float scale = half_at(block + 52);
    for (int trit = 0; trit < 5; ++trit) {
      for (int lane = 0; lane < 32; ++lane) {
        const std::uint8_t rotated =
            static_cast<std::uint8_t>(block[lane] * kPow3[trit]);
        const int quant = (static_cast<unsigned>(rotated) * 3) >> 8;
        total += scale * static_cast<float>(quant - 1) *
                 input[trit * 32 + lane];
      }
    }
    for (int trit = 0; trit < 5; ++trit) {
      for (int lane = 0; lane < 16; ++lane) {
        const std::uint8_t rotated =
            static_cast<std::uint8_t>(block[32 + lane] * kPow3[trit]);
        const int quant = (static_cast<unsigned>(rotated) * 3) >> 8;
        total += scale * static_cast<float>(quant - 1) *
                 input[160 + trit * 16 + lane];
      }
    }
    for (int trit = 0; trit < 4; ++trit) {
      for (int lane = 0; lane < 4; ++lane) {
        const std::uint8_t rotated =
            static_cast<std::uint8_t>(block[48 + lane] * kPow3[trit]);
        const int quant = (static_cast<unsigned>(rotated) * 3) >> 8;
        total += scale * static_cast<float>(quant - 1) *
                 input[240 + trit * 4 + lane];
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kTQ2_0) {
    const float scale = half_at(block + 64);
    for (int half = 0; half < 2; ++half) {
      for (int group = 0; group < 4; ++group) {
        for (int lane = 0; lane < 32; ++lane) {
          const int quant =
              (block[32 * half + lane] >> (2 * group)) & 3;
          const int column = half * 128 + group * 32 + lane;
          total += scale * static_cast<float>(quant - 1) * input[column];
        }
      }
    }
    *result = total;
    return true;
  }
  return false;
}

float dot_i8x16_f32(int8x16_t weights, const float* input) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(weights));
  const int16x8_t hi = vmovl_s8(vget_high_s8(weights));
  float32x4_t sum = vmulq_f32(
      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(input));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                  vld1q_f32(input + 4));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                  vld1q_f32(input + 8));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                  vld1q_f32(input + 12));
  return vaddvq_f32(sum);
}

void q4_rows(const BlockQ4_0* packed, const float* x, float* y,
             long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    const uint8x16_t mask = vdupq_n_u8(15);
    const uint8x16_t offset = vdupq_n_u8(8);
    for (long long row = begin; row < end; ++row) {
      float total = 0.0f;
      const BlockQ4_0* row_weights = packed + row * blocks;
      for (long long block = 0; block < blocks; ++block) {
        const uint8x16_t codes = vld1q_u8(row_weights[block].qs);
        const int8x16_t low = vreinterpretq_s8_u8(
            vsubq_u8(vandq_u8(codes, mask), offset));
        const int8x16_t high = vreinterpretq_s8_u8(
            vsubq_u8(vshrq_n_u8(codes, 4), offset));
        const float* input = x + block * kQ4_0BlockSize;
        const float dot = dot_i8x16_f32(low, input) +
                          dot_i8x16_f32(high, input + 16);
        total += fp16_to_fp32(row_weights[block].d) * dot;
      }
      y[row] = total;
    }
  });
}

}  // namespace

void gguf_gemv_neon(QuantFormat format, const void* packed, const float* x,
                    float* y, long long n, long long k) {
  if (format == QuantFormat::kQ4_0) {
    q4_rows(static_cast<const BlockQ4_0*>(packed), x, y, n, k);
    return;
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  (void)gguf_format_info(format, &block_size, &block_bytes);
  const long long blocks_per_row = k / block_size;
  const auto* bytes = static_cast<const std::uint8_t*>(packed);
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    alignas(64) float decoded[256];
    for (long long row = begin; row < end; ++row) {
      float total = 0.0f;
      for (long long block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* packed_block =
            bytes + (row * blocks_per_row + block) * block_bytes;
        const float* input = x + block * block_size;
        float direct = 0.0f;
        if (direct_block_dot(format, packed_block, input, &direct)) {
          total += direct;
          continue;
        }
        gguf_dequant_block_ref(
            format, packed_block, decoded);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int column = 0;
        for (; column + 7 < block_size; column += 8) {
          acc0 = vfmaq_f32(acc0, vld1q_f32(decoded + column),
                           vld1q_f32(input + column));
          acc1 = vfmaq_f32(acc1, vld1q_f32(decoded + column + 4),
                           vld1q_f32(input + column + 4));
        }
        float block_total = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; column < block_size; ++column) {
          block_total += decoded[column] * input[column];
        }
        total += block_total;
      }
      y[row] = total;
    }
  });
}

}  // namespace quixicore_cpu::quant

#else

namespace quixicore_cpu::quant {
void gguf_gemv_neon(QuantFormat, const void*, const float*, float*, long long,
                    long long) {}
}  // namespace quixicore_cpu::quant

#endif
