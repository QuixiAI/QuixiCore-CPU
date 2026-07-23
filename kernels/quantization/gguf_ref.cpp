#include "kernels/quantization/gguf_ref.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "kernels/common/fp16.h"
#include "kernels/quantization/iq_tables.h"
#include "quixicore_cpu/quantization.h"
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

std::uint32_t u32_at(const std::uint8_t* bytes) {
  std::uint32_t value = 0;
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

float dequant(QuantFormat format, const std::uint8_t* block, int column) {
  static constexpr float kE2M1[8] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const auto e2m1 = [](unsigned code) {
    const float magnitude = kE2M1[code & 7];
    return code & 8 ? -magnitude : magnitude;
  };
  const auto e8m0 = [](std::uint8_t code) {
    return code == 0 ? std::ldexp(1.0f, -127)
                     : std::ldexp(1.0f, static_cast<int>(code) - 127);
  };
  const auto ue4m3 = [](std::uint8_t code) {
    // UE4M3 is unsigned: bit 7 is not a sign bit. GGML reserves 0x7f as
    // another zero representation and uses the remaining low seven bits as
    // E4M3 with exponent bias 7.
    if (code == 0 || code == 0x7f) return 0.0f;
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    return exponent == 0
               ? std::ldexp(static_cast<float>(mantissa), -9)
               : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                            exponent - 7);
  };
  const auto fp6 = [](unsigned code, bool e3m2) {
    const unsigned exponent = e3m2 ? ((code >> 2) & 7) : ((code >> 3) & 3);
    const unsigned mantissa = e3m2 ? (code & 3) : (code & 7);
    float magnitude = 0.0f;
    if (exponent == 0) {
      magnitude = std::ldexp(static_cast<float>(mantissa), e3m2 ? -4 : -3);
    } else {
      magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) /
                                        static_cast<float>(e3m2 ? 4 : 8),
                             static_cast<int>(exponent) - (e3m2 ? 3 : 1));
    }
    return code & 32 ? -magnitude : magnitude;
  };
  switch (format) {
    case QuantFormat::kQ1_0: {
      const int bit = (block[2 + column / 8] >> (column & 7)) & 1;
      return half_at(block) * (bit != 0 ? 1.0f : -1.0f);
    }
    case QuantFormat::kQ2_0: {
      const int quant =
          (block[2 + column / 4] >> (2 * (column & 3))) & 3;
      return half_at(block) * static_cast<float>(quant - 1);
    }
    case QuantFormat::kQ8_0:
      return half_at(block) *
             static_cast<float>(static_cast<std::int8_t>(block[2 + column]));
    case QuantFormat::kQ4_0: {
      const std::uint8_t byte = block[2 + (column & 15)];
      const int quant = column < 16 ? (byte & 15) : (byte >> 4);
      return half_at(block) * (quant - 8);
    }
    case QuantFormat::kQ4_1: {
      const std::uint8_t byte = block[4 + (column & 15)];
      const int quant = column < 16 ? (byte & 15) : (byte >> 4);
      return half_at(block) * quant + half_at(block + 2);
    }
    case QuantFormat::kQ5_0:
    case QuantFormat::kQ5_1: {
      const bool affine = format == QuantFormat::kQ5_1;
      const std::uint8_t* high_bytes = block + (affine ? 4 : 2);
      const std::uint8_t* low_bytes = block + (affine ? 8 : 6);
      const std::uint32_t high = u32_at(high_bytes);
      const int local = column & 15;
      const int low = column < 16 ? (low_bytes[local] & 15)
                                  : (low_bytes[local] >> 4);
      const int high_bit =
          static_cast<int>((high >> (local + (column >= 16 ? 16 : 0))) & 1u);
      const int quant = low | (high_bit << 4);
      return affine ? half_at(block) * quant + half_at(block + 2)
                    : half_at(block) * (quant - 16);
    }
    case QuantFormat::kU4B8: {
      const std::uint8_t* quants = block + 2;
      const int quant = column < 64 ? quants[column] & 15
                                    : quants[column - 64] >> 4;
      return half_at(block) * (quant - 8);
    }
    case QuantFormat::kU4:
    case QuantFormat::kHQQ: {
      const int half_block = format == QuantFormat::kU4 ? 64 : 32;
      const std::uint8_t* quants = block + 4;
      const int quant = column < half_block ? quants[column] & 15
                                            : quants[column - half_block] >> 4;
      return half_at(block) * (quant - half_at(block + 2));
    }
    case QuantFormat::kFP8E4M3:
      return half_at(block) *
             float8_decode(block[2 + column], Float8Format::kE4M3FN);
    case QuantFormat::kFP8E5M2:
      return half_at(block) *
             float8_decode(block[2 + column], Float8Format::kE5M2);
    case QuantFormat::kFP8Block:
      return half_at(block) *
             float8_decode(block[2 + column], Float8Format::kE4M3FN);
    case QuantFormat::kFP8Raw:
      return float8_decode(block[column], Float8Format::kE4M3FN);
    case QuantFormat::kFP4E2M1: {
      const std::uint8_t* quants = block + 2;
      const int code = column < 16 ? quants[column] & 15
                                   : quants[column - 16] >> 4;
      return half_at(block) * e2m1(code);
    }
    case QuantFormat::kMXFP8:
      return e8m0(block[0]) *
             float8_decode(block[1 + column], Float8Format::kE4M3FN);
    case QuantFormat::kNVFP4: {
      // Canonical GGML block_nvfp4: four UE4M3 scales followed by four
      // 16-value E2M1 sub-blocks. kvalues_mxfp4 stores doubled E2M1 values and
      // ggml's UE4M3 decoder returns a half-scale; using ordinary E2M1 here is
      // equivalent, with 0x7f retaining its format-specific zero meaning.
      const int sub = column >> 4;
      const int local = column & 15;
      const std::uint8_t* quants = block + 4 + sub * 8;
      const int code = local < 8 ? quants[local] & 15
                                 : quants[local - 8] >> 4;
      return ue4m3(block[sub]) * e2m1(code);
    }
    case QuantFormat::kMXFP4: {
      const std::uint8_t* quants = block + 1;
      const int code = column < 16 ? quants[column] & 15
                                   : quants[column - 16] >> 4;
      return e8m0(block[0]) * e2m1(code);
    }
    case QuantFormat::kMXFP6E3M2:
    case QuantFormat::kMXFP6E2M3: {
      const int group = column >> 2;
      const int within = column & 3;
      const std::uint8_t* codes = block + 1 + 3 * group;
      const unsigned packed = unsigned(codes[0]) |
                              (unsigned(codes[1]) << 8) |
                              (unsigned(codes[2]) << 16);
      return e8m0(block[0]) * fp6((packed >> (6 * within)) & 63,
                                  format == QuantFormat::kMXFP6E3M2);
    }
    case QuantFormat::kBitnet: {
      const int code = (block[2 + column / 4] >> (2 * (column % 4))) & 3;
      return half_at(block) * (code - 1);
    }
    case QuantFormat::kQ2_K: {
      const std::uint8_t* scales = block;
      const std::uint8_t* quants = block + 16;
      const int chunk = column >> 7;
      const int position = column & 127;
      const int scale_index = position >> 5;
      const int sub = (position >> 4) & 1;
      const int lane = position & 15;
      const int index = chunk * 8 + scale_index * 2 + sub;
      const int quant =
          (quants[chunk * 32 + sub * 16 + lane] >> (2 * scale_index)) & 3;
      return half_at(block + 80) * (scales[index] & 15) * quant -
             half_at(block + 82) * (scales[index] >> 4);
    }
    case QuantFormat::kQ3_K: {
      const std::uint8_t* high_mask = block;
      const std::uint8_t* quants = block + 32;
      const std::uint8_t* scales = block + 96;
      const int chunk = column >> 7;
      const int position = column & 127;
      const int scale_index = position >> 5;
      const int sub = (position >> 4) & 1;
      const int lane = position & 15;
      const int index = chunk * 8 + scale_index * 2 + sub;
      const int low =
          (quants[chunk * 32 + sub * 16 + lane] >> (2 * scale_index)) & 3;
      const int high =
          (high_mask[sub * 16 + lane] & (1 << (chunk * 4 + scale_index))) != 0;
      const int quant = (low | (high << 2)) - 4;
      const int word = index >> 2;
      const int byte = index & 3;
      int scale = 0;
      if (word == 0) {
        scale = (scales[byte] & 15) | ((scales[8 + byte] & 3) << 4);
      } else if (word == 1) {
        scale = (scales[4 + byte] & 15) |
                (((scales[8 + byte] >> 2) & 3) << 4);
      } else if (word == 2) {
        scale = ((scales[byte] >> 4) & 15) |
                (((scales[8 + byte] >> 4) & 3) << 4);
      } else {
        scale = ((scales[4 + byte] >> 4) & 15) |
                (((scales[8 + byte] >> 6) & 3) << 4);
      }
      return half_at(block + 108) * (scale - 32) * quant;
    }
    case QuantFormat::kQ4_K:
    case QuantFormat::kQ5_K: {
      const bool five_bit = format == QuantFormat::kQ5_K;
      const std::uint8_t* scales = block + 4;
      const std::uint8_t* high = five_bit ? block + 16 : nullptr;
      const std::uint8_t* quants = block + (five_bit ? 48 : 16);
      const int chunk = column / 64;
      const int position = column % 64;
      const int sub = 2 * chunk + (position >= 32 ? 1 : 0);
      const int lane = position & 31;
      int quant = position < 32 ? (quants[chunk * 32 + lane] & 15)
                                : (quants[chunk * 32 + lane] >> 4);
      if (five_bit && (high[lane] & (1 << sub)) != 0) quant += 16;
      int scale = 0;
      int minimum = 0;
      scale_min_k4(sub, scales, &scale, &minimum);
      return half_at(block) * scale * quant -
             half_at(block + 2) * minimum;
    }
    case QuantFormat::kQ6_K: {
      const std::uint8_t* low = block;
      const std::uint8_t* high = block + 128;
      const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
      const int chunk = column >> 7;
      const int position = column & 127;
      const int group = position >> 5;
      const int lane = position & 31;
      const int low_byte = low[chunk * 64 + lane + 32 * (group & 1)];
      const int nibble = group & 2 ? (low_byte >> 4) : (low_byte & 15);
      const int high_bits = (high[chunk * 32 + lane] >> (2 * group)) & 3;
      const int quant = (nibble | (high_bits << 4)) - 32;
      const int scale_index = chunk * 8 + (lane >> 4) + group * 2;
      return half_at(block + 208) * scales[scale_index] * quant;
    }
    case QuantFormat::kIQ4_NL: {
      static constexpr int kValues[16] = {
          -127, -104, -83, -65, -49, -35, -22, -10,
          1,    13,   25,  38,  53,  69,  89,  113};
      const std::uint8_t byte = block[2 + (column & 15)];
      const int index = column < 16 ? (byte & 15) : (byte >> 4);
      return half_at(block) * kValues[index];
    }
    case QuantFormat::kIQ4_XS: {
      static constexpr int kValues[16] = {
          -127, -104, -83, -65, -49, -35, -22, -10,
          1,    13,   25,  38,  53,  69,  89,  113};
      const std::uint16_t scales_high = u16_at(block + 2);
      const std::uint8_t* scales_low = block + 4;
      const std::uint8_t* quants = block + 8;
      const int sub = column >> 5;
      const int local = column & 31;
      const int low_scale =
          (scales_low[sub >> 1] >> (4 * (sub & 1))) & 15;
      const int high_scale = (scales_high >> (2 * sub)) & 3;
      const int scale = (low_scale | (high_scale << 4)) - 32;
      const std::uint8_t byte = quants[16 * sub + (local & 15)];
      const int index = local < 16 ? (byte & 15) : (byte >> 4);
      return half_at(block) * scale * kValues[index];
    }
    case QuantFormat::kIQ2_XXS: {
      const int block32 = column >> 5;
      const int position = column & 31;
      const int sub = position >> 3;
      const int element = position & 7;
      const std::uint8_t* values = block + 2 + 8 * block32;
      const std::uint32_t grids = u16_at(values) |
                                  (std::uint32_t(u16_at(values + 2)) << 16);
      const std::uint32_t signs_scale = u16_at(values + 4) |
          (std::uint32_t(u16_at(values + 6)) << 16);
      const std::uint64_t grid = iq_tables::iq2xxs_grid[
          (grids >> (8 * sub)) & 255];
      const std::uint8_t signs = iq_tables::ksigns_iq2xs[
          (signs_scale >> (7 * sub)) & 127];
      const float scale = half_at(block) *
                          (0.5f + float((signs_scale >> 28) & 15)) * 0.25f;
      const float magnitude = float((grid >> (8 * element)) & 255);
      return scale * magnitude *
             ((signs & iq_tables::kmask_iq2xs[element]) ? -1.0f : 1.0f);
    }
    case QuantFormat::kIQ2_XS: {
      const int block32 = column >> 5;
      const int position = column & 31;
      const int half = position >> 4;
      const int sub = (position & 15) >> 3;
      const int element = position & 7;
      const std::uint16_t index =
          u16_at(block + 2 + 2 * (4 * block32 + 2 * half + sub));
      const std::uint64_t grid = iq_tables::iq2xs_grid[index & 511];
      const std::uint8_t signs = iq_tables::ksigns_iq2xs[index >> 9];
      const int local_scale = (block[66 + block32] >> (4 * half)) & 15;
      const float scale = half_at(block) * (0.5f + local_scale) * 0.25f;
      const float magnitude = float((grid >> (8 * element)) & 255);
      return scale * magnitude *
             ((signs & iq_tables::kmask_iq2xs[element]) ? -1.0f : 1.0f);
    }
    case QuantFormat::kIQ3_XXS: {
      const std::uint8_t* quants = block + 2;
      const int block32 = column >> 5;
      const int position = column & 31;
      const int half = position >> 4;
      const int word = position & 15;
      const int group = word >> 2;
      const int element = word & 3;
      const std::uint8_t* local_quants = quants + 8 * block32;
      const std::uint8_t* signs_bytes = quants + 64 + 4 * block32;
      const std::uint32_t signs_scale = u16_at(signs_bytes) |
          (std::uint32_t(u16_at(signs_bytes + 2)) << 16);
      const std::uint32_t grid =
          iq_tables::iq3xxs_grid[local_quants[4 * half + group]];
      const std::uint8_t signs = iq_tables::ksigns_iq2xs[
          (signs_scale >> (14 * half + 7 * (group >> 1))) & 127];
      const float scale = half_at(block) *
                          (0.5f + float(signs_scale >> 28)) * 0.5f;
      const float magnitude = float((grid >> (8 * element)) & 255);
      const int sign_element = element + 4 * (group & 1);
      return scale * magnitude *
             ((signs & iq_tables::kmask_iq2xs[sign_element]) ? -1.0f : 1.0f);
    }
    case QuantFormat::kIQ3_S: {
      const std::uint8_t* quants = block + 2;
      const std::uint8_t* high = block + 66;
      const std::uint8_t* signs = block + 74;
      const std::uint8_t* scales = block + 106;
      const int block32 = column >> 5;
      const int position = column & 31;
      const int group8 = position >> 3;
      const int within8 = position & 7;
      const int group4 = within8 >> 2;
      const int element = within8 & 3;
      const int pair64 = block32 >> 1;
      const int pair_half = block32 & 1;
      const int quant_offset = block32 * 8 + group8 * 2 + group4;
      const int high_byte = high[2 * pair64 + pair_half];
      const int high_shift = group8 * 2 + group4;
      const int grid_index = quants[quant_offset] |
                             (((high_byte >> high_shift) & 1) << 8);
      const std::uint32_t grid = iq_tables::iq3s_grid[grid_index];
      const int sign =
          (signs[block32 * 4 + group8] >> within8) & 1;
      const int local_scale = pair_half == 0 ? (scales[pair64] & 15)
                                             : (scales[pair64] >> 4);
      const float scale =
          half_at(block) * static_cast<float>(1 + 2 * local_scale);
      const float magnitude =
          static_cast<float>((grid >> (8 * element)) & 255);
      return scale * magnitude * (sign != 0 ? -1.0f : 1.0f);
    }
    case QuantFormat::kIQ2_S: {
      const std::uint8_t* quants = block + 2;
      const std::uint8_t* high = block + 66;
      const std::uint8_t* signs = block + 34;
      const std::uint8_t* scales = block + 74;
      const int block32 = column >> 5;
      const int position = column & 31;
      const int group8 = position >> 3;
      const int element = position & 7;
      const int grid_index =
          quants[block32 * 4 + group8] |
          (((high[block32] >> (2 * group8)) & 3) << 8);
      const std::uint64_t grid = iq_tables::iq2s_grid[grid_index];
      const int local_scale = group8 < 2 ? (scales[block32] & 15)
                                         : (scales[block32] >> 4);
      const float scale = half_at(block) * (0.5f + local_scale) * 0.25f;
      const float magnitude =
          static_cast<float>((grid >> (8 * element)) & 255);
      const int sign =
          (signs[block32 * 4 + group8] >> element) & 1;
      return scale * magnitude * (sign != 0 ? -1.0f : 1.0f);
    }
    case QuantFormat::kIQ1_S: {
      static constexpr float kDelta = 0.125f;
      const std::uint8_t* quants = block + 2;
      const int block32 = column >> 5;
      const int position = column & 31;
      const int group8 = position >> 3;
      const int element = position & 7;
      const std::uint16_t high = u16_at(block + 34 + 2 * block32);
      const float scale = half_at(block) * float(2 * ((high >> 12) & 7) + 1);
      const float delta = (high & 0x8000) ? -kDelta : kDelta;
      const int grid_index = quants[4 * block32 + group8] |
          (((high >> (3 * group8)) & 7) << 8);
      const auto value = static_cast<std::int8_t>(
          (iq_tables::iq1s_grid[grid_index] >> (8 * element)) & 255);
      return scale * (static_cast<float>(value) + delta);
    }
    case QuantFormat::kIQ1_M: {
      static constexpr float kDelta = 0.125f;
      const std::uint8_t* quants = block;
      const std::uint8_t* high = block + 32;
      const std::uint8_t* scale_bytes = block + 48;
      const std::uint16_t scales[4] = {
          u16_at(scale_bytes), u16_at(scale_bytes + 2),
          u16_at(scale_bytes + 4), u16_at(scale_bytes + 6)};
      const std::uint16_t scale_bits =
          static_cast<std::uint16_t>((scales[0] >> 12) |
          ((scales[1] >> 8) & 0x00f0) |
          ((scales[2] >> 4) & 0x0f00) | (scales[3] & 0xf000));
      const int block32 = column >> 5;
      const int position = column & 31;
      const int group8 = position >> 3;
      const int element = position & 7;
      const std::uint8_t high_byte = high[block32 * 2 + group8 / 2];
      const int shift = (group8 & 1) == 0 ? 0 : 4;
      const int grid_index = quants[block32 * 4 + group8] |
                             (((high_byte >> shift) & 7) << 8);
      const bool negative_delta =
          (high_byte & ((group8 & 1) == 0 ? 0x08 : 0x80)) != 0;
      const int scale_word = block32 >> 1;
      const int scale_shift = 6 * (block32 & 1) +
                              (group8 < 2 ? 0 : 3);
      const int local_scale = (scales[scale_word] >> scale_shift) & 7;
      const float scale =
          fp16_to_fp32(scale_bits) * static_cast<float>(2 * local_scale + 1);
      const std::uint64_t grid = iq_tables::iq1s_grid[grid_index];
      const float value = static_cast<float>(static_cast<std::int8_t>(
          (grid >> (8 * element)) & 255));
      const float delta = negative_delta ? -kDelta : kDelta;
      return scale * (value + delta);
    }
    case QuantFormat::kTQ1_0: {
      static constexpr std::uint8_t kPow3[5] = {1, 3, 9, 27, 81};
      int encoded = 0;
      int trit = 0;
      if (column < 240) {
        const int section = column / 160;
        const int within = column % 160;
        const int stride = section == 0 ? 32 : 16;
        const int section_base = section == 0 ? 0 : 32;
        trit = within / stride;
        encoded = block[section_base + (within % stride)];
      } else {
        const int within = column - 240;
        trit = within / 4;
        encoded = block[48 + (within & 3)];
      }
      const std::uint8_t rotated =
          static_cast<std::uint8_t>(encoded * kPow3[trit]);
      const int quant = (static_cast<unsigned>(rotated) * 3) >> 8;
      return half_at(block + 52) * static_cast<float>(quant - 1);
    }
    case QuantFormat::kTQ2_0: {
      const int half = column >> 7;
      const int position = column & 127;
      const int group = position >> 5;
      const int lane = position & 31;
      const int quant =
          (block[32 * half + lane] >> (2 * group)) & 3;
      return half_at(block + 64) * (quant - 1);
    }
  }
  return 0.0f;
}

}  // namespace

float gguf_dequant_element(QuantFormat format, const std::uint8_t* block,
                           int column) {
  return dequant(format, block, column);
}

void gguf_dequant_block_ref(QuantFormat format, const std::uint8_t* block,
                            float* values) {
  long long block_size = 0;
  (void)gguf_format_info(format, &block_size, nullptr);
  for (int column = 0; column < block_size; ++column) {
    values[column] = dequant(format, block, column);
  }
}

bool gguf_format_info(QuantFormat format, long long* block_size,
                      std::size_t* block_bytes) {
  long long size = 0;
  std::size_t bytes = 0;
  switch (format) {
    case QuantFormat::kQ1_0: size = 128; bytes = 18; break;
    case QuantFormat::kQ2_0: size = 64; bytes = 18; break;
    case QuantFormat::kQ8_0: size = 32; bytes = 34; break;
    case QuantFormat::kQ4_0: size = 32; bytes = 18; break;
    case QuantFormat::kQ4_1: size = 32; bytes = 20; break;
    case QuantFormat::kQ5_0: size = 32; bytes = 22; break;
    case QuantFormat::kQ5_1: size = 32; bytes = 24; break;
    case QuantFormat::kU4B8: size = 128; bytes = 66; break;
    case QuantFormat::kU4: size = 128; bytes = 68; break;
    case QuantFormat::kHQQ: size = 64; bytes = 36; break;
    case QuantFormat::kFP8E4M3: size = 32; bytes = 34; break;
    case QuantFormat::kFP8E5M2: size = 32; bytes = 34; break;
    case QuantFormat::kFP8Block: size = 128; bytes = 130; break;
    case QuantFormat::kFP8Raw: size = 128; bytes = 128; break;
    case QuantFormat::kFP4E2M1: size = 32; bytes = 18; break;
    case QuantFormat::kMXFP8: size = 32; bytes = 33; break;
    case QuantFormat::kNVFP4: size = 64; bytes = 36; break;
    case QuantFormat::kMXFP4: size = 32; bytes = 17; break;
    case QuantFormat::kMXFP6E3M2: size = 32; bytes = 25; break;
    case QuantFormat::kMXFP6E2M3: size = 32; bytes = 25; break;
    case QuantFormat::kBitnet: size = 32; bytes = 10; break;
    case QuantFormat::kQ2_K: size = 256; bytes = 84; break;
    case QuantFormat::kQ3_K: size = 256; bytes = 110; break;
    case QuantFormat::kQ4_K: size = 256; bytes = 144; break;
    case QuantFormat::kQ5_K: size = 256; bytes = 176; break;
    case QuantFormat::kQ6_K: size = 256; bytes = 210; break;
    case QuantFormat::kIQ4_NL: size = 32; bytes = 18; break;
    case QuantFormat::kIQ4_XS: size = 256; bytes = 136; break;
    case QuantFormat::kIQ2_XXS: size = 256; bytes = 66; break;
    case QuantFormat::kIQ2_XS: size = 256; bytes = 74; break;
    case QuantFormat::kIQ3_XXS: size = 256; bytes = 98; break;
    case QuantFormat::kIQ3_S: size = 256; bytes = 110; break;
    case QuantFormat::kIQ2_S: size = 256; bytes = 82; break;
    case QuantFormat::kIQ1_S: size = 256; bytes = 50; break;
    case QuantFormat::kIQ1_M: size = 256; bytes = 56; break;
    case QuantFormat::kTQ1_0: size = 256; bytes = 54; break;
    case QuantFormat::kTQ2_0: size = 256; bytes = 66; break;
  }
  if (size == 0) return false;
  if (block_size != nullptr) *block_size = size;
  if (block_bytes != nullptr) *block_bytes = bytes;
  return true;
}

void gguf_unpack_ref(QuantFormat format, const void* packed, long long n,
                     long long k, float* weights) {
  long long block_size = 0;
  std::size_t block_bytes = 0;
  (void)gguf_format_info(format, &block_size, &block_bytes);
  const long long blocks_per_row = k / block_size;
  const auto* bytes = static_cast<const std::uint8_t*>(packed);
  for (long long row = 0; row < n; ++row) {
    for (long long block = 0; block < blocks_per_row; ++block) {
      const std::uint8_t* source =
          bytes + (row * blocks_per_row + block) * block_bytes;
      for (int column = 0; column < block_size; ++column) {
        weights[row * k + block * block_size + column] =
            dequant(format, source, column);
      }
    }
  }
}

void gguf_gemv_ref(QuantFormat format, const void* packed, const float* x,
                   float* y, long long n, long long k) {
  long long block_size = 0;
  std::size_t block_bytes = 0;
  (void)gguf_format_info(format, &block_size, &block_bytes);
  const long long blocks_per_row = k / block_size;
  const auto* bytes = static_cast<const std::uint8_t*>(packed);
  threading::parallel_ranges(n, 32, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      double accumulator = 0.0;
      for (long long block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* source =
            bytes + (row * blocks_per_row + block) * block_bytes;
        for (int column = 0; column < block_size; ++column) {
          accumulator += dequant(format, source, column) *
                         x[block * block_size + column];
        }
      }
      y[row] = static_cast<float>(accumulator);
    }
  });
}

}  // namespace quixicore_cpu::quant
