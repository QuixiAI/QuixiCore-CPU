#include "kernels/quantization/gguf_pack_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "kernels/common/fp16.h"
#include "kernels/quantization/gguf_ref.h"

namespace quixicore_cpu::quant {
namespace {

constexpr std::int8_t kFp4Values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

void put_half(std::uint8_t* destination, float value) {
  const std::uint16_t bits = fp32_to_fp16(value);
  std::memcpy(destination, &bits, sizeof(bits));
}

void put_u32(std::uint8_t* destination, std::uint32_t value) {
  std::memcpy(destination, &value, sizeof(value));
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

bool finite_block(const float* values, int count) {
  for (int index = 0; index < count; ++index) {
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
                std::uint8_t* destination) {
  switch (format) {
    case QuantFormat::kQ1_0: pack_q1_0(source, destination); break;
    case QuantFormat::kQ2_0: pack_q2_0(source, destination); break;
    case QuantFormat::kQ4_1: pack_q4_1(source, destination); break;
    case QuantFormat::kQ5_0: pack_q5_0(source, destination); break;
    case QuantFormat::kQ5_1: pack_q5_1(source, destination); break;
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
    case QuantFormat::kMXFP4:
    case QuantFormat::kNVFP4:
    case QuantFormat::kTQ1_0:
      return true;
    default:
      return false;
  }
}

bool gguf_pack_ref(QuantFormat format, const float* weights, long long n,
                   long long k, void* packed) {
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!gguf_pack_supported(format) ||
      !gguf_format_info(format, &block_size, &block_bytes)) {
    return false;
  }
  const long long blocks_per_row = k / block_size;
  auto* bytes = static_cast<std::uint8_t*>(packed);
  for (long long row = 0; row < n; ++row) {
    for (long long block = 0; block < blocks_per_row; ++block) {
      const float* source = weights + row * k + block * block_size;
      if (!finite_block(source, static_cast<int>(block_size))) return false;
      pack_block(format, source,
                 bytes + (row * blocks_per_row + block) * block_bytes);
    }
  }
  return true;
}

}  // namespace quixicore_cpu::quant
