#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>

#include "quixicore_cpu/qgemm.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/validation.h"
#include "kernels/quantization/qgemm_canonical_isa.h"
#include "kernels/utils/float_storage_isa.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quant_import.h"
#include "quixicore_cpu/quantization.h"
#include "quixicore_cpu/threading.h"
#include "src/memory/workspace_internal.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

// M=16 remains one tile while prefill rows reuse each decoded weight block in
// 32-row chunks. This bounds stack state to 2 KiB at the widest row panel and
// halves packed decode work at M=128 relative to the correctness-first M16
// schedule.
constexpr long long kMTile = 32;
constexpr long long kKTile = 32;
constexpr long long kMaximumRowTile = 16;
constexpr long long kAccumulatorCount = 4;
constexpr long long kTypedPanelGroup = 4;
constexpr long long kDualFp8PanelGroup = 8;
constexpr long long kDualFp8GemmPanelGroup = 8;
#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
constexpr long long kDualAvx2PanelGroup = 4;
#endif
constexpr std::size_t kMissingOffset = std::numeric_limits<std::size_t>::max();

bool known_storage(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

template <class T>
const T* prepared_table(const std::uint8_t* panel, std::size_t offset) {
  return offset == kMissingOffset ? nullptr
                                  : reinterpret_cast<const T*>(panel + offset);
}

std::uint16_t little_u16(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(
      bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8));
}

float e8m0_decode(std::uint8_t code) {
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

template <Float8Format Format>
float fp8_decode_fast(std::uint8_t code) {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = float8_decode(static_cast<std::uint8_t>(index), Format);
    }
    return values;
  }();
  return table[code];
}

std::uint8_t nibble(const std::uint8_t* bytes, long long element) {
  return static_cast<std::uint8_t>((bytes[element / 2] >> (4 * (element & 1))) &
                                   0x0f);
}

std::size_t scale_index(const QuantTensorMetadata& metadata,
                        const int* group_index, long long row,
                        long long column) {
  switch (metadata.scale_mode) {
    case QuantScaleMode::kTensor:
      return 0;
    case QuantScaleMode::kRow:
    case QuantScaleMode::kChannel:
      return static_cast<std::size_t>(row);
    case QuantScaleMode::kBlock: {
      const long long groups = metadata.logical_columns / metadata.group_size;
      return static_cast<std::size_t>((row / metadata.scale_domain_rows) *
                                          groups +
                                      column / metadata.group_size);
    }
    case QuantScaleMode::kGroup: {
      const long long groups = metadata.scale_count /
                               static_cast<std::size_t>(metadata.logical_rows);
      const long long group = group_index == nullptr
                                  ? column / metadata.group_size
                                  : group_index[column];
      return static_cast<std::size_t>(row * groups + group);
    }
    case QuantScaleMode::kNone:
    case QuantScaleMode::kMicroscaleK32:
    case QuantScaleMode::kNVFP4K16:
    case QuantScaleMode::kTurboQuantK32:
      return 0;
  }
  return 0;
}

const std::uint8_t* panel_block(const CpuPackedWeightsInfo& info,
                                const std::uint8_t* panel, long long row_panel,
                                long long block, long long lane) {
  const std::size_t index = static_cast<std::size_t>(
      (row_panel * info.blocks_per_row + block) * info.row_tile + lane);
  return panel + index * info.block_bytes;
}

template <CanonicalQuantLayout Layout>
void decode_prepared_chunk(const CpuPackedWeightsInfo& info,
                           const std::uint8_t* panel, long long row_panel,
                           long long block, long long lanes,
                           long long item_base, long long items,
                           long long first_weight_row, float* decoded) {
  const QuantTensorMetadata& metadata = info.quant_metadata;
  const auto* scales = prepared_table<float>(panel, info.scale_table_offset);
  const auto* zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* groups = prepared_table<int>(panel, info.group_index_offset);
  const auto* scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long first_column = block * info.block_size + item_base;
  for (long long lane = 0; lane < lanes; ++lane) {
    const long long weight_row = first_weight_row + lane;
    const auto* source = panel_block(info, panel, row_panel, block, lane);
    if constexpr (Layout == CanonicalQuantLayout::kInt4Symmetric ||
                  Layout == CanonicalQuantLayout::kUInt4Affine ||
                  Layout == CanonicalQuantLayout::kFP4E2M1 ||
                  Layout == CanonicalQuantLayout::kInt8Symmetric ||
                  Layout == CanonicalQuantLayout::kInt8Affine ||
                  Layout == CanonicalQuantLayout::kFP8E4M3FN ||
                  Layout == CanonicalQuantLayout::kFP8E5M2) {
      const std::size_t common_index =
          scale_index(metadata, groups, weight_row, first_column);
      for (long long item = 0; item < items; ++item) {
        const std::size_t side_index =
            groups == nullptr ? common_index
                              : scale_index(metadata, groups, weight_row,
                                            first_column + item);
        float value = 0.0f;
        if constexpr (Layout == CanonicalQuantLayout::kInt4Symmetric) {
          const int code = nibble(source, item_base + item);
          value = scales[side_index] *
                  static_cast<float>(code >= 8 ? code - 16 : code);
        } else if constexpr (Layout == CanonicalQuantLayout::kUInt4Affine) {
          value = scales[side_index] *
                  (static_cast<float>(nibble(source, item_base + item)) -
                   zeros[side_index]);
        } else if constexpr (Layout == CanonicalQuantLayout::kFP4E2M1) {
          value = scales[side_index] *
                  fp4_e2m1_decode(nibble(source, item_base + item));
        } else if constexpr (Layout == CanonicalQuantLayout::kInt8Symmetric ||
                             Layout == CanonicalQuantLayout::kInt8Affine) {
          const float code = static_cast<float>(
              static_cast<std::int8_t>(source[item_base + item]));
          const float zero = zeros == nullptr ? 0.0f : zeros[side_index];
          value = scales[side_index] * (code - zero);
        } else {
          constexpr Float8Format format =
              Layout == CanonicalQuantLayout::kFP8E4M3FN ? Float8Format::kE4M3FN
                                                         : Float8Format::kE5M2;
          const float scale = scales == nullptr ? 1.0f : scales[side_index];
          value = scale * fp8_decode_fast<format>(source[item_base + item]);
        }
        decoded[item * kMaximumRowTile + lane] = value;
      }
    } else if constexpr (Layout == CanonicalQuantLayout::kMXFP8E4M3E8M0) {
      const float scale = e8m0_decode(source[0]);
      for (long long item = 0; item < items; ++item) {
        decoded[item * kMaximumRowTile + lane] =
            scale * fp8_decode_fast<Float8Format::kE4M3FN>(
                        source[1 + item_base + item]);
      }
    } else if constexpr (Layout == CanonicalQuantLayout::kMXFP4E2M1E8M0) {
      const float scale = e8m0_decode(source[0]);
      for (long long item = 0; item < items; ++item) {
        decoded[item * kMaximumRowTile + lane] =
            scale * fp4_e2m1_decode(nibble(source + 1, item_base + item));
      }
    } else if constexpr (Layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
      const float scale =
          metadata.global_scale *
          fp8_decode_fast<Float8Format::kE4M3FN>(
              scale_codes[weight_row * info.blocks_per_row + block]);
      for (long long item = 0; item < items; ++item) {
        decoded[item * kMaximumRowTile + lane] =
            scale * fp4_e2m1_decode(nibble(source, item_base + item));
      }
    } else if constexpr (Layout == CanonicalQuantLayout::kBitNetTernary) {
      const float scale = f16_to_float(little_u16(source));
      for (long long item = 0; item < items; ++item) {
        const long long block_item = item_base + item;
        const std::uint8_t code = static_cast<std::uint8_t>(
            (source[2 + block_item / 4] >> (2 * (block_item & 3))) & 3);
        decoded[item * kMaximumRowTile + lane] =
            scale * static_cast<float>(static_cast<int>(code) - 1);
      }
    }
  }
}

std::size_t tensor_scale_index(const CanonicalQuantTensor& tensor,
                               long long row, long long column) {
  const int* groups =
      tensor.group_index.empty() ? nullptr : tensor.group_index.data();
  return scale_index(tensor.metadata, groups, row, column);
}

float tensor_value(const CanonicalQuantTensor& tensor, long long row,
                   long long column) {
  const QuantTensorMetadata& metadata = tensor.metadata;
  const long long columns = metadata.logical_columns;
  const std::size_t element = static_cast<std::size_t>(row * columns + column);
  const std::size_t side_index = tensor_scale_index(tensor, row, column);
  switch (metadata.layout) {
    case CanonicalQuantLayout::kInt4Symmetric: {
      const int code = nibble(tensor.data.data(), element);
      return tensor.scales[side_index] *
             static_cast<float>(code >= 8 ? code - 16 : code);
    }
    case CanonicalQuantLayout::kUInt4Affine:
      return tensor.scales[side_index] *
             (static_cast<float>(nibble(tensor.data.data(), element)) -
              tensor.zero_points[side_index]);
    case CanonicalQuantLayout::kFP4E2M1:
      return tensor.scales[side_index] *
             fp4_e2m1_decode(nibble(tensor.data.data(), element));
    case CanonicalQuantLayout::kInt8Symmetric:
    case CanonicalQuantLayout::kInt8Affine: {
      const float code =
          static_cast<float>(static_cast<std::int8_t>(tensor.data[element]));
      const float zero =
          tensor.zero_points.empty() ? 0.0f : tensor.zero_points[side_index];
      return tensor.scales[side_index] * (code - zero);
    }
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2: {
      const Float8Format format =
          metadata.layout == CanonicalQuantLayout::kFP8E4M3FN
              ? Float8Format::kE4M3FN
              : Float8Format::kE5M2;
      const float scale =
          tensor.scales.empty() ? 1.0f : tensor.scales[side_index];
      return scale * float8_decode(tensor.data[element], format);
    }
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
    case CanonicalQuantLayout::kMXFP4E2M1E8M0: {
      const bool fp4 = metadata.layout == CanonicalQuantLayout::kMXFP4E2M1E8M0;
      const long long block = column / 32;
      const long long item = column % 32;
      const std::size_t block_bytes = fp4 ? 17 : 33;
      const auto* source =
          tensor.data.data() +
          static_cast<std::size_t>((row * (columns / 32) + block) *
                                   static_cast<long long>(block_bytes));
      const std::uint8_t code =
          fp4 ? nibble(source + 1, item) : source[1 + item];
      return e8m0_decode(source[0]) *
             (fp4 ? fp4_e2m1_decode(code)
                  : float8_decode(code, Float8Format::kE4M3FN));
    }
    case CanonicalQuantLayout::kNVFP4E2M1E4M3: {
      const long long blocks = columns / 16;
      const long long block = column / 16;
      const float local = float8_decode(
          tensor.scale_codes[static_cast<std::size_t>(row * blocks + block)],
          Float8Format::kE4M3FN);
      return metadata.global_scale * local *
             fp4_e2m1_decode(nibble(tensor.data.data(), element));
    }
    case CanonicalQuantLayout::kBitNetTernary: {
      const long long block = column / 32;
      const long long item = column % 32;
      const auto* source =
          tensor.data.data() +
          static_cast<std::size_t>((row * (columns / 32) + block) * 10);
      const std::uint8_t code = static_cast<std::uint8_t>(
          (source[2 + item / 4] >> (2 * (item & 3))) & 3);
      return f16_to_float(little_u16(source)) *
             static_cast<float>(static_cast<int>(code) - 1);
    }
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return 0.0f;
  }
  return 0.0f;
}

float tensor_scale(const CanonicalQuantTensor& tensor, long long row,
                   long long column) {
  return tensor.scales.empty()
             ? 1.0f
             : tensor.scales[tensor_scale_index(tensor, row, column)];
}

float tensor_zero(const CanonicalQuantTensor& tensor, long long row,
                  long long column) {
  return tensor.zero_points.empty()
             ? 0.0f
             : tensor.zero_points[tensor_scale_index(tensor, row, column)];
}

bool scale_interval_is_constant(const QuantTensorMetadata& metadata,
                                long long interval) {
  if (metadata.scale_mode != QuantScaleMode::kGroup &&
      metadata.scale_mode != QuantScaleMode::kBlock) {
    return true;
  }
  return metadata.group_size >= interval && metadata.group_size % interval == 0;
}

template <FloatStorageType Type>
float input_value(const void* input, std::size_t index) {
  if constexpr (Type == FloatStorageType::kF32) {
    return static_cast<const float*>(input)[index];
  } else if constexpr (Type == FloatStorageType::kF16) {
    return f16_to_float(static_cast<const std::uint16_t*>(input)[index]);
  } else {
    return bf16_to_float(static_cast<const std::uint16_t*>(input)[index]);
  }
}

void store_output(FloatStorageOutput output, std::size_t index, float value) {
  switch (output.type) {
    case FloatStorageType::kF32:
      static_cast<float*>(output.data)[index] = value;
      break;
    case FloatStorageType::kF16:
      static_cast<std::uint16_t*>(output.data)[index] = float_to_f16(value);
      break;
    case FloatStorageType::kBF16:
      static_cast<std::uint16_t*>(output.data)[index] = float_to_bf16(value);
      break;
  }
}

bool known_linear_activation(LinearActivation activation) {
  return activation == LinearActivation::kNone ||
         activation == LinearActivation::kGeluErf ||
         activation == LinearActivation::kGeluTanh ||
         activation == LinearActivation::kSilu ||
         activation == LinearActivation::kRelu2;
}

float projection_activation(float value, LinearActivation activation) {
  constexpr float kInvSqrt2 = 0.7071067811865475f;
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  switch (activation) {
    case LinearActivation::kNone:
      return value;
    case LinearActivation::kGeluErf:
      return 0.5f * value * (1.0f + std::erf(value * kInvSqrt2));
    case LinearActivation::kGeluTanh:
      return 0.5f * value *
             (1.0f + std::tanh(kSqrt2OverPi *
                               (value + 0.044715f * value * value * value)));
    case LinearActivation::kSilu:
      return value / (1.0f + std::exp(-value));
    case LinearActivation::kRelu2:
      return value > 0.0f ? value * value : 0.0f;
  }
  return value;
}

#if defined(__aarch64__) || defined(_M_ARM64)
float neon_dot_i8x16_f32(int8x16_t weights, const float* input) {
  const int16x8_t low = vmovl_s8(vget_low_s8(weights));
  const int16x8_t high = vmovl_s8(vget_high_s8(weights));
  float32x4_t sum0 =
      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))), vld1q_f32(input));
  float32x4_t sum1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                               vld1q_f32(input + 8));
  sum0 = vfmaq_f32(sum0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                   vld1q_f32(input + 4));
  sum1 = vfmaq_f32(sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
                   vld1q_f32(input + 12));
  return vaddvq_f32(vaddq_f32(sum0, sum1));
}

int neon_dot_i8x16_i8(int8x16_t first, int8x16_t second) {
  const int16x8_t low = vmull_s8(vget_low_s8(first), vget_low_s8(second));
  const int16x8_t high = vmull_s8(vget_high_s8(first), vget_high_s8(second));
  return vaddvq_s32(vaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high)));
}

float neon_sum_f32(const float* values, long long count) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  long long item = 0;
  for (; item + 7 < count; item += 8) {
    sum0 = vaddq_f32(sum0, vld1q_f32(values + item));
    sum1 = vaddq_f32(sum1, vld1q_f32(values + item + 4));
  }
  float total = vaddvq_f32(vaddq_f32(sum0, sum1));
  for (; item < count; ++item) total += values[item];
  return total;
}

void neon_interleaved_nibbles(const std::uint8_t* packed, int8x16_t* first,
                              int8x16_t* second, std::uint8_t offset,
                              const int8x16_t* table = nullptr) {
  const uint8x16_t bytes = vld1q_u8(packed);
  const uint8x16_t mask = vdupq_n_u8(15);
  const uint8x16_t low = vandq_u8(bytes, mask);
  const uint8x16_t high = vshrq_n_u8(bytes, 4);
  uint8x16_t codes0 = vzip1q_u8(low, high);
  uint8x16_t codes1 = vzip2q_u8(low, high);
  if (table != nullptr) {
    *first = vqtbl1q_s8(*table, codes0);
    *second = vqtbl1q_s8(*table, codes1);
  } else {
    const uint8x16_t bias = vdupq_n_u8(offset);
    if (offset == 0) {
      *first = vreinterpretq_s8_u8(codes0);
      *second = vreinterpretq_s8_u8(codes1);
    } else {
      // Canonical signed INT4 is a two's-complement nibble. XOR/subtract
      // sign-extends it without a lane-by-lane branch.
      *first = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(codes0, bias), bias));
      *second = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(codes1, bias), bias));
    }
  }
}

int8x16_t neon_ordered_nibbles16(const std::uint8_t* packed,
                                 const int8x16_t& table) {
  const uint8x8_t bytes = vld1_u8(packed);
  const uint8x8_t low = vand_u8(bytes, vdup_n_u8(15));
  const uint8x8_t high = vshr_n_u8(bytes, 4);
  const uint8x8x2_t zipped = vzip_u8(low, high);
  return vqtbl1q_s8(table, vcombine_u8(zipped.val[0], zipped.val[1]));
}

template <Float8Format Format>
float32x4_t neon_decode_fp8x4(uint32x4_t codes) {
  const uint32x4_t sign = vshlq_n_u32(vandq_u32(codes, vdupq_n_u32(0x80)), 24);
  constexpr int mantissa_bits = Format == Float8Format::kE4M3FN ? 3 : 2;
  constexpr int exponent_mask = Format == Float8Format::kE4M3FN ? 15 : 31;
  constexpr int exponent_bias_delta =
      Format == Float8Format::kE4M3FN ? 120 : 112;
  constexpr float subnormal_scale =
      Format == Float8Format::kE4M3FN ? 1.0f / 512.0f : 1.0f / 65536.0f;
  const uint32x4_t exponent =
      vandq_u32(vshrq_n_u32(codes, mantissa_bits), vdupq_n_u32(exponent_mask));
  const uint32x4_t mantissa =
      vandq_u32(codes, vdupq_n_u32((1U << mantissa_bits) - 1));
  const uint32x4_t normal_bits = vorrq_u32(
      sign,
      vorrq_u32(vshlq_n_u32(
                    vaddq_u32(exponent, vdupq_n_u32(exponent_bias_delta)), 23),
                vshlq_n_u32(mantissa, 23 - mantissa_bits)));
  const float32x4_t subnormal_magnitude =
      vmulq_n_f32(vcvtq_f32_u32(mantissa), subnormal_scale);
  const float32x4_t subnormal = vreinterpretq_f32_u32(
      veorq_u32(vreinterpretq_u32_f32(subnormal_magnitude), sign));
  float32x4_t decoded =
      vbslq_f32(vceqq_u32(exponent, vdupq_n_u32(0)), subnormal,
                vreinterpretq_f32_u32(normal_bits));
  if constexpr (Format == Float8Format::kE4M3FN) {
    const uint32x4_t special = vandq_u32(vceqq_u32(exponent, vdupq_n_u32(15)),
                                         vceqq_u32(mantissa, vdupq_n_u32(7)));
    decoded = vbslq_f32(
        special,
        vreinterpretq_f32_u32(vorrq_u32(sign, vdupq_n_u32(0x7fc00000))),
        decoded);
  } else {
    const uint32x4_t special = vceqq_u32(exponent, vdupq_n_u32(31));
    const uint32x4_t nonzero_mantissa =
        vmvnq_u32(vceqq_u32(mantissa, vdupq_n_u32(0)));
    const uint32x4_t special_bits = vbslq_u32(
        nonzero_mantissa, vdupq_n_u32(0x7fc00000), vdupq_n_u32(0x7f800000));
    decoded = vbslq_f32(
        special, vreinterpretq_f32_u32(vorrq_u32(sign, special_bits)), decoded);
  }
  return decoded;
}

template <Float8Format Format>
float neon_fp8_dot(const std::uint8_t* codes, const float* input,
                   long long count) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  long long item = 0;
  for (; item + 15 < count; item += 16) {
    const uint8x16_t packed = vld1q_u8(codes + item);
    const uint16x8_t low = vmovl_u8(vget_low_u8(packed));
    const uint16x8_t high = vmovl_u8(vget_high_u8(packed));
    sum0 =
        vfmaq_f32(sum0, neon_decode_fp8x4<Format>(vmovl_u16(vget_low_u16(low))),
                  vld1q_f32(input + item));
    sum1 = vfmaq_f32(sum1,
                     neon_decode_fp8x4<Format>(vmovl_u16(vget_high_u16(low))),
                     vld1q_f32(input + item + 4));
    sum2 = vfmaq_f32(sum2,
                     neon_decode_fp8x4<Format>(vmovl_u16(vget_low_u16(high))),
                     vld1q_f32(input + item + 8));
    sum3 = vfmaq_f32(sum3,
                     neon_decode_fp8x4<Format>(vmovl_u16(vget_high_u16(high))),
                     vld1q_f32(input + item + 12));
  }
  float total =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  for (; item < count; ++item)
    total += fp8_decode_fast<Format>(codes[item]) * input[item];
  return total;
}

template <Float8Format WeightFormat, Float8Format ActivationFormat>
float neon_fp8_pair_dot(const std::uint8_t* weights,
                        const std::uint8_t* activations, long long count) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  long long item = 0;
  for (; item + 15 < count; item += 16) {
    const uint8x16_t packed_weights = vld1q_u8(weights + item);
    const uint8x16_t packed_activations = vld1q_u8(activations + item);
    const uint16x8_t weight_low = vmovl_u8(vget_low_u8(packed_weights));
    const uint16x8_t weight_high = vmovl_u8(vget_high_u8(packed_weights));
    const uint16x8_t activation_low = vmovl_u8(vget_low_u8(packed_activations));
    const uint16x8_t activation_high =
        vmovl_u8(vget_high_u8(packed_activations));
    sum0 = vfmaq_f32(
        sum0,
        neon_decode_fp8x4<WeightFormat>(vmovl_u16(vget_low_u16(weight_low))),
        neon_decode_fp8x4<ActivationFormat>(
            vmovl_u16(vget_low_u16(activation_low))));
    sum1 = vfmaq_f32(
        sum1,
        neon_decode_fp8x4<WeightFormat>(vmovl_u16(vget_high_u16(weight_low))),
        neon_decode_fp8x4<ActivationFormat>(
            vmovl_u16(vget_high_u16(activation_low))));
    sum2 = vfmaq_f32(
        sum2,
        neon_decode_fp8x4<WeightFormat>(vmovl_u16(vget_low_u16(weight_high))),
        neon_decode_fp8x4<ActivationFormat>(
            vmovl_u16(vget_low_u16(activation_high))));
    sum3 = vfmaq_f32(
        sum3,
        neon_decode_fp8x4<WeightFormat>(vmovl_u16(vget_high_u16(weight_high))),
        neon_decode_fp8x4<ActivationFormat>(
            vmovl_u16(vget_high_u16(activation_high))));
  }
  float total =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  for (; item < count; ++item) {
    total += fp8_decode_fast<WeightFormat>(weights[item]) *
             fp8_decode_fast<ActivationFormat>(activations[item]);
  }
  return total;
}

template <Float8Format Format>
float neon_fp8_decoded_dot(const std::uint8_t* codes,
                           const float32x4_t* decoded, long long count) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  long long item = 0;
  for (; item + 7 < count; item += 8) {
    const uint8x8_t packed = vld1_u8(codes + item);
    const uint16x8_t wide = vmovl_u8(packed);
    sum0 = vfmaq_f32(sum0,
                     neon_decode_fp8x4<Format>(vmovl_u16(vget_low_u16(wide))),
                     decoded[item / 4]);
    sum1 = vfmaq_f32(sum1,
                     neon_decode_fp8x4<Format>(vmovl_u16(vget_high_u16(wide))),
                     decoded[item / 4 + 1]);
  }
  float total = vaddvq_f32(vaddq_f32(sum0, sum1));
  for (; item < count; ++item) {
    alignas(16) float values[4];
    vst1q_f32(values, decoded[item / 4]);
    total += fp8_decode_fast<Format>(codes[item]) * values[item & 3];
  }
  return total;
}

void neon_unpack_ternary32(const std::uint8_t* packed, int8x16_t* first,
                           int8x16_t* second) {
  const uint8x8_t bytes = vld1_u8(packed);
  const uint8x8_t mask = vdup_n_u8(3);
  const uint8x8_t q0 = vand_u8(bytes, mask);
  const uint8x8_t q1 = vand_u8(vshr_n_u8(bytes, 2), mask);
  const uint8x8_t q2 = vand_u8(vshr_n_u8(bytes, 4), mask);
  const uint8x8_t q3 = vshr_n_u8(bytes, 6);
  const uint8x8x2_t q01 = vzip_u8(q0, q1);
  const uint8x8x2_t q23 = vzip_u8(q2, q3);
  const uint16x4x2_t low = vzip_u16(vreinterpret_u16_u8(q01.val[0]),
                                    vreinterpret_u16_u8(q23.val[0]));
  const uint16x4x2_t high = vzip_u16(vreinterpret_u16_u8(q01.val[1]),
                                     vreinterpret_u16_u8(q23.val[1]));
  const uint8x16_t codes0 = vcombine_u8(vreinterpret_u8_u16(low.val[0]),
                                        vreinterpret_u8_u16(low.val[1]));
  const uint8x16_t codes1 = vcombine_u8(vreinterpret_u8_u16(high.val[0]),
                                        vreinterpret_u8_u16(high.val[1]));
  const uint8x16_t one = vdupq_n_u8(1);
  *first = vreinterpretq_s8_u8(vsubq_u8(codes0, one));
  *second = vreinterpretq_s8_u8(vsubq_u8(codes1, one));
}
#endif

template <CanonicalQuantLayout Layout>
bool prepared_row_block_dot_neon(const CpuPackedWeightsInfo& info,
                                 const std::uint8_t* panel, const float* input,
                                 long long weight_row, long long block,
                                 float* result, long long input_origin = 0) {
#if defined(__aarch64__) || defined(_M_ARM64)
  const QuantTensorMetadata& metadata = info.quant_metadata;
  const auto* groups = prepared_table<int>(panel, info.group_index_offset);
  if (groups != nullptr) return false;
  const auto* scales = prepared_table<float>(panel, info.scale_table_offset);
  const auto* zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long row_panel = weight_row / info.row_tile;
  const long long lane = weight_row % info.row_tile;
  const auto* source = panel_block(info, panel, row_panel, block, lane);
  const long long first_column = block * info.block_size;
  const float* input_block = input + first_column - input_origin;
  const std::size_t side_index =
      scale_index(metadata, nullptr, weight_row, first_column);
  float total = 0.0f;
  if constexpr (Layout == CanonicalQuantLayout::kInt4Symmetric ||
                Layout == CanonicalQuantLayout::kUInt4Affine ||
                Layout == CanonicalQuantLayout::kFP4E2M1) {
    if ((info.block_size % 32) != 0) return false;
    const int8x16_t fp4_table = {0, 1,  2,  3,  4,  6,  8,  12,
                                 0, -1, -2, -3, -4, -6, -8, -12};
    for (long long item = 0; item < info.block_size; item += 32) {
      int8x16_t values0;
      int8x16_t values1;
      if constexpr (Layout == CanonicalQuantLayout::kInt4Symmetric) {
        neon_interleaved_nibbles(source + item / 2, &values0, &values1, 8);
      } else if constexpr (Layout == CanonicalQuantLayout::kUInt4Affine) {
        neon_interleaved_nibbles(source + item / 2, &values0, &values1, 0);
      } else {
        neon_interleaved_nibbles(source + item / 2, &values0, &values1, 0,
                                 &fp4_table);
      }
      const float* values = input_block + item;
      float dot = neon_dot_i8x16_f32(values0, values) +
                  neon_dot_i8x16_f32(values1, values + 16);
      if constexpr (Layout == CanonicalQuantLayout::kUInt4Affine) {
        dot -= zeros[side_index] * neon_sum_f32(values, 32);
      } else if constexpr (Layout == CanonicalQuantLayout::kFP4E2M1) {
        dot *= 0.5f;
      }
      total += scales[side_index] * dot;
    }
  } else if constexpr (Layout == CanonicalQuantLayout::kInt8Symmetric ||
                       Layout == CanonicalQuantLayout::kInt8Affine) {
    float dot = 0.0f;
    long long item = 0;
    for (; item + 15 < info.block_size; item += 16) {
      dot += neon_dot_i8x16_f32(
          vld1q_s8(reinterpret_cast<const std::int8_t*>(source + item)),
          input_block + item);
    }
    for (; item < info.block_size; ++item) {
      dot += static_cast<float>(static_cast<std::int8_t>(source[item])) *
             input_block[item];
    }
    if constexpr (Layout == CanonicalQuantLayout::kInt8Affine) {
      dot -= zeros[side_index] * neon_sum_f32(input_block, info.block_size);
    }
    total = scales[side_index] * dot;
  } else if constexpr (Layout == CanonicalQuantLayout::kFP8E4M3FN ||
                       Layout == CanonicalQuantLayout::kFP8E5M2 ||
                       Layout == CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    constexpr Float8Format format = Layout == CanonicalQuantLayout::kFP8E5M2
                                        ? Float8Format::kE5M2
                                        : Float8Format::kE4M3FN;
    const long long count = info.block_size;
    const bool microscale = Layout == CanonicalQuantLayout::kMXFP8E4M3E8M0;
    const std::uint8_t* codes = source + (microscale ? 1 : 0);
    const float scale = microscale
                            ? e8m0_decode(source[0])
                            : (scales == nullptr ? 1.0f : scales[side_index]);
    total = scale * neon_fp8_dot<format>(codes, input_block, count);
  } else if constexpr (Layout == CanonicalQuantLayout::kMXFP4E2M1E8M0 ||
                       Layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    const int8x16_t fp4_table = {0, 1,  2,  3,  4,  6,  8,  12,
                                 0, -1, -2, -3, -4, -6, -8, -12};
    const std::uint8_t* codes =
        source + (Layout == CanonicalQuantLayout::kMXFP4E2M1E8M0 ? 1 : 0);
    int8x16_t values0;
    int8x16_t values1;
    if constexpr (Layout == CanonicalQuantLayout::kMXFP4E2M1E8M0) {
      neon_interleaved_nibbles(codes, &values0, &values1, 0, &fp4_table);
      const float dot = neon_dot_i8x16_f32(values0, input_block) +
                        neon_dot_i8x16_f32(values1, input_block + 16);
      total = 0.5f * e8m0_decode(source[0]) * dot;
    } else {
      const uint8x8_t packed = vld1_u8(codes);
      const uint8x8_t low = vand_u8(packed, vdup_n_u8(15));
      const uint8x8_t high = vshr_n_u8(packed, 4);
      const uint8x8x2_t zipped = vzip_u8(low, high);
      const uint8x16_t ordered = vcombine_u8(zipped.val[0], zipped.val[1]);
      values0 = vqtbl1q_s8(fp4_table, ordered);
      const float scale =
          metadata.global_scale *
          fp8_decode_fast<Float8Format::kE4M3FN>(
              scale_codes[weight_row * info.blocks_per_row + block]);
      total = 0.5f * scale * neon_dot_i8x16_f32(values0, input_block);
    }
  } else if constexpr (Layout == CanonicalQuantLayout::kBitNetTernary) {
    int8x16_t decoded0;
    int8x16_t decoded1;
    neon_unpack_ternary32(source + 2, &decoded0, &decoded1);
    total = f16_to_float(little_u16(source)) *
            (neon_dot_i8x16_f32(decoded0, input_block) +
             neon_dot_i8x16_f32(decoded1, input_block + 16));
  }
  *result = total;
  return true;
#else
  (void)info;
  (void)panel;
  (void)input;
  (void)weight_row;
  (void)block;
  (void)result;
  (void)input_origin;
  return false;
#endif
}

template <FloatStorageType Type>
bool simd_decode_storage_block(const void* input, long long first,
                               long long count, float* decoded) {
#if defined(__aarch64__) || defined(_M_ARM64)
  const auto* source = static_cast<const std::uint16_t*>(input) + first;
  long long item = 0;
  if constexpr (Type == FloatStorageType::kF16) {
    for (; item + 7 < count; item += 8) {
      const float16x8_t packed =
          vreinterpretq_f16_u16(vld1q_u16(source + item));
      vst1q_f32(decoded + item, vcvt_f32_f16(vget_low_f16(packed)));
      vst1q_f32(decoded + item + 4, vcvt_f32_f16(vget_high_f16(packed)));
    }
  } else if constexpr (Type == FloatStorageType::kBF16) {
    for (; item + 7 < count; item += 8) {
      const uint16x8_t packed = vld1q_u16(source + item);
      vst1q_f32(decoded + item, vreinterpretq_f32_u32(vshlq_n_u32(
                                    vmovl_u16(vget_low_u16(packed)), 16)));
      vst1q_f32(decoded + item + 4, vreinterpretq_f32_u32(vshlq_n_u32(
                                        vmovl_u16(vget_high_u16(packed)), 16)));
    }
  } else {
    return false;
  }
  for (; item < count; ++item) {
    if constexpr (Type == FloatStorageType::kF16)
      decoded[item] = f16_to_float(source[item]);
    else
      decoded[item] = bf16_to_float(source[item]);
  }
  return true;
#endif
#if defined(__x86_64__) || defined(_M_X64)
  if constexpr (Type == FloatStorageType::kF16) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_F16C)
    const auto* source = static_cast<const std::uint16_t*>(input) + first;
    if (cpu_features().f16c) {
      float_storage_detail::f16_to_f32_f16c(source, decoded, 0, count);
      return true;
    }
#endif
  } else if constexpr (Type == FloatStorageType::kBF16) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_AVX2)
    const auto* source = static_cast<const std::uint16_t*>(input) + first;
    if (cpu_features().avx2) {
      float_storage_detail::bf16_to_f32_avx2(source, decoded, 0, count);
      return true;
    }
#endif
  }
#endif
  (void)input;
  (void)first;
  (void)count;
  (void)decoded;
  return false;
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
float prepared_row_block_dot(const CpuPackedWeightsInfo& info,
                             const std::uint8_t* panel, const void* input,
                             const int* order, long long weight_row,
                             long long block) {
  if constexpr (Type == FloatStorageType::kF32) {
    float direct = 0.0f;
    if (order == nullptr && prepared_row_block_dot_neon<Layout>(
                                info, panel, static_cast<const float*>(input),
                                weight_row, block, &direct)) {
      return direct;
    }
  } else {
    const long long first_column = block * info.block_size;
    alignas(64) float decoded[kKTile];
    float direct = 0.0f;
    if (order == nullptr && info.block_size <= kKTile &&
        simd_decode_storage_block<Type>(input, first_column, info.block_size,
                                        decoded) &&
        prepared_row_block_dot_neon<Layout>(info, panel, decoded, weight_row,
                                            block, &direct, first_column)) {
      return direct;
    }
  }
  const QuantTensorMetadata& metadata = info.quant_metadata;
  const auto* scales = prepared_table<float>(panel, info.scale_table_offset);
  const auto* zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* groups = prepared_table<int>(panel, info.group_index_offset);
  const auto* scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long row_panel = weight_row / info.row_tile;
  const long long lane = weight_row % info.row_tile;
  const auto* source = panel_block(info, panel, row_panel, block, lane);
  const long long first_column = block * info.block_size;
  float sums[4] = {};
  if constexpr (Layout == CanonicalQuantLayout::kInt4Symmetric ||
                Layout == CanonicalQuantLayout::kUInt4Affine ||
                Layout == CanonicalQuantLayout::kFP4E2M1 ||
                Layout == CanonicalQuantLayout::kInt8Symmetric ||
                Layout == CanonicalQuantLayout::kInt8Affine ||
                Layout == CanonicalQuantLayout::kFP8E4M3FN ||
                Layout == CanonicalQuantLayout::kFP8E5M2) {
    const std::size_t common_index =
        scale_index(metadata, groups, weight_row, first_column);
    for (long long item = 0; item < info.block_size; ++item) {
      const long long packed_column = first_column + item;
      const long long logical_column =
          order == nullptr ? packed_column : order[packed_column];
      const std::size_t side_index =
          groups == nullptr
              ? common_index
              : scale_index(metadata, groups, weight_row, packed_column);
      float value = 0.0f;
      if constexpr (Layout == CanonicalQuantLayout::kInt4Symmetric) {
        const int code = nibble(source, item);
        value = scales[side_index] *
                static_cast<float>(code >= 8 ? code - 16 : code);
      } else if constexpr (Layout == CanonicalQuantLayout::kUInt4Affine) {
        value = scales[side_index] *
                (static_cast<float>(nibble(source, item)) - zeros[side_index]);
      } else if constexpr (Layout == CanonicalQuantLayout::kFP4E2M1) {
        value = scales[side_index] * fp4_e2m1_decode(nibble(source, item));
      } else if constexpr (Layout == CanonicalQuantLayout::kInt8Symmetric ||
                           Layout == CanonicalQuantLayout::kInt8Affine) {
        const float code =
            static_cast<float>(static_cast<std::int8_t>(source[item]));
        const float zero = zeros == nullptr ? 0.0f : zeros[side_index];
        value = scales[side_index] * (code - zero);
      } else {
        constexpr Float8Format format =
            Layout == CanonicalQuantLayout::kFP8E4M3FN ? Float8Format::kE4M3FN
                                                       : Float8Format::kE5M2;
        const float scale = scales == nullptr ? 1.0f : scales[side_index];
        value = scale * fp8_decode_fast<format>(source[item]);
      }
      sums[item & 3] +=
          value *
          input_value<Type>(input, static_cast<std::size_t>(logical_column));
    }
  } else if constexpr (Layout == CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    const float scale = e8m0_decode(source[0]);
    for (long long item = 0; item < 32; ++item) {
      const long long packed_column = first_column + item;
      const long long logical_column =
          order == nullptr ? packed_column : order[packed_column];
      sums[item & 3] +=
          scale * fp8_decode_fast<Float8Format::kE4M3FN>(source[1 + item]) *
          input_value<Type>(input, static_cast<std::size_t>(logical_column));
    }
  } else if constexpr (Layout == CanonicalQuantLayout::kMXFP4E2M1E8M0) {
    const float scale = e8m0_decode(source[0]);
    for (long long item = 0; item < 32; ++item) {
      const long long packed_column = first_column + item;
      const long long logical_column =
          order == nullptr ? packed_column : order[packed_column];
      sums[item & 3] +=
          scale * fp4_e2m1_decode(nibble(source + 1, item)) *
          input_value<Type>(input, static_cast<std::size_t>(logical_column));
    }
  } else if constexpr (Layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    const float scale =
        metadata.global_scale *
        fp8_decode_fast<Float8Format::kE4M3FN>(
            scale_codes[weight_row * info.blocks_per_row + block]);
    for (long long item = 0; item < 16; ++item) {
      const long long packed_column = first_column + item;
      const long long logical_column =
          order == nullptr ? packed_column : order[packed_column];
      sums[item & 3] +=
          scale * fp4_e2m1_decode(nibble(source, item)) *
          input_value<Type>(input, static_cast<std::size_t>(logical_column));
    }
  } else if constexpr (Layout == CanonicalQuantLayout::kBitNetTernary) {
    const float scale = f16_to_float(little_u16(source));
    for (long long item = 0; item < 32; ++item) {
      const std::uint8_t code = static_cast<std::uint8_t>(
          (source[2 + item / 4] >> (2 * (item & 3))) & 3);
      const long long packed_column = first_column + item;
      const long long logical_column =
          order == nullptr ? packed_column : order[packed_column];
      sums[item & 3] +=
          scale * static_cast<float>(static_cast<int>(code) - 1) *
          input_value<Type>(input, static_cast<std::size_t>(logical_column));
    }
  }
  return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout>
bool dual_block_dot_neon(const CpuPackedWeightsInfo& info,
                         const std::uint8_t* panel,
                         const CanonicalQuantTensor& activation,
                         long long activation_row, long long weight_row,
                         long long block, float* result) {
#if defined(__aarch64__) || defined(_M_ARM64)
  const QuantTensorMetadata& weight_metadata = info.quant_metadata;
  const auto* weight_scales =
      prepared_table<float>(panel, info.scale_table_offset);
  const auto* weight_zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* weight_scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long row_panel = weight_row / info.row_tile;
  const long long lane = weight_row % info.row_tile;
  const auto* weight = panel_block(info, panel, row_panel, block, lane);
  const long long first_column = block * info.block_size;
  const long long activation_element =
      activation_row * activation.metadata.logical_columns + first_column;
  const std::size_t weight_side =
      scale_index(weight_metadata, nullptr, weight_row, first_column);
  const float weight_scale =
      weight_scales == nullptr ? 1.0f : weight_scales[weight_side];
  const float activation_scale =
      tensor_scale(activation, activation_row, first_column);
  const int8x16_t fp4_table = {0, 1,  2,  3,  4,  6,  8,  12,
                               0, -1, -2, -3, -4, -6, -8, -12};
  int dot = 0;
  if constexpr (WeightLayout == CanonicalQuantLayout::kInt4Symmetric &&
                ActivationLayout == CanonicalQuantLayout::kInt4Symmetric) {
    if (info.block_size % 32 != 0) return false;
    for (long long item = 0; item < info.block_size; item += 32) {
      int8x16_t weights0;
      int8x16_t weights1;
      int8x16_t inputs0;
      int8x16_t inputs1;
      neon_interleaved_nibbles(weight + item / 2, &weights0, &weights1, 8);
      neon_interleaved_nibbles(
          activation.data.data() + (activation_element + item) / 2, &inputs0,
          &inputs1, 8);
      dot += neon_dot_i8x16_i8(weights0, inputs0) +
             neon_dot_i8x16_i8(weights1, inputs1);
    }
    *result = weight_scale * activation_scale * static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kInt4Symmetric &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    if (info.block_size % 32 != 0) return false;
    for (long long item = 0; item < info.block_size; item += 32) {
      int8x16_t weights0;
      int8x16_t weights1;
      neon_interleaved_nibbles(weight + item / 2, &weights0, &weights1, 8);
      const auto* input = reinterpret_cast<const std::int8_t*>(
          activation.data.data() + activation_element + item);
      dot += neon_dot_i8x16_i8(weights0, vld1q_s8(input)) +
             neon_dot_i8x16_i8(weights1, vld1q_s8(input + 16));
    }
    *result = weight_scale * activation_scale * static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kUInt4Affine &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    if (info.block_size % 32 != 0) return false;
    int activation_sum = 0;
    for (long long item = 0; item < info.block_size; item += 32) {
      int8x16_t weights0;
      int8x16_t weights1;
      neon_interleaved_nibbles(weight + item / 2, &weights0, &weights1, 0);
      const auto* input = reinterpret_cast<const std::int8_t*>(
          activation.data.data() + activation_element + item);
      const int8x16_t inputs0 = vld1q_s8(input);
      const int8x16_t inputs1 = vld1q_s8(input + 16);
      dot += neon_dot_i8x16_i8(weights0, inputs0) +
             neon_dot_i8x16_i8(weights1, inputs1);
      activation_sum += vaddlvq_s8(inputs0) + vaddlvq_s8(inputs1);
    }
    *result = weight_scale * activation_scale *
              (static_cast<float>(dot) -
               weight_zeros[weight_side] * static_cast<float>(activation_sum));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kInt8Symmetric &&
                       ActivationLayout == CanonicalQuantLayout::kInt8Affine) {
    if (info.block_size % 16 != 0) return false;
    int weight_sum = 0;
    for (long long item = 0; item < info.block_size; item += 16) {
      const int8x16_t weights =
          vld1q_s8(reinterpret_cast<const std::int8_t*>(weight + item));
      const int8x16_t inputs = vld1q_s8(reinterpret_cast<const std::int8_t*>(
          activation.data.data() + activation_element + item));
      dot += neon_dot_i8x16_i8(weights, inputs);
      weight_sum += vaddlvq_s8(weights);
    }
    *result = weight_scale * activation_scale *
              (static_cast<float>(dot) -
               tensor_zero(activation, activation_row, first_column) *
                   static_cast<float>(weight_sum));
  } else if constexpr ((WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ||
                        WeightLayout == CanonicalQuantLayout::kFP8E5M2) &&
                       ActivationLayout == CanonicalQuantLayout::kFP8E4M3FN) {
    constexpr Float8Format weight_format =
        WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ? Float8Format::kE4M3FN
                                                         : Float8Format::kE5M2;
    *result = weight_scale * activation_scale *
              neon_fp8_pair_dot<weight_format, Float8Format::kE4M3FN>(
                  weight, activation.data.data() + activation_element,
                  info.block_size);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    const auto* activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            33);
    *result = e8m0_decode(weight[0]) * e8m0_decode(activation_block[0]) *
              neon_fp8_pair_dot<Float8Format::kE4M3FN, Float8Format::kE4M3FN>(
                  weight + 1, activation_block + 1, 32);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kMXFP4E2M1E8M0 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kMXFP4E2M1E8M0) {
    const auto* activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            17);
    int8x16_t weights0;
    int8x16_t weights1;
    int8x16_t inputs0;
    int8x16_t inputs1;
    neon_interleaved_nibbles(weight + 1, &weights0, &weights1, 0, &fp4_table);
    neon_interleaved_nibbles(activation_block + 1, &inputs0, &inputs1, 0,
                             &fp4_table);
    dot = neon_dot_i8x16_i8(weights0, inputs0) +
          neon_dot_i8x16_i8(weights1, inputs1);
    *result = 0.25f * e8m0_decode(weight[0]) *
              e8m0_decode(activation_block[0]) * static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kNVFP4E2M1E4M3 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    const int8x16_t weights = neon_ordered_nibbles16(weight, fp4_table);
    const int8x16_t inputs = neon_ordered_nibbles16(
        activation.data.data() + activation_element / 2, fp4_table);
    dot = neon_dot_i8x16_i8(weights, inputs);
    const float weight_block_scale =
        weight_metadata.global_scale *
        fp8_decode_fast<Float8Format::kE4M3FN>(
            weight_scale_codes[weight_row * info.blocks_per_row + block]);
    const float activation_block_scale =
        activation.metadata.global_scale *
        fp8_decode_fast<Float8Format::kE4M3FN>(
            activation.scale_codes[static_cast<std::size_t>(
                activation_row * info.blocks_per_row + block)]);
    *result = 0.25f * weight_block_scale * activation_block_scale *
              static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kBitNetTernary &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    int8x16_t weights0;
    int8x16_t weights1;
    neon_unpack_ternary32(weight + 2, &weights0, &weights1);
    const auto* input = reinterpret_cast<const std::int8_t*>(
        activation.data.data() + activation_element);
    dot = neon_dot_i8x16_i8(weights0, vld1q_s8(input)) +
          neon_dot_i8x16_i8(weights1, vld1q_s8(input + 16));
    *result = f16_to_float(little_u16(weight)) * activation_scale *
              static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kBitNetTernary &&
                       ActivationLayout == CanonicalQuantLayout::kFP4E2M1) {
    int8x16_t weights0;
    int8x16_t weights1;
    int8x16_t inputs0;
    int8x16_t inputs1;
    neon_unpack_ternary32(weight + 2, &weights0, &weights1);
    neon_interleaved_nibbles(activation.data.data() + activation_element / 2,
                             &inputs0, &inputs1, 0, &fp4_table);
    dot = neon_dot_i8x16_i8(weights0, inputs0) +
          neon_dot_i8x16_i8(weights1, inputs1);
    *result = 0.5f * f16_to_float(little_u16(weight)) * activation_scale *
              static_cast<float>(dot);
  } else {
    return false;
  }
  return true;
#else
  (void)info;
  (void)panel;
  (void)activation;
  (void)activation_row;
  (void)weight_row;
  (void)block;
  (void)result;
  return false;
#endif
}

template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout>
float portable_dual_block_dot(const CpuPackedWeightsInfo& info,
                              const std::uint8_t* panel,
                              const CanonicalQuantTensor& activation,
                              long long activation_row, long long weight_row,
                              long long block) {
  float direct = 0.0f;
  if (dual_block_dot_neon<WeightLayout, ActivationLayout>(
          info, panel, activation, activation_row, weight_row, block,
          &direct)) {
    return direct;
  }
  const QuantTensorMetadata& weight_metadata = info.quant_metadata;
  const auto* weight_scales =
      prepared_table<float>(panel, info.scale_table_offset);
  const auto* weight_zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* weight_scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long row_panel = weight_row / info.row_tile;
  const long long lane = weight_row % info.row_tile;
  const auto* weight = panel_block(info, panel, row_panel, block, lane);
  const long long first_column = block * info.block_size;
  const long long activation_element =
      activation_row * activation.metadata.logical_columns + first_column;
  const std::size_t weight_side =
      scale_index(weight_metadata, nullptr, weight_row, first_column);
  const float weight_scale =
      weight_scales == nullptr ? 1.0f : weight_scales[weight_side];
  const float activation_scale =
      tensor_scale(activation, activation_row, first_column);

  float sums[4] = {};
  if constexpr (WeightLayout == CanonicalQuantLayout::kInt4Symmetric &&
                ActivationLayout == CanonicalQuantLayout::kInt4Symmetric) {
    for (long long item = 0; item < info.block_size; ++item) {
      const int weight_code = nibble(weight, item);
      const int activation_code =
          nibble(activation.data.data(), activation_element + item);
      sums[item & 3] += static_cast<float>(
          (weight_code >= 8 ? weight_code - 16 : weight_code) *
          (activation_code >= 8 ? activation_code - 16 : activation_code));
    }
    return weight_scale * activation_scale *
           ((sums[0] + sums[1]) + (sums[2] + sums[3]));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kInt4Symmetric &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    int dot = 0;
    for (long long item = 0; item < info.block_size; ++item) {
      const int weight_code = nibble(weight, item);
      const int activation_code = static_cast<std::int8_t>(
          activation.data[static_cast<std::size_t>(activation_element + item)]);
      dot +=
          (weight_code >= 8 ? weight_code - 16 : weight_code) * activation_code;
    }
    return weight_scale * activation_scale * static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kUInt4Affine &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    int dot = 0;
    int activation_sum = 0;
    for (long long item = 0; item < info.block_size; ++item) {
      const int activation_code = static_cast<std::int8_t>(
          activation.data[static_cast<std::size_t>(activation_element + item)]);
      dot += static_cast<int>(nibble(weight, item)) * activation_code;
      activation_sum += activation_code;
    }
    return weight_scale * activation_scale *
           (static_cast<float>(dot) -
            weight_zeros[weight_side] * static_cast<float>(activation_sum));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kInt8Symmetric &&
                       ActivationLayout == CanonicalQuantLayout::kInt8Affine) {
    int dot = 0;
    int weight_sum = 0;
    for (long long item = 0; item < info.block_size; ++item) {
      const int weight_code = static_cast<std::int8_t>(weight[item]);
      const int activation_code = static_cast<std::int8_t>(
          activation.data[static_cast<std::size_t>(activation_element + item)]);
      dot += weight_code * activation_code;
      weight_sum += weight_code;
    }
    return weight_scale * activation_scale *
           (static_cast<float>(dot) -
            tensor_zero(activation, activation_row, first_column) *
                static_cast<float>(weight_sum));
  } else if constexpr ((WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ||
                        WeightLayout == CanonicalQuantLayout::kFP8E5M2) &&
                       ActivationLayout == CanonicalQuantLayout::kFP8E4M3FN) {
    constexpr Float8Format weight_format =
        WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ? Float8Format::kE4M3FN
                                                         : Float8Format::kE5M2;
    for (long long item = 0; item < info.block_size; ++item) {
      sums[item & 3] +=
          fp8_decode_fast<weight_format>(weight[item]) *
          fp8_decode_fast<Float8Format::kE4M3FN>(
              activation
                  .data[static_cast<std::size_t>(activation_element + item)]);
    }
    return weight_scale * activation_scale *
           ((sums[0] + sums[1]) + (sums[2] + sums[3]));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    const auto* activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            33);
    for (long long item = 0; item < 32; ++item) {
      sums[item & 3] +=
          fp8_decode_fast<Float8Format::kE4M3FN>(weight[1 + item]) *
          fp8_decode_fast<Float8Format::kE4M3FN>(activation_block[1 + item]);
    }
    return e8m0_decode(weight[0]) * e8m0_decode(activation_block[0]) *
           ((sums[0] + sums[1]) + (sums[2] + sums[3]));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kMXFP4E2M1E8M0 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kMXFP4E2M1E8M0) {
    const auto* activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            17);
    for (long long item = 0; item < 32; ++item) {
      sums[item & 3] += fp4_e2m1_decode(nibble(weight + 1, item)) *
                        fp4_e2m1_decode(nibble(activation_block + 1, item));
    }
    return e8m0_decode(weight[0]) * e8m0_decode(activation_block[0]) *
           ((sums[0] + sums[1]) + (sums[2] + sums[3]));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kNVFP4E2M1E4M3 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    const auto* activation_codes =
        activation.data.data() + activation_element / 2;
    for (long long item = 0; item < 16; ++item) {
      sums[item & 3] += fp4_e2m1_decode(nibble(weight, item)) *
                        fp4_e2m1_decode(nibble(activation_codes, item));
    }
    const float weight_block_scale =
        weight_metadata.global_scale *
        fp8_decode_fast<Float8Format::kE4M3FN>(
            weight_scale_codes[weight_row * info.blocks_per_row + block]);
    const float activation_block_scale =
        activation.metadata.global_scale *
        fp8_decode_fast<Float8Format::kE4M3FN>(
            activation.scale_codes[static_cast<std::size_t>(
                activation_row * info.blocks_per_row + block)]);
    return weight_block_scale * activation_block_scale *
           ((sums[0] + sums[1]) + (sums[2] + sums[3]));
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kBitNetTernary &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    int dot = 0;
    for (long long item = 0; item < 32; ++item) {
      const int weight_code =
          static_cast<int>((weight[2 + item / 4] >> (2 * (item & 3))) & 3) - 1;
      const int activation_code = static_cast<std::int8_t>(
          activation.data[static_cast<std::size_t>(activation_element + item)]);
      dot += weight_code * activation_code;
    }
    return f16_to_float(little_u16(weight)) * activation_scale *
           static_cast<float>(dot);
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kBitNetTernary &&
                       ActivationLayout == CanonicalQuantLayout::kFP4E2M1) {
    for (long long item = 0; item < 32; ++item) {
      const int weight_code =
          static_cast<int>((weight[2 + item / 4] >> (2 * (item & 3))) & 3) - 1;
      sums[item & 3] += static_cast<float>(weight_code) *
                        fp4_e2m1_decode(nibble(activation.data.data(),
                                               activation_element + item));
    }
    return f16_to_float(little_u16(weight)) * activation_scale *
           ((sums[0] + sums[1]) + (sums[2] + sums[3]));
  }
  return 0.0f;
}

template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout>
bool dual_panel_block_neon(const CpuPackedWeightsInfo& info,
                           const std::uint8_t* panel,
                           const CanonicalQuantTensor& activation,
                           long long activation_row, long long row_panel,
                           long long block, long long lanes,
                           float* accumulators) {
#if defined(__aarch64__) || defined(_M_ARM64)
  if (info.block_size > kKTile) return false;
  const QuantTensorMetadata& weight_metadata = info.quant_metadata;
  const auto* weight_scales =
      prepared_table<float>(panel, info.scale_table_offset);
  const auto* weight_zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* weight_scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long first_output = row_panel * info.row_tile;
  const long long first_column = block * info.block_size;
  const long long activation_element =
      activation_row * activation.metadata.logical_columns + first_column;
  const float activation_scale =
      tensor_scale(activation, activation_row, first_column);
  const int8x16_t fp4_table = {0, 1,  2,  3,  4,  6,  8,  12,
                               0, -1, -2, -3, -4, -6, -8, -12};

  if constexpr (WeightLayout == CanonicalQuantLayout::kInt4Symmetric &&
                ActivationLayout == CanonicalQuantLayout::kInt4Symmetric) {
    if (info.block_size != 32) return false;
    int8x16_t inputs0;
    int8x16_t inputs1;
    neon_interleaved_nibbles(activation.data.data() + activation_element / 2,
                             &inputs0, &inputs1, 8);
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long weight_row = first_output + lane;
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      int8x16_t weights0;
      int8x16_t weights1;
      neon_interleaved_nibbles(weight, &weights0, &weights1, 8);
      const std::size_t side =
          scale_index(weight_metadata, nullptr, weight_row, first_column);
      const int dot = neon_dot_i8x16_i8(weights0, inputs0) +
                      neon_dot_i8x16_i8(weights1, inputs1);
      accumulators[lane] +=
          weight_scales[side] * activation_scale * static_cast<float>(dot);
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kUInt4Affine &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    if (info.block_size != 32) return false;
    const auto* input = reinterpret_cast<const std::int8_t*>(
        activation.data.data() + activation_element);
    const int8x16_t inputs0 = vld1q_s8(input);
    const int8x16_t inputs1 = vld1q_s8(input + 16);
    const int activation_sum = vaddlvq_s8(inputs0) + vaddlvq_s8(inputs1);
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long weight_row = first_output + lane;
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      int8x16_t weights0;
      int8x16_t weights1;
      neon_interleaved_nibbles(weight, &weights0, &weights1, 0);
      const std::size_t side =
          scale_index(weight_metadata, nullptr, weight_row, first_column);
      const int dot = neon_dot_i8x16_i8(weights0, inputs0) +
                      neon_dot_i8x16_i8(weights1, inputs1);
      accumulators[lane] +=
          weight_scales[side] * activation_scale *
          (static_cast<float>(dot) -
           weight_zeros[side] * static_cast<float>(activation_sum));
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kInt8Symmetric &&
                       ActivationLayout == CanonicalQuantLayout::kInt8Affine) {
    if (info.block_size % 16 != 0) return false;
    const float activation_zero =
        tensor_zero(activation, activation_row, first_column);
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long weight_row = first_output + lane;
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      int dot = 0;
      int weight_sum = 0;
      for (long long item = 0; item < info.block_size; item += 16) {
        const int8x16_t weights =
            vld1q_s8(reinterpret_cast<const std::int8_t*>(weight + item));
        const int8x16_t inputs = vld1q_s8(reinterpret_cast<const std::int8_t*>(
            activation.data.data() + activation_element + item));
        dot += neon_dot_i8x16_i8(weights, inputs);
        weight_sum += vaddlvq_s8(weights);
      }
      const std::size_t side =
          scale_index(weight_metadata, nullptr, weight_row, first_column);
      accumulators[lane] += weight_scales[side] * activation_scale *
                            (static_cast<float>(dot) -
                             activation_zero * static_cast<float>(weight_sum));
    }
  } else if constexpr ((WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ||
                        WeightLayout == CanonicalQuantLayout::kFP8E5M2) &&
                       ActivationLayout == CanonicalQuantLayout::kFP8E4M3FN) {
    constexpr Float8Format weight_format =
        WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ? Float8Format::kE4M3FN
                                                         : Float8Format::kE5M2;
    float32x4_t decoded[kKTile / 4];
    for (long long item = 0; item < info.block_size; item += 8) {
      const uint16x8_t wide =
          vmovl_u8(vld1_u8(activation.data.data() + activation_element + item));
      decoded[item / 4] = neon_decode_fp8x4<Float8Format::kE4M3FN>(
          vmovl_u16(vget_low_u16(wide)));
      decoded[item / 4 + 1] = neon_decode_fp8x4<Float8Format::kE4M3FN>(
          vmovl_u16(vget_high_u16(wide)));
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long weight_row = first_output + lane;
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      const std::size_t side =
          scale_index(weight_metadata, nullptr, weight_row, first_column);
      accumulators[lane] +=
          weight_scales[side] * activation_scale *
          neon_fp8_decoded_dot<weight_format>(weight, decoded, info.block_size);
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    const auto* activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            33);
    float32x4_t decoded[8];
    for (long long item = 0; item < 32; item += 8) {
      const uint16x8_t wide = vmovl_u8(vld1_u8(activation_block + 1 + item));
      decoded[item / 4] = neon_decode_fp8x4<Float8Format::kE4M3FN>(
          vmovl_u16(vget_low_u16(wide)));
      decoded[item / 4 + 1] = neon_decode_fp8x4<Float8Format::kE4M3FN>(
          vmovl_u16(vget_high_u16(wide)));
    }
    const float input_scale = e8m0_decode(activation_block[0]);
    for (long long lane = 0; lane < lanes; ++lane) {
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      accumulators[lane] +=
          e8m0_decode(weight[0]) * input_scale *
          neon_fp8_decoded_dot<Float8Format::kE4M3FN>(weight + 1, decoded, 32);
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kMXFP4E2M1E8M0 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kMXFP4E2M1E8M0) {
    const auto* activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            17);
    int8x16_t inputs0;
    int8x16_t inputs1;
    neon_interleaved_nibbles(activation_block + 1, &inputs0, &inputs1, 0,
                             &fp4_table);
    const float input_scale = e8m0_decode(activation_block[0]);
    for (long long lane = 0; lane < lanes; ++lane) {
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      int8x16_t weights0;
      int8x16_t weights1;
      neon_interleaved_nibbles(weight + 1, &weights0, &weights1, 0, &fp4_table);
      const int dot = neon_dot_i8x16_i8(weights0, inputs0) +
                      neon_dot_i8x16_i8(weights1, inputs1);
      accumulators[lane] += 0.25f * e8m0_decode(weight[0]) * input_scale *
                            static_cast<float>(dot);
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kNVFP4E2M1E4M3 &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    const int8x16_t inputs = neon_ordered_nibbles16(
        activation.data.data() + activation_element / 2, fp4_table);
    const float input_scale =
        activation.metadata.global_scale *
        fp8_decode_fast<Float8Format::kE4M3FN>(
            activation.scale_codes[static_cast<std::size_t>(
                activation_row * info.blocks_per_row + block)]);
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long weight_row = first_output + lane;
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      const int8x16_t weights = neon_ordered_nibbles16(weight, fp4_table);
      const float weight_scale =
          weight_metadata.global_scale *
          fp8_decode_fast<Float8Format::kE4M3FN>(
              weight_scale_codes[weight_row * info.blocks_per_row + block]);
      accumulators[lane] +=
          0.25f * weight_scale * input_scale *
          static_cast<float>(neon_dot_i8x16_i8(weights, inputs));
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kBitNetTernary &&
                       ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric) {
    const auto* input = reinterpret_cast<const std::int8_t*>(
        activation.data.data() + activation_element);
    const int8x16_t inputs0 = vld1q_s8(input);
    const int8x16_t inputs1 = vld1q_s8(input + 16);
    for (long long lane = 0; lane < lanes; ++lane) {
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      int8x16_t weights0;
      int8x16_t weights1;
      neon_unpack_ternary32(weight + 2, &weights0, &weights1);
      const int dot = neon_dot_i8x16_i8(weights0, inputs0) +
                      neon_dot_i8x16_i8(weights1, inputs1);
      accumulators[lane] += f16_to_float(little_u16(weight)) *
                            activation_scale * static_cast<float>(dot);
    }
  } else if constexpr (WeightLayout == CanonicalQuantLayout::kBitNetTernary &&
                       ActivationLayout == CanonicalQuantLayout::kFP4E2M1) {
    int8x16_t inputs0;
    int8x16_t inputs1;
    neon_interleaved_nibbles(activation.data.data() + activation_element / 2,
                             &inputs0, &inputs1, 0, &fp4_table);
    for (long long lane = 0; lane < lanes; ++lane) {
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      int8x16_t weights0;
      int8x16_t weights1;
      neon_unpack_ternary32(weight + 2, &weights0, &weights1);
      const int dot = neon_dot_i8x16_i8(weights0, inputs0) +
                      neon_dot_i8x16_i8(weights1, inputs1);
      accumulators[lane] += 0.5f * f16_to_float(little_u16(weight)) *
                            activation_scale * static_cast<float>(dot);
    }
  } else {
    return false;
  }
  return true;
#else
  (void)info;
  (void)panel;
  (void)activation;
  (void)activation_row;
  (void)row_panel;
  (void)block;
  (void)lanes;
  (void)accumulators;
  return false;
#endif
}

struct CanonicalDualContext {
  CpuPackedWeightsInfo info;
  const std::uint8_t* panel = nullptr;
  const CanonicalQuantTensor* activation = nullptr;
  float* output = nullptr;
  long long activation_row_base = 0;
};

template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout>
void canonical_dual_rows(void* opaque, long long begin, long long end, int) {
  const auto& context = *static_cast<const CanonicalDualContext*>(opaque);
  for (long long index = begin; index < end; ++index) {
    const long long local_activation_row = index / context.info.rows;
    const long long activation_row =
        context.activation_row_base + local_activation_row;
    const long long weight_row = index % context.info.rows;
    float sums[4] = {};
    for (long long block = 0; block < context.info.blocks_per_row; ++block) {
      sums[block & 3] +=
          portable_dual_block_dot<WeightLayout, ActivationLayout>(
              context.info, context.panel, *context.activation, activation_row,
              weight_row, block);
    }
    context.output[index] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  }
}

template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout>
void canonical_dual_panel_rows(void* opaque, long long begin, long long end,
                               int) {
  const auto& context = *static_cast<const CanonicalDualContext*>(opaque);
  const long long panels =
      (context.info.rows + context.info.row_tile - 1) / context.info.row_tile;
  for (long long index = begin; index < end; ++index) {
    const long long local_activation_row = index / panels;
    const long long activation_row =
        context.activation_row_base + local_activation_row;
    const long long row_panel = index % panels;
    const long long first_output = row_panel * context.info.row_tile;
    const long long lanes =
        std::min(context.info.row_tile, context.info.rows - first_output);
    alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile] = {};
    for (long long block = 0; block < context.info.blocks_per_row; ++block) {
      float* partial = accumulators + (block & 3) * kMaximumRowTile;
      bool handled = false;
#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
      if constexpr (!(WeightLayout == CanonicalQuantLayout::kInt8Symmetric &&
                      ActivationLayout == CanonicalQuantLayout::kInt8Affine)) {
        if (cpu_features().avx2 &&
            context.activation->metadata.logical_rows > 1) {
          handled = quant::canonical_dual_panel_group_avx2(
              WeightLayout, ActivationLayout, context.info, context.panel,
              *context.activation, activation_row, row_panel, 1, block,
              accumulators);
        }
      }
#endif
      if (!handled) {
        handled = dual_panel_block_neon<WeightLayout, ActivationLayout>(
            context.info, context.panel, *context.activation, activation_row,
            row_panel, block, lanes, partial);
      }
      if (!handled) {
        for (long long lane = 0; lane < lanes; ++lane) {
          partial[lane] +=
              portable_dual_block_dot<WeightLayout, ActivationLayout>(
                  context.info, context.panel, *context.activation,
                  activation_row, first_output + lane, block);
        }
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      context.output[local_activation_row * context.info.rows + first_output +
                     lane] =
          (accumulators[lane] + accumulators[kMaximumRowTile + lane]) +
          (accumulators[2 * kMaximumRowTile + lane] +
           accumulators[3 * kMaximumRowTile + lane]);
    }
  }
}

template <CanonicalQuantLayout WeightLayout, long long PanelGroup>
void canonical_dual_fp8_panel_groups(void* opaque, long long begin,
                                     long long end, int) {
  const auto& context = *static_cast<const CanonicalDualContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
  const long long groups_per_row = (panels + PanelGroup - 1) / PanelGroup;
#if defined(__aarch64__) || defined(_M_ARM64)
  constexpr bool microscale =
      WeightLayout == CanonicalQuantLayout::kMXFP8E4M3E8M0;
  constexpr Float8Format weight_format =
      WeightLayout == CanonicalQuantLayout::kFP8E5M2 ? Float8Format::kE5M2
                                                     : Float8Format::kE4M3FN;
  const auto* weight_scales =
      prepared_table<float>(context.panel, info.scale_table_offset);
  alignas(
      64) float accumulators[PanelGroup * kAccumulatorCount * kMaximumRowTile];
  for (long long index = begin; index < end; ++index) {
    const long long local_activation_row = index / groups_per_row;
    const long long activation_row =
        context.activation_row_base + local_activation_row;
    const long long group = index % groups_per_row;
    const long long first_panel = group * PanelGroup;
    const long long panel_count = std::min(PanelGroup, panels - first_panel);
    std::fill_n(accumulators, PanelGroup * kAccumulatorCount * kMaximumRowTile,
                0.0f);
    for (long long block = 0; block < info.blocks_per_row; ++block) {
      const long long first_column = block * info.block_size;
      const std::uint8_t* activation_codes = nullptr;
      float activation_scale = 1.0f;
      if constexpr (microscale) {
        const auto* activation_block =
            context.activation->data.data() +
            static_cast<std::size_t>(
                (activation_row * info.columns / 32 + block) * 33);
        activation_scale = e8m0_decode(activation_block[0]);
        activation_codes = activation_block + 1;
      } else {
        activation_scale =
            tensor_scale(*context.activation, activation_row, first_column);
        activation_codes = context.activation->data.data() +
                           activation_row * info.columns + first_column;
      }
      float32x4_t decoded[kKTile / 4];
      for (long long item = 0; item < info.block_size; item += 8) {
        const uint16x8_t wide = vmovl_u8(vld1_u8(activation_codes + item));
        decoded[item / 4] = neon_decode_fp8x4<Float8Format::kE4M3FN>(
            vmovl_u16(vget_low_u16(wide)));
        decoded[item / 4 + 1] = neon_decode_fp8x4<Float8Format::kE4M3FN>(
            vmovl_u16(vget_high_u16(wide)));
      }
      for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
        const long long row_panel = first_panel + panel_lane;
        const long long first_output = row_panel * info.row_tile;
        const long long lanes =
            std::min(info.row_tile, info.rows - first_output);
        float* partial =
            accumulators +
            (panel_lane * kAccumulatorCount + (block & 3)) * kMaximumRowTile;
        for (long long lane = 0; lane < lanes; ++lane) {
          const long long weight_row = first_output + lane;
          const auto* weight =
              panel_block(info, context.panel, row_panel, block, lane);
          float scale = activation_scale;
          const std::uint8_t* weight_codes = weight;
          if constexpr (microscale) {
            scale *= e8m0_decode(weight[0]);
            weight_codes = weight + 1;
          } else {
            const std::size_t side = scale_index(info.quant_metadata, nullptr,
                                                 weight_row, first_column);
            scale *= weight_scales == nullptr ? 1.0f : weight_scales[side];
          }
          partial[lane] += scale * neon_fp8_decoded_dot<weight_format>(
                                       weight_codes, decoded, info.block_size);
        }
      }
    }
    for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
      const long long first_output = (first_panel + panel_lane) * info.row_tile;
      const long long lanes = std::min(info.row_tile, info.rows - first_output);
      const float* panel_accumulators =
          accumulators + panel_lane * kAccumulatorCount * kMaximumRowTile;
      for (long long lane = 0; lane < lanes; ++lane) {
        context.output[local_activation_row * info.rows + first_output + lane] =
            (panel_accumulators[lane] +
             panel_accumulators[kMaximumRowTile + lane]) +
            (panel_accumulators[2 * kMaximumRowTile + lane] +
             panel_accumulators[3 * kMaximumRowTile + lane]);
      }
    }
  }
#else
  constexpr CanonicalQuantLayout activation_layout =
      WeightLayout == CanonicalQuantLayout::kMXFP8E4M3E8M0
          ? CanonicalQuantLayout::kMXFP8E4M3E8M0
          : CanonicalQuantLayout::kFP8E4M3FN;
  for (long long index = begin; index < end; ++index) {
    const long long local_activation_row = index / groups_per_row;
    const long long group = index % groups_per_row;
    const long long first_panel = group * PanelGroup;
    canonical_dual_panel_rows<WeightLayout, activation_layout>(
        opaque, local_activation_row * panels + first_panel,
        local_activation_row * panels +
            std::min(first_panel + PanelGroup, panels),
        0);
  }
#endif
}

#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout, long long PanelGroup>
void canonical_dual_avx2_panel_groups(void* opaque, long long begin,
                                      long long end, int) {
  const auto& context = *static_cast<const CanonicalDualContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
  const long long groups_per_row = (panels + PanelGroup - 1) / PanelGroup;
  alignas(
      64) float accumulators[PanelGroup * kAccumulatorCount * kMaximumRowTile];
  for (long long index = begin; index < end; ++index) {
    const long long local_activation_row = index / groups_per_row;
    const long long activation_row =
        context.activation_row_base + local_activation_row;
    const long long group = index % groups_per_row;
    const long long first_panel = group * PanelGroup;
    const long long panel_count = std::min(PanelGroup, panels - first_panel);
    std::fill_n(accumulators, PanelGroup * kAccumulatorCount * kMaximumRowTile,
                0.0f);
    for (long long block = 0; block < info.blocks_per_row; ++block) {
      if (!quant::canonical_dual_panel_group_avx2(
              WeightLayout, ActivationLayout, info, context.panel,
              *context.activation, activation_row, first_panel, panel_count,
              block, accumulators)) {
        for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
          const long long row_panel = first_panel + panel_lane;
          const long long first_output = row_panel * info.row_tile;
          const long long lanes =
              std::min(info.row_tile, info.rows - first_output);
          float* partial =
              accumulators +
              (panel_lane * kAccumulatorCount + (block & 3)) * kMaximumRowTile;
          for (long long lane = 0; lane < lanes; ++lane) {
            partial[lane] +=
                portable_dual_block_dot<WeightLayout, ActivationLayout>(
                    info, context.panel, *context.activation, activation_row,
                    first_output + lane, block);
          }
        }
      }
    }
    for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
      const long long first_output = (first_panel + panel_lane) * info.row_tile;
      const long long lanes = std::min(info.row_tile, info.rows - first_output);
      const float* panel_accumulators =
          accumulators + panel_lane * kAccumulatorCount * kMaximumRowTile;
      for (long long lane = 0; lane < lanes; ++lane) {
        context.output[local_activation_row * info.rows + first_output + lane] =
            (panel_accumulators[lane] +
             panel_accumulators[kMaximumRowTile + lane]) +
            (panel_accumulators[2 * kMaximumRowTile + lane] +
             panel_accumulators[3 * kMaximumRowTile + lane]);
      }
    }
  }
}
#endif

threading::RangeFn dual_gemv_kernel(CanonicalQuantLayout weight,
                                    CanonicalQuantLayout activation) {
#define QUIXICORE_DUAL_CASE(weight_name, activation_name)       \
  if (weight == CanonicalQuantLayout::weight_name &&            \
      activation == CanonicalQuantLayout::activation_name)      \
  return canonical_dual_rows<CanonicalQuantLayout::weight_name, \
                             CanonicalQuantLayout::activation_name>
  QUIXICORE_DUAL_CASE(kInt4Symmetric, kInt4Symmetric);
  QUIXICORE_DUAL_CASE(kInt4Symmetric, kInt8Symmetric);
  QUIXICORE_DUAL_CASE(kUInt4Affine, kInt8Symmetric);
  QUIXICORE_DUAL_CASE(kInt8Symmetric, kInt8Affine);
  QUIXICORE_DUAL_CASE(kFP8E4M3FN, kFP8E4M3FN);
  QUIXICORE_DUAL_CASE(kFP8E5M2, kFP8E4M3FN);
  QUIXICORE_DUAL_CASE(kMXFP8E4M3E8M0, kMXFP8E4M3E8M0);
  QUIXICORE_DUAL_CASE(kMXFP4E2M1E8M0, kMXFP4E2M1E8M0);
  QUIXICORE_DUAL_CASE(kNVFP4E2M1E4M3, kNVFP4E2M1E4M3);
  QUIXICORE_DUAL_CASE(kBitNetTernary, kInt8Symmetric);
  QUIXICORE_DUAL_CASE(kBitNetTernary, kFP4E2M1);
#undef QUIXICORE_DUAL_CASE
  return nullptr;
}

threading::RangeFn dual_panel_kernel(CanonicalQuantLayout weight,
                                     CanonicalQuantLayout activation) {
#define QUIXICORE_DUAL_CASE(weight_name, activation_name)             \
  if (weight == CanonicalQuantLayout::weight_name &&                  \
      activation == CanonicalQuantLayout::activation_name)            \
  return canonical_dual_panel_rows<CanonicalQuantLayout::weight_name, \
                                   CanonicalQuantLayout::activation_name>
  QUIXICORE_DUAL_CASE(kInt4Symmetric, kInt4Symmetric);
  QUIXICORE_DUAL_CASE(kInt4Symmetric, kInt8Symmetric);
  QUIXICORE_DUAL_CASE(kUInt4Affine, kInt8Symmetric);
  QUIXICORE_DUAL_CASE(kInt8Symmetric, kInt8Affine);
  QUIXICORE_DUAL_CASE(kFP8E4M3FN, kFP8E4M3FN);
  QUIXICORE_DUAL_CASE(kFP8E5M2, kFP8E4M3FN);
  QUIXICORE_DUAL_CASE(kMXFP8E4M3E8M0, kMXFP8E4M3E8M0);
  QUIXICORE_DUAL_CASE(kMXFP4E2M1E8M0, kMXFP4E2M1E8M0);
  QUIXICORE_DUAL_CASE(kNVFP4E2M1E4M3, kNVFP4E2M1E4M3);
  QUIXICORE_DUAL_CASE(kBitNetTernary, kInt8Symmetric);
  QUIXICORE_DUAL_CASE(kBitNetTernary, kFP4E2M1);
#undef QUIXICORE_DUAL_CASE
  return nullptr;
}

template <long long PanelGroup>
threading::RangeFn dual_fp8_panel_group_kernel(
    CanonicalQuantLayout weight, CanonicalQuantLayout activation) {
  if (activation == CanonicalQuantLayout::kFP8E4M3FN) {
    if (weight == CanonicalQuantLayout::kFP8E4M3FN) {
      return canonical_dual_fp8_panel_groups<CanonicalQuantLayout::kFP8E4M3FN,
                                             PanelGroup>;
    }
    if (weight == CanonicalQuantLayout::kFP8E5M2) {
      return canonical_dual_fp8_panel_groups<CanonicalQuantLayout::kFP8E5M2,
                                             PanelGroup>;
    }
  }
  if (weight == CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
      activation == CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    return canonical_dual_fp8_panel_groups<CanonicalQuantLayout::kMXFP8E4M3E8M0,
                                           PanelGroup>;
  }
  return nullptr;
}

#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
template <long long PanelGroup>
threading::RangeFn dual_avx2_panel_group_kernel(
    CanonicalQuantLayout weight, CanonicalQuantLayout activation) {
#define QUIXICORE_DUAL_AVX2_CASE(weight_name, activation_name) \
  if (weight == CanonicalQuantLayout::weight_name &&           \
      activation == CanonicalQuantLayout::activation_name) {   \
    return canonical_dual_avx2_panel_groups<                   \
        CanonicalQuantLayout::weight_name,                     \
        CanonicalQuantLayout::activation_name, PanelGroup>;    \
  }
  QUIXICORE_DUAL_AVX2_CASE(kInt4Symmetric, kInt4Symmetric);
  QUIXICORE_DUAL_AVX2_CASE(kInt4Symmetric, kInt8Symmetric);
  QUIXICORE_DUAL_AVX2_CASE(kUInt4Affine, kInt8Symmetric);
  QUIXICORE_DUAL_AVX2_CASE(kInt8Symmetric, kInt8Affine);
  QUIXICORE_DUAL_AVX2_CASE(kFP8E4M3FN, kFP8E4M3FN);
  QUIXICORE_DUAL_AVX2_CASE(kFP8E5M2, kFP8E4M3FN);
  QUIXICORE_DUAL_AVX2_CASE(kMXFP8E4M3E8M0, kMXFP8E4M3E8M0);
  QUIXICORE_DUAL_AVX2_CASE(kMXFP4E2M1E8M0, kMXFP4E2M1E8M0);
  QUIXICORE_DUAL_AVX2_CASE(kNVFP4E2M1E4M3, kNVFP4E2M1E4M3);
  QUIXICORE_DUAL_AVX2_CASE(kBitNetTernary, kInt8Symmetric);
  QUIXICORE_DUAL_AVX2_CASE(kBitNetTernary, kFP4E2M1);
#undef QUIXICORE_DUAL_AVX2_CASE
  return nullptr;
}
#endif

bool direct_dual_gemv_supported(const CpuPackedWeightsInfo& info,
                                const CanonicalQuantTensor& activation) {
  if (info.act_order_offset != kMissingOffset ||
      info.group_index_offset != kMissingOffset ||
      !activation.act_order.empty() || !activation.group_index.empty() ||
      !scale_interval_is_constant(info.quant_metadata, info.block_size) ||
      !scale_interval_is_constant(activation.metadata, info.block_size)) {
    return false;
  }
  const CanonicalQuantLayout weight = info.quant_metadata.layout;
  const CanonicalQuantLayout input = activation.metadata.layout;
  if ((weight == CanonicalQuantLayout::kMXFP8E4M3E8M0 ||
       weight == CanonicalQuantLayout::kMXFP4E2M1E8M0 ||
       weight == CanonicalQuantLayout::kBitNetTernary) &&
      info.block_size != 32) {
    return false;
  }
  if (weight == CanonicalQuantLayout::kNVFP4E2M1E4M3 && info.block_size != 16) {
    return false;
  }
  return dual_gemv_kernel(weight, input) != nullptr;
}

bool dual_fp8_panel_pair(CanonicalQuantLayout weight,
                         CanonicalQuantLayout activation) {
  return ((weight == CanonicalQuantLayout::kFP8E4M3FN ||
           weight == CanonicalQuantLayout::kFP8E5M2) &&
          activation == CanonicalQuantLayout::kFP8E4M3FN) ||
         (weight == CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
          activation == CanonicalQuantLayout::kMXFP8E4M3E8M0);
}

void accumulate_output_lanes(float* accumulators, const float* weights,
                             float activation, long long lanes) {
  long long lane = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  const float32x4_t value = vdupq_n_f32(activation);
  for (; lane + 3 < lanes; lane += 4) {
    vst1q_f32(accumulators + lane, vfmaq_f32(vld1q_f32(accumulators + lane),
                                             vld1q_f32(weights + lane), value));
  }
#endif
  for (; lane < lanes; ++lane) {
    accumulators[lane] += weights[lane] * activation;
  }
}

struct CanonicalStorageContext {
  CpuPackedWeightsInfo info;
  const std::uint8_t* panel = nullptr;
  FloatStorageInput input;
  FloatStorageOutput output;
  long long m = 0;
  const float* bias = nullptr;
  LinearActivation activation = LinearActivation::kNone;
};

struct CanonicalGateUpContext {
  CpuPackedWeightsInfo gate_info;
  CpuPackedWeightsInfo up_info;
  const std::uint8_t* gate_panel = nullptr;
  const std::uint8_t* up_panel = nullptr;
  FloatStorageInput input;
  FloatStorageOutput gate_output;
  FloatStorageOutput up_output;
  FloatStorageOutput fused_output;
  float* fused_output_f32 = nullptr;
  long long m = 0;
  bool fuse_swiglu = false;
};

struct CanonicalQKVContext {
  CanonicalStorageContext planes[3];
  threading::RangeFn kernel = nullptr;
  long long task_counts[3] = {};
  long long panel_tasks = 0;
  long long panel_stride = 1;
};

struct CanonicalQKVRopeContext {
  CpuPackedWeightsInfo infos[3];
  const std::uint8_t* panels[3] = {};
  FloatStorageInput input;
  const float* cosine = nullptr;
  const float* sine = nullptr;
  FloatStorageOutput q_output;
  FloatStorageOutput key_cache;
  FloatStorageOutput value_cache;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long head_dim = 0;
  long long half = 0;
  long long query_pairs = 0;
  long long key_pairs = 0;
  long long value_rows = 0;
  long long cache_base = 0;
  long long query_panel_tasks = 0;
  long long key_panel_tasks = 0;
  long long value_panel_tasks = 0;
};

template <FloatStorageType Type, CanonicalQuantLayout Layout>
float canonical_projection_row(const CpuPackedWeightsInfo& info,
                               const std::uint8_t* panel,
                               FloatStorageInput input, long long row) {
  const auto* order = prepared_table<int>(panel, info.act_order_offset);
  float sums[kAccumulatorCount] = {};
  for (long long block = 0; block < info.blocks_per_row; ++block) {
    sums[block & 3] += prepared_row_block_dot<Type, Layout>(
        info, panel, input.data, order, row, block);
  }
  return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_projection_row_pair(const CpuPackedWeightsInfo& info,
                                   const std::uint8_t* panel,
                                   FloatStorageInput input, long long row0,
                                   long long row1, float* first,
                                   float* second) {
  const auto* order = prepared_table<int>(panel, info.act_order_offset);
  float sums0[kAccumulatorCount] = {};
  float sums1[kAccumulatorCount] = {};
  for (long long block = 0; block < info.blocks_per_row; ++block) {
    bool handled = false;
    if constexpr (Type != FloatStorageType::kF32) {
      if (order == nullptr && info.block_size <= kKTile) {
        const long long first_column = block * info.block_size;
        alignas(64) float decoded[kKTile];
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        handled =
            simd_decode_storage_block<Type>(input.data, first_column,
                                            info.block_size, decoded) &&
            prepared_row_block_dot_neon<Layout>(
                info, panel, decoded, row0, block, &partial0, first_column) &&
            prepared_row_block_dot_neon<Layout>(info, panel, decoded, row1,
                                                block, &partial1, first_column);
        if (handled) {
          sums0[block & 3] += partial0;
          sums1[block & 3] += partial1;
        }
      }
    }
    if (!handled) {
      sums0[block & 3] += prepared_row_block_dot<Type, Layout>(
          info, panel, input.data, order, row0, block);
      sums1[block & 3] += prepared_row_block_dot<Type, Layout>(
          info, panel, input.data, order, row1, block);
    }
  }
  *first = (sums0[0] + sums0[1]) + (sums0[2] + sums0[3]);
  *second = (sums1[0] + sums1[1]) + (sums1[2] + sums1[3]);
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_qkv_rope_rows(void* opaque, long long begin, long long end,
                             int) {
  const auto& context = *static_cast<const CanonicalQKVRopeContext*>(opaque);
  for (long long item = begin; item < end; ++item) {
    if (item < context.query_pairs) {
      const long long head = item / context.half;
      const long long dim = item - head * context.half;
      const long long row0 = head * context.head_dim + dim;
      const long long row1 = row0 + context.half;
      float first = 0.0f;
      float second = 0.0f;
      canonical_projection_row_pair<Type, Layout>(
          context.infos[0], context.panels[0], context.input, row0, row1,
          &first, &second);
      store_output(context.q_output, static_cast<std::size_t>(row0),
                   first * context.cosine[dim] - second * context.sine[dim]);
      store_output(context.q_output, static_cast<std::size_t>(row1),
                   second * context.cosine[dim] + first * context.sine[dim]);
      continue;
    }

    const long long kv_item = item - context.query_pairs;
    if (kv_item < context.key_pairs) {
      const long long head = kv_item / context.half;
      const long long dim = kv_item - head * context.half;
      const long long row0 = head * context.head_dim + dim;
      const long long row1 = row0 + context.half;
      float first = 0.0f;
      float second = 0.0f;
      canonical_projection_row_pair<Type, Layout>(
          context.infos[1], context.panels[1], context.input, row0, row1,
          &first, &second);
      store_output(context.key_cache,
                   static_cast<std::size_t>(context.cache_base + row0),
                   first * context.cosine[dim] - second * context.sine[dim]);
      store_output(context.key_cache,
                   static_cast<std::size_t>(context.cache_base + row1),
                   second * context.cosine[dim] + first * context.sine[dim]);
      continue;
    }

    const long long row = kv_item - context.key_pairs;
    const float value = canonical_projection_row<Type, Layout>(
        context.infos[2], context.panels[2], context.input, row);
    store_output(context.value_cache,
                 static_cast<std::size_t>(context.cache_base + row), value);
  }
}

template <CanonicalQuantLayout Layout>
void canonical_qkv_rope_f32_tiles(void* opaque, long long begin, long long end,
                                  int) {
  const auto& context = *static_cast<const CanonicalQKVRopeContext*>(opaque);
  constexpr long long kPairTile = 64;
  alignas(64) float first_values[kPairTile];
  alignas(64) float second_values[kPairTile];
  long long item = begin;
  while (item < end) {
    const bool query = item < context.query_pairs;
    const long long after_query = item - context.query_pairs;
    const bool key = !query && after_query < context.key_pairs;
    if (!query && !key) {
      const long long row = after_query - context.key_pairs;
      store_output(
          context.value_cache,
          static_cast<std::size_t>(context.cache_base + row),
          canonical_projection_row<FloatStorageType::kF32, Layout>(
              context.infos[2], context.panels[2], context.input, row));
      ++item;
      continue;
    }

    const long long local_item = query ? item : after_query;
    const long long head = local_item / context.half;
    const long long dim = local_item - head * context.half;
    const long long region_end =
        query ? std::min(end, context.query_pairs)
              : std::min(end, context.query_pairs + context.key_pairs);
    const long long count =
        std::min({kPairTile, context.half - dim, region_end - item});
    const int plane = query ? 0 : 1;
    const long long first_row = head * context.head_dim + dim;
    for (long long lane = 0; lane < count; ++lane) {
      first_values[lane] =
          canonical_projection_row<FloatStorageType::kF32, Layout>(
              context.infos[plane], context.panels[plane], context.input,
              first_row + lane);
    }
    for (long long lane = 0; lane < count; ++lane) {
      second_values[lane] =
          canonical_projection_row<FloatStorageType::kF32, Layout>(
              context.infos[plane], context.panels[plane], context.input,
              first_row + context.half + lane);
    }
    for (long long lane = 0; lane < count; ++lane) {
      const long long output_row = first_row + lane;
      const long long rope_dim = dim + lane;
      const float first = first_values[lane];
      const float second = second_values[lane];
      const float rotated0 =
          first * context.cosine[rope_dim] - second * context.sine[rope_dim];
      const float rotated1 =
          second * context.cosine[rope_dim] + first * context.sine[rope_dim];
      if (query) {
        store_output(context.q_output, static_cast<std::size_t>(output_row),
                     rotated0);
        store_output(context.q_output,
                     static_cast<std::size_t>(output_row + context.half),
                     rotated1);
      } else {
        store_output(context.key_cache,
                     static_cast<std::size_t>(context.cache_base + output_row),
                     rotated0);
        store_output(context.key_cache,
                     static_cast<std::size_t>(context.cache_base + output_row +
                                              context.half),
                     rotated1);
      }
    }
    item += count;
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_qkv_rope_typed_panels(void* opaque, long long begin,
                                     long long end, int) {
  static_assert(Type != FloatStorageType::kF32);
  const auto& context = *static_cast<const CanonicalQKVRopeContext*>(opaque);
  const long long pair_panels_per_head =
      context.half / context.infos[0].row_tile;
  alignas(64) float activations[kKTile];
  alignas(64) float sums0[kAccumulatorCount * kMaximumRowTile];
  alignas(64) float sums1[kAccumulatorCount * kMaximumRowTile];
  for (long long task = begin; task < end; ++task) {
    const bool query = task < context.query_panel_tasks;
    const long long after_query = task - context.query_panel_tasks;
    const bool key = !query && after_query < context.key_panel_tasks;
    const bool pair = query || key;
    const int plane = query ? 0 : (key ? 1 : 2);
    const CpuPackedWeightsInfo& info = context.infos[plane];
    const std::uint8_t* panel = context.panels[plane];
    long long first_output = 0;
    if (pair) {
      const long long local_task = query ? task : after_query;
      const long long head = local_task / pair_panels_per_head;
      const long long head_panel = local_task - head * pair_panels_per_head;
      first_output = head * context.head_dim + head_panel * info.row_tile;
    } else {
      const long long local_task = after_query - context.key_panel_tasks;
      first_output = local_task * info.row_tile;
    }
    const long long lanes =
        pair ? info.row_tile
             : std::min(info.row_tile, info.rows - first_output);
    std::fill_n(sums0, kAccumulatorCount * kMaximumRowTile, 0.0f);
    if (pair) std::fill_n(sums1, kAccumulatorCount * kMaximumRowTile, 0.0f);
    const auto* order = prepared_table<int>(panel, info.act_order_offset);
    for (long long block = 0; block < info.blocks_per_row; ++block) {
      const long long first_column = block * info.block_size;
      const bool decoded = simd_decode_storage_block<Type>(
          context.input.data, first_column, info.block_size, activations);
      for (long long lane = 0; lane < lanes; ++lane) {
        float partial0 = 0.0f;
        bool handled0 =
            decoded && prepared_row_block_dot_neon<Layout>(
                           info, panel, activations, first_output + lane, block,
                           &partial0, first_column);
        if (!handled0) {
          partial0 = prepared_row_block_dot<Type, Layout>(
              info, panel, context.input.data, order, first_output + lane,
              block);
        }
        sums0[(block & 3) * kMaximumRowTile + lane] += partial0;
        if (pair) {
          float partial1 = 0.0f;
          bool handled1 = decoded && prepared_row_block_dot_neon<Layout>(
                                         info, panel, activations,
                                         first_output + context.half + lane,
                                         block, &partial1, first_column);
          if (!handled1) {
            partial1 = prepared_row_block_dot<Type, Layout>(
                info, panel, context.input.data, order,
                first_output + context.half + lane, block);
          }
          sums1[(block & 3) * kMaximumRowTile + lane] += partial1;
        }
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      const float first = (sums0[lane] + sums0[kMaximumRowTile + lane]) +
                          (sums0[2 * kMaximumRowTile + lane] +
                           sums0[3 * kMaximumRowTile + lane]);
      if (pair) {
        const float second = (sums1[lane] + sums1[kMaximumRowTile + lane]) +
                             (sums1[2 * kMaximumRowTile + lane] +
                              sums1[3 * kMaximumRowTile + lane]);
        const long long dim = (first_output + lane) % context.head_dim;
        const float rotated0 =
            first * context.cosine[dim] - second * context.sine[dim];
        const float rotated1 =
            second * context.cosine[dim] + first * context.sine[dim];
        if (query) {
          store_output(context.q_output,
                       static_cast<std::size_t>(first_output + lane), rotated0);
          store_output(
              context.q_output,
              static_cast<std::size_t>(first_output + context.half + lane),
              rotated1);
        } else {
          store_output(context.key_cache,
                       static_cast<std::size_t>(context.cache_base +
                                                first_output + lane),
                       rotated0);
          store_output(
              context.key_cache,
              static_cast<std::size_t>(context.cache_base + first_output +
                                       context.half + lane),
              rotated1);
        }
      } else {
        store_output(
            context.value_cache,
            static_cast<std::size_t>(context.cache_base + first_output + lane),
            first);
      }
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_qkv_rope_panels(void* opaque, long long begin, long long end,
                               int) {
  const auto& context = *static_cast<const CanonicalQKVRopeContext*>(opaque);
  const long long pair_panels_per_head =
      (context.half + context.infos[0].row_tile - 1) /
      context.infos[0].row_tile;
  alignas(64) float decoded0[kMaximumRowTile * kKTile];
  alignas(64) float decoded1[kMaximumRowTile * kKTile];
  alignas(64) float activations[kKTile];
  alignas(64) float sums0[kAccumulatorCount * kMaximumRowTile];
  alignas(64) float sums1[kAccumulatorCount * kMaximumRowTile];
  for (long long task = begin; task < end; ++task) {
    const bool query = task < context.query_panel_tasks;
    const long long after_query = task - context.query_panel_tasks;
    const bool key = !query && after_query < context.key_panel_tasks;
    const bool pair = query || key;
    const int plane = query ? 0 : (key ? 1 : 2);
    const CpuPackedWeightsInfo& info = context.infos[plane];
    const std::uint8_t* panel = context.panels[plane];
    long long first_output = 0;
    long long row_panel0 = 0;
    long long row_panel1 = 0;
    long long lanes = 0;
    if (pair) {
      const long long local_task = query ? task : after_query;
      const long long head = local_task / pair_panels_per_head;
      const long long head_panel = local_task - head * pair_panels_per_head;
      first_output = head * context.head_dim + head_panel * info.row_tile;
      lanes =
          std::min(info.row_tile, context.half - head_panel * info.row_tile);
      row_panel0 = first_output / info.row_tile;
      row_panel1 = (first_output + context.half) / info.row_tile;
    } else {
      const long long local_task = after_query - context.key_panel_tasks;
      row_panel0 = local_task;
      first_output = row_panel0 * info.row_tile;
      lanes = std::min(info.row_tile, info.rows - first_output);
    }
    std::fill_n(sums0, kAccumulatorCount * kMaximumRowTile, 0.0f);
    if (pair) std::fill_n(sums1, kAccumulatorCount * kMaximumRowTile, 0.0f);

    for (long long block = 0; block < info.blocks_per_row; ++block) {
      for (long long item_base = 0; item_base < info.block_size;
           item_base += kKTile) {
        const long long items = std::min(kKTile, info.block_size - item_base);
        const long long first_column = block * info.block_size + item_base;
        if (!simd_decode_storage_block<Type>(context.input.data, first_column,
                                             items, activations)) {
          for (long long item = 0; item < items; ++item) {
            activations[item] = input_value<Type>(
                context.input.data,
                static_cast<std::size_t>(first_column + item));
          }
        }
        decode_prepared_chunk<Layout>(info, panel, row_panel0, block, lanes,
                                      item_base, items, first_output, decoded0);
        if (pair) {
          decode_prepared_chunk<Layout>(info, panel, row_panel1, block, lanes,
                                        item_base, items,
                                        first_output + context.half, decoded1);
        }
        for (long long item = 0; item < items; ++item) {
          const long long accumulator = (first_column + item) & 3;
          accumulate_output_lanes(sums0 + accumulator * kMaximumRowTile,
                                  decoded0 + item * kMaximumRowTile,
                                  activations[item], lanes);
          if (pair) {
            accumulate_output_lanes(sums1 + accumulator * kMaximumRowTile,
                                    decoded1 + item * kMaximumRowTile,
                                    activations[item], lanes);
          }
        }
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      const float first = (sums0[lane] + sums0[kMaximumRowTile + lane]) +
                          (sums0[2 * kMaximumRowTile + lane] +
                           sums0[3 * kMaximumRowTile + lane]);
      if (query) {
        const float second = (sums1[lane] + sums1[kMaximumRowTile + lane]) +
                             (sums1[2 * kMaximumRowTile + lane] +
                              sums1[3 * kMaximumRowTile + lane]);
        const long long dim = (first_output + lane) % context.head_dim;
        store_output(context.q_output,
                     static_cast<std::size_t>(first_output + lane),
                     first * context.cosine[dim] - second * context.sine[dim]);
        store_output(
            context.q_output,
            static_cast<std::size_t>(first_output + context.half + lane),
            second * context.cosine[dim] + first * context.sine[dim]);
      } else if (key) {
        const float second = (sums1[lane] + sums1[kMaximumRowTile + lane]) +
                             (sums1[2 * kMaximumRowTile + lane] +
                              sums1[3 * kMaximumRowTile + lane]);
        const long long dim = (first_output + lane) % context.head_dim;
        store_output(
            context.key_cache,
            static_cast<std::size_t>(context.cache_base + first_output + lane),
            first * context.cosine[dim] - second * context.sine[dim]);
        store_output(
            context.key_cache,
            static_cast<std::size_t>(context.cache_base + first_output +
                                     context.half + lane),
            second * context.cosine[dim] + first * context.sine[dim]);
      } else {
        store_output(
            context.value_cache,
            static_cast<std::size_t>(context.cache_base + first_output + lane),
            first);
      }
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout,
          long long PanelGroup>
void canonical_qkv_rope_panel_groups(void* opaque, long long begin,
                                     long long end, int) {
  static_assert(Type != FloatStorageType::kF32);
  const auto& context = *static_cast<const CanonicalQKVRopeContext*>(opaque);
  const long long pair_panels_per_head =
      context.half / context.infos[0].row_tile;
  const long long pair_groups_per_head =
      (pair_panels_per_head + PanelGroup - 1) / PanelGroup;
  alignas(64) float decoded0[kMaximumRowTile * kKTile];
  alignas(64) float decoded1[kMaximumRowTile * kKTile];
  alignas(64) float activations[kKTile];
  alignas(64) float sums0[PanelGroup][kAccumulatorCount * kMaximumRowTile];
  alignas(64) float sums1[PanelGroup][kAccumulatorCount * kMaximumRowTile];
  for (long long task = begin; task < end; ++task) {
    const bool query = task < context.query_panel_tasks;
    const long long after_query = task - context.query_panel_tasks;
    const bool key = !query && after_query < context.key_panel_tasks;
    const bool pair = query || key;
    const int plane = query ? 0 : (key ? 1 : 2);
    const CpuPackedWeightsInfo& info = context.infos[plane];
    const std::uint8_t* panel = context.panels[plane];
    long long head = 0;
    long long first_panel = 0;
    long long panel_count = 0;
    if (pair) {
      const long long local_task = query ? task : after_query;
      head = local_task / pair_groups_per_head;
      const long long group = local_task - head * pair_groups_per_head;
      const long long first_head_panel = group * PanelGroup;
      panel_count =
          std::min(PanelGroup, pair_panels_per_head - first_head_panel);
      first_panel =
          head * (context.head_dim / info.row_tile) + first_head_panel;
    } else {
      const long long local_task = after_query - context.key_panel_tasks;
      first_panel = local_task * PanelGroup;
      const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
      panel_count = std::min(PanelGroup, panels - first_panel);
    }
    std::fill_n(&sums0[0][0], PanelGroup * kAccumulatorCount * kMaximumRowTile,
                0.0f);
    if (pair) {
      std::fill_n(&sums1[0][0],
                  PanelGroup * kAccumulatorCount * kMaximumRowTile, 0.0f);
    }

    for (long long block = 0; block < info.blocks_per_row; ++block) {
      for (long long item_base = 0; item_base < info.block_size;
           item_base += kKTile) {
        const long long items = std::min(kKTile, info.block_size - item_base);
        const long long first_column = block * info.block_size + item_base;
        if (!simd_decode_storage_block<Type>(context.input.data, first_column,
                                             items, activations)) {
          for (long long item = 0; item < items; ++item) {
            activations[item] = input_value<Type>(
                context.input.data,
                static_cast<std::size_t>(first_column + item));
          }
        }
        for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
          const long long row_panel0 = first_panel + panel_lane;
          const long long first_output = row_panel0 * info.row_tile;
          const long long lanes =
              pair ? info.row_tile
                   : std::min(info.row_tile, info.rows - first_output);
          decode_prepared_chunk<Layout>(info, panel, row_panel0, block, lanes,
                                        item_base, items, first_output,
                                        decoded0);
          if (pair) {
            const long long row_panel1 =
                row_panel0 + context.half / info.row_tile;
            decode_prepared_chunk<Layout>(
                info, panel, row_panel1, block, lanes, item_base, items,
                first_output + context.half, decoded1);
          }
          for (long long item = 0; item < items; ++item) {
            const long long accumulator = (first_column + item) & 3;
            accumulate_output_lanes(
                sums0[panel_lane] + accumulator * kMaximumRowTile,
                decoded0 + item * kMaximumRowTile, activations[item], lanes);
            if (pair) {
              accumulate_output_lanes(
                  sums1[panel_lane] + accumulator * kMaximumRowTile,
                  decoded1 + item * kMaximumRowTile, activations[item], lanes);
            }
          }
        }
      }
    }
    for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
      const long long row_panel0 = first_panel + panel_lane;
      const long long first_output = row_panel0 * info.row_tile;
      const long long lanes =
          pair ? info.row_tile
               : std::min(info.row_tile, info.rows - first_output);
      for (long long lane = 0; lane < lanes; ++lane) {
        const float first = (sums0[panel_lane][lane] +
                             sums0[panel_lane][kMaximumRowTile + lane]) +
                            (sums0[panel_lane][2 * kMaximumRowTile + lane] +
                             sums0[panel_lane][3 * kMaximumRowTile + lane]);
        if (query || key) {
          const float second = (sums1[panel_lane][lane] +
                                sums1[panel_lane][kMaximumRowTile + lane]) +
                               (sums1[panel_lane][2 * kMaximumRowTile + lane] +
                                sums1[panel_lane][3 * kMaximumRowTile + lane]);
          const long long dim = (first_output + lane) % context.head_dim;
          const float rotated0 =
              first * context.cosine[dim] - second * context.sine[dim];
          const float rotated1 =
              second * context.cosine[dim] + first * context.sine[dim];
          if (query) {
            store_output(context.q_output,
                         static_cast<std::size_t>(first_output + lane),
                         rotated0);
            store_output(
                context.q_output,
                static_cast<std::size_t>(first_output + context.half + lane),
                rotated1);
          } else {
            store_output(context.key_cache,
                         static_cast<std::size_t>(context.cache_base +
                                                  first_output + lane),
                         rotated0);
            store_output(
                context.key_cache,
                static_cast<std::size_t>(context.cache_base + first_output +
                                         context.half + lane),
                rotated1);
          }
        } else {
          store_output(context.value_cache,
                       static_cast<std::size_t>(context.cache_base +
                                                first_output + lane),
                       first);
        }
      }
    }
  }
}

void canonical_qkv_schedule(void* opaque, long long begin, long long end,
                            int worker) {
  auto& context = *static_cast<CanonicalQKVContext*>(opaque);
  long long offset = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const long long plane_end = offset + context.task_counts[plane];
    const long long overlap_begin = std::max(begin, offset);
    const long long overlap_end = std::min(end, plane_end);
    if (overlap_begin < overlap_end) {
      context.kernel(&context.planes[plane], overlap_begin - offset,
                     overlap_end - offset, worker);
    }
    offset = plane_end;
  }
}

template <bool FuseSwiGLU>
void store_gate_up_output(const CanonicalGateUpContext& context,
                          std::size_t index, float gate, float up) {
  if constexpr (FuseSwiGLU) {
    const float silu = gate / (1.0f + std::exp(-gate));
    const float value = silu * up;
    if (context.fused_output_f32 != nullptr) {
      context.fused_output_f32[index] = value;
    } else {
      store_output(context.fused_output, index, value);
    }
    return;
  }
  store_output(context.gate_output, index, gate);
  store_output(context.up_output, index, up);
}

void store_projection_output(const CanonicalStorageContext& context,
                             std::size_t index, long long output_column,
                             float value) {
  if (context.bias == nullptr &&
      context.activation == LinearActivation::kNone) {
    store_output(context.output, index, value);
    return;
  }
  if (context.bias != nullptr) value += context.bias[output_column];
  store_output(context.output, index,
               projection_activation(value, context.activation));
}

bool store_relu2_lanes(const CanonicalStorageContext& context,
                       std::size_t first_index, long long first_output,
                       const float* partial, long long reduction_stride,
                       long long lanes) {
#if defined(__aarch64__) || defined(_M_ARM64)
  if (context.output.type != FloatStorageType::kF32 ||
      context.activation != LinearActivation::kRelu2) {
    return false;
  }
  auto* destination = static_cast<float*>(context.output.data) + first_index;
  const float32x4_t zero = vdupq_n_f32(0.0f);
  long long lane = 0;
  for (; lane + 3 < lanes; lane += 4) {
    float32x4_t value =
        vaddq_f32(vaddq_f32(vld1q_f32(partial + lane),
                            vld1q_f32(partial + reduction_stride + lane)),
                  vaddq_f32(vld1q_f32(partial + 2 * reduction_stride + lane),
                            vld1q_f32(partial + 3 * reduction_stride + lane)));
    if (context.bias != nullptr)
      value = vaddq_f32(value, vld1q_f32(context.bias + first_output + lane));
    value = vmaxq_f32(value, zero);
    vst1q_f32(destination + lane, vmulq_f32(value, value));
  }
  for (; lane < lanes; ++lane) {
    const float sum = (partial[lane] + partial[reduction_stride + lane]) +
                      (partial[2 * reduction_stride + lane] +
                       partial[3 * reduction_stride + lane]);
    store_projection_output(context, first_index + lane, first_output + lane,
                            sum);
  }
  return true;
#else
  (void)context;
  (void)first_index;
  (void)first_output;
  (void)partial;
  (void)reduction_stride;
  (void)lanes;
  return false;
#endif
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_gemv_storage_rows(void* opaque, long long begin, long long end,
                                 int) {
  const auto& context = *static_cast<const CanonicalStorageContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  for (long long row = begin; row < end; ++row) {
    float sums[4] = {};
    for (long long block = 0; block < info.blocks_per_row; ++block) {
      sums[block & 3] += prepared_row_block_dot<Type, Layout>(
          info, context.panel, context.input.data, order, row, block);
    }
    store_projection_output(context, static_cast<std::size_t>(row), row,
                            (sums[0] + sums[1]) + (sums[2] + sums[3]));
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_typed_gemv_panels(void* opaque, long long begin, long long end,
                                 int) {
  const auto& context = *static_cast<const CanonicalStorageContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  for (long long row_panel = begin; row_panel < end; ++row_panel) {
    const long long first_output = row_panel * info.row_tile;
    const long long lanes = std::min(info.row_tile, info.rows - first_output);
    alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile] = {};
    for (long long block = 0; block < info.blocks_per_row; ++block) {
      const long long first_column = block * info.block_size;
      float* partial = accumulators + (block & 3) * kMaximumRowTile;
      bool handled = false;
      if (info.block_size <= kKTile) {
        alignas(64) float decoded[kKTile];
        handled = simd_decode_storage_block<Type>(
            context.input.data, first_column, info.block_size, decoded);
        if (handled) {
          for (long long lane = 0; lane < lanes; ++lane) {
            float direct = 0.0f;
            if (!prepared_row_block_dot_neon<Layout>(
                    info, context.panel, decoded, first_output + lane, block,
                    &direct, first_column)) {
              handled = false;
              break;
            }
            partial[lane] += direct;
          }
        }
      } else if constexpr (Layout == CanonicalQuantLayout::kInt8Symmetric ||
                           Layout == CanonicalQuantLayout::kInt8Affine) {
#if defined(__aarch64__) || defined(_M_ARM64)
        const auto* scales =
            prepared_table<float>(context.panel, info.scale_table_offset);
        const auto* zeros =
            prepared_table<float>(context.panel, info.zero_point_table_offset);
        alignas(64) float block_sums[kMaximumRowTile] = {};
        handled = true;
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          alignas(64) float decoded[kKTile];
          if (!simd_decode_storage_block<Type>(context.input.data,
                                               first_column + item_base, items,
                                               decoded)) {
            handled = false;
            break;
          }
          const float input_sum = neon_sum_f32(decoded, items);
          for (long long lane = 0; lane < lanes; ++lane) {
            const long long weight_row = first_output + lane;
            const auto* source =
                panel_block(info, context.panel, row_panel, block, lane);
            float dot = 0.0f;
            long long item = 0;
            for (; item + 15 < items; item += 16) {
              dot += neon_dot_i8x16_f32(
                  vld1q_s8(reinterpret_cast<const std::int8_t*>(
                      source + item_base + item)),
                  decoded + item);
            }
            for (; item < items; ++item) {
              dot += static_cast<float>(
                         static_cast<std::int8_t>(source[item_base + item])) *
                     decoded[item];
            }
            if constexpr (Layout == CanonicalQuantLayout::kInt8Affine) {
              const std::size_t side = scale_index(info.quant_metadata, nullptr,
                                                   weight_row, first_column);
              dot -= zeros[side] * input_sum;
            }
            block_sums[lane] += dot;
          }
        }
        if (handled) {
          for (long long lane = 0; lane < lanes; ++lane) {
            const std::size_t side =
                scale_index(info.quant_metadata, nullptr, first_output + lane,
                            first_column);
            partial[lane] += scales[side] * block_sums[lane];
          }
        }
#endif
      }
      if (!handled) {
        for (long long lane = 0; lane < lanes; ++lane) {
          partial[lane] += prepared_row_block_dot<Type, Layout>(
              info, context.panel, context.input.data, nullptr,
              first_output + lane, block);
        }
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      store_projection_output(
          context, static_cast<std::size_t>(first_output + lane),
          first_output + lane,
          (accumulators[lane] + accumulators[kMaximumRowTile + lane]) +
              (accumulators[2 * kMaximumRowTile + lane] +
               accumulators[3 * kMaximumRowTile + lane]));
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_storage_panels(void* opaque, long long begin, long long end,
                              int) {
  const auto& context = *static_cast<const CanonicalStorageContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  alignas(64) float decoded[kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long row_panel = begin; row_panel < end; ++row_panel) {
    const long long first_output = row_panel * info.row_tile;
    const long long lanes = std::min(info.row_tile, info.rows - first_output);
    for (long long m_base = 0; m_base < context.m; m_base += kMTile) {
      const long long m_count = std::min(kMTile, context.m - m_base);
      std::fill_n(accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                        lanes, item_base, items, first_output,
                                        decoded);
          bool typed_activations_decoded = false;
          if constexpr (Type != FloatStorageType::kF32) {
            if (order == nullptr) {
              typed_activations_decoded = true;
              const long long first_column =
                  block * info.block_size + item_base;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                const std::size_t input_index = static_cast<std::size_t>(
                    (m_base + m_lane) * info.columns + first_column);
                if (!simd_decode_storage_block<Type>(
                        context.input.data, static_cast<long long>(input_index),
                        items, decoded_activations + m_lane * kKTile)) {
                  typed_activations_decoded = false;
                  break;
                }
              }
            }
          }
          bool avx2_accumulated = false;
#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
          if (cpu_features().avx2 && order == nullptr && lanes == 8) {
            const long long first_column = block * info.block_size + item_base;
            const float* activations = nullptr;
            long long activation_stride = 0;
            if constexpr (Type == FloatStorageType::kF32) {
              activations = static_cast<const float*>(context.input.data) +
                            m_base * info.columns + first_column;
              activation_stride = info.columns;
            } else if (typed_activations_decoded) {
              activations = decoded_activations;
              activation_stride = kKTile;
            }
            if (activations != nullptr) {
              quant::canonical_accumulate_tile_avx2(
                  accumulators, decoded, activations, activation_stride,
                  first_column, items, m_count, lanes);
              avx2_accumulated = true;
            }
          }
#endif
          if (!avx2_accumulated) {
            for (long long item = 0; item < items; ++item) {
              const long long packed_column =
                  block * info.block_size + item_base + item;
              const long long logical_column =
                  order == nullptr ? packed_column : order[packed_column];
              const long long accumulator = packed_column & 3;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                const float activation =
                    typed_activations_decoded
                        ? decoded_activations[m_lane * kKTile + item]
                        : input_value<Type>(
                              context.input.data,
                              static_cast<std::size_t>((m_base + m_lane) *
                                                           info.columns +
                                                       logical_column));
                accumulate_output_lanes(
                    accumulators +
                        (accumulator * kMTile + m_lane) * kMaximumRowTile,
                    decoded + item * kMaximumRowTile, activation, lanes);
              }
            }
          }
        }
      }
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        const std::size_t first_output_index = static_cast<std::size_t>(
            (m_base + m_lane) * info.rows + first_output);
        const float* row_partial = accumulators + m_lane * kMaximumRowTile;
        constexpr long long stride = kMTile * kMaximumRowTile;
        if (store_relu2_lanes(context, first_output_index, first_output,
                              row_partial, stride, lanes)) {
          continue;
        }
        for (long long lane = 0; lane < lanes; ++lane) {
          const long long output_column = first_output + lane;
          const std::size_t output_index = static_cast<std::size_t>(
              (m_base + m_lane) * info.rows + output_column);
          const float* partial = accumulators + m_lane * kMaximumRowTile + lane;
          const float sum = (partial[0] + partial[stride]) +
                            (partial[2 * stride] + partial[3 * stride]);
          store_projection_output(context, output_index, output_column, sum);
        }
      }
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_qkv_shared_panels(void* opaque, long long begin, long long end,
                                 int worker) {
  auto& context = *static_cast<CanonicalQKVContext*>(opaque);
  long long panel_counts[3];
  for (int plane = 0; plane < 3; ++plane) {
    const CpuPackedWeightsInfo& info = context.planes[plane].info;
    panel_counts[plane] = (info.rows + info.row_tile - 1) / info.row_tile;
  }

  alignas(64) float decoded_weights[3][kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(
      64) float accumulators[3][kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long task = begin; task < end; ++task) {
    const long long row_panel =
        context.panel_stride == 1 ||
                task >
                    std::numeric_limits<long long>::max() / context.panel_stride
            ? task
            : (task * context.panel_stride) % context.panel_tasks;
    bool active[3];
    int active_count = 0;
    for (int plane = 0; plane < 3; ++plane) {
      active[plane] = row_panel < panel_counts[plane];
      active_count += active[plane] ? 1 : 0;
    }
    if (active_count == 1) {
      for (int plane = 0; plane < 3; ++plane) {
        if (active[plane]) {
          canonical_storage_panels<Type, Layout>(
              &context.planes[plane], row_panel, row_panel + 1, worker);
          break;
        }
      }
      continue;
    }

    const CpuPackedWeightsInfo& common = context.planes[0].info;
    for (long long m_base = 0; m_base < context.planes[0].m; m_base += kMTile) {
      const long long m_count = std::min(kMTile, context.planes[0].m - m_base);
      for (int plane = 0; plane < 3; ++plane) {
        if (active[plane]) {
          std::fill_n(accumulators[plane],
                      kAccumulatorCount * kMaximumRowTile * kMTile, 0.0f);
        }
      }
      for (long long block = 0; block < common.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < common.block_size;
             item_base += kKTile) {
          const long long items =
              std::min(kKTile, common.block_size - item_base);
          const long long first_column = block * common.block_size + item_base;
          for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
            const long long input_index =
                (m_base + m_lane) * common.columns + first_column;
            if (!simd_decode_storage_block<Type>(
                    context.planes[0].input.data, input_index, items,
                    decoded_activations + m_lane * kKTile)) {
              for (long long item = 0; item < items; ++item) {
                decoded_activations[m_lane * kKTile + item] = input_value<Type>(
                    context.planes[0].input.data,
                    static_cast<std::size_t>(input_index + item));
              }
            }
          }
          for (int plane = 0; plane < 3; ++plane) {
            if (!active[plane]) continue;
            const CanonicalStorageContext& projection = context.planes[plane];
            const CpuPackedWeightsInfo& info = projection.info;
            const long long first_output = row_panel * info.row_tile;
            const long long lanes =
                std::min(info.row_tile, info.rows - first_output);
            decode_prepared_chunk<Layout>(info, projection.panel, row_panel,
                                          block, lanes, item_base, items,
                                          first_output, decoded_weights[plane]);
            for (long long item = 0; item < items; ++item) {
              const long long accumulator = (first_column + item) & 3;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                accumulate_output_lanes(
                    accumulators[plane] +
                        (accumulator * kMTile + m_lane) * kMaximumRowTile,
                    decoded_weights[plane] + item * kMaximumRowTile,
                    decoded_activations[m_lane * kKTile + item], lanes);
              }
            }
          }
        }
      }
      for (int plane = 0; plane < 3; ++plane) {
        if (!active[plane]) continue;
        const CanonicalStorageContext& projection = context.planes[plane];
        const CpuPackedWeightsInfo& info = projection.info;
        const long long first_output = row_panel * info.row_tile;
        const long long lanes =
            std::min(info.row_tile, info.rows - first_output);
        for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
          const float* row_partial =
              accumulators[plane] + m_lane * kMaximumRowTile;
          constexpr long long stride = kMTile * kMaximumRowTile;
          for (long long lane = 0; lane < lanes; ++lane) {
            const long long output_column = first_output + lane;
            const float* partial = row_partial + lane;
            store_projection_output(
                projection,
                static_cast<std::size_t>((m_base + m_lane) * info.rows +
                                         output_column),
                output_column,
                (partial[0] + partial[stride]) +
                    (partial[2 * stride] + partial[3 * stride]));
          }
        }
      }
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout, bool FuseSwiGLU>
void canonical_gate_up_panels(void* opaque, long long begin, long long end,
                              int) {
  const auto& context = *static_cast<const CanonicalGateUpContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.gate_info;
  const auto* gate_order =
      prepared_table<int>(context.gate_panel, info.act_order_offset);
  const auto* up_order =
      prepared_table<int>(context.up_panel, context.up_info.act_order_offset);
  alignas(64) float gate_decoded[kMaximumRowTile * kKTile];
  alignas(64) float up_decoded[kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(
      64) float gate_accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  alignas(
      64) float up_accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long row_panel = begin; row_panel < end; ++row_panel) {
    const long long first_output = row_panel * info.row_tile;
    const long long lanes = std::min(info.row_tile, info.rows - first_output);
    for (long long m_base = 0; m_base < context.m; m_base += kMTile) {
      const long long m_count = std::min(kMTile, context.m - m_base);
      std::fill_n(gate_accumulators,
                  kAccumulatorCount * kMaximumRowTile * kMTile, 0.0f);
      std::fill_n(up_accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);

      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          decode_prepared_chunk<Layout>(context.gate_info, context.gate_panel,
                                        row_panel, block, lanes, item_base,
                                        items, first_output, gate_decoded);
          decode_prepared_chunk<Layout>(context.up_info, context.up_panel,
                                        row_panel, block, lanes, item_base,
                                        items, first_output, up_decoded);
          const long long first_column = block * info.block_size + item_base;
          bool typed_activations_decoded = false;
          if constexpr (Type != FloatStorageType::kF32) {
            if (gate_order == nullptr && up_order == nullptr) {
              typed_activations_decoded = true;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                const long long input_index =
                    (m_base + m_lane) * info.columns + first_column;
                if (!simd_decode_storage_block<Type>(
                        context.input.data, input_index, items,
                        decoded_activations + m_lane * kKTile)) {
                  typed_activations_decoded = false;
                  break;
                }
              }
            }
          }
          for (long long item = 0; item < items; ++item) {
            const long long packed_column = first_column + item;
            const long long gate_column = gate_order == nullptr
                                              ? packed_column
                                              : gate_order[packed_column];
            const long long up_column =
                up_order == nullptr ? packed_column : up_order[packed_column];
            const long long accumulator = packed_column & 3;
            for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
              const std::size_t input_row =
                  static_cast<std::size_t>((m_base + m_lane) * info.columns);
              const float gate_activation =
                  typed_activations_decoded
                      ? decoded_activations[m_lane * kKTile + item]
                      : input_value<Type>(context.input.data,
                                          input_row + gate_column);
              const float up_activation =
                  up_column == gate_column
                      ? gate_activation
                      : input_value<Type>(context.input.data,
                                          input_row + up_column);
              accumulate_output_lanes(
                  gate_accumulators +
                      (accumulator * kMTile + m_lane) * kMaximumRowTile,
                  gate_decoded + item * kMaximumRowTile, gate_activation,
                  lanes);
              accumulate_output_lanes(
                  up_accumulators +
                      (accumulator * kMTile + m_lane) * kMaximumRowTile,
                  up_decoded + item * kMaximumRowTile, up_activation, lanes);
            }
          }
        }
      }

      constexpr long long stride = kMTile * kMaximumRowTile;
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        for (long long lane = 0; lane < lanes; ++lane) {
          const long long output_column = first_output + lane;
          const std::size_t output_index = static_cast<std::size_t>(
              (m_base + m_lane) * info.rows + output_column);
          const float* gate_partial =
              gate_accumulators + m_lane * kMaximumRowTile + lane;
          const float* up_partial =
              up_accumulators + m_lane * kMaximumRowTile + lane;
          const float gate =
              (gate_partial[0] + gate_partial[stride]) +
              (gate_partial[2 * stride] + gate_partial[3 * stride]);
          const float up = (up_partial[0] + up_partial[stride]) +
                           (up_partial[2 * stride] + up_partial[3 * stride]);
          store_gate_up_output<FuseSwiGLU>(context, output_index, gate, up);
        }
      }
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout, bool FuseSwiGLU>
void canonical_gate_up_rows(void* opaque, long long begin, long long end, int) {
  const auto& context = *static_cast<const CanonicalGateUpContext*>(opaque);
#if defined(__x86_64__) || defined(_M_X64)
  if constexpr (FuseSwiGLU) {
    const auto* gate_order = prepared_table<int>(
        context.gate_panel, context.gate_info.act_order_offset);
    const auto* up_order =
        prepared_table<int>(context.up_panel, context.up_info.act_order_offset);
    constexpr long long kFusionRowTile = 16;
    alignas(64) float gate_values[kFusionRowTile];
    alignas(64) float up_values[kFusionRowTile];
    for (long long row_base = begin; row_base < end;
         row_base += kFusionRowTile) {
      const long long rows = std::min(kFusionRowTile, end - row_base);
      for (long long lane = 0; lane < rows; ++lane) {
        float sums[kAccumulatorCount] = {};
        for (long long block = 0; block < context.gate_info.blocks_per_row;
             ++block) {
          sums[block & 3] += prepared_row_block_dot<Type, Layout>(
              context.gate_info, context.gate_panel, context.input.data,
              gate_order, row_base + lane, block);
        }
        gate_values[lane] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
      }
      for (long long lane = 0; lane < rows; ++lane) {
        float sums[kAccumulatorCount] = {};
        for (long long block = 0; block < context.up_info.blocks_per_row;
             ++block) {
          sums[block & 3] += prepared_row_block_dot<Type, Layout>(
              context.up_info, context.up_panel, context.input.data, up_order,
              row_base + lane, block);
        }
        up_values[lane] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
      }
      for (long long lane = 0; lane < rows; ++lane) {
        store_gate_up_output<FuseSwiGLU>(
            context, static_cast<std::size_t>(row_base + lane),
            gate_values[lane], up_values[lane]);
      }
    }
    return;
  }
  CanonicalStorageContext gate_context{context.gate_info,
                                       context.gate_panel,
                                       context.input,
                                       context.gate_output,
                                       1,
                                       nullptr,
                                       LinearActivation::kNone};
  CanonicalStorageContext up_context{context.up_info,
                                     context.up_panel,
                                     context.input,
                                     context.up_output,
                                     1,
                                     nullptr,
                                     LinearActivation::kNone};
  canonical_gemv_storage_rows<Type, Layout>(&gate_context, begin, end, 0);
  canonical_gemv_storage_rows<Type, Layout>(&up_context, begin, end, 0);
#else
  const CpuPackedWeightsInfo& info = context.gate_info;
  const auto* gate_order =
      prepared_table<int>(context.gate_panel, info.act_order_offset);
  const auto* up_order =
      prepared_table<int>(context.up_panel, context.up_info.act_order_offset);
  for (long long row = begin; row < end; ++row) {
    float gate_sums[kAccumulatorCount] = {};
    float up_sums[kAccumulatorCount] = {};
    for (long long block = 0; block < info.blocks_per_row; ++block) {
      gate_sums[block & 3] += prepared_row_block_dot<Type, Layout>(
          context.gate_info, context.gate_panel, context.input.data, gate_order,
          row, block);
      up_sums[block & 3] += prepared_row_block_dot<Type, Layout>(
          context.up_info, context.up_panel, context.input.data, up_order, row,
          block);
    }
    store_gate_up_output<FuseSwiGLU>(
        context, static_cast<std::size_t>(row),
        (gate_sums[0] + gate_sums[1]) + (gate_sums[2] + gate_sums[3]),
        (up_sums[0] + up_sums[1]) + (up_sums[2] + up_sums[3]));
  }
#endif
}

struct CanonicalSwiGLUQuantContext {
  CpuPackedWeightsInfo gate_info;
  CpuPackedWeightsInfo up_info;
  const std::uint8_t* gate_panel = nullptr;
  const std::uint8_t* up_panel = nullptr;
  FloatStorageInput input;
  CanonicalQuantTensor* output = nullptr;
  float* scratch = nullptr;
  float* worker_maximum = nullptr;
  std::atomic<bool>* invalid = nullptr;
  long long m = 0;
  long long quant_group = 0;
  long long groups_per_row = 0;
  long long m_tile = 1;
  long long scratch_elements = 0;
  float global_scale = 0.0f;
  bool scale_2d = false;
  bool measure_only = false;
  long long output_row_base = 0;
};

template <FloatStorageType Type>
const void* storage_row(const FloatStorageInput& input, long long row,
                        long long columns) {
  const std::size_t offset = static_cast<std::size_t>(row * columns);
  if constexpr (Type == FloatStorageType::kF32) {
    return static_cast<const float*>(input.data) + offset;
  } else {
    return static_cast<const std::uint16_t*>(input.data) + offset;
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void project_swiglu_group(const CanonicalSwiGLUQuantContext& context,
                          long long input_row, long long first_output,
                          long long outputs, float* gate_values,
                          float* up_values) {
  const auto* gate_order = prepared_table<int>(
      context.gate_panel, context.gate_info.act_order_offset);
  const auto* up_order =
      prepared_table<int>(context.up_panel, context.up_info.act_order_offset);
  const void* input =
      storage_row<Type>(context.input, input_row, context.gate_info.columns);
  for (long long lane = 0; lane < outputs; ++lane) {
    const long long output = first_output + lane;
    float sums[kAccumulatorCount] = {};
    for (long long block = 0; block < context.gate_info.blocks_per_row;
         ++block) {
      sums[block & 3] += prepared_row_block_dot<Type, Layout>(
          context.gate_info, context.gate_panel, input, gate_order, output,
          block);
    }
    gate_values[lane] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  }
  for (long long lane = 0; lane < outputs; ++lane) {
    const long long output = first_output + lane;
    float sums[kAccumulatorCount] = {};
    for (long long block = 0; block < context.up_info.blocks_per_row; ++block) {
      sums[block & 3] += prepared_row_block_dot<Type, Layout>(
          context.up_info, context.up_panel, input, up_order, output, block);
    }
    up_values[lane] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  }
  for (long long lane = 0; lane < outputs; ++lane) {
    const float gate = gate_values[lane];
    gate_values[lane] = gate / (1.0f + std::exp(-gate)) * up_values[lane];
  }
}

float group_absmax(const float* values, long long count,
                   std::atomic<bool>* invalid) {
  float maximum = 0.0f;
  for (long long index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      invalid->store(true, std::memory_order_relaxed);
      continue;
    }
    maximum = std::max(maximum, std::fabs(values[index]));
  }
  return maximum;
}

std::uint8_t encode_e8m0_scale(float requested) {
  if (!(requested > 0.0f)) return 0;
  const int exponent = static_cast<int>(std::ceil(std::log2(requested)));
  return static_cast<std::uint8_t>(std::clamp(exponent + 127, 0, 254));
}

void encode_swiglu_quant_group(const CanonicalSwiGLUQuantContext& context,
                               long long row, long long group,
                               const float* values) {
  CanonicalQuantTensor& output = *context.output;
  row += context.output_row_base;
  const long long first = group * context.quant_group;
  const std::size_t flat =
      static_cast<std::size_t>(row * context.gate_info.rows + first);
  const std::size_t scale_index =
      static_cast<std::size_t>(row * context.groups_per_row + group);
  const float maximum =
      group_absmax(values, context.quant_group, context.invalid);
  if (context.invalid->load(std::memory_order_relaxed)) return;
  switch (output.metadata.layout) {
    case CanonicalQuantLayout::kInt4Symmetric: {
      const float scale = maximum / 7.0f;
      output.scales[scale_index] = scale;
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      for (long long pair = 0; pair < context.quant_group / 2; ++pair) {
        const int low = std::clamp(
            static_cast<int>(std::nearbyint(values[2 * pair] * inverse)), -8,
            7);
        const int high = std::clamp(
            static_cast<int>(std::nearbyint(values[2 * pair + 1] * inverse)),
            -8, 7);
        output.data[flat / 2 + static_cast<std::size_t>(pair)] =
            static_cast<std::uint8_t>(
                (static_cast<unsigned>(low) & 15U) |
                ((static_cast<unsigned>(high) & 15U) << 4));
      }
      break;
    }
    case CanonicalQuantLayout::kUInt4Affine: {
      float minimum = values[0];
      float upper = values[0];
      for (long long item = 1; item < context.quant_group; ++item) {
        minimum = std::min(minimum, values[item]);
        upper = std::max(upper, values[item]);
      }
      float scale = (upper - minimum) / 15.0f;
      float zero = 0.0f;
      if (scale > 0.0f) {
        zero = -minimum / scale;
      } else if (upper > 0.0f) {
        scale = upper / 15.0f;
      } else if (minimum < 0.0f) {
        scale = -minimum / 15.0f;
        zero = 15.0f;
      }
      output.scales[scale_index] = scale;
      output.zero_points[scale_index] = zero;
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      for (long long pair = 0; pair < context.quant_group / 2; ++pair) {
        const int low = std::clamp(
            static_cast<int>(std::nearbyint(values[2 * pair] * inverse + zero)),
            0, 15);
        const int high = std::clamp(static_cast<int>(std::nearbyint(
                                        values[2 * pair + 1] * inverse + zero)),
                                    0, 15);
        output.data[flat / 2 + static_cast<std::size_t>(pair)] =
            static_cast<std::uint8_t>(low | (high << 4));
      }
      break;
    }
    case CanonicalQuantLayout::kInt8Symmetric: {
      const float scale = maximum / 127.0f;
      output.scales[scale_index] = scale;
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      auto* codes = reinterpret_cast<std::int8_t*>(output.data.data()) + flat;
      for (long long item = 0; item < context.quant_group; ++item) {
        codes[item] = static_cast<std::int8_t>(
            std::clamp(static_cast<int>(std::nearbyint(values[item] * inverse)),
                       -127, 127));
      }
      break;
    }
    case CanonicalQuantLayout::kInt8Affine: {
      float minimum = values[0];
      float upper = values[0];
      for (long long item = 1; item < context.quant_group; ++item) {
        minimum = std::min(minimum, values[item]);
        upper = std::max(upper, values[item]);
      }
      const float range = upper - minimum;
      const float scale = range > 0.0f
                              ? range / 255.0f
                              : std::max(std::fabs(minimum) / 127.0f, 1e-7f);
      const int zero =
          static_cast<int>(std::nearbyint(-128.0f - minimum / scale));
      output.scales[scale_index] = scale;
      output.zero_points[scale_index] = static_cast<float>(zero);
      auto* codes = reinterpret_cast<std::int8_t*>(output.data.data()) + flat;
      for (long long item = 0; item < context.quant_group; ++item) {
        codes[item] = static_cast<std::int8_t>(std::clamp(
            static_cast<int>(std::nearbyint(values[item] / scale)) + zero, -128,
            127));
      }
      break;
    }
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
    case CanonicalQuantLayout::kFP4E2M1: {
      const bool fp4 = output.metadata.layout == CanonicalQuantLayout::kFP4E2M1;
      const Float8Format format =
          output.metadata.layout == CanonicalQuantLayout::kFP8E5M2
              ? Float8Format::kE5M2
              : Float8Format::kE4M3FN;
      const float maximum_code =
          fp4 ? 6.0f : (format == Float8Format::kE4M3FN ? 448.0f : 57344.0f);
      float scale = maximum == 0.0f ? 0.0f : maximum / maximum_code;
      if (output.metadata.scale_mode == QuantScaleMode::kTensor) {
        scale = output.scales[0];
      } else {
        if (fp4) scale = f16_to_float(float_to_f16(scale));
        output.scales[scale_index] = scale;
      }
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      if (fp4) {
        for (long long pair = 0; pair < context.quant_group / 2; ++pair) {
          const std::uint8_t low = fp4_e2m1_encode(values[2 * pair] * inverse);
          const std::uint8_t high =
              fp4_e2m1_encode(values[2 * pair + 1] * inverse);
          output.data[flat / 2 + static_cast<std::size_t>(pair)] =
              static_cast<std::uint8_t>(low | (high << 4));
        }
      } else {
        for (long long item = 0; item < context.quant_group; ++item) {
          output.data[flat + static_cast<std::size_t>(item)] =
              float8_encode(values[item] * inverse, format);
        }
      }
      break;
    }
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
    case CanonicalQuantLayout::kMXFP4E2M1E8M0: {
      const bool fp4 =
          output.metadata.layout == CanonicalQuantLayout::kMXFP4E2M1E8M0;
      const std::size_t block_bytes = fp4 ? 17 : 33;
      std::uint8_t* destination =
          output.data.data() + scale_index * block_bytes;
      const std::uint8_t scale_code =
          encode_e8m0_scale(maximum / (fp4 ? 6.0f : 448.0f));
      destination[0] = scale_code;
      const float scale = maximum == 0.0f ? 0.0f : e8m0_decode(scale_code);
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      if (fp4) {
        for (long long pair = 0; pair < 16; ++pair) {
          const std::uint8_t low = fp4_e2m1_encode(values[2 * pair] * inverse);
          const std::uint8_t high =
              fp4_e2m1_encode(values[2 * pair + 1] * inverse);
          destination[1 + pair] = static_cast<std::uint8_t>(low | (high << 4));
        }
      } else {
        for (long long item = 0; item < 32; ++item) {
          destination[1 + item] =
              float8_encode(values[item] * inverse, Float8Format::kE4M3FN);
        }
      }
      break;
    }
    case CanonicalQuantLayout::kNVFP4E2M1E4M3:
    case CanonicalQuantLayout::kBitNetTernary:
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      break;
  }
}

void encode_swiglu_nvfp4_group(const CanonicalSwiGLUQuantContext& context,
                               long long first_row, long long rows,
                               long long group, const float* values) {
  CanonicalQuantTensor& output = *context.output;
  first_row += context.output_row_base;
  const long long columns = context.gate_info.rows;
  float maximum = group_absmax(values, rows * 16, context.invalid);
  if (context.invalid->load(std::memory_order_relaxed)) return;
  const float requested = context.global_scale > 0.0f
                              ? maximum / (6.0f * context.global_scale)
                              : 0.0f;
  const std::uint8_t scale_code =
      float8_encode(requested, Float8Format::kE4M3FN);
  const float scale =
      context.global_scale * float8_decode(scale_code, Float8Format::kE4M3FN);
  const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
  for (long long row_lane = 0; row_lane < rows; ++row_lane) {
    const long long row = first_row + row_lane;
    const std::size_t block =
        static_cast<std::size_t>(row * context.groups_per_row + group);
    output.scale_codes[block] = scale_code;
    std::uint8_t* destination =
        output.data.data() +
        static_cast<std::size_t>(row * (columns / 2) + group * 8);
    const float* row_values = values + row_lane * 16;
    for (long long pair = 0; pair < 8; ++pair) {
      const std::uint8_t low = fp4_e2m1_encode(row_values[2 * pair] * inverse);
      const std::uint8_t high =
          fp4_e2m1_encode(row_values[2 * pair + 1] * inverse);
      destination[pair] = static_cast<std::uint8_t>(low | (high << 4));
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_swiglu_quant_groups(void* opaque, long long begin, long long end,
                                   int worker) {
  auto& context = *static_cast<CanonicalSwiGLUQuantContext*>(opaque);
  float* gate_values = context.scratch + static_cast<long long>(worker) *
                                             context.scratch_elements * 2;
  float* up_values = gate_values + context.scratch_elements;
  float local_maximum = context.worker_maximum[worker];
  const bool nvfp4 =
      context.output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;
  const long long row_domain = nvfp4 && context.scale_2d ? 16 : 1;
  for (long long task = begin; task < end; ++task) {
    const long long row_domain_index = task / context.groups_per_row;
    const long long group = task % context.groups_per_row;
    const long long first_row = row_domain_index * row_domain;
    const long long rows = std::min(row_domain, context.m - first_row);
    for (long long row_lane = 0; row_lane < rows; ++row_lane) {
      project_swiglu_group<Type, Layout>(
          context, first_row + row_lane, group * context.quant_group,
          context.quant_group, gate_values + row_lane * context.quant_group,
          up_values + row_lane * context.quant_group);
    }
    if (context.measure_only) {
      local_maximum = std::max(
          local_maximum, group_absmax(gate_values, rows * context.quant_group,
                                      context.invalid));
    } else if (nvfp4) {
      encode_swiglu_nvfp4_group(context, first_row, rows, group, gate_values);
    } else {
      encode_swiglu_quant_group(context, first_row, group, gate_values);
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_swiglu_quant_full_rows(void* opaque, long long begin,
                                      long long end, int worker) {
  auto& context = *static_cast<CanonicalSwiGLUQuantContext*>(opaque);
  float* gate_values = context.scratch + static_cast<long long>(worker) *
                                             context.scratch_elements * 2;
  float* up_values = gate_values + context.scratch_elements;
  float local_maximum = context.worker_maximum[worker];
  const bool nvfp4 =
      context.output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;
  for (long long row = begin; row < end; ++row) {
    project_swiglu_group<Type, Layout>(context, row, 0, context.gate_info.rows,
                                       gate_values, up_values);
    if (context.measure_only) {
      local_maximum = std::max(
          local_maximum,
          group_absmax(gate_values, context.gate_info.rows, context.invalid));
      continue;
    }
    for (long long group = 0; group < context.groups_per_row; ++group) {
      const float* values = gate_values + group * context.quant_group;
      if (nvfp4) {
        encode_swiglu_nvfp4_group(context, row, 1, group, values);
      } else {
        encode_swiglu_quant_group(context, row, group, values);
      }
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void canonical_swiglu_quant_panel_groups(void* opaque, long long begin,
                                         long long end, int worker) {
  auto& context = *static_cast<CanonicalSwiGLUQuantContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.gate_info;
  const auto* gate_order =
      prepared_table<int>(context.gate_panel, info.act_order_offset);
  const auto* up_order =
      prepared_table<int>(context.up_panel, context.up_info.act_order_offset);
  float* gate_values = context.scratch + static_cast<long long>(worker) *
                                             context.scratch_elements * 2;
  alignas(64) float gate_decoded[kMaximumRowTile * kKTile];
  alignas(64) float up_decoded[kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(
      64) float gate_accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  alignas(
      64) float up_accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  float local_maximum = context.worker_maximum[worker];
  const bool nvfp4 =
      context.output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;

  for (long long task = begin; task < end; ++task) {
    const long long m_tile_index = task / context.groups_per_row;
    const long long group = task % context.groups_per_row;
    const long long m_base = m_tile_index * context.m_tile;
    const long long m_count = std::min(context.m_tile, context.m - m_base);
    const long long group_first = group * context.quant_group;
    const long long first_panel = group_first / info.row_tile;
    const long long panel_count = context.quant_group / info.row_tile;

    for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
      const long long row_panel = first_panel + panel_lane;
      const long long first_output = row_panel * info.row_tile;
      const long long lanes = std::min(info.row_tile, info.rows - first_output);
      std::fill_n(gate_accumulators,
                  kAccumulatorCount * kMaximumRowTile * kMTile, 0.0f);
      std::fill_n(up_accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          decode_prepared_chunk<Layout>(context.gate_info, context.gate_panel,
                                        row_panel, block, lanes, item_base,
                                        items, first_output, gate_decoded);
          decode_prepared_chunk<Layout>(context.up_info, context.up_panel,
                                        row_panel, block, lanes, item_base,
                                        items, first_output, up_decoded);
          const long long first_column = block * info.block_size + item_base;
          bool typed_activations_decoded = false;
          if constexpr (Type != FloatStorageType::kF32) {
            if (gate_order == nullptr && up_order == nullptr) {
              typed_activations_decoded = true;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                const long long input_index =
                    (m_base + m_lane) * info.columns + first_column;
                if (!simd_decode_storage_block<Type>(
                        context.input.data, input_index, items,
                        decoded_activations + m_lane * kKTile)) {
                  typed_activations_decoded = false;
                  break;
                }
              }
            }
          }
          for (long long item = 0; item < items; ++item) {
            const long long packed_column = first_column + item;
            const long long gate_column = gate_order == nullptr
                                              ? packed_column
                                              : gate_order[packed_column];
            const long long up_column =
                up_order == nullptr ? packed_column : up_order[packed_column];
            const long long accumulator = packed_column & 3;
            for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
              const std::size_t input_row =
                  static_cast<std::size_t>((m_base + m_lane) * info.columns);
              const float gate_activation =
                  typed_activations_decoded
                      ? decoded_activations[m_lane * kKTile + item]
                      : input_value<Type>(context.input.data,
                                          input_row + gate_column);
              const float up_activation =
                  up_column == gate_column
                      ? gate_activation
                      : input_value<Type>(context.input.data,
                                          input_row + up_column);
              accumulate_output_lanes(
                  gate_accumulators +
                      (accumulator * kMTile + m_lane) * kMaximumRowTile,
                  gate_decoded + item * kMaximumRowTile, gate_activation,
                  lanes);
              accumulate_output_lanes(
                  up_accumulators +
                      (accumulator * kMTile + m_lane) * kMaximumRowTile,
                  up_decoded + item * kMaximumRowTile, up_activation, lanes);
            }
          }
        }
      }
      constexpr long long stride = kMTile * kMaximumRowTile;
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        for (long long lane = 0; lane < lanes; ++lane) {
          const float* gate_partial =
              gate_accumulators + m_lane * kMaximumRowTile + lane;
          const float* up_partial =
              up_accumulators + m_lane * kMaximumRowTile + lane;
          const float gate =
              (gate_partial[0] + gate_partial[stride]) +
              (gate_partial[2 * stride] + gate_partial[3 * stride]);
          const float up = (up_partial[0] + up_partial[stride]) +
                           (up_partial[2 * stride] + up_partial[3 * stride]);
          const long long value_index =
              m_lane * context.quant_group + panel_lane * info.row_tile + lane;
          gate_values[value_index] = gate / (1.0f + std::exp(-gate)) * up;
        }
      }
    }

    if (context.measure_only) {
      local_maximum =
          std::max(local_maximum,
                   group_absmax(gate_values, m_count * context.quant_group,
                                context.invalid));
    } else if (nvfp4 && context.scale_2d) {
      encode_swiglu_nvfp4_group(context, m_base, m_count, group, gate_values);
    } else {
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        if (nvfp4) {
          encode_swiglu_nvfp4_group(context, m_base + m_lane, 1, group,
                                    gate_values + m_lane * context.quant_group);
        } else {
          encode_swiglu_quant_group(context, m_base + m_lane, group,
                                    gate_values + m_lane * context.quant_group);
        }
      }
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

template <FloatStorageType Type, CanonicalQuantLayout Layout,
          long long PanelGroup>
void canonical_typed_storage_panel_groups(void* opaque, long long begin,
                                          long long end, int) {
  const auto& context = *static_cast<const CanonicalStorageContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
  alignas(64) float decoded_weights[kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(64) float
      accumulators[PanelGroup * kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long group = begin; group < end; ++group) {
    const long long first_panel = group * PanelGroup;
    const long long panel_count = std::min(PanelGroup, panels - first_panel);
    for (long long m_base = 0; m_base < context.m; m_base += kMTile) {
      const long long m_count = std::min(kMTile, context.m - m_base);
      std::fill_n(accumulators,
                  PanelGroup * kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          const long long first_column = block * info.block_size + item_base;
          for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
            const long long input_index =
                (m_base + m_lane) * info.columns + first_column;
            if (!simd_decode_storage_block<Type>(
                    context.input.data, input_index, items,
                    decoded_activations + m_lane * kKTile)) {
              for (long long item = 0; item < items; ++item) {
                decoded_activations[m_lane * kKTile + item] = input_value<Type>(
                    context.input.data,
                    static_cast<std::size_t>(input_index + item));
              }
            }
          }
          for (long long panel_lane = 0; panel_lane < panel_count;
               ++panel_lane) {
            const long long row_panel = first_panel + panel_lane;
            const long long first_output = row_panel * info.row_tile;
            const long long lanes =
                std::min(info.row_tile, info.rows - first_output);
            decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                          lanes, item_base, items, first_output,
                                          decoded_weights);
            bool avx2_accumulated = false;
#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
            if (cpu_features().avx2 && lanes == 8) {
              quant::canonical_accumulate_tile_avx2(
                  accumulators +
                      panel_lane * kAccumulatorCount * kMTile * kMaximumRowTile,
                  decoded_weights, decoded_activations, kKTile, first_column,
                  items, m_count, lanes);
              avx2_accumulated = true;
            }
#endif
            if (!avx2_accumulated) {
              for (long long item = 0; item < items; ++item) {
                const long long accumulator = (first_column + item) & 3;
                for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                  accumulate_output_lanes(
                      accumulators +
                          ((panel_lane * kAccumulatorCount + accumulator) *
                               kMTile +
                           m_lane) *
                              kMaximumRowTile,
                      decoded_weights + item * kMaximumRowTile,
                      decoded_activations[m_lane * kKTile + item], lanes);
                }
              }
            }
          }
        }
      }
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
          const long long first_output =
              (first_panel + panel_lane) * info.row_tile;
          const long long lanes =
              std::min(info.row_tile, info.rows - first_output);
          const float* panel_accumulators =
              accumulators +
              panel_lane * kAccumulatorCount * kMTile * kMaximumRowTile;
          const std::size_t first_output_index = static_cast<std::size_t>(
              (m_base + m_lane) * info.rows + first_output);
          const float* row_partial =
              panel_accumulators + m_lane * kMaximumRowTile;
          constexpr long long stride = kMTile * kMaximumRowTile;
          if (store_relu2_lanes(context, first_output_index, first_output,
                                row_partial, stride, lanes)) {
            continue;
          }
          for (long long lane = 0; lane < lanes; ++lane) {
            const long long output_column = first_output + lane;
            const float* partial =
                panel_accumulators + m_lane * kMaximumRowTile + lane;
            store_projection_output(
                context,
                static_cast<std::size_t>((m_base + m_lane) * info.rows +
                                         output_column),
                output_column,
                (partial[0] + partial[stride]) +
                    (partial[2 * stride] + partial[3 * stride]));
          }
        }
      }
    }
  }
}

struct CanonicalQuantizedContext {
  CpuPackedWeightsInfo info;
  const std::uint8_t* panel = nullptr;
  const CanonicalQuantTensor* activation = nullptr;
  float* output = nullptr;
};

template <CanonicalQuantLayout Layout>
void canonical_quantized_panels(void* opaque, long long begin, long long end,
                                int) {
  const auto& context = *static_cast<const CanonicalQuantizedContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const long long m = context.activation->metadata.logical_rows;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  alignas(64) float decoded_weights[kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long row_panel = begin; row_panel < end; ++row_panel) {
    const long long first_output = row_panel * info.row_tile;
    const long long lanes = std::min(info.row_tile, info.rows - first_output);
    for (long long m_base = 0; m_base < m; m_base += kMTile) {
      const long long m_count = std::min(kMTile, m - m_base);
      std::fill_n(accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          for (long long item = 0; item < items; ++item) {
            const long long packed_column =
                block * info.block_size + item_base + item;
            const long long logical_column =
                order == nullptr ? packed_column : order[packed_column];
            for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
              decoded_activations[m_lane * kKTile + item] = tensor_value(
                  *context.activation, m_base + m_lane, logical_column);
            }
          }
          decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                        lanes, item_base, items, first_output,
                                        decoded_weights);
          bool avx2_accumulated = false;
#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
          if (cpu_features().avx2 && order == nullptr && lanes == 8) {
            quant::canonical_accumulate_tile_avx2(
                accumulators, decoded_weights, decoded_activations, kKTile,
                block * info.block_size + item_base, items, m_count, lanes);
            avx2_accumulated = true;
          }
#endif
          if (!avx2_accumulated) {
            for (long long item = 0; item < items; ++item) {
              const long long packed_column =
                  block * info.block_size + item_base + item;
              const long long accumulator = packed_column & 3;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                const float activation =
                    decoded_activations[m_lane * kKTile + item];
                accumulate_output_lanes(
                    accumulators +
                        (accumulator * kMTile + m_lane) * kMaximumRowTile,
                    decoded_weights + item * kMaximumRowTile, activation,
                    lanes);
              }
            }
          }
        }
      }
      for (long long lane = 0; lane < lanes; ++lane) {
        const long long output_column = first_output + lane;
        for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
          const float* partial = accumulators + m_lane * kMaximumRowTile + lane;
          constexpr long long stride = kMTile * kMaximumRowTile;
          context.output[(m_base + m_lane) * info.rows + output_column] =
              (partial[0] + partial[stride]) +
              (partial[2 * stride] + partial[3 * stride]);
        }
      }
    }
  }
}

bool projection_metadata_supported(const QuantTensorMetadata& metadata) {
  return metadata.layout != CanonicalQuantLayout::kTurboQuantKey &&
         metadata.layout != CanonicalQuantLayout::kTurboQuantValue;
}

#if defined(__x86_64__) || defined(_M_X64)
bool x86_f32_panel_layout(CanonicalQuantLayout layout) {
  return layout == CanonicalQuantLayout::kFP8E4M3FN ||
         layout == CanonicalQuantLayout::kFP8E5M2;
}
#endif

template <FloatStorageType Type>
threading::RangeFn storage_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_STORAGE_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:   \
    return canonical_storage_panels<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_STORAGE_CASE(kInt4Symmetric);
    QUIXICORE_STORAGE_CASE(kUInt4Affine);
    QUIXICORE_STORAGE_CASE(kInt8Symmetric);
    QUIXICORE_STORAGE_CASE(kInt8Affine);
    QUIXICORE_STORAGE_CASE(kFP8E4M3FN);
    QUIXICORE_STORAGE_CASE(kFP8E5M2);
    QUIXICORE_STORAGE_CASE(kFP4E2M1);
    QUIXICORE_STORAGE_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_STORAGE_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_STORAGE_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_STORAGE_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_STORAGE_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn qkv_shared_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QKV_SHARED_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:      \
    return canonical_qkv_shared_panels<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_QKV_SHARED_CASE(kInt4Symmetric);
    QUIXICORE_QKV_SHARED_CASE(kUInt4Affine);
    QUIXICORE_QKV_SHARED_CASE(kInt8Symmetric);
    QUIXICORE_QKV_SHARED_CASE(kInt8Affine);
    QUIXICORE_QKV_SHARED_CASE(kFP8E4M3FN);
    QUIXICORE_QKV_SHARED_CASE(kFP8E5M2);
    QUIXICORE_QKV_SHARED_CASE(kFP4E2M1);
    QUIXICORE_QKV_SHARED_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QKV_SHARED_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QKV_SHARED_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QKV_SHARED_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QKV_SHARED_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn qkv_rope_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QKV_ROPE_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:    \
    return canonical_qkv_rope_rows<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_QKV_ROPE_CASE(kInt4Symmetric);
    QUIXICORE_QKV_ROPE_CASE(kUInt4Affine);
    QUIXICORE_QKV_ROPE_CASE(kInt8Symmetric);
    QUIXICORE_QKV_ROPE_CASE(kInt8Affine);
    QUIXICORE_QKV_ROPE_CASE(kFP8E4M3FN);
    QUIXICORE_QKV_ROPE_CASE(kFP8E5M2);
    QUIXICORE_QKV_ROPE_CASE(kFP4E2M1);
    QUIXICORE_QKV_ROPE_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QKV_ROPE_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QKV_ROPE_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QKV_ROPE_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QKV_ROPE_CASE
  return nullptr;
}

threading::RangeFn qkv_rope_f32_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QKV_ROPE_F32_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:        \
    return canonical_qkv_rope_f32_tiles<CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_QKV_ROPE_F32_CASE(kInt4Symmetric);
    QUIXICORE_QKV_ROPE_F32_CASE(kUInt4Affine);
    QUIXICORE_QKV_ROPE_F32_CASE(kInt8Symmetric);
    QUIXICORE_QKV_ROPE_F32_CASE(kInt8Affine);
    QUIXICORE_QKV_ROPE_F32_CASE(kFP8E4M3FN);
    QUIXICORE_QKV_ROPE_F32_CASE(kFP8E5M2);
    QUIXICORE_QKV_ROPE_F32_CASE(kFP4E2M1);
    QUIXICORE_QKV_ROPE_F32_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QKV_ROPE_F32_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QKV_ROPE_F32_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QKV_ROPE_F32_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QKV_ROPE_F32_CASE
  return nullptr;
}

threading::RangeFn qkv_rope_f32_panel_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QKV_ROPE_F32_PANEL_CASE(layout_name)       \
  case CanonicalQuantLayout::layout_name:                    \
    return canonical_qkv_rope_panels<FloatStorageType::kF32, \
                                     CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kInt4Symmetric);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kUInt4Affine);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kInt8Symmetric);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kInt8Affine);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kFP8E4M3FN);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kFP8E5M2);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kFP4E2M1);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QKV_ROPE_F32_PANEL_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QKV_ROPE_F32_PANEL_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn qkv_rope_panel_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QKV_ROPE_PANEL_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:          \
    return canonical_qkv_rope_panel_groups<        \
        Type, CanonicalQuantLayout::layout_name, kTypedPanelGroup>
  switch (layout) {
    QUIXICORE_QKV_ROPE_PANEL_CASE(kInt4Symmetric);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kUInt4Affine);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kInt8Symmetric);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kInt8Affine);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kFP8E4M3FN);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kFP8E5M2);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kFP4E2M1);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QKV_ROPE_PANEL_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QKV_ROPE_PANEL_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn qkv_rope_typed_panel_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:                \
    return canonical_qkv_rope_typed_panels<Type,         \
                                           CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kInt4Symmetric);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kUInt4Affine);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kInt8Symmetric);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kInt8Affine);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kFP8E4M3FN);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kFP8E5M2);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kFP4E2M1);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QKV_ROPE_TYPED_PANEL_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn gemv_storage_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_GEMV_STORAGE_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:        \
    return canonical_gemv_storage_rows<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_GEMV_STORAGE_CASE(kInt4Symmetric);
    QUIXICORE_GEMV_STORAGE_CASE(kUInt4Affine);
    QUIXICORE_GEMV_STORAGE_CASE(kInt8Symmetric);
    QUIXICORE_GEMV_STORAGE_CASE(kInt8Affine);
    QUIXICORE_GEMV_STORAGE_CASE(kFP8E4M3FN);
    QUIXICORE_GEMV_STORAGE_CASE(kFP8E5M2);
    QUIXICORE_GEMV_STORAGE_CASE(kFP4E2M1);
    QUIXICORE_GEMV_STORAGE_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_GEMV_STORAGE_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_GEMV_STORAGE_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_GEMV_STORAGE_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_GEMV_STORAGE_CASE
  return nullptr;
}

template <FloatStorageType Type, bool FuseSwiGLU>
threading::RangeFn gate_up_storage_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_GATE_UP_CASE(layout_name)                                  \
  case CanonicalQuantLayout::layout_name:                                    \
    return canonical_gate_up_panels<Type, CanonicalQuantLayout::layout_name, \
                                    FuseSwiGLU>
  switch (layout) {
    QUIXICORE_GATE_UP_CASE(kInt4Symmetric);
    QUIXICORE_GATE_UP_CASE(kUInt4Affine);
    QUIXICORE_GATE_UP_CASE(kInt8Symmetric);
    QUIXICORE_GATE_UP_CASE(kInt8Affine);
    QUIXICORE_GATE_UP_CASE(kFP8E4M3FN);
    QUIXICORE_GATE_UP_CASE(kFP8E5M2);
    QUIXICORE_GATE_UP_CASE(kFP4E2M1);
    QUIXICORE_GATE_UP_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_GATE_UP_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_GATE_UP_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_GATE_UP_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_GATE_UP_CASE
  return nullptr;
}

template <FloatStorageType Type, bool FuseSwiGLU>
threading::RangeFn gate_up_row_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_GATE_UP_ROW_CASE(layout_name)                            \
  case CanonicalQuantLayout::layout_name:                                  \
    return canonical_gate_up_rows<Type, CanonicalQuantLayout::layout_name, \
                                  FuseSwiGLU>
  switch (layout) {
    QUIXICORE_GATE_UP_ROW_CASE(kInt4Symmetric);
    QUIXICORE_GATE_UP_ROW_CASE(kUInt4Affine);
    QUIXICORE_GATE_UP_ROW_CASE(kInt8Symmetric);
    QUIXICORE_GATE_UP_ROW_CASE(kInt8Affine);
    QUIXICORE_GATE_UP_ROW_CASE(kFP8E4M3FN);
    QUIXICORE_GATE_UP_ROW_CASE(kFP8E5M2);
    QUIXICORE_GATE_UP_ROW_CASE(kFP4E2M1);
    QUIXICORE_GATE_UP_ROW_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_GATE_UP_ROW_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_GATE_UP_ROW_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_GATE_UP_ROW_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_GATE_UP_ROW_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn swiglu_quant_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_SWIGLU_QUANT_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:        \
    return canonical_swiglu_quant_groups<Type,   \
                                         CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_SWIGLU_QUANT_CASE(kInt4Symmetric);
    QUIXICORE_SWIGLU_QUANT_CASE(kUInt4Affine);
    QUIXICORE_SWIGLU_QUANT_CASE(kInt8Symmetric);
    QUIXICORE_SWIGLU_QUANT_CASE(kInt8Affine);
    QUIXICORE_SWIGLU_QUANT_CASE(kFP8E4M3FN);
    QUIXICORE_SWIGLU_QUANT_CASE(kFP8E5M2);
    QUIXICORE_SWIGLU_QUANT_CASE(kFP4E2M1);
    QUIXICORE_SWIGLU_QUANT_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_SWIGLU_QUANT_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_SWIGLU_QUANT_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_SWIGLU_QUANT_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_SWIGLU_QUANT_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn swiglu_quant_panel_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_SWIGLU_QUANT_PANEL_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:              \
    return canonical_swiglu_quant_panel_groups<        \
        Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kInt4Symmetric);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kUInt4Affine);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kInt8Symmetric);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kInt8Affine);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kFP8E4M3FN);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kFP8E5M2);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kFP4E2M1);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_SWIGLU_QUANT_PANEL_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_SWIGLU_QUANT_PANEL_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn swiglu_quant_full_row_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:                 \
    return canonical_swiglu_quant_full_rows<Type,         \
                                            CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kInt4Symmetric);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kUInt4Affine);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kInt8Symmetric);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kInt8Affine);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kFP8E4M3FN);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kFP8E5M2);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kFP4E2M1);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_SWIGLU_QUANT_FULL_ROW_CASE
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn typed_gemv_panel_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_TYPED_GEMV_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:      \
    return canonical_typed_gemv_panels<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_TYPED_GEMV_CASE(kInt4Symmetric);
    QUIXICORE_TYPED_GEMV_CASE(kUInt4Affine);
    QUIXICORE_TYPED_GEMV_CASE(kInt8Symmetric);
    QUIXICORE_TYPED_GEMV_CASE(kInt8Affine);
    QUIXICORE_TYPED_GEMV_CASE(kFP8E4M3FN);
    QUIXICORE_TYPED_GEMV_CASE(kFP8E5M2);
    QUIXICORE_TYPED_GEMV_CASE(kFP4E2M1);
    QUIXICORE_TYPED_GEMV_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_TYPED_GEMV_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_TYPED_GEMV_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_TYPED_GEMV_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_TYPED_GEMV_CASE
  return nullptr;
}

template <FloatStorageType Type, long long PanelGroup>
threading::RangeFn typed_gemm_panel_group_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_TYPED_GEMM_CASE(layout_name)   \
  case CanonicalQuantLayout::layout_name:        \
    return canonical_typed_storage_panel_groups< \
        Type, CanonicalQuantLayout::layout_name, PanelGroup>
  switch (layout) {
    QUIXICORE_TYPED_GEMM_CASE(kInt4Symmetric);
    QUIXICORE_TYPED_GEMM_CASE(kUInt4Affine);
    QUIXICORE_TYPED_GEMM_CASE(kInt8Symmetric);
    QUIXICORE_TYPED_GEMM_CASE(kInt8Affine);
    QUIXICORE_TYPED_GEMM_CASE(kFP8E4M3FN);
    QUIXICORE_TYPED_GEMM_CASE(kFP8E5M2);
    QUIXICORE_TYPED_GEMM_CASE(kFP4E2M1);
    QUIXICORE_TYPED_GEMM_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_TYPED_GEMM_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_TYPED_GEMM_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_TYPED_GEMM_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_TYPED_GEMM_CASE
  return nullptr;
}

threading::RangeFn quantized_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_QUANTIZED_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:     \
    return canonical_quantized_panels<CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_QUANTIZED_CASE(kInt4Symmetric);
    QUIXICORE_QUANTIZED_CASE(kUInt4Affine);
    QUIXICORE_QUANTIZED_CASE(kInt8Symmetric);
    QUIXICORE_QUANTIZED_CASE(kInt8Affine);
    QUIXICORE_QUANTIZED_CASE(kFP8E4M3FN);
    QUIXICORE_QUANTIZED_CASE(kFP8E5M2);
    QUIXICORE_QUANTIZED_CASE(kFP4E2M1);
    QUIXICORE_QUANTIZED_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_QUANTIZED_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_QUANTIZED_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_QUANTIZED_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_QUANTIZED_CASE
  return nullptr;
}

Status validate_prepared_projection(const CpuPackedWeights& weights,
                                    CpuPackedWeightsInfo* info) {
  if (!weights.ready() || info == nullptr) return Status::kInvalidArgument;
  *info = weights.info();
  if (!info->has_canonical_layout ||
      !projection_metadata_supported(info->quant_metadata)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::valid_product({info->rows, info->columns}) ||
      info->row_tile <= 0 || info->row_tile > kMaximumRowTile ||
      info->block_size <= 0 || info->blocks_per_row <= 0 ||
      info->columns / info->block_size != info->blocks_per_row ||
      weights.panel_data() == nullptr) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

bool compatible_gate_up(const CpuPackedWeightsInfo& gate,
                        const CpuPackedWeightsInfo& up) {
  return gate.quant_metadata.layout == up.quant_metadata.layout &&
         gate.rows == up.rows && gate.columns == up.columns &&
         gate.row_tile == up.row_tile && gate.block_size == up.block_size &&
         gate.blocks_per_row == up.blocks_per_row &&
         gate.block_bytes == up.block_bytes;
}

bool compatible_qkv_plane(const CpuPackedWeightsInfo& q,
                          const CpuPackedWeightsInfo& other) {
  return q.quant_metadata.layout == other.quant_metadata.layout &&
         q.columns == other.columns && q.row_tile == other.row_tile &&
         q.block_size == other.block_size &&
         q.blocks_per_row == other.blocks_per_row &&
         q.block_bytes == other.block_bytes;
}

Status configure_swiglu_quant_output(CanonicalQuantTensor* output,
                                     CanonicalQuantLayout layout,
                                     long long rows, long long columns,
                                     long long requested_group, bool scale_2d,
                                     long long* quant_group,
                                     bool* needs_global_maximum) {
  if (output == nullptr || quant_group == nullptr ||
      needs_global_maximum == nullptr ||
      !detail::valid_product({rows, columns})) {
    return output == nullptr ? Status::kInvalidArgument : Status::kInvalidShape;
  }
  const std::size_t elements =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
  *needs_global_maximum = false;
  output->metadata = {};
  output->metadata.layout = layout;
  output->metadata.logical_rows = rows;
  output->metadata.logical_columns = columns;
  output->metadata.group_size = requested_group;
  output->provenance = QuantImportProvenance::kCanonical;
  output->scale_codes.clear();
  output->scales.clear();
  output->zero_points.clear();
  output->activation_scales.clear();
  output->activation_zero_points.clear();
  output->group_index.clear();
  output->act_order.clear();
  output->row_sums.clear();

  switch (layout) {
    case CanonicalQuantLayout::kInt4Symmetric:
    case CanonicalQuantLayout::kUInt4Affine: {
      if (requested_group <= 0 || columns % requested_group != 0 ||
          (requested_group & 1) != 0 || (columns & 1) != 0)
        return Status::kInvalidShape;
      *quant_group = requested_group;
      const std::size_t groups =
          elements / static_cast<std::size_t>(requested_group);
      output->data.resize(elements / 2);
      output->scales.resize(groups);
      output->metadata.scale_mode = QuantScaleMode::kGroup;
      output->metadata.scale_encoding = QuantScaleEncoding::kFP32;
      output->metadata.scale_count = groups;
      if (layout == CanonicalQuantLayout::kUInt4Affine) {
        output->zero_points.resize(groups);
        output->metadata.zero_point_mode = QuantZeroPointMode::kFractional;
        output->metadata.zero_point_count = groups;
      }
      break;
    }
    case CanonicalQuantLayout::kInt8Symmetric: {
      *quant_group = requested_group == 0 ? columns : requested_group;
      if (*quant_group <= 0 || columns % *quant_group != 0)
        return Status::kInvalidShape;
      const std::size_t groups =
          elements / static_cast<std::size_t>(*quant_group);
      output->data.resize(elements);
      output->scales.resize(groups);
      output->metadata.group_size = *quant_group;
      output->metadata.scale_mode = *quant_group == columns
                                        ? QuantScaleMode::kRow
                                        : QuantScaleMode::kGroup;
      output->metadata.scale_encoding = QuantScaleEncoding::kFP32;
      output->metadata.scale_count = groups;
      break;
    }
    case CanonicalQuantLayout::kInt8Affine: {
      if (requested_group != 0 && requested_group != columns)
        return Status::kInvalidShape;
      *quant_group = columns;
      output->data.resize(elements);
      output->scales.resize(static_cast<std::size_t>(rows));
      output->zero_points.resize(static_cast<std::size_t>(rows));
      output->metadata.group_size = columns;
      output->metadata.scale_mode = QuantScaleMode::kRow;
      output->metadata.scale_encoding = QuantScaleEncoding::kFP32;
      output->metadata.zero_point_mode = QuantZeroPointMode::kInteger;
      output->metadata.scale_count = static_cast<std::size_t>(rows);
      output->metadata.zero_point_count = static_cast<std::size_t>(rows);
      break;
    }
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
    case CanonicalQuantLayout::kFP4E2M1: {
      const bool fp4 = layout == CanonicalQuantLayout::kFP4E2M1;
      *quant_group = requested_group == 0 ? columns : requested_group;
      if (*quant_group <= 0 || columns % *quant_group != 0 ||
          (fp4 && (((*quant_group) & 1) != 0 || (columns & 1) != 0))) {
        return Status::kInvalidShape;
      }
      output->data.resize(fp4 ? elements / 2 : elements);
      if (requested_group == 0) {
        output->scales.resize(1);
        output->metadata.scale_mode = QuantScaleMode::kTensor;
        output->metadata.group_size = 0;
        output->metadata.scale_count = 1;
        output->metadata.scale_domain_rows = static_cast<int>(rows);
        *needs_global_maximum = true;
      } else {
        const std::size_t groups =
            elements / static_cast<std::size_t>(*quant_group);
        output->scales.resize(groups);
        output->metadata.scale_mode =
            fp4 ? QuantScaleMode::kBlock : QuantScaleMode::kGroup;
        output->metadata.scale_count = groups;
        output->metadata.scale_domain_rows = 1;
      }
      output->metadata.scale_encoding =
          fp4 ? QuantScaleEncoding::kFP16 : QuantScaleEncoding::kFP32;
      break;
    }
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
    case CanonicalQuantLayout::kMXFP4E2M1E8M0: {
      if (columns % 32 != 0) return Status::kInvalidShape;
      *quant_group = 32;
      const std::size_t blocks = elements / 32;
      output->data.resize(
          blocks * (layout == CanonicalQuantLayout::kMXFP4E2M1E8M0 ? 17 : 33));
      output->metadata.group_size = 32;
      output->metadata.scale_mode = QuantScaleMode::kMicroscaleK32;
      output->metadata.scale_encoding = QuantScaleEncoding::kE8M0;
      output->metadata.scale_count = blocks;
      break;
    }
    case CanonicalQuantLayout::kNVFP4E2M1E4M3: {
      if (columns % 16 != 0) return Status::kInvalidShape;
      *quant_group = 16;
      const std::size_t blocks = elements / 16;
      output->data.resize(elements / 2);
      output->scale_codes.resize(blocks);
      output->metadata.group_size = 16;
      output->metadata.scale_mode = QuantScaleMode::kNVFP4K16;
      output->metadata.scale_encoding = QuantScaleEncoding::kE4M3FN;
      output->metadata.scale_count = blocks;
      output->metadata.scale_2d = scale_2d;
      output->metadata.scale_domain_rows = scale_2d ? 16 : 1;
      *needs_global_maximum = true;
      break;
    }
    case CanonicalQuantLayout::kBitNetTernary:
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return Status::kUnsupportedFormat;
    default:
      return Status::kUnsupportedFormat;
  }
  return Status::kOk;
}

}  // namespace

static Status qgemm_prepacked_storage_impl(const CpuPackedWeights& weights,
                                           FloatStorageInput x,
                                           FloatStorageOutput y, long long m,
                                           const float* bias,
                                           LinearActivation activation,
                                           Workspace* workspace) {
  (void)workspace;
  if (!known_linear_activation(activation)) return Status::kInvalidArgument;
  CpuPackedWeightsInfo info;
  Status status = validate_prepared_projection(weights, &info);
  if (status != Status::kOk) return status;
  if (!detail::valid_product({m, info.columns}) ||
      !detail::valid_product({m, info.rows}) || x.count != m * info.columns ||
      y.count != m * info.rows) {
    return Status::kInvalidShape;
  }
  if (!known_storage(x.type) || !known_storage(y.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(x.data, y.data)) return Status::kInvalidArgument;
  CanonicalStorageContext context{
      info,      static_cast<const std::uint8_t*>(weights.panel_data()),
      x,         y,
      m,         bias,
      activation};
  threading::RangeFn kernel = nullptr;
  if (m == 1) {
    bool use_panels = false;
    long long panel_group = 1;
    switch (x.type) {
      case FloatStorageType::kF32:
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && info.act_order_offset == kMissingOffset &&
            x86_f32_panel_layout(info.quant_metadata.layout)) {
          kernel = storage_kernel<FloatStorageType::kF32>(
              info.quant_metadata.layout);
          use_panels = true;
          break;
        }
#endif
        kernel = gemv_storage_kernel<FloatStorageType::kF32>(
            info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (info.act_order_offset == kMissingOffset &&
            info.group_index_offset == kMissingOffset) {
          kernel = typed_gemv_panel_kernel<FloatStorageType::kF16>(
              info.quant_metadata.layout);
          use_panels = true;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().f16c && info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF16,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panels = true;
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
        kernel = gemv_storage_kernel<FloatStorageType::kF16>(
            info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (info.act_order_offset == kMissingOffset &&
            info.group_index_offset == kMissingOffset) {
          kernel = typed_gemv_panel_kernel<FloatStorageType::kBF16>(
              info.quant_metadata.layout);
          use_panels = true;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kBF16,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panels = true;
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
        kernel = gemv_storage_kernel<FloatStorageType::kBF16>(
            info.quant_metadata.layout);
        break;
    }
    if (use_panels) {
      const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
      threading::parallel_ranges_impl((panels + panel_group - 1) / panel_group,
                                      1, kernel, &context);
    } else {
      threading::parallel_ranges_impl(info.rows, 16, kernel, &context);
    }
    return Status::kOk;
  } else {
    bool use_panel_groups = false;
    switch (x.type) {
      case FloatStorageType::kF32:
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF32,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panel_groups = true;
          break;
        }
#endif
        kernel =
            storage_kernel<FloatStorageType::kF32>(info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF16,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panel_groups = true;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().f16c && info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF16,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panel_groups = true;
          break;
        }
#endif
        kernel =
            storage_kernel<FloatStorageType::kF16>(info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kBF16,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panel_groups = true;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && info.act_order_offset == kMissingOffset) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kBF16,
                                                 kTypedPanelGroup>(
              info.quant_metadata.layout);
          use_panel_groups = true;
          break;
        }
#endif
        kernel =
            storage_kernel<FloatStorageType::kBF16>(info.quant_metadata.layout);
        break;
    }
    const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
    threading::parallel_ranges_impl(
        use_panel_groups ? (panels + kTypedPanelGroup - 1) / kTypedPanelGroup
                         : panels,
        1, kernel, &context);
    return Status::kOk;
  }
  return Status::kOk;
}

Status qgemm_prepacked_storage(const CpuPackedWeights& weights,
                               FloatStorageInput x, FloatStorageOutput y,
                               long long m, Workspace* workspace) {
  return qgemm_prepacked_storage_impl(weights, x, y, m, nullptr,
                                      LinearActivation::kNone, workspace);
}

Status qgemm_prepacked_epilogue_storage(const CpuPackedWeights& weights,
                                        FloatStorageInput x, const float* bias,
                                        FloatStorageOutput y, long long m,
                                        LinearActivation activation,
                                        Workspace* workspace) {
  return qgemm_prepacked_storage_impl(weights, x, y, m, bias, activation,
                                      workspace);
}

Status qgemv_prepacked(const CpuPackedWeights& weights, const float* x,
                       float* y, Workspace* workspace) {
  const CpuPackedWeightsInfo info = weights.info();
  return qgemm_prepacked_storage(
      weights, {x, FloatStorageType::kF32, info.columns},
      {y, FloatStorageType::kF32, info.rows}, 1, workspace);
}

Status qgemv_prepacked_storage(const CpuPackedWeights& weights,
                               FloatStorageInput x, FloatStorageOutput y,
                               Workspace* workspace) {
  return qgemm_prepacked_storage(weights, x, y, 1, workspace);
}

static Status qgemm_prepacked_gate_up_impl(
    const CpuPackedWeights& gate_weights, const CpuPackedWeights& up_weights,
    FloatStorageInput x, FloatStorageOutput gate_output,
    FloatStorageOutput up_output, FloatStorageOutput fused_output, long long m,
    bool fuse_swiglu, Workspace* workspace) {
  (void)workspace;
  CpuPackedWeightsInfo gate_info;
  CpuPackedWeightsInfo up_info;
  Status status = validate_prepared_projection(gate_weights, &gate_info);
  if (status != Status::kOk) return status;
  status = validate_prepared_projection(up_weights, &up_info);
  if (status != Status::kOk) return status;
  if (!compatible_gate_up(gate_info, up_info) ||
      !detail::valid_product({m, gate_info.columns}) ||
      !detail::valid_product({m, gate_info.rows}) ||
      x.count != m * gate_info.columns ||
      (fuse_swiglu ? fused_output.count != m * gate_info.rows
                   : (gate_output.count != m * gate_info.rows ||
                      up_output.count != m * gate_info.rows))) {
    return Status::kInvalidShape;
  }
  if (!known_storage(x.type) ||
      (fuse_swiglu ? !known_storage(fused_output.type)
                   : (!known_storage(gate_output.type) ||
                      !known_storage(up_output.type)))) {
    return Status::kUnsupportedFormat;
  }
  if (x.data == nullptr ||
      (fuse_swiglu ? fused_output.data == nullptr
                   : !detail::all_nonnull(gate_output.data, up_output.data)))
    return Status::kInvalidArgument;

  CanonicalGateUpContext context{
      gate_info,
      up_info,
      static_cast<const std::uint8_t*>(gate_weights.panel_data()),
      static_cast<const std::uint8_t*>(up_weights.panel_data()),
      x,
      gate_output,
      up_output,
      fused_output,
      fuse_swiglu && fused_output.type == FloatStorageType::kF32
          ? static_cast<float*>(fused_output.data)
          : nullptr,
      m,
      fuse_swiglu};
  bool use_row_kernel = m == 1;
#if defined(__x86_64__) || defined(_M_X64)
  if (m == 1 && (x.type != FloatStorageType::kF32 ||
                 x86_f32_panel_layout(gate_info.quant_metadata.layout))) {
    use_row_kernel = false;
  }
#endif
  threading::RangeFn kernel = nullptr;
  switch (x.type) {
    case FloatStorageType::kF32:
      kernel =
          fuse_swiglu
              ? (use_row_kernel
                     ? gate_up_row_kernel<FloatStorageType::kF32, true>(
                           gate_info.quant_metadata.layout)
                     : gate_up_storage_kernel<FloatStorageType::kF32, true>(
                           gate_info.quant_metadata.layout))
              : (use_row_kernel
                     ? gate_up_row_kernel<FloatStorageType::kF32, false>(
                           gate_info.quant_metadata.layout)
                     : gate_up_storage_kernel<FloatStorageType::kF32, false>(
                           gate_info.quant_metadata.layout));
      break;
    case FloatStorageType::kF16:
      kernel =
          fuse_swiglu
              ? (use_row_kernel
                     ? gate_up_row_kernel<FloatStorageType::kF16, true>(
                           gate_info.quant_metadata.layout)
                     : gate_up_storage_kernel<FloatStorageType::kF16, true>(
                           gate_info.quant_metadata.layout))
              : (use_row_kernel
                     ? gate_up_row_kernel<FloatStorageType::kF16, false>(
                           gate_info.quant_metadata.layout)
                     : gate_up_storage_kernel<FloatStorageType::kF16, false>(
                           gate_info.quant_metadata.layout));
      break;
    case FloatStorageType::kBF16:
      kernel =
          fuse_swiglu
              ? (use_row_kernel
                     ? gate_up_row_kernel<FloatStorageType::kBF16, true>(
                           gate_info.quant_metadata.layout)
                     : gate_up_storage_kernel<FloatStorageType::kBF16, true>(
                           gate_info.quant_metadata.layout))
              : (use_row_kernel
                     ? gate_up_row_kernel<FloatStorageType::kBF16, false>(
                           gate_info.quant_metadata.layout)
                     : gate_up_storage_kernel<FloatStorageType::kBF16, false>(
                           gate_info.quant_metadata.layout));
      break;
  }
  if (kernel == nullptr) return Status::kUnsupportedFormat;
  if (use_row_kernel) {
    threading::parallel_ranges_impl(gate_info.rows, 16, kernel, &context);
  } else {
    const long long panels =
        (gate_info.rows + gate_info.row_tile - 1) / gate_info.row_tile;
    threading::parallel_ranges_impl(panels, 1, kernel, &context);
  }
  return Status::kOk;
}

Status qgemm_prepacked_gate_up_storage(const CpuPackedWeights& gate_weights,
                                       const CpuPackedWeights& up_weights,
                                       FloatStorageInput x,
                                       FloatStorageOutput gate_output,
                                       FloatStorageOutput up_output,
                                       long long m, Workspace* workspace) {
  return qgemm_prepacked_gate_up_impl(gate_weights, up_weights, x, gate_output,
                                      up_output, {}, m, false, workspace);
}

Status qgemv_prepacked_gate_up_storage(const CpuPackedWeights& gate_weights,
                                       const CpuPackedWeights& up_weights,
                                       FloatStorageInput x,
                                       FloatStorageOutput gate_output,
                                       FloatStorageOutput up_output,
                                       Workspace* workspace) {
  return qgemm_prepacked_gate_up_storage(gate_weights, up_weights, x,
                                         gate_output, up_output, 1, workspace);
}

Status qgemm_prepacked_qkv_storage(
    const CpuPackedWeights& q_weights, const CpuPackedWeights& k_weights,
    const CpuPackedWeights& v_weights, FloatStorageInput x,
    FloatStorageOutput q_output, FloatStorageOutput k_output,
    FloatStorageOutput v_output, long long m, Workspace* workspace) {
  (void)workspace;
  CpuPackedWeightsInfo q_info;
  CpuPackedWeightsInfo k_info;
  CpuPackedWeightsInfo v_info;
  Status status = validate_prepared_projection(q_weights, &q_info);
  if (status != Status::kOk) return status;
  status = validate_prepared_projection(k_weights, &k_info);
  if (status != Status::kOk) return status;
  status = validate_prepared_projection(v_weights, &v_info);
  if (status != Status::kOk) return status;
  if (!compatible_qkv_plane(q_info, k_info) ||
      !compatible_qkv_plane(q_info, v_info) ||
      !detail::valid_product({m, q_info.columns}) ||
      !detail::valid_product({m, q_info.rows}) ||
      !detail::valid_product({m, k_info.rows}) ||
      !detail::valid_product({m, v_info.rows}) ||
      x.count != m * q_info.columns || q_output.count != m * q_info.rows ||
      k_output.count != m * k_info.rows || v_output.count != m * v_info.rows) {
    return Status::kInvalidShape;
  }
  if (!known_storage(x.type) || !known_storage(q_output.type) ||
      !known_storage(k_output.type) || !known_storage(v_output.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(x.data, q_output.data, k_output.data,
                           v_output.data)) {
    return Status::kInvalidArgument;
  }

  const bool no_act_order = q_info.act_order_offset == kMissingOffset &&
                            k_info.act_order_offset == kMissingOffset &&
                            v_info.act_order_offset == kMissingOffset;
  [[maybe_unused]] const bool no_group_index =
      q_info.group_index_offset == kMissingOffset &&
      k_info.group_index_offset == kMissingOffset &&
      v_info.group_index_offset == kMissingOffset;
  bool use_rows = m == 1;
  long long panel_group = 1;
  [[maybe_unused]] const bool conventional_fp8 =
      q_info.quant_metadata.layout == CanonicalQuantLayout::kFP8E4M3FN ||
      q_info.quant_metadata.layout == CanonicalQuantLayout::kFP8E5M2;
  bool share_panels =
      m >= 64 && no_act_order && x.type == FloatStorageType::kF32;
#if defined(__x86_64__) || defined(_M_X64)
  share_panels =
      m > 1 && no_act_order &&
      ((x.type == FloatStorageType::kF32 &&
        ((m < 64 && !conventional_fp8) || (m >= 64 && num_threads() == 1))) ||
       (x.type == FloatStorageType::kF16 && !conventional_fp8));
#endif
  threading::RangeFn kernel = nullptr;
  if (m == 1) {
    switch (x.type) {
      case FloatStorageType::kF32:
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && no_act_order &&
            x86_f32_panel_layout(q_info.quant_metadata.layout)) {
          kernel = storage_kernel<FloatStorageType::kF32>(
              q_info.quant_metadata.layout);
          use_rows = false;
          break;
        }
#endif
        kernel = gemv_storage_kernel<FloatStorageType::kF32>(
            q_info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (no_act_order && no_group_index) {
          kernel = typed_gemv_panel_kernel<FloatStorageType::kF16>(
              q_info.quant_metadata.layout);
          use_rows = false;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().f16c && no_act_order) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF16,
                                                 kTypedPanelGroup>(
              q_info.quant_metadata.layout);
          use_rows = false;
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
        kernel = gemv_storage_kernel<FloatStorageType::kF16>(
            q_info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (no_act_order && no_group_index) {
          kernel = typed_gemv_panel_kernel<FloatStorageType::kBF16>(
              q_info.quant_metadata.layout);
          use_rows = false;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && no_act_order) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kBF16,
                                                 kTypedPanelGroup>(
              q_info.quant_metadata.layout);
          use_rows = false;
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
        kernel = gemv_storage_kernel<FloatStorageType::kBF16>(
            q_info.quant_metadata.layout);
        break;
    }
  } else if (share_panels) {
    switch (x.type) {
      case FloatStorageType::kF32:
        kernel = qkv_shared_kernel<FloatStorageType::kF32>(
            q_info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
        kernel = qkv_shared_kernel<FloatStorageType::kF16>(
            q_info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
        kernel = qkv_shared_kernel<FloatStorageType::kBF16>(
            q_info.quant_metadata.layout);
        break;
    }
  } else {
    switch (x.type) {
      case FloatStorageType::kF32:
        kernel = storage_kernel<FloatStorageType::kF32>(
            q_info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (no_act_order) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF16,
                                                 kTypedPanelGroup>(
              q_info.quant_metadata.layout);
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().f16c && no_act_order) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kF16,
                                                 kTypedPanelGroup>(
              q_info.quant_metadata.layout);
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
        kernel = storage_kernel<FloatStorageType::kF16>(
            q_info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
#if defined(__aarch64__) || defined(_M_ARM64)
        if (no_act_order) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kBF16,
                                                 kTypedPanelGroup>(
              q_info.quant_metadata.layout);
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_features().avx2 && no_act_order) {
          kernel = typed_gemm_panel_group_kernel<FloatStorageType::kBF16,
                                                 kTypedPanelGroup>(
              q_info.quant_metadata.layout);
          panel_group = kTypedPanelGroup;
          break;
        }
#endif
        kernel = storage_kernel<FloatStorageType::kBF16>(
            q_info.quant_metadata.layout);
        break;
    }
  }
  if (kernel == nullptr) return Status::kUnsupportedFormat;

  CanonicalQKVContext context{
      {{q_info, static_cast<const std::uint8_t*>(q_weights.panel_data()), x,
        q_output, m, nullptr, LinearActivation::kNone},
       {k_info, static_cast<const std::uint8_t*>(k_weights.panel_data()), x,
        k_output, m, nullptr, LinearActivation::kNone},
       {v_info, static_cast<const std::uint8_t*>(v_weights.panel_data()), x,
        v_output, m, nullptr, LinearActivation::kNone}},
      kernel,
      {use_rows ? q_info.rows
                : ((q_info.rows + q_info.row_tile - 1) / q_info.row_tile +
                   panel_group - 1) /
                      panel_group,
       use_rows ? k_info.rows
                : ((k_info.rows + k_info.row_tile - 1) / k_info.row_tile +
                   panel_group - 1) /
                      panel_group,
       use_rows ? v_info.rows
                : ((v_info.rows + v_info.row_tile - 1) / v_info.row_tile +
                   panel_group - 1) /
                      panel_group},
      0,
      1};
  const long long total_tasks =
      context.task_counts[0] + context.task_counts[1] + context.task_counts[2];
  if (share_panels) {
    context.panel_tasks =
        std::max({context.task_counts[0], context.task_counts[1],
                  context.task_counts[2]});
    const long long common_panels =
        std::min({context.task_counts[0], context.task_counts[1],
                  context.task_counts[2]});
    if (num_threads() > 1 && common_panels < context.panel_tasks) {
      long long stride = common_panels + 1;
      while (stride < context.panel_tasks &&
             std::gcd(stride, context.panel_tasks) != 1) {
        ++stride;
      }
      if (stride < context.panel_tasks) context.panel_stride = stride;
    }
    threading::parallel_ranges_impl(context.panel_tasks, 1, kernel, &context);
  } else {
    threading::parallel_ranges_impl(total_tasks, use_rows ? 16 : 1,
                                    canonical_qkv_schedule, &context);
  }
  return Status::kOk;
}

Status qgemv_prepacked_qkv_storage(
    const CpuPackedWeights& q_weights, const CpuPackedWeights& k_weights,
    const CpuPackedWeights& v_weights, FloatStorageInput x,
    FloatStorageOutput q_output, FloatStorageOutput k_output,
    FloatStorageOutput v_output, Workspace* workspace) {
  return qgemm_prepacked_qkv_storage(q_weights, k_weights, v_weights, x,
                                     q_output, k_output, v_output, 1,
                                     workspace);
}

Status qgemv_prepacked_qkv_rope_kv_storage(
    const CpuPackedWeights& q_weights, const CpuPackedWeights& k_weights,
    const CpuPackedWeights& v_weights, FloatStorageInput x, const float* cosine,
    const float* sine, FloatStorageOutput q_output,
    FloatStorageOutput key_cache, FloatStorageOutput value_cache,
    long long query_heads, long long kv_heads, long long head_dim,
    long long slots, long long max_position, int position, int slot,
    Workspace* workspace) {
  (void)workspace;
  CpuPackedWeightsInfo q_info;
  CpuPackedWeightsInfo k_info;
  CpuPackedWeightsInfo v_info;
  Status status = validate_prepared_projection(q_weights, &q_info);
  if (status != Status::kOk) return status;
  status = validate_prepared_projection(k_weights, &k_info);
  if (status != Status::kOk) return status;
  status = validate_prepared_projection(v_weights, &v_info);
  if (status != Status::kOk) return status;
  if (!compatible_qkv_plane(q_info, k_info) ||
      !compatible_qkv_plane(q_info, v_info) || head_dim % 2 != 0 ||
      !detail::valid_product({query_heads, head_dim}) ||
      !detail::valid_product({kv_heads, head_dim}) ||
      !detail::valid_product({slots, kv_heads, head_dim}) ||
      !detail::valid_product({max_position, head_dim / 2}) || position < 0 ||
      position >= max_position || slot < -1 || slot >= slots) {
    return Status::kInvalidShape;
  }
  const long long query_dim = query_heads * head_dim;
  const long long kv_dim = kv_heads * head_dim;
  const long long half = head_dim / 2;
  const long long cache_count = slots * kv_dim;
  if (q_info.rows != query_dim || k_info.rows != kv_dim ||
      v_info.rows != kv_dim || x.count != q_info.columns ||
      q_output.count != query_dim || key_cache.count != cache_count ||
      value_cache.count != cache_count) {
    return Status::kInvalidShape;
  }
  if (!known_storage(x.type) || !known_storage(q_output.type) ||
      !known_storage(key_cache.type) || !known_storage(value_cache.type)) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(x.data, cosine, sine, q_output.data, key_cache.data,
                           value_cache.data)) {
    return Status::kInvalidArgument;
  }

  const bool conventional_fp8 =
      q_info.quant_metadata.layout == CanonicalQuantLayout::kFP8E4M3FN ||
      q_info.quant_metadata.layout == CanonicalQuantLayout::kFP8E5M2;
  bool use_inplace_f32_outputs = false;
#if defined(__x86_64__) || defined(_M_X64)
  use_inplace_f32_outputs = x.type == FloatStorageType::kF32 &&
                            (!conventional_fp8 || num_threads() > 1);
#elif defined(__aarch64__) || defined(_M_ARM64)
  use_inplace_f32_outputs = !conventional_fp8 || num_threads() > 1;
#endif
  if (slot >= 0 && use_inplace_f32_outputs &&
      q_output.type == FloatStorageType::kF32 &&
      key_cache.type == FloatStorageType::kF32 &&
      value_cache.type == FloatStorageType::kF32) {
    auto* query = static_cast<float*>(q_output.data);
    auto* keys = static_cast<float*>(key_cache.data);
    auto* values = static_cast<float*>(value_cache.data);
    const long long cache_base = static_cast<long long>(slot) * kv_dim;
    status = qgemv_prepacked_qkv_storage(
        q_weights, k_weights, v_weights, x,
        {query, FloatStorageType::kF32, query_dim},
        {keys + cache_base, FloatStorageType::kF32, kv_dim},
        {values + cache_base, FloatStorageType::kF32, kv_dim}, workspace);
    if (status != Status::kOk) return status;
    const float* cos_row = cosine + static_cast<long long>(position) * half;
    const float* sin_row = sine + static_cast<long long>(position) * half;
    for (long long head = 0; head < query_heads; ++head) {
      for (long long dim = 0; dim < half; ++dim) {
        const long long row0 = head * head_dim + dim;
        const long long row1 = row0 + half;
        const float first = query[row0];
        const float second = query[row1];
        query[row0] = first * cos_row[dim] - second * sin_row[dim];
        query[row1] = second * cos_row[dim] + first * sin_row[dim];
      }
    }
    for (long long head = 0; head < kv_heads; ++head) {
      for (long long dim = 0; dim < half; ++dim) {
        const long long row0 = cache_base + head * head_dim + dim;
        const long long row1 = row0 + half;
        const float first = keys[row0];
        const float second = keys[row1];
        keys[row0] = first * cos_row[dim] - second * sin_row[dim];
        keys[row1] = second * cos_row[dim] + first * sin_row[dim];
      }
    }
    return Status::kOk;
  }

  const bool use_panels = x.type != FloatStorageType::kF32 &&
                          q_info.act_order_offset == kMissingOffset &&
                          k_info.act_order_offset == kMissingOffset &&
                          v_info.act_order_offset == kMissingOffset &&
                          half % q_info.row_tile == 0;
  bool use_f32_panels = false;
#if defined(__x86_64__) || defined(_M_X64)
  use_f32_panels =
      x.type == FloatStorageType::kF32 &&
      (q_info.quant_metadata.layout == CanonicalQuantLayout::kFP8E4M3FN ||
       q_info.quant_metadata.layout == CanonicalQuantLayout::kFP8E5M2) &&
      num_threads() == 1 && q_info.act_order_offset == kMissingOffset &&
      k_info.act_order_offset == kMissingOffset &&
      v_info.act_order_offset == kMissingOffset && half % q_info.row_tile == 0;
#endif
  bool use_typed_panels = false;
#if defined(__aarch64__) || defined(_M_ARM64)
  use_typed_panels = use_panels && q_info.block_size <= kKTile &&
                     q_info.group_index_offset == kMissingOffset &&
                     k_info.group_index_offset == kMissingOffset &&
                     v_info.group_index_offset == kMissingOffset;
#endif
  const long long panel_group =
      (use_typed_panels || use_f32_panels) ? 1 : kTypedPanelGroup;
  threading::RangeFn kernel = nullptr;
  switch (x.type) {
    case FloatStorageType::kF32:
      kernel = use_f32_panels
                   ? qkv_rope_f32_panel_kernel(q_info.quant_metadata.layout)
                   : qkv_rope_f32_kernel(q_info.quant_metadata.layout);
      break;
    case FloatStorageType::kF16:
      kernel = use_panels
                   ? (use_typed_panels
                          ? qkv_rope_typed_panel_kernel<FloatStorageType::kF16>(
                                q_info.quant_metadata.layout)
                          : qkv_rope_panel_kernel<FloatStorageType::kF16>(
                                q_info.quant_metadata.layout))
                   : qkv_rope_kernel<FloatStorageType::kF16>(
                         q_info.quant_metadata.layout);
      break;
    case FloatStorageType::kBF16:
      kernel =
          use_panels
              ? (use_typed_panels
                     ? qkv_rope_typed_panel_kernel<FloatStorageType::kBF16>(
                           q_info.quant_metadata.layout)
                     : qkv_rope_panel_kernel<FloatStorageType::kBF16>(
                           q_info.quant_metadata.layout))
              : qkv_rope_kernel<FloatStorageType::kBF16>(
                    q_info.quant_metadata.layout);
      break;
  }
  if (kernel == nullptr) return Status::kUnsupportedFormat;

  const long long query_pairs = query_heads * half;
  const long long key_pairs = slot >= 0 ? kv_heads * half : 0;
  const long long value_rows = slot >= 0 ? kv_dim : 0;
  if (query_pairs > std::numeric_limits<long long>::max() - key_pairs ||
      query_pairs + key_pairs >
          std::numeric_limits<long long>::max() - value_rows) {
    return Status::kInvalidShape;
  }
  CanonicalQKVRopeContext context{
      {q_info, k_info, v_info},
      {static_cast<const std::uint8_t*>(q_weights.panel_data()),
       static_cast<const std::uint8_t*>(k_weights.panel_data()),
       static_cast<const std::uint8_t*>(v_weights.panel_data())},
      x,
      cosine + static_cast<long long>(position) * half,
      sine + static_cast<long long>(position) * half,
      q_output,
      key_cache,
      value_cache,
      query_heads,
      kv_heads,
      head_dim,
      half,
      query_pairs,
      key_pairs,
      value_rows,
      static_cast<long long>(slot) * kv_dim};
  long long tasks = query_pairs + key_pairs + value_rows;
  if (use_panels || use_f32_panels) {
    const long long pair_panels = half / q_info.row_tile;
    const long long pair_groups = (pair_panels + panel_group - 1) / panel_group;
    context.query_panel_tasks = query_heads * pair_groups;
    context.key_panel_tasks = slot >= 0 ? kv_heads * pair_groups : 0;
    context.value_panel_tasks =
        slot >= 0 ? ((kv_dim + v_info.row_tile - 1) / v_info.row_tile +
                     panel_group - 1) /
                        panel_group
                  : 0;
    tasks = context.query_panel_tasks + context.key_panel_tasks +
            context.value_panel_tasks;
  }
  threading::parallel_ranges_impl(tasks, (use_panels || use_f32_panels) ? 1 : 8,
                                  kernel, &context);
  return Status::kOk;
}

Status qgemm_prepacked_swiglu_storage(const CpuPackedWeights& gate_weights,
                                      const CpuPackedWeights& up_weights,
                                      FloatStorageInput x,
                                      FloatStorageOutput output, long long m,
                                      Workspace* workspace) {
  return qgemm_prepacked_gate_up_impl(gate_weights, up_weights, x, {}, {},
                                      output, m, true, workspace);
}

Status qgemv_prepacked_swiglu_storage(const CpuPackedWeights& gate_weights,
                                      const CpuPackedWeights& up_weights,
                                      FloatStorageInput x,
                                      FloatStorageOutput output,
                                      Workspace* workspace) {
  return qgemm_prepacked_swiglu_storage(gate_weights, up_weights, x, output, 1,
                                        workspace);
}

Status qgemm_prepacked_swiglu_quantized(
    const CpuPackedWeights& gate_weights, const CpuPackedWeights& up_weights,
    FloatStorageInput x, CanonicalQuantLayout output_layout,
    long long output_group_size, CanonicalQuantTensor* output, long long m,
    bool scale_2d, Workspace* workspace) {
  CpuPackedWeightsInfo gate_info;
  CpuPackedWeightsInfo up_info;
  Status status = validate_prepared_projection(gate_weights, &gate_info);
  if (status != Status::kOk) return status;
  status = validate_prepared_projection(up_weights, &up_info);
  if (status != Status::kOk) return status;
  if (!compatible_gate_up(gate_info, up_info) ||
      !detail::valid_product({m, gate_info.columns}) ||
      x.count != m * gate_info.columns) {
    return Status::kInvalidShape;
  }
  if (!known_storage(x.type)) return Status::kUnsupportedFormat;
  if (x.data == nullptr || output == nullptr) return Status::kInvalidArgument;

  long long quant_group = 0;
  bool needs_global_maximum = false;
  try {
    status = configure_swiglu_quant_output(
        output, output_layout, m, gate_info.rows, output_group_size, scale_2d,
        &quant_group, &needs_global_maximum);
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
  if (status != Status::kOk) return status;

  const long long groups_per_row = gate_info.rows / quant_group;
  const long long row_domain =
      output_layout == CanonicalQuantLayout::kNVFP4E2M1E4M3 && scale_2d ? 16
                                                                        : 1;
  const CanonicalQuantLayout weight_layout = gate_info.quant_metadata.layout;
  const std::size_t workers = static_cast<std::size_t>(num_threads());
  bool a8_row_route =
      output_layout == CanonicalQuantLayout::kInt8Symmetric &&
      (weight_layout == CanonicalQuantLayout::kInt4Symmetric ||
       weight_layout == CanonicalQuantLayout::kBitNetTernary ||
       (weight_layout == CanonicalQuantLayout::kMXFP4E2M1E8M0 && m <= 32));
#if defined(__x86_64__) || defined(_M_X64)
  if (m > 1) a8_row_route = false;
#endif
  bool full_row_groups = false;
  bool x86_m1_panel = false;
#if defined(__x86_64__) || defined(_M_X64)
  x86_m1_panel = m == 1 && x86_f32_panel_layout(weight_layout);
  full_row_groups = m == 1 && workers == 1 && !x86_m1_panel;
#endif
  const bool use_panel_groups = quant_group % gate_info.row_tile == 0 &&
                                (x86_m1_panel || (m != 1 && !a8_row_route));
  const long long m_tile =
      use_panel_groups ? (row_domain == 16 ? 16 : kMTile) : row_domain;
  const long long m_tiles = (m + m_tile - 1) / m_tile;
  const long long tasks = full_row_groups ? m : m_tiles * groups_per_row;
  const long long scratch_elements =
      full_row_groups ? gate_info.rows : quant_group * m_tile;
  if (!detail::valid_product({tasks, scratch_elements}) ||
      static_cast<unsigned long long>(scratch_elements) >
          std::numeric_limits<std::size_t>::max() / (2 * sizeof(float)) /
              workers) {
    return Status::kInvalidShape;
  }

  detail::WorkspaceFrame frame(workspace);
  float* scratch = frame.allocate<float>(
      workers * static_cast<std::size_t>(scratch_elements) * 2);
  float* worker_maximum = frame.allocate<float>(workers);
  if (scratch == nullptr || worker_maximum == nullptr)
    return Status::kOutOfMemory;
  std::fill_n(worker_maximum, workers, 0.0f);
  std::atomic<bool> invalid{false};
  CanonicalSwiGLUQuantContext context{
      gate_info,
      up_info,
      static_cast<const std::uint8_t*>(gate_weights.panel_data()),
      static_cast<const std::uint8_t*>(up_weights.panel_data()),
      x,
      output,
      scratch,
      worker_maximum,
      &invalid,
      m,
      quant_group,
      groups_per_row,
      m_tile,
      scratch_elements,
      0.0f,
      scale_2d,
      needs_global_maximum};

  threading::RangeFn kernel = nullptr;
  switch (x.type) {
    case FloatStorageType::kF32:
      kernel = full_row_groups
                   ? swiglu_quant_full_row_kernel<FloatStorageType::kF32>(
                         gate_info.quant_metadata.layout)
                   : (use_panel_groups
                          ? swiglu_quant_panel_kernel<FloatStorageType::kF32>(
                                gate_info.quant_metadata.layout)
                          : swiglu_quant_kernel<FloatStorageType::kF32>(
                                gate_info.quant_metadata.layout));
      break;
    case FloatStorageType::kF16:
      kernel = full_row_groups
                   ? swiglu_quant_full_row_kernel<FloatStorageType::kF16>(
                         gate_info.quant_metadata.layout)
                   : (use_panel_groups
                          ? swiglu_quant_panel_kernel<FloatStorageType::kF16>(
                                gate_info.quant_metadata.layout)
                          : swiglu_quant_kernel<FloatStorageType::kF16>(
                                gate_info.quant_metadata.layout));
      break;
    case FloatStorageType::kBF16:
      kernel = full_row_groups
                   ? swiglu_quant_full_row_kernel<FloatStorageType::kBF16>(
                         gate_info.quant_metadata.layout)
                   : (use_panel_groups
                          ? swiglu_quant_panel_kernel<FloatStorageType::kBF16>(
                                gate_info.quant_metadata.layout)
                          : swiglu_quant_kernel<FloatStorageType::kBF16>(
                                gate_info.quant_metadata.layout));
      break;
  }
  if (kernel == nullptr) return Status::kUnsupportedFormat;

  if (needs_global_maximum) {
    threading::parallel_ranges_impl(tasks, 1, kernel, &context);
    if (invalid.load(std::memory_order_relaxed))
      return Status::kInvalidArgument;
    float maximum = 0.0f;
    for (std::size_t worker = 0; worker < workers; ++worker)
      maximum = std::max(maximum, worker_maximum[worker]);
    if (output_layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
      context.global_scale = maximum / (6.0f * 448.0f);
      output->metadata.global_scale = context.global_scale;
    } else {
      const bool fp4 = output_layout == CanonicalQuantLayout::kFP4E2M1;
      const float maximum_code =
          fp4 ? 6.0f
              : (output_layout == CanonicalQuantLayout::kFP8E4M3FN ? 448.0f
                                                                   : 57344.0f);
      float scale = maximum == 0.0f ? 0.0f : maximum / maximum_code;
      if (fp4) scale = f16_to_float(float_to_f16(scale));
      output->scales[0] = scale;
    }
    context.measure_only = false;
  }

  threading::parallel_ranges_impl(tasks, 1, kernel, &context);
  if (invalid.load(std::memory_order_relaxed)) return Status::kInvalidArgument;
  return validate_canonical_quant_tensor(*output);
}

Status qgemv_prepacked_swiglu_quantized(const CpuPackedWeights& gate_weights,
                                        const CpuPackedWeights& up_weights,
                                        FloatStorageInput x,
                                        CanonicalQuantLayout output_layout,
                                        long long output_group_size,
                                        CanonicalQuantTensor* output,
                                        bool scale_2d, Workspace* workspace) {
  return qgemm_prepacked_swiglu_quantized(gate_weights, up_weights, x,
                                          output_layout, output_group_size,
                                          output, 1, scale_2d, workspace);
}

Status qgemm_prepacked_quantized(const CpuPackedWeights& weights,
                                 const CanonicalQuantTensor& activation,
                                 float* y, Workspace* workspace) {
  (void)workspace;
  CpuPackedWeightsInfo info;
  Status status = validate_prepared_projection(weights, &info);
  if (status != Status::kOk) return status;
  status = validate_canonical_quant_tensor(activation);
  if (status != Status::kOk) return status;
  if (!projection_metadata_supported(activation.metadata)) {
    return Status::kUnsupportedFormat;
  }
  const long long m = activation.metadata.logical_rows;
  if (activation.metadata.logical_columns != info.columns ||
      !detail::valid_product({m, info.rows}) || y == nullptr) {
    return activation.metadata.logical_columns != info.columns
               ? Status::kInvalidShape
               : Status::kInvalidArgument;
  }
  if (direct_dual_gemv_supported(info, activation)) {
    CanonicalDualContext context{
        info, static_cast<const std::uint8_t*>(weights.panel_data()),
        &activation, y};
    if (m == 1) {
      if (info.block_size <= kKTile &&
          dual_fp8_panel_pair(info.quant_metadata.layout,
                              activation.metadata.layout)) {
        const long long panels =
            (info.rows + info.row_tile - 1) / info.row_tile;
        threading::parallel_ranges_impl(
            (panels + kDualFp8PanelGroup - 1) / kDualFp8PanelGroup, 1,
            dual_fp8_panel_group_kernel<kDualFp8PanelGroup>(
                info.quant_metadata.layout, activation.metadata.layout),
            &context);
      } else {
        threading::parallel_ranges_impl(
            info.rows, 16,
            dual_gemv_kernel(info.quant_metadata.layout,
                             activation.metadata.layout),
            &context);
      }
    } else {
      const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)
      const bool avx2_dual_pair = !(
          info.quant_metadata.layout == CanonicalQuantLayout::kInt8Symmetric &&
          activation.metadata.layout == CanonicalQuantLayout::kInt8Affine);
      if (cpu_features().avx2 && avx2_dual_pair && info.block_size <= kKTile) {
        const long long groups =
            (panels + kDualAvx2PanelGroup - 1) / kDualAvx2PanelGroup;
        threading::parallel_ranges_impl(
            m * groups, 1,
            dual_avx2_panel_group_kernel<kDualAvx2PanelGroup>(
                info.quant_metadata.layout, activation.metadata.layout),
            &context);
        return Status::kOk;
      }
#endif
      if (info.block_size <= kKTile &&
          dual_fp8_panel_pair(info.quant_metadata.layout,
                              activation.metadata.layout)) {
        const long long groups =
            (panels + kDualFp8GemmPanelGroup - 1) / kDualFp8GemmPanelGroup;
        threading::parallel_ranges_impl(
            m * groups, 1,
            dual_fp8_panel_group_kernel<kDualFp8GemmPanelGroup>(
                info.quant_metadata.layout, activation.metadata.layout),
            &context);
      } else {
        threading::parallel_ranges_impl(
            m * panels, 1,
            dual_panel_kernel(info.quant_metadata.layout,
                              activation.metadata.layout),
            &context);
      }
    }
    return Status::kOk;
  }
  CanonicalQuantizedContext context{
      info, static_cast<const std::uint8_t*>(weights.panel_data()), &activation,
      y};
  const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
  threading::parallel_ranges_impl(
      panels, 1, quantized_kernel(info.quant_metadata.layout), &context);
  return Status::kOk;
}

Status qgemv_prepacked_quantized(const CpuPackedWeights& weights,
                                 const CanonicalQuantTensor& activation,
                                 float* y, Workspace* workspace) {
  if (activation.metadata.logical_rows != 1) return Status::kInvalidShape;
  return qgemm_prepacked_quantized(weights, activation, y, workspace);
}

Status canonical_quantized_embedding_storage(const CanonicalQuantTensor& table,
                                             const int* ids,
                                             FloatStorageInput add,
                                             FloatStorageOutput output,
                                             long long count, float scale) {
  Status status = validate_canonical_quant_tensor(table);
  if (status != Status::kOk) return status;
  const long long vocab = table.metadata.logical_rows;
  const long long dim = table.metadata.logical_columns;
  if (!detail::valid_product({vocab, count, dim}) || !std::isfinite(scale) ||
      output.count != count * dim ||
      (add.data == nullptr ? add.count != 0 : add.count != count * dim)) {
    return Status::kInvalidShape;
  }
  if (ids == nullptr || output.data == nullptr) return Status::kInvalidArgument;
  if (!known_storage(output.type) ||
      (add.data != nullptr && !known_storage(add.type))) {
    return Status::kUnsupportedFormat;
  }
  for (long long item = 0; item < count; ++item) {
    if (ids[item] < 0 || ids[item] >= vocab) return Status::kInvalidArgument;
  }
  if (output.type == FloatStorageType::kF32 && add.data == nullptr) {
    auto* destination = static_cast<float*>(output.data);
    threading::parallel_ranges(
        count, 4, [&](long long begin, long long end, int) {
          for (long long item = begin; item < end; ++item) {
            const long long row = ids[item];
            const long long base = item * dim;
            for (long long column = 0; column < dim; ++column) {
              destination[base + column] =
                  scale * tensor_value(table, row, column);
            }
          }
        });
    return Status::kOk;
  }
  threading::parallel_ranges(
      count, 4, [&](long long begin, long long end, int) {
        for (long long item = begin; item < end; ++item) {
          const long long row = ids[item];
          const long long base = item * dim;
          for (long long column = 0; column < dim; ++column) {
            float value = scale * tensor_value(table, row, column);
            if (add.data != nullptr) {
              switch (add.type) {
                case FloatStorageType::kF32:
                  value += input_value<FloatStorageType::kF32>(add.data,
                                                               base + column);
                  break;
                case FloatStorageType::kF16:
                  value += input_value<FloatStorageType::kF16>(add.data,
                                                               base + column);
                  break;
                case FloatStorageType::kBF16:
                  value += input_value<FloatStorageType::kBF16>(add.data,
                                                                base + column);
                  break;
              }
            }
            store_output(output, static_cast<std::size_t>(base + column),
                         value);
          }
        }
      });
  return Status::kOk;
}

Status canonical_quantized_embedding_bag_storage(
    const CanonicalQuantTensor& table, const int* ids, const long long* offsets,
    const float* sample_weights, FloatStorageOutput output, long long id_count,
    long long bags, float scale, bool use_weights, bool mean_mode,
    Workspace* workspace) {
  Status status = validate_canonical_quant_tensor(table);
  if (status != Status::kOk) return status;
  const long long vocab = table.metadata.logical_rows;
  const long long dim = table.metadata.logical_columns;
  if (!detail::valid_product({vocab, id_count, bags, dim}) ||
      !std::isfinite(scale) || output.count != bags * dim) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(ids, offsets, output.data) ||
      (use_weights && sample_weights == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (!known_storage(output.type)) return Status::kUnsupportedFormat;
  if (offsets[0] != 0 || offsets[bags] != id_count) {
    return Status::kInvalidArgument;
  }
  for (long long bag = 0; bag < bags; ++bag) {
    if (offsets[bag] > offsets[bag + 1]) return Status::kInvalidArgument;
  }
  for (long long item = 0; item < id_count; ++item) {
    if (ids[item] < 0 || ids[item] >= vocab ||
        (use_weights && !std::isfinite(sample_weights[item]))) {
      return Status::kInvalidArgument;
    }
  }
  const std::size_t workers = static_cast<std::size_t>(num_threads());
  if (static_cast<unsigned long long>(dim) >
      std::numeric_limits<std::size_t>::max() / sizeof(float) / workers) {
    return Status::kInvalidShape;
  }
  detail::WorkspaceFrame frame(workspace);
  float* scratch =
      frame.allocate<float>(workers * static_cast<std::size_t>(dim));
  if (scratch == nullptr) return Status::kOutOfMemory;
  threading::parallel_ranges(
      bags, 2, [&](long long begin, long long end, int worker) {
        float* sums = scratch + static_cast<long long>(worker) * dim;
        for (long long bag = begin; bag < end; ++bag) {
          const long long start = offsets[bag];
          const long long stop = offsets[bag + 1];
          const float divisor = mean_mode && stop > start
                                    ? 1.0f / static_cast<float>(stop - start)
                                    : 1.0f;
          std::fill_n(sums, dim, 0.0f);
          for (long long item = start; item < stop; ++item) {
            const float coefficient = use_weights ? sample_weights[item] : 1.0f;
            const long long row = ids[item];
            for (long long column = 0; column < dim; ++column) {
              sums[column] += coefficient * tensor_value(table, row, column);
            }
          }
          for (long long column = 0; column < dim; ++column) {
            const float value = scale * divisor * sums[column];
            if (output.type == FloatStorageType::kF32) {
              static_cast<float*>(output.data)[bag * dim + column] = value;
            } else {
              store_output(output, static_cast<std::size_t>(bag * dim + column),
                           value);
            }
          }
        }
      });
  return Status::kOk;
}

namespace {

using LmHeadRowDot = float (*)(const CpuPackedWeightsInfo&, const std::uint8_t*,
                               FloatStorageInput, long long);

template <FloatStorageType Type, CanonicalQuantLayout Layout>
float lm_head_row_dot(const CpuPackedWeightsInfo& info,
                      const std::uint8_t* panel, FloatStorageInput input,
                      long long row) {
  return canonical_projection_row<Type, Layout>(info, panel, input, row);
}

template <FloatStorageType Type>
LmHeadRowDot lm_head_row_dot_for_layout(CanonicalQuantLayout layout) {
#define QUIXICORE_LM_HEAD_DOT_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:       \
    return &lm_head_row_dot<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_LM_HEAD_DOT_CASE(kInt4Symmetric);
    QUIXICORE_LM_HEAD_DOT_CASE(kUInt4Affine);
    QUIXICORE_LM_HEAD_DOT_CASE(kInt8Symmetric);
    QUIXICORE_LM_HEAD_DOT_CASE(kInt8Affine);
    QUIXICORE_LM_HEAD_DOT_CASE(kFP8E4M3FN);
    QUIXICORE_LM_HEAD_DOT_CASE(kFP8E5M2);
    QUIXICORE_LM_HEAD_DOT_CASE(kFP4E2M1);
    QUIXICORE_LM_HEAD_DOT_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_LM_HEAD_DOT_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_LM_HEAD_DOT_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_LM_HEAD_DOT_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_LM_HEAD_DOT_CASE
  return nullptr;
}

LmHeadRowDot lm_head_row_dot_for(FloatStorageType type,
                                 CanonicalQuantLayout layout) {
  switch (type) {
    case FloatStorageType::kF32:
      return lm_head_row_dot_for_layout<FloatStorageType::kF32>(layout);
    case FloatStorageType::kF16:
      return lm_head_row_dot_for_layout<FloatStorageType::kF16>(layout);
    case FloatStorageType::kBF16:
      return lm_head_row_dot_for_layout<FloatStorageType::kBF16>(layout);
  }
  return nullptr;
}

FloatStorageInput lm_head_input_row(FloatStorageInput input, long long row,
                                    long long hidden) {
  const std::size_t offset = static_cast<std::size_t>(row * hidden);
  const void* data = nullptr;
  switch (input.type) {
    case FloatStorageType::kF32:
      data = static_cast<const float*>(input.data) + offset;
      break;
    case FloatStorageType::kF16:
    case FloatStorageType::kBF16:
      data = static_cast<const std::uint16_t*>(input.data) + offset;
      break;
  }
  return {data, input.type, hidden};
}

bool lm_head_valid_logit(float value) {
  return !std::isnan(value) && value != std::numeric_limits<float>::infinity();
}

bool lm_head_better(float value, int token, float incumbent,
                    int incumbent_token) {
  return value > incumbent || (value == incumbent && token < incumbent_token);
}

void lm_head_insert(float value, int token, float* values, int* ids, int* count,
                    int capacity) {
  int position = *count;
  while (position > 0 && lm_head_better(value, token, values[position - 1],
                                        ids[position - 1])) {
    --position;
  }
  if (position >= capacity) return;
  const int upper = std::min(*count, capacity - 1);
  for (int item = upper; item > position; --item) {
    values[item] = values[item - 1];
    ids[item] = ids[item - 1];
  }
  values[position] = value;
  ids[position] = token;
  if (*count < capacity) ++*count;
}

void lm_head_lse_add(float value, double* maximum, double* sum) {
  if (value == -std::numeric_limits<float>::infinity()) return;
  if (static_cast<double>(value) > *maximum) {
    *sum = *sum * std::exp(*maximum - value) + 1.0;
    *maximum = value;
  } else {
    *sum += std::exp(static_cast<double>(value) - *maximum);
  }
}

double lm_head_uniform01(std::uint32_t seed, std::uint64_t row) {
  std::uint64_t z =
      (static_cast<std::uint64_t>(seed) << 32) ^ row ^ 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

int lm_head_sample_sorted(const float* values, const int* ids, int count,
                          float temperature, double uniform, double* weights) {
  const float maximum = values[0];
  if (!std::isfinite(maximum)) return ids[0];
  double total = 0.0;
  for (int item = 0; item < count; ++item) {
    weights[item] =
        std::exp((static_cast<double>(values[item]) - maximum) / temperature);
    total += weights[item];
  }
  const double target = uniform * total;
  double cumulative = 0.0;
  for (int item = 0; item < count; ++item) {
    cumulative += weights[item];
    if (target < cumulative || item + 1 == count) return ids[item];
  }
  return ids[count - 1];
}

Status validate_prepared_lm_head(const CpuPackedWeights& weights,
                                 FloatStorageInput hidden_states,
                                 long long rows, CpuPackedWeightsInfo* info,
                                 LmHeadRowDot* dot) {
  Status status = validate_prepared_projection(weights, info);
  if (status != Status::kOk) return status;
  if (!detail::valid_product({rows, info->rows, info->columns}) ||
      info->rows > std::numeric_limits<int>::max() ||
      hidden_states.count != rows * info->columns) {
    return Status::kInvalidShape;
  }
  if (!known_storage(hidden_states.type)) return Status::kUnsupportedFormat;
  if (hidden_states.data == nullptr) return Status::kInvalidArgument;
  *dot = lm_head_row_dot_for(hidden_states.type, info->quant_metadata.layout);
  return *dot == nullptr ? Status::kUnsupportedFormat : Status::kOk;
}

Status lm_head_project_row(const CpuPackedWeightsInfo& info,
                           const std::uint8_t* panel, LmHeadRowDot dot,
                           FloatStorageInput input, const float* bias,
                           float* logits) {
  for (long long token = 0; token < info.rows; ++token) {
    const float value =
        dot(info, panel, input, token) + (bias == nullptr ? 0.0f : bias[token]);
    if (!lm_head_valid_logit(value)) return Status::kInvalidArgument;
    logits[token] = value;
  }
  return Status::kOk;
}

struct LmHeadStreamContext {
  CpuPackedWeightsInfo info;
  const std::uint8_t* panel = nullptr;
  FloatStorageInput input;
  const float* bias = nullptr;
  const std::uint8_t* mask = nullptr;
  long long mask_stride = 0;
  long long rows = 0;
  int retained = 0;
  bool normalize_allowed = false;
  float* best_values = nullptr;
  int* best_ids = nullptr;
  int* counts = nullptr;
  double* maximum = nullptr;
  double* denominator = nullptr;
  std::atomic<bool>* invalid = nullptr;
};

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void lm_head_stream_row_tiles(void* opaque, long long begin, long long end,
                              int) {
  auto& context = *static_cast<LmHeadStreamContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
  alignas(64) float decoded[kMaximumRowTile * kKTile];
  alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long task = begin; task < end; ++task) {
    const long long m_base = task * kMTile;
    const long long m_count = std::min(kMTile, context.rows - m_base);
    for (long long row_panel = 0; row_panel < panels; ++row_panel) {
      const long long first_output = row_panel * info.row_tile;
      const long long lanes = std::min(info.row_tile, info.rows - first_output);
      std::fill_n(accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                        lanes, item_base, items, first_output,
                                        decoded);
          for (long long item = 0; item < items; ++item) {
            const long long packed_column =
                block * info.block_size + item_base + item;
            const long long logical_column =
                order == nullptr ? packed_column : order[packed_column];
            const long long accumulator = packed_column & 3;
            for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
              const float activation = input_value<Type>(
                  context.input.data,
                  static_cast<std::size_t>((m_base + m_lane) * info.columns +
                                           logical_column));
              accumulate_output_lanes(
                  accumulators +
                      (accumulator * kMTile + m_lane) * kMaximumRowTile,
                  decoded + item * kMaximumRowTile, activation, lanes);
            }
          }
        }
      }
      constexpr long long stride = kMTile * kMaximumRowTile;
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        const long long row = m_base + m_lane;
        float* row_values = context.best_values + row * context.retained;
        int* row_ids = context.best_ids + row * context.retained;
        for (long long lane = 0; lane < lanes; ++lane) {
          const int token = static_cast<int>(first_output + lane);
          const float* partial = accumulators + m_lane * kMaximumRowTile + lane;
          const float value =
              (partial[0] + partial[stride]) +
              (partial[2 * stride] + partial[3 * stride]) +
              (context.bias == nullptr ? 0.0f : context.bias[token]);
          if (!lm_head_valid_logit(value)) {
            context.invalid->store(true, std::memory_order_relaxed);
            continue;
          }
          const bool allowed =
              context.mask == nullptr ||
              (context.mask[row * context.mask_stride + token / 8] &
               static_cast<std::uint8_t>(0x80u >> (token & 7))) != 0;
          if (context.maximum != nullptr &&
              (context.mask == nullptr || !context.normalize_allowed ||
               allowed)) {
            lm_head_lse_add(value, context.maximum + row,
                            context.denominator + row);
          }
          if (allowed) {
            lm_head_insert(value, token, row_values, row_ids,
                           context.counts + row, context.retained);
          }
        }
      }
    }
  }
}

template <FloatStorageType Type>
threading::RangeFn lm_head_stream_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_LM_HEAD_STREAM_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:          \
    return lm_head_stream_row_tiles<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_LM_HEAD_STREAM_CASE(kInt4Symmetric);
    QUIXICORE_LM_HEAD_STREAM_CASE(kUInt4Affine);
    QUIXICORE_LM_HEAD_STREAM_CASE(kInt8Symmetric);
    QUIXICORE_LM_HEAD_STREAM_CASE(kInt8Affine);
    QUIXICORE_LM_HEAD_STREAM_CASE(kFP8E4M3FN);
    QUIXICORE_LM_HEAD_STREAM_CASE(kFP8E5M2);
    QUIXICORE_LM_HEAD_STREAM_CASE(kFP4E2M1);
    QUIXICORE_LM_HEAD_STREAM_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_LM_HEAD_STREAM_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_LM_HEAD_STREAM_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_LM_HEAD_STREAM_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_LM_HEAD_STREAM_CASE
  return nullptr;
}

threading::RangeFn lm_head_stream_kernel_for(FloatStorageType type,
                                             CanonicalQuantLayout layout) {
  switch (type) {
    case FloatStorageType::kF32:
      return lm_head_stream_kernel<FloatStorageType::kF32>(layout);
    case FloatStorageType::kF16:
      return lm_head_stream_kernel<FloatStorageType::kF16>(layout);
    case FloatStorageType::kBF16:
      return lm_head_stream_kernel<FloatStorageType::kBF16>(layout);
  }
  return nullptr;
}

Status lm_head_stream_topk(const CpuPackedWeights& weights,
                           const CpuPackedWeightsInfo& info,
                           FloatStorageInput input, const float* bias,
                           const std::uint8_t* mask, long long mask_stride,
                           long long rows, int retained, bool normalize_allowed,
                           float* best_values, int* best_ids, int* counts,
                           double* maximum, double* denominator) {
  std::fill_n(counts, rows, 0);
  if (maximum != nullptr) {
    std::fill_n(maximum, rows, -std::numeric_limits<double>::infinity());
    std::fill_n(denominator, rows, 0.0);
  }
  std::atomic<bool> invalid{false};
  LmHeadStreamContext context{
      info,
      static_cast<const std::uint8_t*>(weights.panel_data()),
      input,
      bias,
      mask,
      mask_stride,
      rows,
      retained,
      normalize_allowed,
      best_values,
      best_ids,
      counts,
      maximum,
      denominator,
      &invalid};
  threading::RangeFn kernel =
      lm_head_stream_kernel_for(input.type, info.quant_metadata.layout);
  if (kernel == nullptr) return Status::kUnsupportedFormat;
  threading::parallel_ranges_impl((rows + kMTile - 1) / kMTile, 1, kernel,
                                  &context);
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

bool lm_head_reuse_row_tiles(const CpuPackedWeightsInfo& info, long long rows) {
  if (rows <= 1) return false;
#if defined(__x86_64__) || defined(_M_X64)
  (void)info;
  return true;
#else
  return info.quant_metadata.layout == CanonicalQuantLayout::kFP8E4M3FN ||
         info.quant_metadata.layout == CanonicalQuantLayout::kFP8E5M2;
#endif
}

struct LmHeadPartialContext {
  CpuPackedWeightsInfo info;
  const std::uint8_t* panel = nullptr;
  FloatStorageInput input;
  const float* bias = nullptr;
  const std::uint8_t* mask = nullptr;
  long long mask_stride = 0;
  long long tiles = 0;
  long long rows = 0;
  int retained = 0;
  bool normalize_allowed = false;
  LmHeadRowDot dot = nullptr;
  float* values = nullptr;
  int* ids = nullptr;
  int* counts = nullptr;
  double* maximum = nullptr;
  double* denominator = nullptr;
  std::atomic<bool>* invalid = nullptr;
};

void lm_head_partial_tiles(void* opaque, long long begin, long long end, int) {
  auto& context = *static_cast<LmHeadPartialContext*>(opaque);
  constexpr long long kVocabularyTile = 256;
  for (long long task = begin; task < end; ++task) {
    const long long row = task / context.tiles;
    const long long tile = task % context.tiles;
    const long long first = tile * kVocabularyTile;
    const long long stop = std::min(first + kVocabularyTile, context.info.rows);
    const FloatStorageInput input =
        lm_head_input_row(context.input, row, context.info.columns);
    float* values = context.values + task * context.retained;
    int* ids = context.ids + task * context.retained;
    int count = 0;
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    for (long long token = first; token < stop; ++token) {
      const float value =
          context.dot(context.info, context.panel, input, token) +
          (context.bias == nullptr ? 0.0f : context.bias[token]);
      if (!lm_head_valid_logit(value)) {
        context.invalid->store(true, std::memory_order_relaxed);
        continue;
      }
      const bool allowed =
          context.mask == nullptr ||
          (context.mask[row * context.mask_stride + token / 8] &
           static_cast<std::uint8_t>(0x80u >> (token & 7))) != 0;
      if (context.maximum != nullptr &&
          (context.mask == nullptr || !context.normalize_allowed || allowed)) {
        lm_head_lse_add(value, &maximum, &denominator);
      }
      if (allowed) {
        lm_head_insert(value, static_cast<int>(token), values, ids, &count,
                       context.retained);
      }
    }
    context.counts[task] = count;
    if (context.maximum != nullptr) {
      context.maximum[task] = maximum;
      context.denominator[task] = denominator;
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void lm_head_partial_panel_tiles(void* opaque, long long begin, long long end,
                                 int) {
  auto& context = *static_cast<LmHeadPartialContext*>(opaque);
  constexpr long long kVocabularyTile = 256;
  const CpuPackedWeightsInfo& info = context.info;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  alignas(64) float decoded[kMaximumRowTile * kKTile];
  alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile];
  for (long long task = begin; task < end; ++task) {
    const long long row = task / context.tiles;
    const long long tile = task % context.tiles;
    const long long first = tile * kVocabularyTile;
    const long long stop = std::min(first + kVocabularyTile, info.rows);
    const long long first_panel = first / info.row_tile;
    const long long stop_panel = (stop + info.row_tile - 1) / info.row_tile;
    float* values = context.values + task * context.retained;
    int* ids = context.ids + task * context.retained;
    int count = 0;
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    for (long long row_panel = first_panel; row_panel < stop_panel;
         ++row_panel) {
      const long long first_output = row_panel * info.row_tile;
      const long long lanes = std::min(info.row_tile, info.rows - first_output);
      std::fill_n(accumulators, kAccumulatorCount * kMaximumRowTile, 0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                        lanes, item_base, items, first_output,
                                        decoded);
          for (long long item = 0; item < items; ++item) {
            const long long packed_column =
                block * info.block_size + item_base + item;
            const long long logical_column =
                order == nullptr ? packed_column : order[packed_column];
            const float activation = input_value<Type>(
                context.input.data,
                static_cast<std::size_t>(row * info.columns + logical_column));
            accumulate_output_lanes(
                accumulators + (packed_column & 3) * kMaximumRowTile,
                decoded + item * kMaximumRowTile, activation, lanes);
          }
        }
      }
      for (long long lane = 0; lane < lanes; ++lane) {
        const int token = static_cast<int>(first_output + lane);
        if (token < first || token >= stop) continue;
        const float value =
            (accumulators[lane] + accumulators[kMaximumRowTile + lane]) +
            (accumulators[2 * kMaximumRowTile + lane] +
             accumulators[3 * kMaximumRowTile + lane]) +
            (context.bias == nullptr ? 0.0f : context.bias[token]);
        if (!lm_head_valid_logit(value)) {
          context.invalid->store(true, std::memory_order_relaxed);
          continue;
        }
        const bool allowed =
            context.mask == nullptr ||
            (context.mask[row * context.mask_stride + token / 8] &
             static_cast<std::uint8_t>(0x80u >> (token & 7))) != 0;
        if (context.maximum != nullptr &&
            (context.mask == nullptr || !context.normalize_allowed ||
             allowed)) {
          lm_head_lse_add(value, &maximum, &denominator);
        }
        if (allowed) {
          lm_head_insert(value, token, values, ids, &count, context.retained);
        }
      }
    }
    context.counts[task] = count;
    if (context.maximum != nullptr) {
      context.maximum[task] = maximum;
      context.denominator[task] = denominator;
    }
  }
}

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void lm_head_partial_mrow_panel_tiles(void* opaque, long long begin,
                                      long long end, int) {
  auto& context = *static_cast<LmHeadPartialContext*>(opaque);
  constexpr long long kVocabularyTile = 256;
  const CpuPackedWeightsInfo& info = context.info;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  alignas(64) float decoded[kMaximumRowTile * kKTile];
  alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long tile = begin; tile < end; ++tile) {
    const long long first = tile * kVocabularyTile;
    const long long stop = std::min(first + kVocabularyTile, info.rows);
    const long long first_panel = first / info.row_tile;
    const long long stop_panel = (stop + info.row_tile - 1) / info.row_tile;
    for (long long row = 0; row < context.rows; ++row) {
      const long long partial = row * context.tiles + tile;
      context.counts[partial] = 0;
      if (context.maximum != nullptr) {
        context.maximum[partial] = -std::numeric_limits<double>::infinity();
        context.denominator[partial] = 0.0;
      }
    }
    for (long long m_base = 0; m_base < context.rows; m_base += kMTile) {
      const long long m_count = std::min(kMTile, context.rows - m_base);
      for (long long row_panel = first_panel; row_panel < stop_panel;
           ++row_panel) {
        const long long first_output = row_panel * info.row_tile;
        const long long lanes =
            std::min(info.row_tile, info.rows - first_output);
        std::fill_n(accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                    0.0f);
        for (long long block = 0; block < info.blocks_per_row; ++block) {
          for (long long item_base = 0; item_base < info.block_size;
               item_base += kKTile) {
            const long long items =
                std::min(kKTile, info.block_size - item_base);
            decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                          lanes, item_base, items, first_output,
                                          decoded);
            for (long long item = 0; item < items; ++item) {
              const long long packed_column =
                  block * info.block_size + item_base + item;
              const long long logical_column =
                  order == nullptr ? packed_column : order[packed_column];
              const long long accumulator = packed_column & 3;
              for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
                const float activation = input_value<Type>(
                    context.input.data,
                    static_cast<std::size_t>((m_base + m_lane) * info.columns +
                                             logical_column));
                accumulate_output_lanes(
                    accumulators +
                        (accumulator * kMTile + m_lane) * kMaximumRowTile,
                    decoded + item * kMaximumRowTile, activation, lanes);
              }
            }
          }
        }
        constexpr long long stride = kMTile * kMaximumRowTile;
        for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
          const long long row = m_base + m_lane;
          const long long partial_index = row * context.tiles + tile;
          float* values = context.values + partial_index * context.retained;
          int* ids = context.ids + partial_index * context.retained;
          for (long long lane = 0; lane < lanes; ++lane) {
            const int token = static_cast<int>(first_output + lane);
            if (token < first || token >= stop) continue;
            const float* parts = accumulators + m_lane * kMaximumRowTile + lane;
            const float value =
                (parts[0] + parts[stride]) +
                (parts[2 * stride] + parts[3 * stride]) +
                (context.bias == nullptr ? 0.0f : context.bias[token]);
            if (!lm_head_valid_logit(value)) {
              context.invalid->store(true, std::memory_order_relaxed);
              continue;
            }
            const bool allowed =
                context.mask == nullptr ||
                (context.mask[row * context.mask_stride + token / 8] &
                 static_cast<std::uint8_t>(0x80u >> (token & 7))) != 0;
            if (context.maximum != nullptr &&
                (context.mask == nullptr || !context.normalize_allowed ||
                 allowed)) {
              lm_head_lse_add(value, context.maximum + partial_index,
                              context.denominator + partial_index);
            }
            if (allowed) {
              lm_head_insert(value, token, values, ids,
                             context.counts + partial_index, context.retained);
            }
          }
        }
      }
    }
  }
}

template <FloatStorageType Type>
threading::RangeFn lm_head_partial_panel_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_LM_HEAD_PARTIAL_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:           \
    return lm_head_partial_panel_tiles<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kInt4Symmetric);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kUInt4Affine);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kInt8Symmetric);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kInt8Affine);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kFP8E4M3FN);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kFP8E5M2);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kFP4E2M1);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_LM_HEAD_PARTIAL_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_LM_HEAD_PARTIAL_CASE
  return nullptr;
}

[[maybe_unused]] threading::RangeFn lm_head_partial_panel_kernel_for(
    FloatStorageType type, CanonicalQuantLayout layout) {
  switch (type) {
    case FloatStorageType::kF32:
      return lm_head_partial_panel_kernel<FloatStorageType::kF32>(layout);
    case FloatStorageType::kF16:
      return lm_head_partial_panel_kernel<FloatStorageType::kF16>(layout);
    case FloatStorageType::kBF16:
      return lm_head_partial_panel_kernel<FloatStorageType::kBF16>(layout);
  }
  return nullptr;
}

template <FloatStorageType Type>
threading::RangeFn lm_head_partial_mrow_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_LM_HEAD_MROW_CASE(layout_name)  \
  case CanonicalQuantLayout::layout_name:         \
    return lm_head_partial_mrow_panel_tiles<Type, \
                                            CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_LM_HEAD_MROW_CASE(kInt4Symmetric);
    QUIXICORE_LM_HEAD_MROW_CASE(kUInt4Affine);
    QUIXICORE_LM_HEAD_MROW_CASE(kInt8Symmetric);
    QUIXICORE_LM_HEAD_MROW_CASE(kInt8Affine);
    QUIXICORE_LM_HEAD_MROW_CASE(kFP8E4M3FN);
    QUIXICORE_LM_HEAD_MROW_CASE(kFP8E5M2);
    QUIXICORE_LM_HEAD_MROW_CASE(kFP4E2M1);
    QUIXICORE_LM_HEAD_MROW_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_LM_HEAD_MROW_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_LM_HEAD_MROW_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_LM_HEAD_MROW_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_LM_HEAD_MROW_CASE
  return nullptr;
}

[[maybe_unused]] threading::RangeFn lm_head_partial_mrow_kernel_for(
    FloatStorageType type, CanonicalQuantLayout layout) {
  switch (type) {
    case FloatStorageType::kF32:
      return lm_head_partial_mrow_kernel<FloatStorageType::kF32>(layout);
    case FloatStorageType::kF16:
      return lm_head_partial_mrow_kernel<FloatStorageType::kF16>(layout);
    case FloatStorageType::kBF16:
      return lm_head_partial_mrow_kernel<FloatStorageType::kBF16>(layout);
  }
  return nullptr;
}

Status lm_head_parallel_topk(const CpuPackedWeights& weights,
                             const CpuPackedWeightsInfo& info,
                             FloatStorageInput input, const float* bias,
                             const std::uint8_t* mask, long long mask_stride,
                             long long rows, int retained,
                             bool normalize_allowed, float* best_values,
                             int* best_ids, int* counts, double* maximum,
                             double* denominator, Workspace* workspace) {
  constexpr long long kVocabularyTile = 256;
  const long long tiles = (info.rows + kVocabularyTile - 1) / kVocabularyTile;
  if (!detail::valid_product({rows, tiles, retained}))
    return Status::kInvalidShape;
  const std::size_t tasks = static_cast<std::size_t>(rows * tiles);
  detail::WorkspaceFrame frame(workspace);
  float* partial_values = frame.allocate<float>(tasks * retained);
  int* partial_ids = frame.allocate<int>(tasks * retained);
  int* partial_counts = frame.allocate<int>(tasks);
  double* partial_maximum =
      maximum == nullptr ? nullptr : frame.allocate<double>(tasks);
  double* partial_denominator =
      maximum == nullptr ? nullptr : frame.allocate<double>(tasks);
  if (partial_values == nullptr || partial_ids == nullptr ||
      partial_counts == nullptr ||
      (maximum != nullptr &&
       (partial_maximum == nullptr || partial_denominator == nullptr))) {
    return Status::kOutOfMemory;
  }
  std::atomic<bool> invalid{false};
  LmHeadPartialContext context{
      info,
      static_cast<const std::uint8_t*>(weights.panel_data()),
      input,
      bias,
      mask,
      mask_stride,
      tiles,
      rows,
      retained,
      normalize_allowed,
      lm_head_row_dot_for(input.type, info.quant_metadata.layout),
      partial_values,
      partial_ids,
      partial_counts,
      partial_maximum,
      partial_denominator,
      &invalid};
  if (context.dot == nullptr) return Status::kUnsupportedFormat;
  threading::RangeFn kernel = nullptr;
#if defined(__x86_64__) || defined(_M_X64)
  kernel =
      lm_head_partial_mrow_kernel_for(input.type, info.quant_metadata.layout);
#endif
  threading::parallel_ranges_impl(
      kernel == nullptr ? static_cast<long long>(tasks) : tiles, 1,
      kernel == nullptr ? lm_head_partial_tiles : kernel, &context);
  if (invalid.load(std::memory_order_relaxed)) return Status::kInvalidArgument;
  std::fill_n(counts, rows, 0);
  if (maximum != nullptr) {
    std::fill_n(maximum, rows, -std::numeric_limits<double>::infinity());
    std::fill_n(denominator, rows, 0.0);
  }
  for (long long row = 0; row < rows; ++row) {
    for (long long tile = 0; tile < tiles; ++tile) {
      const long long task = row * tiles + tile;
      for (int item = 0; item < partial_counts[task]; ++item) {
        lm_head_insert(partial_values[task * retained + item],
                       partial_ids[task * retained + item],
                       best_values + row * retained, best_ids + row * retained,
                       counts + row, retained);
      }
      if (maximum != nullptr && partial_denominator[task] > 0.0) {
        const double tile_maximum = partial_maximum[task];
        const double tile_sum = partial_denominator[task];
        if (tile_maximum > maximum[row]) {
          denominator[row] =
              denominator[row] * std::exp(maximum[row] - tile_maximum) +
              tile_sum;
          maximum[row] = tile_maximum;
        } else {
          denominator[row] += tile_sum * std::exp(tile_maximum - maximum[row]);
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace

Status qgemm_prepacked_lm_head_sample_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, int* token_ids, long long rows, LmHeadSampling mode,
    int k, float top_p, float temperature, std::uint32_t seed,
    Workspace* workspace) {
  CpuPackedWeightsInfo info;
  LmHeadRowDot dot = nullptr;
  Status status =
      validate_prepared_lm_head(weights, hidden_states, rows, &info, &dot);
  if (status != Status::kOk) return status;
  if (token_ids == nullptr) return Status::kInvalidArgument;
  if (!std::isfinite(temperature) || temperature < 0.0f ||
      (mode == LmHeadSampling::kTopK && (k <= 0 || k > info.rows)) ||
      (mode == LmHeadSampling::kTopP &&
       (!std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f))) {
    return Status::kInvalidShape;
  }
  const int retained = mode == LmHeadSampling::kTopK ? k : 1;
  detail::WorkspaceFrame frame(workspace);
  int* result = frame.allocate<int>(static_cast<std::size_t>(rows));
  float* best_values =
      frame.allocate<float>(static_cast<std::size_t>(rows * retained));
  int* best_ids =
      frame.allocate<int>(static_cast<std::size_t>(rows * retained));
  int* counts = frame.allocate<int>(static_cast<std::size_t>(rows));
  double* sample_weights = frame.allocate<double>(static_cast<std::size_t>(
      mode == LmHeadSampling::kTopK ? retained : info.rows));
  float* row_logits = nullptr;
  int* order = nullptr;
  if (mode == LmHeadSampling::kCategorical || mode == LmHeadSampling::kTopP) {
    row_logits = frame.allocate<float>(static_cast<std::size_t>(info.rows));
    order = frame.allocate<int>(static_cast<std::size_t>(info.rows));
  }
  if (result == nullptr || best_values == nullptr || best_ids == nullptr ||
      counts == nullptr || sample_weights == nullptr ||
      ((mode == LmHeadSampling::kCategorical ||
        mode == LmHeadSampling::kTopP) &&
       (row_logits == nullptr || order == nullptr))) {
    return Status::kOutOfMemory;
  }
  const auto* panel = static_cast<const std::uint8_t*>(weights.panel_data());
  const bool reuse_row_tiles = lm_head_reuse_row_tiles(info, rows);
  if ((mode == LmHeadSampling::kArgmax || mode == LmHeadSampling::kTopK) &&
      (reuse_row_tiles || num_threads() > 1)) {
    status =
        num_threads() > 1
            ? lm_head_parallel_topk(weights, info, hidden_states, bias, nullptr,
                                    0, rows, retained, false, best_values,
                                    best_ids, counts, nullptr, nullptr,
                                    workspace)
            : lm_head_stream_topk(weights, info, hidden_states, bias, nullptr,
                                  0, rows, retained, false, best_values,
                                  best_ids, counts, nullptr, nullptr);
    if (status != Status::kOk) return status;
    for (long long row = 0; row < rows; ++row) {
      if (counts[row] != retained) return Status::kInvalidArgument;
      const float* row_values = best_values + row * retained;
      const int* row_ids = best_ids + row * retained;
      result[row] =
          mode == LmHeadSampling::kTopK && temperature > 0.0f
              ? lm_head_sample_sorted(
                    row_values, row_ids, retained, temperature,
                    lm_head_uniform01(seed, static_cast<std::uint64_t>(row)),
                    sample_weights)
              : row_ids[0];
    }
    std::copy_n(result, rows, token_ids);
    return Status::kOk;
  }
  for (long long row = 0; row < rows; ++row) {
    const FloatStorageInput input =
        lm_head_input_row(hidden_states, row, info.columns);
    if (mode == LmHeadSampling::kArgmax || mode == LmHeadSampling::kTopK ||
        temperature == 0.0f) {
      int count = 0;
      float* row_values = best_values + row * retained;
      int* row_ids = best_ids + row * retained;
      for (long long token = 0; token < info.rows; ++token) {
        const float value = dot(info, panel, input, token) +
                            (bias == nullptr ? 0.0f : bias[token]);
        if (!lm_head_valid_logit(value)) return Status::kInvalidArgument;
        lm_head_insert(value, static_cast<int>(token), row_values, row_ids,
                       &count, retained);
      }
      result[row] =
          mode == LmHeadSampling::kTopK && temperature > 0.0f
              ? lm_head_sample_sorted(
                    row_values, row_ids, retained, temperature,
                    lm_head_uniform01(seed, static_cast<std::uint64_t>(row)),
                    sample_weights)
              : row_ids[0];
      continue;
    }
    if (num_threads() > 1) {
      status = qgemm_prepacked_storage(
          weights, input, {row_logits, FloatStorageType::kF32, info.rows}, 1,
          workspace);
      if (status == Status::kOk) {
        for (long long token = 0; token < info.rows; ++token) {
          row_logits[token] += bias == nullptr ? 0.0f : bias[token];
          if (!lm_head_valid_logit(row_logits[token])) {
            status = Status::kInvalidArgument;
            break;
          }
        }
      }
    } else {
      status = lm_head_project_row(info, panel, dot, input, bias, row_logits);
    }
    if (status != Status::kOk) return status;
    std::iota(order, order + info.rows, 0);
    std::stable_sort(order, order + info.rows, [&](int lhs, int rhs) {
      return lm_head_better(row_logits[lhs], lhs, row_logits[rhs], rhs);
    });
    const float maximum = row_logits[order[0]];
    if (!std::isfinite(maximum)) {
      result[row] = order[0];
      continue;
    }
    double total = 0.0;
    for (long long token = 0; token < info.rows; ++token) {
      sample_weights[token] = std::exp(
          (static_cast<double>(row_logits[token]) - maximum) / temperature);
      total += sample_weights[token];
    }
    long long selected = info.rows;
    if (mode == LmHeadSampling::kTopP) {
      double cumulative = 0.0;
      selected = 0;
      do {
        cumulative += sample_weights[order[selected]];
        ++selected;
      } while (selected < info.rows && cumulative < top_p * total);
    }
    const double target =
        lm_head_uniform01(seed, static_cast<std::uint64_t>(row)) *
        std::accumulate(
            order, order + selected, 0.0,
            [&](double sum, int token) { return sum + sample_weights[token]; });
    double cumulative = 0.0;
    result[row] = order[selected - 1];
    for (long long item = 0; item < selected; ++item) {
      cumulative += sample_weights[order[item]];
      if (target < cumulative) {
        result[row] = order[item];
        break;
      }
    }
  }
  std::copy_n(result, rows, token_ids);
  return Status::kOk;
}

Status qgemm_prepacked_lm_head_masked_topk_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, const std::uint8_t* allow_mask, int* token_ids,
    float* log_probabilities, long long rows, int top_k, bool normalize_allowed,
    Workspace* workspace) {
  CpuPackedWeightsInfo info;
  LmHeadRowDot dot = nullptr;
  Status status =
      validate_prepared_lm_head(weights, hidden_states, rows, &info, &dot);
  if (status != Status::kOk) return status;
  if (top_k <= 0 || top_k > info.rows) return Status::kInvalidShape;
  if (!detail::all_nonnull(allow_mask, token_ids, log_probabilities))
    return Status::kInvalidArgument;
  const long long mask_stride = (info.rows + 7) / 8;
  for (long long row = 0; row < rows; ++row) {
    int allowed = 0;
    for (long long token = 0; token < info.rows; ++token) {
      allowed += (allow_mask[row * mask_stride + token / 8] &
                  static_cast<std::uint8_t>(0x80u >> (token & 7))) != 0;
    }
    if (allowed < top_k) return Status::kInvalidArgument;
  }
  detail::WorkspaceFrame frame(workspace);
  const std::size_t output_count = static_cast<std::size_t>(rows * top_k);
  int* result_ids = frame.allocate<int>(output_count);
  float* result_probabilities = frame.allocate<float>(output_count);
  float* best_values = frame.allocate<float>(output_count);
  int* best_ids = frame.allocate<int>(output_count);
  int* counts = frame.allocate<int>(static_cast<std::size_t>(rows));
  double* maxima = frame.allocate<double>(static_cast<std::size_t>(rows));
  double* denominators = frame.allocate<double>(static_cast<std::size_t>(rows));
  if (result_ids == nullptr || result_probabilities == nullptr ||
      best_values == nullptr || best_ids == nullptr || counts == nullptr ||
      maxima == nullptr || denominators == nullptr) {
    return Status::kOutOfMemory;
  }
  const bool reuse_row_tiles = lm_head_reuse_row_tiles(info, rows);
  if (reuse_row_tiles || num_threads() > 1) {
    status = num_threads() > 1
                 ? lm_head_parallel_topk(
                       weights, info, hidden_states, bias, allow_mask,
                       mask_stride, rows, top_k, normalize_allowed, best_values,
                       best_ids, counts, maxima, denominators, workspace)
                 : lm_head_stream_topk(weights, info, hidden_states, bias,
                                       allow_mask, mask_stride, rows, top_k,
                                       normalize_allowed, best_values, best_ids,
                                       counts, maxima, denominators);
    if (status != Status::kOk) return status;
    for (long long row = 0; row < rows; ++row) {
      if (counts[row] != top_k || !(denominators[row] > 0.0))
        return Status::kInvalidArgument;
      const double lse = maxima[row] + std::log(denominators[row]);
      for (int item = 0; item < top_k; ++item) {
        const long long output = row * top_k + item;
        result_ids[output] = best_ids[output];
        result_probabilities[output] =
            static_cast<float>(static_cast<double>(best_values[output]) - lse);
      }
    }
    std::copy_n(result_ids, output_count, token_ids);
    std::copy_n(result_probabilities, output_count, log_probabilities);
    return Status::kOk;
  }
  const auto* panel = static_cast<const std::uint8_t*>(weights.panel_data());
  for (long long row = 0; row < rows; ++row) {
    const FloatStorageInput input =
        lm_head_input_row(hidden_states, row, info.columns);
    int count = 0;
    float* row_values = best_values + row * top_k;
    int* row_ids = best_ids + row * top_k;
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    for (long long token = 0; token < info.rows; ++token) {
      const float value = dot(info, panel, input, token) +
                          (bias == nullptr ? 0.0f : bias[token]);
      if (!lm_head_valid_logit(value)) return Status::kInvalidArgument;
      const bool allowed =
          (allow_mask[row * mask_stride + token / 8] &
           static_cast<std::uint8_t>(0x80u >> (token & 7))) != 0;
      if (!normalize_allowed || allowed)
        lm_head_lse_add(value, &maximum, &denominator);
      if (allowed)
        lm_head_insert(value, static_cast<int>(token), row_values, row_ids,
                       &count, top_k);
    }
    if (count != top_k || !(denominator > 0.0)) return Status::kInvalidArgument;
    const double lse = maximum + std::log(denominator);
    for (int item = 0; item < top_k; ++item) {
      const long long output = row * top_k + item;
      result_ids[output] = row_ids[item];
      result_probabilities[output] =
          static_cast<float>(static_cast<double>(row_values[item]) - lse);
    }
  }
  std::copy_n(result_ids, output_count, token_ids);
  std::copy_n(result_probabilities, output_count, log_probabilities);
  return Status::kOk;
}

Status qgemm_prepacked_lm_head_candidates_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, const int* candidate_ids, const long long* offsets,
    int* token_ids, float* log_probabilities, long long rows,
    long long candidates, int top_k, Workspace* workspace) {
  CpuPackedWeightsInfo info;
  LmHeadRowDot dot = nullptr;
  Status status =
      validate_prepared_lm_head(weights, hidden_states, rows, &info, &dot);
  if (status != Status::kOk) return status;
  if (candidates < 0 || top_k <= 0 || top_k > info.rows)
    return Status::kInvalidShape;
  if (!detail::all_nonnull(candidate_ids, offsets, token_ids,
                           log_probabilities)) {
    return Status::kInvalidArgument;
  }
  if (offsets[0] != 0 || offsets[rows] != candidates)
    return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    if (offsets[row] > offsets[row + 1] ||
        offsets[row + 1] - offsets[row] < top_k)
      return Status::kInvalidArgument;
    for (long long item = offsets[row]; item < offsets[row + 1]; ++item) {
      const int id = candidate_ids[item];
      if (id < 0 || id >= info.rows) return Status::kInvalidArgument;
      for (long long prior = offsets[row]; prior < item; ++prior) {
        if (candidate_ids[prior] == id) return Status::kInvalidArgument;
      }
    }
  }
  detail::WorkspaceFrame frame(workspace);
  const std::size_t output_count = static_cast<std::size_t>(rows * top_k);
  int* result_ids = frame.allocate<int>(output_count);
  float* result_probabilities = frame.allocate<float>(output_count);
  float* best_values = frame.allocate<float>(static_cast<std::size_t>(top_k));
  int* best_ids = frame.allocate<int>(static_cast<std::size_t>(top_k));
  if (result_ids == nullptr || result_probabilities == nullptr ||
      best_values == nullptr || best_ids == nullptr) {
    return Status::kOutOfMemory;
  }
  const auto* panel = static_cast<const std::uint8_t*>(weights.panel_data());
  for (long long row = 0; row < rows; ++row) {
    const FloatStorageInput input =
        lm_head_input_row(hidden_states, row, info.columns);
    int count = 0;
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    for (long long item = offsets[row]; item < offsets[row + 1]; ++item) {
      const int token = candidate_ids[item];
      const float value = dot(info, panel, input, token) +
                          (bias == nullptr ? 0.0f : bias[token]);
      if (!lm_head_valid_logit(value)) return Status::kInvalidArgument;
      lm_head_lse_add(value, &maximum, &denominator);
      lm_head_insert(value, token, best_values, best_ids, &count, top_k);
    }
    if (count != top_k || !(denominator > 0.0)) return Status::kInvalidArgument;
    const double lse = maximum + std::log(denominator);
    for (int item = 0; item < top_k; ++item) {
      const long long output = row * top_k + item;
      result_ids[output] = best_ids[item];
      result_probabilities[output] =
          static_cast<float>(static_cast<double>(best_values[item]) - lse);
    }
  }
  std::copy_n(result_ids, output_count, token_ids);
  std::copy_n(result_probabilities, output_count, log_probabilities);
  return Status::kOk;
}

Status qgemm_prepacked_lm_head_beam_advance_storage(
    const CpuPackedWeights& weights, FloatStorageInput hidden_states,
    const float* bias, const float* cumulative_log_probabilities,
    int* next_token, int* parent_beam, float* next_cumulative, long long batch,
    long long beam_width, Workspace* workspace) {
  if (!detail::valid_product({batch, beam_width})) return Status::kInvalidShape;
  const long long rows = batch * beam_width;
  CpuPackedWeightsInfo info;
  LmHeadRowDot dot = nullptr;
  Status status =
      validate_prepared_lm_head(weights, hidden_states, rows, &info, &dot);
  if (status != Status::kOk) return status;
  if (beam_width > info.rows || beam_width > std::numeric_limits<int>::max())
    return Status::kInvalidShape;
  if (!detail::all_nonnull(cumulative_log_probabilities, next_token,
                           parent_beam, next_cumulative)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (!lm_head_valid_logit(cumulative_log_probabilities[row]))
      return Status::kInvalidArgument;
  }
  detail::WorkspaceFrame frame(workspace);
  const std::size_t per_request =
      static_cast<std::size_t>(beam_width * beam_width);
  float* parent_values = frame.allocate<float>(per_request);
  int* parent_tokens = frame.allocate<int>(per_request);
  double* parent_lse =
      frame.allocate<double>(static_cast<std::size_t>(beam_width));
  double* parent_maximum =
      frame.allocate<double>(static_cast<std::size_t>(beam_width));
  double* parent_denominator =
      frame.allocate<double>(static_cast<std::size_t>(beam_width));
  int* parent_counts =
      frame.allocate<int>(static_cast<std::size_t>(beam_width));
  const std::size_t output_count = static_cast<std::size_t>(rows);
  int* result_tokens = frame.allocate<int>(output_count);
  int* result_parents = frame.allocate<int>(output_count);
  float* result_scores = frame.allocate<float>(output_count);
  if (parent_values == nullptr || parent_tokens == nullptr ||
      parent_lse == nullptr || parent_maximum == nullptr ||
      parent_denominator == nullptr || parent_counts == nullptr ||
      result_tokens == nullptr || result_parents == nullptr ||
      result_scores == nullptr) {
    return Status::kOutOfMemory;
  }
  const auto* panel = static_cast<const std::uint8_t*>(weights.panel_data());
  for (long long request = 0; request < batch; ++request) {
    if (num_threads() > 1) {
      FloatStorageInput request_input =
          lm_head_input_row(hidden_states, request * beam_width, info.columns);
      request_input.count = beam_width * info.columns;
      status = lm_head_parallel_topk(
          weights, info, request_input, bias, nullptr, 0, beam_width,
          static_cast<int>(beam_width), false, parent_values, parent_tokens,
          parent_counts, parent_maximum, parent_denominator, workspace);
      if (status != Status::kOk) return status;
      for (long long parent = 0; parent < beam_width; ++parent) {
        if (parent_counts[parent] != beam_width ||
            !(parent_denominator[parent] > 0.0)) {
          return Status::kInvalidArgument;
        }
        parent_lse[parent] =
            parent_maximum[parent] + std::log(parent_denominator[parent]);
      }
    } else if (lm_head_reuse_row_tiles(info, beam_width)) {
      FloatStorageInput request_input =
          lm_head_input_row(hidden_states, request * beam_width, info.columns);
      request_input.count = beam_width * info.columns;
      status = lm_head_stream_topk(
          weights, info, request_input, bias, nullptr, 0, beam_width,
          static_cast<int>(beam_width), false, parent_values, parent_tokens,
          parent_counts, parent_maximum, parent_denominator);
      if (status != Status::kOk) return status;
      for (long long parent = 0; parent < beam_width; ++parent) {
        if (parent_counts[parent] != beam_width ||
            !(parent_denominator[parent] > 0.0)) {
          return Status::kInvalidArgument;
        }
        parent_lse[parent] =
            parent_maximum[parent] + std::log(parent_denominator[parent]);
      }
    } else {
      for (long long parent = 0; parent < beam_width; ++parent) {
        const long long row = request * beam_width + parent;
        const FloatStorageInput input =
            lm_head_input_row(hidden_states, row, info.columns);
        float* values = parent_values + parent * beam_width;
        int* ids = parent_tokens + parent * beam_width;
        int count = 0;
        double maximum = -std::numeric_limits<double>::infinity();
        double denominator = 0.0;
        for (long long token = 0; token < info.rows; ++token) {
          const float value = dot(info, panel, input, token) +
                              (bias == nullptr ? 0.0f : bias[token]);
          if (!lm_head_valid_logit(value)) return Status::kInvalidArgument;
          lm_head_lse_add(value, &maximum, &denominator);
          lm_head_insert(value, static_cast<int>(token), values, ids, &count,
                         static_cast<int>(beam_width));
        }
        if (count != beam_width || !(denominator > 0.0))
          return Status::kInvalidArgument;
        parent_lse[parent] = maximum + std::log(denominator);
      }
    }
    float* scores = result_scores + request * beam_width;
    int* tokens = result_tokens + request * beam_width;
    int* parents = result_parents + request * beam_width;
    int count = 0;
    for (long long parent = 0; parent < beam_width; ++parent) {
      for (long long item = 0; item < beam_width; ++item) {
        const float score = static_cast<float>(
            cumulative_log_probabilities[request * beam_width + parent] +
            static_cast<double>(parent_values[parent * beam_width + item]) -
            parent_lse[parent]);
        const int token = parent_tokens[parent * beam_width + item];
        int position = count;
        while (position > 0 && (score > scores[position - 1] ||
                                (score == scores[position - 1] &&
                                 (parent < parents[position - 1] ||
                                  (parent == parents[position - 1] &&
                                   token < tokens[position - 1]))))) {
          --position;
        }
        if (position >= beam_width) continue;
        const int upper = std::min(count, static_cast<int>(beam_width) - 1);
        for (int slot = upper; slot > position; --slot) {
          scores[slot] = scores[slot - 1];
          tokens[slot] = tokens[slot - 1];
          parents[slot] = parents[slot - 1];
        }
        scores[position] = score;
        tokens[position] = token;
        parents[position] = static_cast<int>(parent);
        if (count < beam_width) ++count;
      }
    }
  }
  std::copy_n(result_tokens, output_count, next_token);
  std::copy_n(result_parents, output_count, parent_beam);
  std::copy_n(result_scores, output_count, next_cumulative);
  return Status::kOk;
}

namespace {

std::size_t float_storage_bytes(FloatStorageType type) {
  return type == FloatStorageType::kF32 ? sizeof(float) : sizeof(std::uint16_t);
}

const void* float_storage_offset(const void* data, FloatStorageType type,
                                 long long elements) {
  return static_cast<const std::uint8_t*>(data) +
         static_cast<std::size_t>(elements) * float_storage_bytes(type);
}

void* float_storage_offset(void* data, FloatStorageType type,
                           long long elements) {
  return static_cast<std::uint8_t*>(data) +
         static_cast<std::size_t>(elements) * float_storage_bytes(type);
}

Status validate_moe_prepared_arrays(const CpuPackedWeights* first,
                                    const CpuPackedWeights* second,
                                    long long experts, FloatStorageInput x,
                                    const int* expert_ids,
                                    FloatStorageOutput output, long long rows,
                                    bool swiglu,
                                    CpuPackedWeightsInfo* first_info) {
  if (experts <= 0 || rows <= 0 || first == nullptr || expert_ids == nullptr ||
      x.data == nullptr || output.data == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!known_storage(x.type) || !known_storage(output.type))
    return Status::kUnsupportedFormat;
  Status status = validate_prepared_projection(first[0], first_info);
  if (status != Status::kOk) return status;
  const long long columns = first_info->columns;
  const long long output_columns = first_info->rows;
  if (!detail::valid_product({rows, columns}) ||
      !detail::valid_product({rows, output_columns}) ||
      !detail::valid_product({experts, output_columns}) ||
      x.count != rows * columns || output.count != rows * output_columns) {
    return Status::kInvalidShape;
  }
  for (long long expert = 0; expert < experts; ++expert) {
    CpuPackedWeightsInfo info;
    status = validate_prepared_projection(first[expert], &info);
    if (status != Status::kOk) return status;
    if (info.columns != columns || info.rows != output_columns)
      return Status::kInvalidShape;
    if (swiglu) {
      if (second == nullptr) return Status::kInvalidArgument;
      CpuPackedWeightsInfo other;
      status = validate_prepared_projection(second[expert], &other);
      if (status != Status::kOk) return status;
      if (other.columns != columns || other.rows != output_columns ||
          !compatible_gate_up(info, other))
        return Status::kInvalidShape;
    }
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < -1 || expert_ids[row] >= experts)
      return Status::kInvalidArgument;
  }
  return Status::kOk;
}

bool moe_prefers_token_rows(const CpuPackedWeights* weights,
                            long long experts) {
#if !defined(__aarch64__) && !defined(_M_ARM64)
  (void)weights;
  (void)experts;
  return false;
#else
  if (num_threads() != 1) return false;
  for (long long expert = 0; expert < experts; ++expert) {
    const CanonicalQuantLayout layout =
        weights[expert].info().quant_metadata.layout;
    // The single-token kernels for compact 4-bit/8-bit panels are materially
    // faster than the generic multi-row decoder at small expert batches. FP8
    // is deliberately excluded: its multi-row panel decoder amortizes table
    // conversion well even at roughly 16 tokens per expert.
    if (layout == CanonicalQuantLayout::kFP8E4M3FN ||
        layout == CanonicalQuantLayout::kFP8E5M2 ||
        layout == CanonicalQuantLayout::kMXFP8E4M3E8M0) {
      return false;
    }
  }
  return true;
#endif
}

struct MoeParallelProjectionContext {
  CanonicalStorageContext* projections = nullptr;
  threading::RangeFn* kernels = nullptr;
  long long* task_offsets = nullptr;
  long long groups = 0;
};

void moe_parallel_projection_tasks(void* opaque, long long begin, long long end,
                                   int worker) {
  const auto& context =
      *static_cast<const MoeParallelProjectionContext*>(opaque);
  long long group = static_cast<long long>(
      std::upper_bound(context.task_offsets,
                       context.task_offsets + context.groups + 1, begin) -
      context.task_offsets - 1);
  while (begin < end) {
    const long long group_end = std::min(end, context.task_offsets[group + 1]);
    context.kernels[group](&context.projections[group],
                           begin - context.task_offsets[group],
                           group_end - context.task_offsets[group], worker);
    begin = group_end;
    ++group;
  }
}

struct MoeParallelSwiGLUContext {
  CanonicalGateUpContext* projections = nullptr;
  threading::RangeFn* kernels = nullptr;
  long long* task_offsets = nullptr;
  long long groups = 0;
};

struct MoeQuantizedProjectionContext {
  CpuPackedWeightsInfo info;
  const std::uint8_t* panel = nullptr;
  const CanonicalQuantTensor* activation = nullptr;
  const long long* input_rows = nullptr;
  float* output = nullptr;
  long long rows = 0;
};

template <CanonicalQuantLayout Layout>
void moe_quantized_projection_panels(void* opaque, long long begin,
                                     long long end, int) {
  const auto& context =
      *static_cast<const MoeQuantizedProjectionContext*>(opaque);
  const CpuPackedWeightsInfo& info = context.info;
  const auto* order = prepared_table<int>(context.panel, info.act_order_offset);
  alignas(64) float decoded_weights[kMaximumRowTile * kKTile];
  alignas(64) float decoded_activations[kMTile * kKTile];
  alignas(64) float accumulators[kAccumulatorCount * kMaximumRowTile * kMTile];
  for (long long row_panel = begin; row_panel < end; ++row_panel) {
    const long long first_output = row_panel * info.row_tile;
    const long long lanes = std::min(info.row_tile, info.rows - first_output);
    for (long long m_base = 0; m_base < context.rows; m_base += kMTile) {
      const long long m_count = std::min(kMTile, context.rows - m_base);
      std::fill_n(accumulators, kAccumulatorCount * kMaximumRowTile * kMTile,
                  0.0f);
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long item_base = 0; item_base < info.block_size;
             item_base += kKTile) {
          const long long items = std::min(kKTile, info.block_size - item_base);
          const long long first_column = block * info.block_size + item_base;
          for (long long item = 0; item < items; ++item) {
            const long long packed_column = first_column + item;
            const long long logical_column =
                order == nullptr ? packed_column : order[packed_column];
            for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
              decoded_activations[m_lane * kKTile + item] = tensor_value(
                  *context.activation, context.input_rows[m_base + m_lane],
                  logical_column);
            }
          }
          decode_prepared_chunk<Layout>(info, context.panel, row_panel, block,
                                        lanes, item_base, items, first_output,
                                        decoded_weights);
          for (long long item = 0; item < items; ++item) {
            const long long accumulator = (first_column + item) & 3;
            for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
              accumulate_output_lanes(
                  accumulators +
                      (accumulator * kMTile + m_lane) * kMaximumRowTile,
                  decoded_weights + item * kMaximumRowTile,
                  decoded_activations[m_lane * kKTile + item], lanes);
            }
          }
        }
      }
      constexpr long long stride = kMTile * kMaximumRowTile;
      for (long long m_lane = 0; m_lane < m_count; ++m_lane) {
        const long long input_row = context.input_rows[m_base + m_lane];
        const float* row_partial = accumulators + m_lane * kMaximumRowTile;
        for (long long lane = 0; lane < lanes; ++lane) {
          const float* partial = row_partial + lane;
          context.output[input_row * info.rows + first_output + lane] =
              (partial[0] + partial[stride]) +
              (partial[2 * stride] + partial[3 * stride]);
        }
      }
    }
  }
}

threading::RangeFn moe_quantized_projection_kernel(
    CanonicalQuantLayout layout) {
#define QUIXICORE_MOE_QUANTIZED_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:         \
    return moe_quantized_projection_panels<CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_MOE_QUANTIZED_CASE(kInt4Symmetric);
    QUIXICORE_MOE_QUANTIZED_CASE(kUInt4Affine);
    QUIXICORE_MOE_QUANTIZED_CASE(kInt8Symmetric);
    QUIXICORE_MOE_QUANTIZED_CASE(kInt8Affine);
    QUIXICORE_MOE_QUANTIZED_CASE(kFP8E4M3FN);
    QUIXICORE_MOE_QUANTIZED_CASE(kFP8E5M2);
    QUIXICORE_MOE_QUANTIZED_CASE(kFP4E2M1);
    QUIXICORE_MOE_QUANTIZED_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_MOE_QUANTIZED_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_MOE_QUANTIZED_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_MOE_QUANTIZED_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_MOE_QUANTIZED_CASE
  return nullptr;
}

struct MoeParallelQuantizedContext {
  MoeQuantizedProjectionContext* projections = nullptr;
  threading::RangeFn* kernels = nullptr;
  long long* task_offsets = nullptr;
  long long groups = 0;
};

struct MoeParallelDirectDualContext {
  CanonicalDualContext* projections = nullptr;
  threading::RangeFn* kernels = nullptr;
  long long* task_offsets = nullptr;
  long long groups = 0;
};

void moe_parallel_direct_dual_tasks(void* opaque, long long begin,
                                    long long end, int worker) {
  const auto& context =
      *static_cast<const MoeParallelDirectDualContext*>(opaque);
  long long group = static_cast<long long>(
      std::upper_bound(context.task_offsets,
                       context.task_offsets + context.groups + 1, begin) -
      context.task_offsets - 1);
  while (begin < end) {
    const long long group_end = std::min(end, context.task_offsets[group + 1]);
    context.kernels[group](&context.projections[group],
                           begin - context.task_offsets[group],
                           group_end - context.task_offsets[group], worker);
    begin = group_end;
    ++group;
  }
}

bool try_moe_sorted_direct_dual(const CpuPackedWeights* expert_weights,
                                long long experts,
                                const CanonicalQuantTensor& activation,
                                const int* expert_ids, float* output,
                                Workspace* workspace) {
  (void)experts;
  const long long rows = activation.metadata.logical_rows;
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < 0 ||
        (row > 0 && expert_ids[row] < expert_ids[row - 1])) {
      return false;
    }
  }
  long long groups = 0;
  for (long long begin = 0; begin < rows;) {
    long long end = begin + 1;
    while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
    const CpuPackedWeightsInfo info = expert_weights[expert_ids[begin]].info();
    if (!direct_dual_gemv_supported(info, activation)) return false;
    ++groups;
    begin = end;
  }

  detail::WorkspaceFrame frame(workspace);
  auto* contexts =
      frame.allocate<CanonicalDualContext>(static_cast<std::size_t>(groups));
  auto* kernels =
      frame.allocate<threading::RangeFn>(static_cast<std::size_t>(groups));
  long long* task_offsets =
      frame.allocate<long long>(static_cast<std::size_t>(groups + 1));
  if (contexts == nullptr || kernels == nullptr || task_offsets == nullptr)
    return false;
  task_offsets[0] = 0;
  long long group = 0;
  for (long long begin = 0; begin < rows; ++group) {
    long long end = begin + 1;
    while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
    const int expert = expert_ids[begin];
    const CpuPackedWeightsInfo info = expert_weights[expert].info();
    contexts[group] = {
        info,
        static_cast<const std::uint8_t*>(expert_weights[expert].panel_data()),
        &activation, output + begin * info.rows, begin};
    const long long count = end - begin;
    const long long panels = (info.rows + info.row_tile - 1) / info.row_tile;
    if (info.block_size <= kKTile &&
        dual_fp8_panel_pair(info.quant_metadata.layout,
                            activation.metadata.layout)) {
      kernels[group] = dual_fp8_panel_group_kernel<kDualFp8GemmPanelGroup>(
          info.quant_metadata.layout, activation.metadata.layout);
      task_offsets[group + 1] =
          task_offsets[group] + count * ((panels + kDualFp8GemmPanelGroup - 1) /
                                         kDualFp8GemmPanelGroup);
    } else {
      kernels[group] = dual_panel_kernel(info.quant_metadata.layout,
                                         activation.metadata.layout);
      task_offsets[group + 1] = task_offsets[group] + count * panels;
    }
    if (kernels[group] == nullptr) return false;
    begin = end;
  }
  MoeParallelDirectDualContext context{contexts, kernels, task_offsets, groups};
  threading::parallel_ranges_impl(task_offsets[groups], 1,
                                  moe_parallel_direct_dual_tasks, &context);
  return true;
}

using MoeSwiGLUProjectFn = void (*)(const CanonicalSwiGLUQuantContext&,
                                    long long, long long, long long, float*,
                                    float*);

template <FloatStorageType Type, CanonicalQuantLayout Layout>
void moe_project_swiglu_group(const CanonicalSwiGLUQuantContext& context,
                              long long input_row, long long first_output,
                              long long outputs, float* gate_values,
                              float* up_values) {
  project_swiglu_group<Type, Layout>(context, input_row, first_output, outputs,
                                     gate_values, up_values);
}

template <FloatStorageType Type>
MoeSwiGLUProjectFn moe_swiglu_project_kernel(CanonicalQuantLayout layout) {
#define QUIXICORE_MOE_SWIGLU_PROJECT_CASE(layout_name) \
  case CanonicalQuantLayout::layout_name:              \
    return moe_project_swiglu_group<Type, CanonicalQuantLayout::layout_name>
  switch (layout) {
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kInt4Symmetric);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kUInt4Affine);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kInt8Symmetric);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kInt8Affine);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kFP8E4M3FN);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kFP8E5M2);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kFP4E2M1);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kMXFP8E4M3E8M0);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kMXFP4E2M1E8M0);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kNVFP4E2M1E4M3);
    QUIXICORE_MOE_SWIGLU_PROJECT_CASE(kBitNetTernary);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return nullptr;
  }
#undef QUIXICORE_MOE_SWIGLU_PROJECT_CASE
  return nullptr;
}

struct MoeSwiGLUQuantizedContext {
  CanonicalSwiGLUQuantContext* experts = nullptr;
  MoeSwiGLUProjectFn* project = nullptr;
  const int* expert_ids = nullptr;
  float* scratch = nullptr;
  float* worker_maximum = nullptr;
  std::atomic<bool>* invalid = nullptr;
  long long rows = 0;
  long long quant_group = 0;
  long long groups_per_row = 0;
  long long row_domain = 1;
  long long scratch_elements = 0;
  bool scale_2d = false;
  bool measure_only = false;
};

void moe_swiglu_quantized_groups(void* opaque, long long begin, long long end,
                                 int worker) {
  auto& context = *static_cast<MoeSwiGLUQuantizedContext*>(opaque);
  float* gate_values = context.scratch + static_cast<long long>(worker) *
                                             context.scratch_elements * 2;
  float* up_values = gate_values + context.scratch_elements;
  float local_maximum = context.worker_maximum[worker];
  const bool nvfp4 = context.experts[0].output->metadata.layout ==
                     CanonicalQuantLayout::kNVFP4E2M1E4M3;
  for (long long task = begin; task < end; ++task) {
    const long long domain = task / context.groups_per_row;
    const long long group = task % context.groups_per_row;
    const long long first_row = domain * context.row_domain;
    const long long domain_rows =
        std::min(context.row_domain, context.rows - first_row);
    for (long long row_lane = 0; row_lane < domain_rows; ++row_lane) {
      const long long row = first_row + row_lane;
      float* gate = gate_values + row_lane * context.quant_group;
      float* up = up_values + row_lane * context.quant_group;
      const int expert = context.expert_ids[row];
      if (expert < 0) {
        std::fill_n(gate, context.quant_group, 0.0f);
        std::fill_n(up, context.quant_group, 0.0f);
      } else {
        context.project[expert](context.experts[expert], row,
                                group * context.quant_group,
                                context.quant_group, gate, up);
      }
    }
    if (context.measure_only) {
      local_maximum =
          std::max(local_maximum,
                   group_absmax(gate_values, domain_rows * context.quant_group,
                                context.invalid));
    } else if (nvfp4) {
      encode_swiglu_nvfp4_group(context.experts[0], first_row, domain_rows,
                                group, gate_values);
    } else {
      encode_swiglu_quant_group(context.experts[0], first_row, group,
                                gate_values);
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

void moe_swiglu_quantized_full_rows(void* opaque, long long begin,
                                    long long end, int worker) {
  auto& context = *static_cast<MoeSwiGLUQuantizedContext*>(opaque);
  float* gate_values = context.scratch + static_cast<long long>(worker) *
                                             context.scratch_elements * 2;
  float* up_values = gate_values + context.scratch_elements;
  float local_maximum = context.worker_maximum[worker];
  const bool nvfp4 = context.experts[0].output->metadata.layout ==
                     CanonicalQuantLayout::kNVFP4E2M1E4M3;
  for (long long row = begin; row < end; ++row) {
    const int expert = context.expert_ids[row];
    if (expert < 0) {
      std::fill_n(gate_values, context.scratch_elements, 0.0f);
    } else {
      context.project[expert](context.experts[expert], row, 0,
                              context.scratch_elements, gate_values, up_values);
    }
    if (context.measure_only) {
      local_maximum = std::max(
          local_maximum,
          group_absmax(gate_values, context.scratch_elements, context.invalid));
      continue;
    }
    for (long long group = 0; group < context.groups_per_row; ++group) {
      const float* values = gate_values + group * context.quant_group;
      if (nvfp4) {
        encode_swiglu_nvfp4_group(context.experts[0], row, 1, group, values);
      } else {
        encode_swiglu_quant_group(context.experts[0], row, group, values);
      }
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

struct MoeParallelSwiGLUQuantizedContext {
  CanonicalSwiGLUQuantContext* projections = nullptr;
  threading::RangeFn* kernels = nullptr;
  long long* task_offsets = nullptr;
  long long groups = 0;
};

void moe_parallel_swiglu_quantized_tasks(void* opaque, long long begin,
                                         long long end, int worker) {
  auto& context = *static_cast<MoeParallelSwiGLUQuantizedContext*>(opaque);
  long long group = static_cast<long long>(
      std::upper_bound(context.task_offsets,
                       context.task_offsets + context.groups + 1, begin) -
      context.task_offsets - 1);
  while (begin < end) {
    const long long group_end = std::min(end, context.task_offsets[group + 1]);
    context.kernels[group](&context.projections[group],
                           begin - context.task_offsets[group],
                           group_end - context.task_offsets[group], worker);
    begin = group_end;
    ++group;
  }
}

bool moe_swiglu_quant_prefers_panels(CanonicalQuantLayout layout) {
#if defined(__aarch64__) || defined(_M_ARM64)
  return layout == CanonicalQuantLayout::kFP8E4M3FN ||
         layout == CanonicalQuantLayout::kFP8E5M2 ||
         layout == CanonicalQuantLayout::kMXFP8E4M3E8M0;
#elif defined(__x86_64__) || defined(_M_X64)
  (void)layout;
  return true;
#else
  (void)layout;
  return false;
#endif
}

Status try_moe_sorted_swiglu_quantized_panels(
    const CpuPackedWeights* gate_weights, const CpuPackedWeights* up_weights,
    long long experts, FloatStorageInput x, const int* expert_ids,
    CanonicalQuantTensor* output, long long rows, long long quant_group,
    bool scale_2d, bool needs_global_maximum, Workspace* workspace,
    bool* handled) {
  (void)experts;
  *handled = false;
  if (scale_2d &&
      output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    return Status::kOk;
  }
  long long groups = 0;
  for (long long begin = 0; begin < rows;) {
    if (expert_ids[begin] < 0 ||
        (begin > 0 && expert_ids[begin] < expert_ids[begin - 1])) {
      return Status::kOk;
    }
    long long end = begin + 1;
    while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
    const CpuPackedWeightsInfo info = gate_weights[expert_ids[begin]].info();
    if (end - begin <= 1 || quant_group % info.row_tile != 0 ||
        !moe_swiglu_quant_prefers_panels(info.quant_metadata.layout)) {
      return Status::kOk;
    }
    ++groups;
    begin = end;
  }

  const std::size_t workers = static_cast<std::size_t>(num_threads());
  const long long groups_per_row =
      gate_weights[expert_ids[0]].info().rows / quant_group;
  const long long scratch_elements = quant_group * kMTile;
  if (!detail::valid_product({groups_per_row, scratch_elements}) ||
      static_cast<unsigned long long>(scratch_elements) >
          std::numeric_limits<std::size_t>::max() / (2 * sizeof(float)) /
              workers) {
    return Status::kInvalidShape;
  }

  detail::WorkspaceFrame frame(workspace);
  auto* contexts = frame.allocate<CanonicalSwiGLUQuantContext>(
      static_cast<std::size_t>(groups));
  auto* kernels =
      frame.allocate<threading::RangeFn>(static_cast<std::size_t>(groups));
  long long* task_offsets =
      frame.allocate<long long>(static_cast<std::size_t>(groups + 1));
  float* scratch = frame.allocate<float>(
      workers * static_cast<std::size_t>(scratch_elements) * 2);
  float* worker_maximum = frame.allocate<float>(workers);
  if (contexts == nullptr || kernels == nullptr || task_offsets == nullptr ||
      scratch == nullptr || worker_maximum == nullptr) {
    return Status::kOutOfMemory;
  }
  std::fill_n(worker_maximum, workers, 0.0f);
  std::atomic<bool> invalid{false};
  task_offsets[0] = 0;
  long long group = 0;
  for (long long begin = 0; begin < rows; ++group) {
    long long end = begin + 1;
    while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
    const int expert = expert_ids[begin];
    const CpuPackedWeightsInfo gate_info = gate_weights[expert].info();
    const CpuPackedWeightsInfo up_info = up_weights[expert].info();
    switch (x.type) {
      case FloatStorageType::kF32:
        kernels[group] = swiglu_quant_panel_kernel<FloatStorageType::kF32>(
            gate_info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
        kernels[group] = swiglu_quant_panel_kernel<FloatStorageType::kF16>(
            gate_info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
        kernels[group] = swiglu_quant_panel_kernel<FloatStorageType::kBF16>(
            gate_info.quant_metadata.layout);
        break;
    }
    if (kernels[group] == nullptr) return Status::kUnsupportedFormat;
    contexts[group] = {
        gate_info,
        up_info,
        static_cast<const std::uint8_t*>(gate_weights[expert].panel_data()),
        static_cast<const std::uint8_t*>(up_weights[expert].panel_data()),
        {float_storage_offset(x.data, x.type, begin * gate_info.columns),
         x.type, (end - begin) * gate_info.columns},
        output,
        scratch,
        worker_maximum,
        &invalid,
        end - begin,
        quant_group,
        groups_per_row,
        kMTile,
        scratch_elements,
        0.0f,
        scale_2d,
        needs_global_maximum,
        begin};
    const long long m_tiles = (end - begin + kMTile - 1) / kMTile;
    task_offsets[group + 1] = task_offsets[group] + m_tiles * groups_per_row;
    begin = end;
  }
  MoeParallelSwiGLUQuantizedContext context{contexts, kernels, task_offsets,
                                            groups};
  if (needs_global_maximum) {
    threading::parallel_ranges_impl(
        task_offsets[groups], 1, moe_parallel_swiglu_quantized_tasks, &context);
    if (invalid.load(std::memory_order_relaxed))
      return Status::kInvalidArgument;
    float maximum = 0.0f;
    for (std::size_t worker = 0; worker < workers; ++worker)
      maximum = std::max(maximum, worker_maximum[worker]);
    if (output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
      const float global_scale = maximum / (6.0f * 448.0f);
      output->metadata.global_scale = global_scale;
      for (long long item = 0; item < groups; ++item) {
        contexts[item].global_scale = global_scale;
        contexts[item].measure_only = false;
      }
    } else {
      const bool fp4 =
          output->metadata.layout == CanonicalQuantLayout::kFP4E2M1;
      const float maximum_code =
          fp4 ? 6.0f
              : (output->metadata.layout == CanonicalQuantLayout::kFP8E4M3FN
                     ? 448.0f
                     : 57344.0f);
      float scale = maximum == 0.0f ? 0.0f : maximum / maximum_code;
      if (fp4) scale = f16_to_float(float_to_f16(scale));
      output->scales[0] = scale;
      for (long long item = 0; item < groups; ++item)
        contexts[item].measure_only = false;
    }
  }
  threading::parallel_ranges_impl(
      task_offsets[groups], 1, moe_parallel_swiglu_quantized_tasks, &context);
  if (invalid.load(std::memory_order_relaxed)) return Status::kInvalidArgument;
  *handled = true;
  return Status::kOk;
}

void moe_parallel_quantized_tasks(void* opaque, long long begin, long long end,
                                  int worker) {
  const auto& context =
      *static_cast<const MoeParallelQuantizedContext*>(opaque);
  long long group = static_cast<long long>(
      std::upper_bound(context.task_offsets,
                       context.task_offsets + context.groups + 1, begin) -
      context.task_offsets - 1);
  while (begin < end) {
    const long long group_end = std::min(end, context.task_offsets[group + 1]);
    context.kernels[group](&context.projections[group],
                           begin - context.task_offsets[group],
                           group_end - context.task_offsets[group], worker);
    begin = group_end;
    ++group;
  }
}

void moe_parallel_swiglu_tasks(void* opaque, long long begin, long long end,
                               int worker) {
  const auto& context = *static_cast<const MoeParallelSwiGLUContext*>(opaque);
  long long group = static_cast<long long>(
      std::upper_bound(context.task_offsets,
                       context.task_offsets + context.groups + 1, begin) -
      context.task_offsets - 1);
  while (begin < end) {
    const long long group_end = std::min(end, context.task_offsets[group + 1]);
    context.kernels[group](&context.projections[group],
                           begin - context.task_offsets[group],
                           group_end - context.task_offsets[group], worker);
    begin = group_end;
    ++group;
  }
}

bool try_moe_sorted_parallel(const CpuPackedWeights* first,
                             const CpuPackedWeights* second,
                             FloatStorageInput x, const int* expert_ids,
                             const float* bias, FloatStorageOutput output,
                             long long rows, LinearActivation activation,
                             bool swiglu, Workspace* workspace) {
  if (num_threads() <= 1 || x.type != FloatStorageType::kF32) return false;
  long long groups = 0;
  for (long long begin = 0; begin < rows;) {
    long long end = begin + 1;
    while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
    if (end - begin == 1) return false;
    ++groups;
    begin = end;
  }

  detail::WorkspaceFrame frame(workspace);
  auto* kernels =
      frame.allocate<threading::RangeFn>(static_cast<std::size_t>(groups));
  auto* task_offsets =
      frame.allocate<long long>(static_cast<std::size_t>(groups + 1));
  if (kernels == nullptr || task_offsets == nullptr) return false;
  task_offsets[0] = 0;

  if (!swiglu) {
    auto* contexts = frame.allocate<CanonicalStorageContext>(
        static_cast<std::size_t>(groups));
    if (contexts == nullptr) return false;
    long long group = 0;
    for (long long begin = 0; begin < rows; ++group) {
      long long end = begin + 1;
      while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
      const int expert = expert_ids[begin];
      const CpuPackedWeightsInfo info = first[expert].info();
      kernels[group] =
          storage_kernel<FloatStorageType::kF32>(info.quant_metadata.layout);
      if (kernels[group] == nullptr) return false;
      contexts[group] = {
          info,
          static_cast<const std::uint8_t*>(first[expert].panel_data()),
          {float_storage_offset(x.data, x.type, begin * info.columns), x.type,
           (end - begin) * info.columns},
          {float_storage_offset(output.data, output.type, begin * info.rows),
           output.type, (end - begin) * info.rows},
          end - begin,
          bias == nullptr ? nullptr
                          : bias + static_cast<long long>(expert) * info.rows,
          activation};
      task_offsets[group + 1] =
          task_offsets[group] + (info.rows + info.row_tile - 1) / info.row_tile;
      begin = end;
    }
    MoeParallelProjectionContext context{contexts, kernels, task_offsets,
                                         groups};
    threading::parallel_ranges_impl(task_offsets[groups], 1,
                                    moe_parallel_projection_tasks, &context);
    return true;
  }

  auto* contexts =
      frame.allocate<CanonicalGateUpContext>(static_cast<std::size_t>(groups));
  if (contexts == nullptr) return false;
  long long group = 0;
  for (long long begin = 0; begin < rows; ++group) {
    long long end = begin + 1;
    while (end < rows && expert_ids[end] == expert_ids[begin]) ++end;
    const int expert = expert_ids[begin];
    const CpuPackedWeightsInfo gate_info = first[expert].info();
    const CpuPackedWeightsInfo up_info = second[expert].info();
    kernels[group] = gate_up_storage_kernel<FloatStorageType::kF32, true>(
        gate_info.quant_metadata.layout);
    if (kernels[group] == nullptr) return false;
    FloatStorageOutput destination{
        float_storage_offset(output.data, output.type, begin * gate_info.rows),
        output.type, (end - begin) * gate_info.rows};
    contexts[group] = {
        gate_info,
        up_info,
        static_cast<const std::uint8_t*>(first[expert].panel_data()),
        static_cast<const std::uint8_t*>(second[expert].panel_data()),
        {float_storage_offset(x.data, x.type, begin * gate_info.columns),
         x.type, (end - begin) * gate_info.columns},
        {},
        {},
        destination,
        output.type == FloatStorageType::kF32
            ? static_cast<float*>(destination.data)
            : nullptr,
        end - begin,
        true};
    task_offsets[group + 1] =
        task_offsets[group] +
        (gate_info.rows + gate_info.row_tile - 1) / gate_info.row_tile;
    begin = end;
  }
  MoeParallelSwiGLUContext context{contexts, kernels, task_offsets, groups};
  threading::parallel_ranges_impl(task_offsets[groups], 1,
                                  moe_parallel_swiglu_tasks, &context);
  return true;
}

Status moe_grouped_prepacked_impl(const CpuPackedWeights* first,
                                  const CpuPackedWeights* second,
                                  long long experts, FloatStorageInput x,
                                  const int* expert_ids, const float* bias,
                                  FloatStorageOutput output, long long rows,
                                  LinearActivation activation, bool swiglu,
                                  Workspace* workspace) {
  if (!swiglu && !known_linear_activation(activation))
    return Status::kInvalidArgument;
  CpuPackedWeightsInfo info;
  Status status = validate_moe_prepared_arrays(
      first, second, experts, x, expert_ids, output, rows, swiglu, &info);
  if (status != Status::kOk) return status;
  const long long input_columns = info.columns;
  const long long output_columns = info.rows;

  if (moe_prefers_token_rows(first, experts) &&
      (!swiglu || moe_prefers_token_rows(second, experts))) {
    const std::size_t output_row_bytes =
        static_cast<std::size_t>(output_columns) *
        float_storage_bytes(output.type);
    for (long long row = 0; row < rows; ++row) {
      void* destination_data =
          float_storage_offset(output.data, output.type, row * output_columns);
      if (expert_ids[row] < 0) {
        std::memset(destination_data, 0, output_row_bytes);
        continue;
      }
      const int expert = expert_ids[row];
      const FloatStorageInput input{
          float_storage_offset(x.data, x.type, row * input_columns), x.type,
          input_columns};
      const FloatStorageOutput destination{destination_data, output.type,
                                           output_columns};
      status =
          swiglu ? qgemv_prepacked_swiglu_storage(first[expert], second[expert],
                                                  input, destination, workspace)
                 : qgemm_prepacked_epilogue_storage(
                       first[expert], input,
                       bias == nullptr ? nullptr
                                       : bias + static_cast<long long>(expert) *
                                                    output_columns,
                       destination, 1, activation, workspace);
      if (status != Status::kOk) return status;
    }
    return Status::kOk;
  }

  bool sorted = true;
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < 0 ||
        (row > 0 && expert_ids[row] < expert_ids[row - 1])) {
      sorted = false;
      break;
    }
  }
  if (sorted) {
    if (try_moe_sorted_parallel(first, second, x, expert_ids, bias, output,
                                rows, activation, swiglu, workspace)) {
      return Status::kOk;
    }
    long long begin = 0;
    while (begin < rows) {
      const int expert = expert_ids[begin];
      long long end = begin + 1;
      while (end < rows && expert_ids[end] == expert) ++end;
      const long long count = end - begin;
      FloatStorageInput input{
          float_storage_offset(x.data, x.type, begin * input_columns), x.type,
          count * input_columns};
      FloatStorageOutput destination{
          float_storage_offset(output.data, output.type,
                               begin * output_columns),
          output.type, count * output_columns};
      status =
          swiglu ? qgemm_prepacked_swiglu_storage(first[expert], second[expert],
                                                  input, destination, count,
                                                  workspace)
                 : qgemm_prepacked_epilogue_storage(
                       first[expert], input,
                       bias == nullptr ? nullptr
                                       : bias + static_cast<long long>(expert) *
                                                    output_columns,
                       destination, count, activation, workspace);
      if (status != Status::kOk) return status;
      begin = end;
    }
    return Status::kOk;
  }

  detail::WorkspaceFrame frame(workspace);
  long long* offsets =
      frame.allocate<long long>(static_cast<std::size_t>(experts + 1));
  long long* cursor =
      frame.allocate<long long>(static_cast<std::size_t>(experts));
  long long* order = frame.allocate<long long>(static_cast<std::size_t>(rows));
  if (offsets == nullptr || cursor == nullptr || order == nullptr)
    return Status::kOutOfMemory;
  std::fill_n(offsets, experts + 1, 0);
  long long active_rows = 0;
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] >= 0) {
      ++offsets[expert_ids[row] + 1];
      ++active_rows;
    }
  }
  for (long long expert = 0; expert < experts; ++expert)
    offsets[expert + 1] += offsets[expert];
  std::copy_n(offsets, experts, cursor);
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] >= 0) order[cursor[expert_ids[row]]++] = row;
  }
  if (active_rows == 0) {
    std::memset(output.data, 0,
                static_cast<std::size_t>(rows * output_columns) *
                    float_storage_bytes(output.type));
    return Status::kOk;
  }
  const std::size_t input_elements =
      static_cast<std::size_t>(active_rows * input_columns);
  const std::size_t output_elements =
      static_cast<std::size_t>(active_rows * output_columns);
  void* gathered_input =
      frame.allocate_bytes(input_elements * float_storage_bytes(x.type));
  void* gathered_output =
      frame.allocate_bytes(output_elements * float_storage_bytes(output.type));
  if (gathered_input == nullptr || gathered_output == nullptr)
    return Status::kOutOfMemory;
  const std::size_t input_row_bytes =
      static_cast<std::size_t>(input_columns) * float_storage_bytes(x.type);
  const std::size_t output_row_bytes =
      static_cast<std::size_t>(output_columns) *
      float_storage_bytes(output.type);
  for (long long item = 0; item < active_rows; ++item) {
    std::memcpy(
        float_storage_offset(gathered_input, x.type, item * input_columns),
        float_storage_offset(x.data, x.type, order[item] * input_columns),
        input_row_bytes);
  }
  for (long long expert = 0; expert < experts; ++expert) {
    const long long begin = offsets[expert];
    const long long count = offsets[expert + 1] - begin;
    if (count == 0) continue;
    FloatStorageInput input{
        float_storage_offset(gathered_input, x.type, begin * input_columns),
        x.type, count * input_columns};
    FloatStorageOutput destination{
        float_storage_offset(gathered_output, output.type,
                             begin * output_columns),
        output.type, count * output_columns};
    status =
        swiglu ? qgemm_prepacked_swiglu_storage(first[expert], second[expert],
                                                input, destination, count,
                                                workspace)
               : qgemm_prepacked_epilogue_storage(
                     first[expert], input,
                     bias == nullptr ? nullptr : bias + expert * output_columns,
                     destination, count, activation, workspace);
    if (status != Status::kOk) return status;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < 0) {
      std::memset(
          float_storage_offset(output.data, output.type, row * output_columns),
          0, output_row_bytes);
    }
  }
  for (long long item = 0; item < active_rows; ++item) {
    std::memcpy(float_storage_offset(output.data, output.type,
                                     order[item] * output_columns),
                float_storage_offset(gathered_output, output.type,
                                     item * output_columns),
                output_row_bytes);
  }
  return Status::kOk;
}

bool moe_layout_matches(const CpuPackedWeights* weights, long long experts,
                        bool (*predicate)(CanonicalQuantLayout)) {
  for (long long expert = 0; expert < experts; ++expert) {
    if (!weights[expert].ready() ||
        !predicate(weights[expert].info().quant_metadata.layout)) {
      return false;
    }
  }
  return true;
}

}  // namespace

Status moe_grouped_prepacked_storage(const CpuPackedWeights* expert_weights,
                                     long long experts, FloatStorageInput x,
                                     const int* expert_ids, const float* bias,
                                     FloatStorageOutput output, long long rows,
                                     LinearActivation activation,
                                     Workspace* workspace) {
  return moe_grouped_prepacked_impl(expert_weights, nullptr, experts, x,
                                    expert_ids, bias, output, rows, activation,
                                    false, workspace);
}

Status moe_grouped_prepacked_swiglu_storage(
    const CpuPackedWeights* gate_weights, const CpuPackedWeights* up_weights,
    long long experts, FloatStorageInput x, const int* expert_ids,
    FloatStorageOutput output, long long rows, Workspace* workspace) {
  return moe_grouped_prepacked_impl(gate_weights, up_weights, experts, x,
                                    expert_ids, nullptr, output, rows,
                                    LinearActivation::kNone, true, workspace);
}

Status moe_grouped_prepacked_quantized(const CpuPackedWeights* expert_weights,
                                       long long experts,
                                       const CanonicalQuantTensor& activation,
                                       const int* expert_ids, float* output,
                                       Workspace* workspace) {
  if (expert_weights == nullptr || experts <= 0 || expert_ids == nullptr ||
      output == nullptr) {
    return Status::kInvalidArgument;
  }
  Status status = validate_canonical_quant_tensor(activation);
  if (status != Status::kOk) return status;
  if (!projection_metadata_supported(activation.metadata))
    return Status::kUnsupportedFormat;
  const long long rows = activation.metadata.logical_rows;
  const long long columns = activation.metadata.logical_columns;
  if (rows <= 0) return Status::kInvalidShape;
  CpuPackedWeightsInfo first_info;
  status = validate_prepared_projection(expert_weights[0], &first_info);
  if (status != Status::kOk) return status;
  if (first_info.columns != columns ||
      !detail::valid_product({rows, first_info.rows})) {
    return Status::kInvalidShape;
  }
  for (long long expert = 0; expert < experts; ++expert) {
    CpuPackedWeightsInfo info;
    status = validate_prepared_projection(expert_weights[expert], &info);
    if (status != Status::kOk) return status;
    if (info.columns != columns || info.rows != first_info.rows)
      return Status::kInvalidShape;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < -1 || expert_ids[row] >= experts)
      return Status::kInvalidArgument;
  }
#if defined(__x86_64__) || defined(_M_X64)
  if (rows > 1) {
    detail::WorkspaceFrame frame(workspace);
    float* decoded =
        frame.allocate<float>(static_cast<std::size_t>(rows * columns));
    if (decoded == nullptr) return Status::kOutOfMemory;
    status = dequantize_canonical(activation, decoded, rows * columns);
    if (status != Status::kOk) return status;
    return moe_grouped_prepacked_storage(
        expert_weights, experts,
        {decoded, FloatStorageType::kF32, rows * columns}, expert_ids, nullptr,
        {output, FloatStorageType::kF32, rows * first_info.rows}, rows,
        LinearActivation::kNone, workspace);
  }
#endif
  if (try_moe_sorted_direct_dual(expert_weights, experts, activation,
                                 expert_ids, output, workspace)) {
    return Status::kOk;
  }

  detail::WorkspaceFrame frame(workspace);
  long long* offsets =
      frame.allocate<long long>(static_cast<std::size_t>(experts + 1));
  long long* cursor =
      frame.allocate<long long>(static_cast<std::size_t>(experts));
  long long* order = frame.allocate<long long>(static_cast<std::size_t>(rows));
  auto* contexts = frame.allocate<MoeQuantizedProjectionContext>(
      static_cast<std::size_t>(experts));
  auto* kernels =
      frame.allocate<threading::RangeFn>(static_cast<std::size_t>(experts));
  long long* task_offsets =
      frame.allocate<long long>(static_cast<std::size_t>(experts + 1));
  if (offsets == nullptr || cursor == nullptr || order == nullptr ||
      contexts == nullptr || kernels == nullptr || task_offsets == nullptr) {
    return Status::kOutOfMemory;
  }
  std::fill_n(offsets, experts + 1, 0);
  long long active_rows = 0;
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] >= 0) {
      ++offsets[expert_ids[row] + 1];
      ++active_rows;
    }
  }
  for (long long expert = 0; expert < experts; ++expert)
    offsets[expert + 1] += offsets[expert];
  std::copy_n(offsets, experts, cursor);
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] >= 0) order[cursor[expert_ids[row]]++] = row;
  }
  if (active_rows == 0) {
    std::fill_n(output, rows * first_info.rows, 0.0f);
    return Status::kOk;
  }

  long long groups = 0;
  task_offsets[0] = 0;
  for (long long expert = 0; expert < experts; ++expert) {
    const long long count = offsets[expert + 1] - offsets[expert];
    if (count == 0) continue;
    const CpuPackedWeightsInfo info = expert_weights[expert].info();
    kernels[groups] =
        moe_quantized_projection_kernel(info.quant_metadata.layout);
    if (kernels[groups] == nullptr) return Status::kUnsupportedFormat;
    contexts[groups] = {
        info,
        static_cast<const std::uint8_t*>(expert_weights[expert].panel_data()),
        &activation,
        order + offsets[expert],
        output,
        count};
    task_offsets[groups + 1] =
        task_offsets[groups] + (info.rows + info.row_tile - 1) / info.row_tile;
    ++groups;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < 0)
      std::fill_n(output + row * first_info.rows, first_info.rows, 0.0f);
  }
  MoeParallelQuantizedContext context{contexts, kernels, task_offsets, groups};
  threading::parallel_ranges_impl(task_offsets[groups], 1,
                                  moe_parallel_quantized_tasks, &context);
  return Status::kOk;
}

Status moe_grouped_prepacked_swiglu_quantized(
    const CpuPackedWeights* gate_weights, const CpuPackedWeights* up_weights,
    long long experts, FloatStorageInput x, const int* expert_ids,
    CanonicalQuantLayout output_layout, long long output_group_size,
    CanonicalQuantTensor* output, long long rows, bool scale_2d,
    Workspace* workspace) {
  if (gate_weights == nullptr || up_weights == nullptr || experts <= 0 ||
      expert_ids == nullptr || output == nullptr) {
    return Status::kInvalidArgument;
  }
  CpuPackedWeightsInfo first_info;
  Status status = validate_prepared_projection(gate_weights[0], &first_info);
  if (status != Status::kOk) return status;
  if (!detail::valid_product({rows, first_info.columns}) ||
      !detail::valid_product({rows, first_info.rows}) ||
      x.count != rows * first_info.columns) {
    return Status::kInvalidShape;
  }
  float sentinel = 0.0f;
  status = validate_moe_prepared_arrays(
      gate_weights, up_weights, experts, x, expert_ids,
      {&sentinel, FloatStorageType::kF32, rows * first_info.rows}, rows, true,
      &first_info);
  if (status != Status::kOk) return status;

  long long quant_group = 0;
  bool needs_global_maximum = false;
  try {
    status = configure_swiglu_quant_output(
        output, output_layout, rows, first_info.rows, output_group_size,
        scale_2d, &quant_group, &needs_global_maximum);
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
  if (status != Status::kOk) return status;

  bool handled = false;
  status = try_moe_sorted_swiglu_quantized_panels(
      gate_weights, up_weights, experts, x, expert_ids, output, rows,
      quant_group, scale_2d, needs_global_maximum, workspace, &handled);
  if (status != Status::kOk) return status;
  if (handled) return validate_canonical_quant_tensor(*output);

  const long long groups_per_row = first_info.rows / quant_group;
  const long long row_domain =
      output_layout == CanonicalQuantLayout::kNVFP4E2M1E4M3 && scale_2d ? 16
                                                                        : 1;
  const long long domains = (rows + row_domain - 1) / row_domain;
  if (!detail::valid_product(
          {domains, groups_per_row, row_domain, quant_group})) {
    return Status::kInvalidShape;
  }
  const std::size_t workers = static_cast<std::size_t>(num_threads());
  const bool full_rows = workers == 1 && row_domain == 1;
  const long long tasks = full_rows ? rows : domains * groups_per_row;
  const long long scratch_elements =
      full_rows ? first_info.rows : row_domain * quant_group;
  if (static_cast<unsigned long long>(scratch_elements) >
      std::numeric_limits<std::size_t>::max() / (2 * sizeof(float)) / workers) {
    return Status::kInvalidShape;
  }

  detail::WorkspaceFrame frame(workspace);
  auto* contexts = frame.allocate<CanonicalSwiGLUQuantContext>(
      static_cast<std::size_t>(experts));
  auto* project =
      frame.allocate<MoeSwiGLUProjectFn>(static_cast<std::size_t>(experts));
  float* scratch = frame.allocate<float>(
      workers * static_cast<std::size_t>(scratch_elements) * 2);
  float* worker_maximum = frame.allocate<float>(workers);
  if (contexts == nullptr || project == nullptr || scratch == nullptr ||
      worker_maximum == nullptr) {
    return Status::kOutOfMemory;
  }
  std::fill_n(worker_maximum, workers, 0.0f);
  std::atomic<bool> invalid{false};
  for (long long expert = 0; expert < experts; ++expert) {
    const CpuPackedWeightsInfo gate_info = gate_weights[expert].info();
    const CpuPackedWeightsInfo up_info = up_weights[expert].info();
    switch (x.type) {
      case FloatStorageType::kF32:
        project[expert] = moe_swiglu_project_kernel<FloatStorageType::kF32>(
            gate_info.quant_metadata.layout);
        break;
      case FloatStorageType::kF16:
        project[expert] = moe_swiglu_project_kernel<FloatStorageType::kF16>(
            gate_info.quant_metadata.layout);
        break;
      case FloatStorageType::kBF16:
        project[expert] = moe_swiglu_project_kernel<FloatStorageType::kBF16>(
            gate_info.quant_metadata.layout);
        break;
    }
    if (project[expert] == nullptr) return Status::kUnsupportedFormat;
    contexts[expert] = {
        gate_info,
        up_info,
        static_cast<const std::uint8_t*>(gate_weights[expert].panel_data()),
        static_cast<const std::uint8_t*>(up_weights[expert].panel_data()),
        x,
        output,
        scratch,
        worker_maximum,
        &invalid,
        rows,
        quant_group,
        groups_per_row,
        row_domain,
        scratch_elements,
        0.0f,
        scale_2d,
        needs_global_maximum};
  }
  MoeSwiGLUQuantizedContext context{contexts,
                                    project,
                                    expert_ids,
                                    scratch,
                                    worker_maximum,
                                    &invalid,
                                    rows,
                                    quant_group,
                                    groups_per_row,
                                    row_domain,
                                    scratch_elements,
                                    scale_2d,
                                    needs_global_maximum};

  threading::RangeFn kernel =
      full_rows ? moe_swiglu_quantized_full_rows : moe_swiglu_quantized_groups;
  if (needs_global_maximum) {
    threading::parallel_ranges_impl(tasks, 1, kernel, &context);
    if (invalid.load(std::memory_order_relaxed))
      return Status::kInvalidArgument;
    float maximum = 0.0f;
    for (std::size_t worker = 0; worker < workers; ++worker)
      maximum = std::max(maximum, worker_maximum[worker]);
    if (output_layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
      contexts[0].global_scale = maximum / (6.0f * 448.0f);
      output->metadata.global_scale = contexts[0].global_scale;
    } else {
      const bool fp4 = output_layout == CanonicalQuantLayout::kFP4E2M1;
      const float maximum_code =
          fp4 ? 6.0f
              : (output_layout == CanonicalQuantLayout::kFP8E4M3FN ? 448.0f
                                                                   : 57344.0f);
      float scale = maximum == 0.0f ? 0.0f : maximum / maximum_code;
      if (fp4) scale = f16_to_float(float_to_f16(scale));
      output->scales[0] = scale;
    }
    context.measure_only = false;
    contexts[0].measure_only = false;
  }
  threading::parallel_ranges_impl(tasks, 1, kernel, &context);
  if (invalid.load(std::memory_order_relaxed)) return Status::kInvalidArgument;
  return validate_canonical_quant_tensor(*output);
}

Status moe_grouped_fp8_prepacked_storage(const CpuPackedWeights* expert_weights,
                                         long long experts, FloatStorageInput x,
                                         const int* expert_ids,
                                         FloatStorageOutput output,
                                         long long rows, Workspace* workspace) {
  if (expert_weights == nullptr || experts <= 0)
    return Status::kInvalidArgument;
  if (!moe_layout_matches(
          expert_weights, experts, [](CanonicalQuantLayout layout) {
            return layout == CanonicalQuantLayout::kFP8E4M3FN ||
                   layout == CanonicalQuantLayout::kFP8E5M2 ||
                   layout == CanonicalQuantLayout::kMXFP8E4M3E8M0;
          }))
    return Status::kUnsupportedFormat;
  return moe_grouped_prepacked_storage(expert_weights, experts, x, expert_ids,
                                       nullptr, output, rows,
                                       LinearActivation::kNone, workspace);
}

Status moe_grouped_wna16_prepacked_storage(
    const CpuPackedWeights* expert_weights, long long experts,
    FloatStorageInput x, const int* expert_ids, FloatStorageOutput output,
    long long rows, Workspace* workspace) {
  if (expert_weights == nullptr || experts <= 0)
    return Status::kInvalidArgument;
  if (!moe_layout_matches(
          expert_weights, experts, [](CanonicalQuantLayout layout) {
            return layout == CanonicalQuantLayout::kInt4Symmetric ||
                   layout == CanonicalQuantLayout::kUInt4Affine ||
                   layout == CanonicalQuantLayout::kFP4E2M1 ||
                   layout == CanonicalQuantLayout::kMXFP4E2M1E8M0;
          }))
    return Status::kUnsupportedFormat;
  return moe_grouped_prepacked_storage(expert_weights, experts, x, expert_ids,
                                       nullptr, output, rows,
                                       LinearActivation::kNone, workspace);
}

Status moe_grouped_nvfp4_prepacked_storage(
    const CpuPackedWeights* expert_weights, long long experts,
    FloatStorageInput x, const int* expert_ids, FloatStorageOutput output,
    long long rows, Workspace* workspace) {
  if (expert_weights == nullptr || experts <= 0)
    return Status::kInvalidArgument;
  if (!moe_layout_matches(
          expert_weights, experts, [](CanonicalQuantLayout layout) {
            return layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;
          }))
    return Status::kUnsupportedFormat;
  return moe_grouped_prepacked_storage(expert_weights, experts, x, expert_ids,
                                       nullptr, output, rows,
                                       LinearActivation::kNone, workspace);
}

}  // namespace quixicore_cpu
