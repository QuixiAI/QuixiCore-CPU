#if (defined(__x86_64__) || defined(_M_X64)) && defined(QUIXICORE_CPU_ISA_AVX2)

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kernels/quantization/qgemm_canonical_isa.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/quantization.h"

namespace quixicore_cpu::quant {
namespace {

constexpr long long kMaximumRowTile = 16;
constexpr long long kAccumulatorCount = 4;
constexpr std::size_t kMissingOffset = static_cast<std::size_t>(-1);

template <class T>
const T* prepared_table(const std::uint8_t* panel, std::size_t offset) {
  return offset == kMissingOffset ? nullptr
                                  : reinterpret_cast<const T*>(panel + offset);
}

std::uint16_t little_u16(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(
      bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8));
}

std::uint8_t nibble(const std::uint8_t* bytes, long long element) {
  return static_cast<std::uint8_t>(
      (bytes[element / 2] >> (4 * static_cast<unsigned>(element & 1))) & 15);
}

float e8m0_decode(std::uint8_t code) {
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

template <Float8Format Format>
float fp8_decode(std::uint8_t code) {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = float8_decode(static_cast<std::uint8_t>(index), Format);
    }
    return values;
  }();
  return table[code];
}

std::size_t side_index(const QuantTensorMetadata& metadata, long long row,
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
      return static_cast<std::size_t>(row * groups +
                                      column / metadata.group_size);
    }
    case QuantScaleMode::kNone:
    case QuantScaleMode::kMicroscaleK32:
    case QuantScaleMode::kNVFP4K16:
    case QuantScaleMode::kTurboQuantK32:
      return 0;
  }
  return 0;
}

float tensor_scale(const CanonicalQuantTensor& tensor, long long row,
                   long long column) {
  return tensor.scales.empty()
             ? 1.0f
             : tensor.scales[side_index(tensor.metadata, row, column)];
}

float tensor_zero(const CanonicalQuantTensor& tensor, long long row,
                  long long column) {
  return tensor.zero_points.empty()
             ? 0.0f
             : tensor.zero_points[side_index(tensor.metadata, row, column)];
}

const std::uint8_t* panel_block(const CpuPackedWeightsInfo& info,
                                const std::uint8_t* panel, long long row_panel,
                                long long block, long long lane) {
  const std::size_t index = static_cast<std::size_t>(
      (row_panel * info.blocks_per_row + block) * info.row_tile + lane);
  return panel + index * info.block_bytes;
}

int horizontal_sum(__m256i value) {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

int signed_dot32(const std::int8_t* first, const std::int8_t* second) {
  const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(first));
  const __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(second));
  const __m256i products_low =
      _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)));
  const __m256i products_high =
      _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)),
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)));
  return horizontal_sum(_mm256_add_epi32(products_low, products_high));
}

int sum_i8_32(const std::int8_t* values) {
  const __m256i packed =
      _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
  const __m128i low = _mm256_castsi256_si128(packed);
  const __m128i high = _mm256_extracti128_si256(packed, 1);
  const __m256i wide_low = _mm256_cvtepi8_epi16(low);
  const __m256i wide_high = _mm256_cvtepi8_epi16(high);
  return horizontal_sum(
      _mm256_add_epi32(_mm256_madd_epi16(wide_low, _mm256_set1_epi16(1)),
                       _mm256_madd_epi16(wide_high, _mm256_set1_epi16(1))));
}

float dot_f32_32(const float* first, const float* second) {
  __m256 sum0 = _mm256_mul_ps(_mm256_load_ps(first), _mm256_load_ps(second));
  __m256 sum1 =
      _mm256_mul_ps(_mm256_load_ps(first + 8), _mm256_load_ps(second + 8));
  sum0 = _mm256_fmadd_ps(_mm256_load_ps(first + 16),
                         _mm256_load_ps(second + 16), sum0);
  sum1 = _mm256_fmadd_ps(_mm256_load_ps(first + 24),
                         _mm256_load_ps(second + 24), sum1);
  const __m256 sum = _mm256_add_ps(sum0, sum1);
  __m128 reduced =
      _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  return _mm_cvtss_f32(reduced);
}

void unpack_signed_nibbles(const std::uint8_t* source, std::int8_t* values,
                           long long count) {
  for (long long item = 0; item < count; ++item) {
    const int code = nibble(source, item);
    values[item] = static_cast<std::int8_t>(code >= 8 ? code - 16 : code);
  }
  std::fill(values + count, values + 32, std::int8_t{0});
}

void unpack_unsigned_nibbles(const std::uint8_t* source, std::int8_t* values,
                             long long count) {
  for (long long item = 0; item < count; ++item) {
    values[item] = static_cast<std::int8_t>(nibble(source, item));
  }
  std::fill(values + count, values + 32, std::int8_t{0});
}

void unpack_fp4_nibbles(const std::uint8_t* source, std::int8_t* values,
                        long long count) {
  static constexpr std::int8_t table[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                            0, -1, -2, -3, -4, -6, -8, -12};
  for (long long item = 0; item < count; ++item) {
    values[item] = table[nibble(source, item)];
  }
  std::fill(values + count, values + 32, std::int8_t{0});
}

void unpack_ternary(const std::uint8_t* source, std::int8_t* values) {
  for (long long item = 0; item < 32; ++item) {
    values[item] = static_cast<std::int8_t>(
        static_cast<int>((source[item / 4] >> (2 * (item & 3))) & 3) - 1);
  }
}

template <CanonicalQuantLayout WeightLayout,
          CanonicalQuantLayout ActivationLayout>
bool dual_panel_group(const CpuPackedWeightsInfo& info,
                      const std::uint8_t* panel,
                      const CanonicalQuantTensor& activation,
                      long long activation_row, long long first_panel,
                      long long panel_count, long long block,
                      float* accumulators) {
  if (info.block_size > 32) return false;
  const auto* weight_scales =
      prepared_table<float>(panel, info.scale_table_offset);
  const auto* weight_zeros =
      prepared_table<float>(panel, info.zero_point_table_offset);
  const auto* weight_scale_codes =
      prepared_table<std::uint8_t>(panel, info.scale_code_table_offset);
  const long long first_column = block * info.block_size;
  const long long activation_element =
      activation_row * activation.metadata.logical_columns + first_column;
  const float activation_scale =
      tensor_scale(activation, activation_row, first_column);
  alignas(32) std::int8_t activation_codes[32] = {};
  alignas(32) float activation_values[32] = {};
  const std::uint8_t* activation_block = nullptr;

  if constexpr (ActivationLayout == CanonicalQuantLayout::kInt4Symmetric) {
    unpack_signed_nibbles(activation.data.data() + activation_element / 2,
                          activation_codes, info.block_size);
  } else if constexpr (ActivationLayout ==
                           CanonicalQuantLayout::kInt8Symmetric ||
                       ActivationLayout == CanonicalQuantLayout::kInt8Affine) {
    std::copy_n(reinterpret_cast<const std::int8_t*>(activation.data.data()) +
                    activation_element,
                info.block_size, activation_codes);
  } else if constexpr (ActivationLayout == CanonicalQuantLayout::kFP8E4M3FN) {
    for (long long item = 0; item < info.block_size; ++item) {
      activation_values[item] = fp8_decode<Float8Format::kE4M3FN>(
          activation.data[static_cast<std::size_t>(activation_element + item)]);
    }
  } else if constexpr (ActivationLayout ==
                       CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            33);
    for (long long item = 0; item < 32; ++item) {
      activation_values[item] =
          fp8_decode<Float8Format::kE4M3FN>(activation_block[1 + item]);
    }
  } else if constexpr (ActivationLayout ==
                       CanonicalQuantLayout::kMXFP4E2M1E8M0) {
    activation_block =
        activation.data.data() +
        static_cast<std::size_t>(
            (activation_row * activation.metadata.logical_columns / 32 +
             block) *
            17);
    unpack_fp4_nibbles(activation_block + 1, activation_codes, 32);
  } else if constexpr (ActivationLayout ==
                           CanonicalQuantLayout::kNVFP4E2M1E4M3 ||
                       ActivationLayout == CanonicalQuantLayout::kFP4E2M1) {
    unpack_fp4_nibbles(activation.data.data() + activation_element / 2,
                       activation_codes, info.block_size);
  } else {
    return false;
  }

  const int activation_sum =
      ActivationLayout == CanonicalQuantLayout::kInt8Symmetric
          ? sum_i8_32(activation_codes)
          : 0;
  const float activation_zero =
      ActivationLayout == CanonicalQuantLayout::kInt8Affine
          ? tensor_zero(activation, activation_row, first_column)
          : 0.0f;
  for (long long panel_lane = 0; panel_lane < panel_count; ++panel_lane) {
    const long long row_panel = first_panel + panel_lane;
    const long long first_output = row_panel * info.row_tile;
    const long long lanes = std::min(info.row_tile, info.rows - first_output);
    float* partial =
        accumulators +
        (panel_lane * kAccumulatorCount + (block & 3)) * kMaximumRowTile;
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long weight_row = first_output + lane;
      const auto* weight = panel_block(info, panel, row_panel, block, lane);
      const std::size_t side =
          side_index(info.quant_metadata, weight_row, first_column);
      const float weight_scale =
          weight_scales == nullptr ? 1.0f : weight_scales[side];
      alignas(32) std::int8_t weight_codes[32] = {};
      float value = 0.0f;
      if constexpr (WeightLayout == CanonicalQuantLayout::kInt4Symmetric &&
                    (ActivationLayout == CanonicalQuantLayout::kInt4Symmetric ||
                     ActivationLayout ==
                         CanonicalQuantLayout::kInt8Symmetric)) {
        unpack_signed_nibbles(weight, weight_codes, info.block_size);
        value =
            weight_scale * activation_scale *
            static_cast<float>(signed_dot32(weight_codes, activation_codes));
      } else if constexpr (WeightLayout == CanonicalQuantLayout::kUInt4Affine &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kInt8Symmetric) {
        unpack_unsigned_nibbles(weight, weight_codes, info.block_size);
        value =
            weight_scale * activation_scale *
            (static_cast<float>(signed_dot32(weight_codes, activation_codes)) -
             weight_zeros[side] * static_cast<float>(activation_sum));
      } else if constexpr (WeightLayout ==
                               CanonicalQuantLayout::kInt8Symmetric &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kInt8Affine) {
        std::copy_n(reinterpret_cast<const std::int8_t*>(weight),
                    info.block_size, weight_codes);
        value =
            weight_scale * activation_scale *
            (static_cast<float>(signed_dot32(weight_codes, activation_codes)) -
             activation_zero * static_cast<float>(sum_i8_32(weight_codes)));
      } else if constexpr ((WeightLayout == CanonicalQuantLayout::kFP8E4M3FN ||
                            WeightLayout == CanonicalQuantLayout::kFP8E5M2) &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kFP8E4M3FN) {
        alignas(32) float weight_values[32] = {};
        for (long long item = 0; item < info.block_size; ++item) {
          if constexpr (WeightLayout == CanonicalQuantLayout::kFP8E4M3FN) {
            weight_values[item] =
                fp8_decode<Float8Format::kE4M3FN>(weight[item]);
          } else {
            weight_values[item] = fp8_decode<Float8Format::kE5M2>(weight[item]);
          }
        }
        value = weight_scale * activation_scale *
                dot_f32_32(weight_values, activation_values);
      } else if constexpr (WeightLayout ==
                               CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kMXFP8E4M3E8M0) {
        alignas(32) float weight_values[32];
        for (long long item = 0; item < 32; ++item) {
          weight_values[item] =
              fp8_decode<Float8Format::kE4M3FN>(weight[1 + item]);
        }
        value = e8m0_decode(weight[0]) * e8m0_decode(activation_block[0]) *
                dot_f32_32(weight_values, activation_values);
      } else if constexpr (WeightLayout ==
                               CanonicalQuantLayout::kMXFP4E2M1E8M0 &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kMXFP4E2M1E8M0) {
        unpack_fp4_nibbles(weight + 1, weight_codes, 32);
        value =
            0.25f * e8m0_decode(weight[0]) * e8m0_decode(activation_block[0]) *
            static_cast<float>(signed_dot32(weight_codes, activation_codes));
      } else if constexpr (WeightLayout ==
                               CanonicalQuantLayout::kNVFP4E2M1E4M3 &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kNVFP4E2M1E4M3) {
        unpack_fp4_nibbles(weight, weight_codes, 16);
        const float weight_block_scale =
            info.quant_metadata.global_scale *
            fp8_decode<Float8Format::kE4M3FN>(
                weight_scale_codes[weight_row * info.blocks_per_row + block]);
        const float activation_block_scale =
            activation.metadata.global_scale *
            fp8_decode<Float8Format::kE4M3FN>(
                activation.scale_codes[static_cast<std::size_t>(
                    activation_row * info.blocks_per_row + block)]);
        value =
            0.25f * weight_block_scale * activation_block_scale *
            static_cast<float>(signed_dot32(weight_codes, activation_codes));
      } else if constexpr (WeightLayout ==
                               CanonicalQuantLayout::kBitNetTernary &&
                           ActivationLayout ==
                               CanonicalQuantLayout::kInt8Symmetric) {
        unpack_ternary(weight + 2, weight_codes);
        value =
            f16_to_float(little_u16(weight)) * activation_scale *
            static_cast<float>(signed_dot32(weight_codes, activation_codes));
      } else if constexpr (WeightLayout ==
                               CanonicalQuantLayout::kBitNetTernary &&
                           ActivationLayout == CanonicalQuantLayout::kFP4E2M1) {
        unpack_ternary(weight + 2, weight_codes);
        value =
            0.5f * f16_to_float(little_u16(weight)) * activation_scale *
            static_cast<float>(signed_dot32(weight_codes, activation_codes));
      } else {
        return false;
      }
      partial[lane] += value;
    }
  }
  return true;
}

}  // namespace

void canonical_accumulate_tile_avx2(float* accumulators,
                                    const float* decoded_weights,
                                    const float* activations,
                                    long long activation_row_stride,
                                    long long first_column, long long items,
                                    long long rows, long long lanes) {
  constexpr long long kMTile = 32;
  if (lanes != 8) return;
  for (long long item = 0; item < items; ++item) {
    const long long accumulator = (first_column + item) & 3;
    const __m256 weights =
        _mm256_load_ps(decoded_weights + item * kMaximumRowTile);
    for (long long row = 0; row < rows; ++row) {
      float* partial =
          accumulators + (accumulator * kMTile + row) * kMaximumRowTile;
      const __m256 activation =
          _mm256_set1_ps(activations[row * activation_row_stride + item]);
      _mm256_store_ps(partial, _mm256_fmadd_ps(weights, activation,
                                               _mm256_load_ps(partial)));
    }
  }
}

bool canonical_dual_panel_group_avx2(
    CanonicalQuantLayout weight_layout, CanonicalQuantLayout activation_layout,
    const CpuPackedWeightsInfo& info, const std::uint8_t* panel,
    const CanonicalQuantTensor& activation, long long activation_row,
    long long first_panel, long long panel_count, long long block,
    float* accumulators) {
#define QUIXICORE_DUAL_AVX2_CASE(weight_name, activation_name)             \
  if (weight_layout == CanonicalQuantLayout::weight_name &&                \
      activation_layout == CanonicalQuantLayout::activation_name) {        \
    return dual_panel_group<CanonicalQuantLayout::weight_name,             \
                            CanonicalQuantLayout::activation_name>(        \
        info, panel, activation, activation_row, first_panel, panel_count, \
        block, accumulators);                                              \
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
  return false;
}

}  // namespace quixicore_cpu::quant

#endif
