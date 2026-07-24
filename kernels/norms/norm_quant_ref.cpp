#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

#include "quixicore_cpu/quantization.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/validation.h"
#include "quixicore_cpu/quant_import.h"
#include "quixicore_cpu/threading.h"
#include "src/memory/workspace_internal.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

template <bool LayerNorm>
struct RowNorm {
  double mean = 0.0;
  float inverse = 0.0f;
};

template <bool LayerNorm>
RowNorm<LayerNorm> add_and_measure(const float* x, const float* residual,
                                   float* residual_out, long long hidden,
                                   float eps) {
  RowNorm<LayerNorm> norm;
  if constexpr (LayerNorm) {
    double m2 = 0.0;
    for (long long item = 0; item < hidden; ++item) {
      const float value = x[item] + residual[item];
      residual_out[item] = value;
      const double delta = static_cast<double>(value) - norm.mean;
      norm.mean += delta / static_cast<double>(item + 1);
      m2 += delta * (static_cast<double>(value) - norm.mean);
    }
    norm.inverse = static_cast<float>(
        1.0 / std::sqrt(m2 / static_cast<double>(hidden) + eps));
  } else {
    double sumsq = 0.0;
#if defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t squares0 = vdupq_n_f32(0.0f);
    float32x4_t squares1 = vdupq_n_f32(0.0f);
    float32x4_t squares2 = vdupq_n_f32(0.0f);
    float32x4_t squares3 = vdupq_n_f32(0.0f);
    long long item = 0;
    for (; item + 15 < hidden; item += 16) {
      const float32x4_t value0 =
          vaddq_f32(vld1q_f32(x + item), vld1q_f32(residual + item));
      const float32x4_t value1 =
          vaddq_f32(vld1q_f32(x + item + 4), vld1q_f32(residual + item + 4));
      const float32x4_t value2 =
          vaddq_f32(vld1q_f32(x + item + 8), vld1q_f32(residual + item + 8));
      const float32x4_t value3 =
          vaddq_f32(vld1q_f32(x + item + 12), vld1q_f32(residual + item + 12));
      vst1q_f32(residual_out + item, value0);
      vst1q_f32(residual_out + item + 4, value1);
      vst1q_f32(residual_out + item + 8, value2);
      vst1q_f32(residual_out + item + 12, value3);
      squares0 = vfmaq_f32(squares0, value0, value0);
      squares1 = vfmaq_f32(squares1, value1, value1);
      squares2 = vfmaq_f32(squares2, value2, value2);
      squares3 = vfmaq_f32(squares3, value3, value3);
    }
    sumsq = static_cast<double>(vaddvq_f32(vaddq_f32(
        vaddq_f32(squares0, squares1), vaddq_f32(squares2, squares3))));
    for (; item < hidden; ++item) {
      const float value = x[item] + residual[item];
      residual_out[item] = value;
      sumsq += static_cast<double>(value) * value;
    }
#else
    for (long long item = 0; item < hidden; ++item) {
      const float value = x[item] + residual[item];
      residual_out[item] = value;
      sumsq += static_cast<double>(value) * value;
    }
#endif
    norm.inverse = static_cast<float>(
        1.0 / std::sqrt(sumsq / static_cast<double>(hidden) + eps));
  }
  return norm;
}

template <bool LayerNorm>
inline float normalized_at(const float* residual, const float* weight,
                           const float* bias, long long item,
                           const RowNorm<LayerNorm>& norm) {
  if constexpr (LayerNorm) {
    const float normalized = static_cast<float>(
        (static_cast<double>(residual[item]) - norm.mean) * norm.inverse);
    return normalized * weight[item] + (bias != nullptr ? bias[item] : 0.0f);
  }
  return (residual[item] * norm.inverse) * weight[item];
}

bool valid_dynamic_shape(long long rows, long long hidden,
                         long long* group_size, float eps) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return false;
  }
  if (*group_size == 0) *group_size = hidden;
  return *group_size > 0 && hidden % *group_size == 0;
}

template <bool LayerNorm, typename Code>
Status norm_add_quant_dynamic(const float* x, const float* residual,
                              const float* weight, const float* bias,
                              Code* codes, float* residual_out, float* scales,
                              long long rows, long long hidden, float eps,
                              long long group_size, bool power_of_two_scale,
                              Float8Format format) {
  if (!valid_dynamic_shape(rows, hidden, &group_size, eps)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, residual, weight, codes, residual_out, scales)) {
    return Status::kInvalidArgument;
  }

  const long long groups = hidden / group_size;
  const float maximum_code =
      std::is_same_v<Code, std::int8_t>
          ? 127.0f
          : (format == Float8Format::kE4M3FN ? 448.0f : 57344.0f);
  const std::size_t threads = static_cast<std::size_t>(num_threads());
  if (static_cast<std::size_t>(group_size) >
      std::numeric_limits<std::size_t>::max() / threads) {
    return Status::kInvalidShape;
  }
  detail::WorkspaceFrame workspace;
  float* scratch =
      workspace.allocate<float>(threads * static_cast<std::size_t>(group_size));
  if (scratch == nullptr) return Status::kOutOfMemory;
  std::atomic<bool> invalid{false};
  threading::parallel_ranges(
      rows, 4, [&](long long begin, long long end, int worker) {
        float* group_values =
            scratch + static_cast<long long>(worker) * group_size;
        for (long long row = begin; row < end; ++row) {
          const long long row_base = row * hidden;
          float* residual_row = residual_out + row_base;
          const RowNorm<LayerNorm> norm = add_and_measure<LayerNorm>(
              x + row_base, residual + row_base, residual_row, hidden, eps);
          for (long long group = 0; group < groups; ++group) {
            const long long group_base = group * group_size;
            float maximum = 0.0f;
            bool group_invalid = false;
#if defined(__aarch64__) || defined(_M_ARM64)
            if constexpr (!LayerNorm && std::is_same_v<Code, std::int8_t>) {
              float32x4_t maximum0 = vdupq_n_f32(0.0f);
              float32x4_t maximum1 = vdupq_n_f32(0.0f);
              float32x4_t maximum2 = vdupq_n_f32(0.0f);
              float32x4_t maximum3 = vdupq_n_f32(0.0f);
              long long item = 0;
              const float32x4_t inverse_norm = vdupq_n_f32(norm.inverse);
              for (; item + 15 < group_size; item += 16) {
                const long long column = group_base + item;
                const float32x4_t value0 = vmulq_f32(
                    vmulq_f32(vld1q_f32(residual_row + column), inverse_norm),
                    vld1q_f32(weight + column));
                const float32x4_t value1 =
                    vmulq_f32(vmulq_f32(vld1q_f32(residual_row + column + 4),
                                        inverse_norm),
                              vld1q_f32(weight + column + 4));
                const float32x4_t value2 =
                    vmulq_f32(vmulq_f32(vld1q_f32(residual_row + column + 8),
                                        inverse_norm),
                              vld1q_f32(weight + column + 8));
                const float32x4_t value3 =
                    vmulq_f32(vmulq_f32(vld1q_f32(residual_row + column + 12),
                                        inverse_norm),
                              vld1q_f32(weight + column + 12));
                vst1q_f32(group_values + item, value0);
                vst1q_f32(group_values + item + 4, value1);
                vst1q_f32(group_values + item + 8, value2);
                vst1q_f32(group_values + item + 12, value3);
                maximum0 = vmaxq_f32(maximum0, vabsq_f32(value0));
                maximum1 = vmaxq_f32(maximum1, vabsq_f32(value1));
                maximum2 = vmaxq_f32(maximum2, vabsq_f32(value2));
                maximum3 = vmaxq_f32(maximum3, vabsq_f32(value3));
              }
              for (; item + 3 < group_size; item += 4) {
                const long long column = group_base + item;
                const float32x4_t value = vmulq_f32(
                    vmulq_f32(vld1q_f32(residual_row + column), inverse_norm),
                    vld1q_f32(weight + column));
                vst1q_f32(group_values + item, value);
                maximum0 = vmaxq_f32(maximum0, vabsq_f32(value));
              }
              maximum = vmaxvq_f32(vmaxq_f32(vmaxq_f32(maximum0, maximum1),
                                             vmaxq_f32(maximum2, maximum3)));
              group_invalid = !std::isfinite(maximum);
              for (; item < group_size; ++item) {
                const long long column = group_base + item;
                const float value =
                    (residual_row[column] * norm.inverse) * weight[column];
                group_values[item] = value;
                if (!std::isfinite(value)) group_invalid = true;
                maximum = std::max(maximum, std::fabs(value));
              }
            } else
#endif
            {
              for (long long item = 0; item < group_size; ++item) {
                const long long column = group_base + item;
                const float value = normalized_at<LayerNorm>(
                    residual_row, weight, bias, column, norm);
                group_values[item] = value;
                if (!std::isfinite(value)) group_invalid = true;
                maximum = std::max(maximum, std::fabs(value));
              }
            }
            if (group_invalid) {
              invalid.store(true, std::memory_order_relaxed);
              continue;
            }
            float scale = maximum / maximum_code;
            if constexpr (!std::is_same_v<Code, std::int8_t>) {
              if (power_of_two_scale && maximum > 0.0f) {
                scale = std::exp2(std::ceil(
                    std::log2(std::max(maximum, 1e-10f) / maximum_code)));
              }
            }
            scales[row * groups + group] = scale;
            const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
#if defined(__aarch64__) || defined(_M_ARM64)
            if constexpr (std::is_same_v<Code, std::int8_t>) {
              long long item = 0;
              const float32x4_t inverse_scale = vdupq_n_f32(inverse);
              const int32x4_t lower = vdupq_n_s32(-127);
              const int32x4_t upper = vdupq_n_s32(127);
              for (; item + 15 < group_size; item += 16) {
                int32x4_t quantized0 = vcvtnq_s32_f32(
                    vmulq_f32(vld1q_f32(group_values + item), inverse_scale));
                int32x4_t quantized1 = vcvtnq_s32_f32(vmulq_f32(
                    vld1q_f32(group_values + item + 4), inverse_scale));
                int32x4_t quantized2 = vcvtnq_s32_f32(vmulq_f32(
                    vld1q_f32(group_values + item + 8), inverse_scale));
                int32x4_t quantized3 = vcvtnq_s32_f32(vmulq_f32(
                    vld1q_f32(group_values + item + 12), inverse_scale));
                quantized0 = vmaxq_s32(lower, vminq_s32(upper, quantized0));
                quantized1 = vmaxq_s32(lower, vminq_s32(upper, quantized1));
                quantized2 = vmaxq_s32(lower, vminq_s32(upper, quantized2));
                quantized3 = vmaxq_s32(lower, vminq_s32(upper, quantized3));
                const int16x8_t low = vcombine_s16(vqmovn_s32(quantized0),
                                                   vqmovn_s32(quantized1));
                const int16x8_t high = vcombine_s16(vqmovn_s32(quantized2),
                                                    vqmovn_s32(quantized3));
                vst1q_s8(codes + row_base + group_base + item,
                         vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)));
              }
              for (; item + 3 < group_size; item += 4) {
                int32x4_t quantized = vcvtnq_s32_f32(
                    vmulq_f32(vld1q_f32(group_values + item), inverse_scale));
                quantized = vmaxq_s32(lower, vminq_s32(upper, quantized));
                const int16x4_t narrowed = vqmovn_s32(quantized);
                const int8x8_t bytes =
                    vqmovn_s16(vcombine_s16(narrowed, vdup_n_s16(0)));
                const std::uint32_t word =
                    vget_lane_u32(vreinterpret_u32_s8(bytes), 0);
                std::memcpy(codes + row_base + group_base + item, &word,
                            sizeof(word));
              }
              for (; item < group_size; ++item) {
                codes[row_base + group_base + item] = static_cast<std::int8_t>(
                    std::clamp(static_cast<int>(std::nearbyint(
                                   group_values[item] * inverse)),
                               -127, 127));
              }
            } else
#endif
            {
              for (long long item = 0; item < group_size; ++item) {
                const long long column = group_base + item;
                const float value = group_values[item];
                if constexpr (std::is_same_v<Code, std::int8_t>) {
                  codes[row_base + column] =
                      static_cast<std::int8_t>(std::clamp(
                          static_cast<int>(std::nearbyint(value * inverse)),
                          -127, 127));
                } else {
                  codes[row_base + column] =
                      float8_encode(value * inverse, format);
                }
              }
            }
          }
        }
      });
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

template <bool LayerNorm>
Status norm_add_quant_float8_static_impl(const float* x, const float* residual,
                                         const float* weight, const float* bias,
                                         std::uint8_t* codes,
                                         float* residual_out, long long rows,
                                         long long hidden, float scale,
                                         float eps, Float8Format format) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f || !std::isfinite(scale) || scale <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, residual, weight, codes, residual_out)) {
    return Status::kInvalidArgument;
  }

  std::atomic<bool> invalid{false};
  const float inverse_scale = 1.0f / scale;
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const long long base = row * hidden;
      float* residual_row = residual_out + base;
      const RowNorm<LayerNorm> norm = add_and_measure<LayerNorm>(
          x + base, residual + base, residual_row, hidden, eps);
      for (long long item = 0; item < hidden; ++item) {
        const float value =
            normalized_at<LayerNorm>(residual_row, weight, bias, item, norm);
        if (!std::isfinite(value))
          invalid.store(true, std::memory_order_relaxed);
        codes[base + item] = float8_encode(value * inverse_scale, format);
      }
    }
  });
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

bool known_storage(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

float storage_load(FloatStorageInput input, long long index) {
  if (input.type == FloatStorageType::kF32) {
    return static_cast<const float*>(input.data)[index];
  }
  const std::uint16_t bits =
      static_cast<const std::uint16_t*>(input.data)[index];
  return input.type == FloatStorageType::kF16 ? f16_to_float(bits)
                                              : bf16_to_float(bits);
}

float storage_store_and_reload(FloatStorageOutput output, long long index,
                               float value) {
  if (output.type == FloatStorageType::kF32) {
    static_cast<float*>(output.data)[index] = value;
    return value;
  }
  const std::uint16_t bits = output.type == FloatStorageType::kF16
                                 ? float_to_f16(value)
                                 : float_to_bf16(value);
  static_cast<std::uint16_t*>(output.data)[index] = bits;
  return output.type == FloatStorageType::kF16 ? f16_to_float(bits)
                                               : bf16_to_float(bits);
}

std::uint8_t encode_e8m0_scale(float requested) {
  if (!(requested > 0.0f)) return 0;
  const int exponent = static_cast<int>(std::ceil(std::log2(requested)));
  return static_cast<std::uint8_t>(std::clamp(exponent + 127, 0, 254));
}

float decode_e8m0_scale(std::uint8_t code) {
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

Status configure_norm_quant_output(CanonicalQuantTensor* output,
                                   CanonicalQuantLayout layout, long long rows,
                                   long long columns, long long requested_group,
                                   bool scale_2d, long long* quant_group,
                                   bool* needs_global_maximum) {
  if (output == nullptr || quant_group == nullptr ||
      needs_global_maximum == nullptr ||
      !detail::valid_product({rows, columns})) {
    return output == nullptr ? Status::kInvalidArgument : Status::kInvalidShape;
  }
  if (scale_2d && layout != CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    return Status::kInvalidArgument;
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
  output->data.clear();
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
          (requested_group & 1) != 0 || (columns & 1) != 0) {
        return Status::kInvalidShape;
      }
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
      if (*quant_group <= 0 || columns % *quant_group != 0) {
        return Status::kInvalidShape;
      }
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
      if (requested_group != 0 && requested_group != columns) {
        return Status::kInvalidShape;
      }
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
  }
  return Status::kOk;
}

template <bool LayerNorm>
struct CanonicalNormQuantContext {
  FloatStorageInput x;
  FloatStorageInput residual;
  FloatStorageInput weight;
  FloatStorageInput bias;
  FloatStorageOutput residual_out;
  CanonicalQuantTensor* output = nullptr;
  RowNorm<LayerNorm>* norms = nullptr;
  float* scratch = nullptr;
  float* worker_maximum = nullptr;
  std::atomic<bool>* invalid = nullptr;
  long long rows = 0;
  long long hidden = 0;
  long long quant_group = 0;
  long long groups_per_row = 0;
  long long row_domain = 1;
  long long scratch_elements = 0;
  float global_scale = 0.0f;
  bool measure_only = false;
};

template <bool LayerNorm>
void fill_normalized_group(const CanonicalNormQuantContext<LayerNorm>& context,
                           long long row, long long first, long long count,
                           float* values) {
  const RowNorm<LayerNorm>& norm = context.norms[row];
  const bool direct_f32 = context.residual_out.type == FloatStorageType::kF32 &&
                          context.weight.type == FloatStorageType::kF32 &&
                          (!LayerNorm || context.bias.data == nullptr ||
                           context.bias.type == FloatStorageType::kF32);
  if (direct_f32) {
    const float* residual =
        static_cast<const float*>(context.residual_out.data) +
        row * context.hidden + first;
    const float* weight =
        static_cast<const float*>(context.weight.data) + first;
    const float* bias =
        context.bias.data == nullptr
            ? nullptr
            : static_cast<const float*>(context.bias.data) + first;
    for (long long item = 0; item < count; ++item) {
      float value;
      if constexpr (LayerNorm) {
        value = static_cast<float>(
                    (static_cast<double>(residual[item]) - norm.mean) *
                    norm.inverse) *
                    weight[item] +
                (bias == nullptr ? 0.0f : bias[item]);
      } else {
        value = residual[item] * norm.inverse * weight[item];
      }
      values[item] = value;
      if (!std::isfinite(value)) {
        context.invalid->store(true, std::memory_order_relaxed);
      }
    }
    return;
  }
  for (long long item = 0; item < count; ++item) {
    const long long column = first + item;
    const long long flat = row * context.hidden + column;
    const float residual_value =
        storage_load({context.residual_out.data, context.residual_out.type,
                      context.residual_out.count},
                     flat);
    const float weight = storage_load(context.weight, column);
    float value;
    if constexpr (LayerNorm) {
      const float normalized = static_cast<float>(
          (static_cast<double>(residual_value) - norm.mean) * norm.inverse);
      const float bias = context.bias.data == nullptr
                             ? 0.0f
                             : storage_load(context.bias, column);
      value = normalized * weight + bias;
    } else {
      value = residual_value * norm.inverse * weight;
    }
    values[item] = value;
    if (!std::isfinite(value)) {
      context.invalid->store(true, std::memory_order_relaxed);
    }
  }
}

float finite_absmax(const float* values, long long count,
                    std::atomic<bool>* invalid) {
  float maximum = 0.0f;
  for (long long item = 0; item < count; ++item) {
    if (!std::isfinite(values[item])) {
      invalid->store(true, std::memory_order_relaxed);
    } else {
      maximum = std::max(maximum, std::fabs(values[item]));
    }
  }
  return maximum;
}

template <bool LayerNorm>
void encode_norm_quant_group(
    const CanonicalNormQuantContext<LayerNorm>& context, long long row,
    long long group, const float* values) {
  CanonicalQuantTensor& output = *context.output;
  const long long first = group * context.quant_group;
  const std::size_t flat =
      static_cast<std::size_t>(row * context.hidden + first);
  const std::size_t scale_index =
      static_cast<std::size_t>(row * context.groups_per_row + group);
  const float maximum =
      finite_absmax(values, context.quant_group, context.invalid);
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
      const float scale =
          maximum == 0.0f ? 0.0f : decode_e8m0_scale(scale_code);
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

template <bool LayerNorm>
void encode_norm_nvfp4_group(
    const CanonicalNormQuantContext<LayerNorm>& context, long long first_row,
    long long row_count, long long group, const float* values) {
  CanonicalQuantTensor& output = *context.output;
  const float maximum = finite_absmax(values, row_count * 16, context.invalid);
  if (context.invalid->load(std::memory_order_relaxed)) return;
  const float requested = context.global_scale > 0.0f
                              ? maximum / (6.0f * context.global_scale)
                              : 0.0f;
  const std::uint8_t scale_code =
      float8_encode(requested, Float8Format::kE4M3FN);
  const float scale =
      context.global_scale * float8_decode(scale_code, Float8Format::kE4M3FN);
  const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
  for (long long lane = 0; lane < row_count; ++lane) {
    const long long row = first_row + lane;
    const std::size_t block =
        static_cast<std::size_t>(row * context.groups_per_row + group);
    output.scale_codes[block] = scale_code;
    std::uint8_t* destination =
        output.data.data() +
        static_cast<std::size_t>(row * (context.hidden / 2) + group * 8);
    const float* row_values = values + lane * 16;
    for (long long pair = 0; pair < 8; ++pair) {
      const std::uint8_t low = fp4_e2m1_encode(row_values[2 * pair] * inverse);
      const std::uint8_t high =
          fp4_e2m1_encode(row_values[2 * pair + 1] * inverse);
      destination[pair] = static_cast<std::uint8_t>(low | (high << 4));
    }
  }
}

template <bool LayerNorm>
void canonical_norm_quant_groups(void* opaque, long long begin, long long end,
                                 int worker) {
  auto& context = *static_cast<CanonicalNormQuantContext<LayerNorm>*>(opaque);
  float* values = context.scratch +
                  static_cast<long long>(worker) * context.scratch_elements;
  float local_maximum = 0.0f;
  const bool nvfp4 =
      context.output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;
  for (long long task = begin; task < end; ++task) {
    const long long row_domain_index = task / context.groups_per_row;
    const long long group = task % context.groups_per_row;
    const long long first_row = row_domain_index * context.row_domain;
    const long long row_count =
        std::min(context.row_domain, context.rows - first_row);
    for (long long lane = 0; lane < row_count; ++lane) {
      fill_normalized_group(context, first_row + lane,
                            group * context.quant_group, context.quant_group,
                            values + lane * context.quant_group);
    }
    if (context.measure_only) {
      local_maximum = std::max(
          local_maximum, finite_absmax(values, row_count * context.quant_group,
                                       context.invalid));
    } else if (nvfp4) {
      encode_norm_nvfp4_group(context, first_row, row_count, group, values);
    } else {
      for (long long lane = 0; lane < row_count; ++lane) {
        encode_norm_quant_group(context, first_row + lane, group,
                                values + lane * context.quant_group);
      }
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

template <bool LayerNorm>
void canonical_norm_quant_rows(void* opaque, long long begin, long long end,
                               int worker) {
  auto& context = *static_cast<CanonicalNormQuantContext<LayerNorm>*>(opaque);
  float* values = context.scratch +
                  static_cast<long long>(worker) * context.scratch_elements;
  float local_maximum = 0.0f;
  const bool nvfp4 =
      context.output->metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;
  for (long long row = begin; row < end; ++row) {
    fill_normalized_group(context, row, 0, context.hidden, values);
    if (context.measure_only) {
      local_maximum =
          std::max(local_maximum,
                   finite_absmax(values, context.hidden, context.invalid));
      continue;
    }
    for (long long group = 0; group < context.groups_per_row; ++group) {
      const float* group_values = values + group * context.quant_group;
      if (nvfp4) {
        encode_norm_nvfp4_group(context, row, 1, group, group_values);
      } else {
        encode_norm_quant_group(context, row, group, group_values);
      }
    }
  }
  if (context.measure_only) context.worker_maximum[worker] = local_maximum;
}

template <bool LayerNorm>
Status norm_add_quantized_storage_impl(
    FloatStorageInput x, FloatStorageInput residual, FloatStorageInput weight,
    FloatStorageInput bias, FloatStorageOutput residual_out,
    CanonicalQuantLayout output_layout, long long output_group_size,
    CanonicalQuantTensor* output, long long rows, long long hidden, float eps,
    bool scale_2d) {
  if (!detail::valid_product({rows, hidden}) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  const long long elements = rows * hidden;
  if (x.count != elements || residual.count != elements ||
      weight.count != hidden || residual_out.count != elements ||
      (bias.data != nullptr && bias.count != hidden) ||
      (bias.data == nullptr && bias.count != 0)) {
    return Status::kInvalidShape;
  }
  if (!known_storage(x.type) || !known_storage(residual.type) ||
      !known_storage(weight.type) || !known_storage(residual_out.type) ||
      (bias.data != nullptr && !known_storage(bias.type))) {
    return Status::kUnsupportedFormat;
  }
  if (!detail::all_nonnull(x.data, residual.data, weight.data,
                           residual_out.data, output)) {
    return Status::kInvalidArgument;
  }

  long long quant_group = 0;
  bool needs_global_maximum = false;
  Status status;
  try {
    status = configure_norm_quant_output(output, output_layout, rows, hidden,
                                         output_group_size, scale_2d,
                                         &quant_group, &needs_global_maximum);
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
  if (status != Status::kOk) return status;

  const std::size_t workers = static_cast<std::size_t>(num_threads());
  const long long groups_per_row = hidden / quant_group;
  const long long row_domain =
      output_layout == CanonicalQuantLayout::kNVFP4E2M1E4M3 && scale_2d ? 16
                                                                        : 1;
  const long long row_domains = (rows + row_domain - 1) / row_domain;
  if (!detail::valid_product({row_domains, groups_per_row}) ||
      quant_group > std::numeric_limits<long long>::max() / row_domain) {
    return Status::kInvalidShape;
  }
  const bool full_row_route = row_domain == 1;
  const long long scratch_elements =
      full_row_route ? hidden : quant_group * row_domain;
  if (static_cast<unsigned long long>(scratch_elements) >
      std::numeric_limits<std::size_t>::max() / sizeof(float) / workers) {
    return Status::kInvalidShape;
  }
  detail::WorkspaceFrame frame;
  auto* norms =
      frame.allocate<RowNorm<LayerNorm>>(static_cast<std::size_t>(rows));
  float* scratch = frame.allocate<float>(
      workers * static_cast<std::size_t>(scratch_elements));
  float* worker_maximum = frame.allocate<float>(workers);
  if (norms == nullptr || scratch == nullptr || worker_maximum == nullptr) {
    return Status::kOutOfMemory;
  }
  std::fill_n(worker_maximum, workers, 0.0f);
  std::atomic<bool> invalid{false};

  const bool direct_f32 = x.type == FloatStorageType::kF32 &&
                          residual.type == FloatStorageType::kF32 &&
                          residual_out.type == FloatStorageType::kF32;
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const long long base = row * hidden;
      if (direct_f32) {
        norms[row] = add_and_measure<LayerNorm>(
            static_cast<const float*>(x.data) + base,
            static_cast<const float*>(residual.data) + base,
            static_cast<float*>(residual_out.data) + base, hidden, eps);
        continue;
      }
      RowNorm<LayerNorm> norm;
      double m2 = 0.0;
      double sumsq = 0.0;
      for (long long item = 0; item < hidden; ++item) {
        const long long flat = base + item;
        const float value = storage_store_and_reload(
            residual_out, flat,
            storage_load(x, flat) + storage_load(residual, flat));
        if constexpr (LayerNorm) {
          const double delta = static_cast<double>(value) - norm.mean;
          norm.mean += delta / static_cast<double>(item + 1);
          m2 += delta * (static_cast<double>(value) - norm.mean);
        } else {
          sumsq += static_cast<double>(value) * value;
        }
      }
      norm.inverse =
          static_cast<float>(1.0 / std::sqrt((LayerNorm ? m2 : sumsq) /
                                                 static_cast<double>(hidden) +
                                             eps));
      norms[row] = norm;
    }
  });

  CanonicalNormQuantContext<LayerNorm> context{x,
                                               residual,
                                               weight,
                                               bias,
                                               residual_out,
                                               output,
                                               norms,
                                               scratch,
                                               worker_maximum,
                                               &invalid,
                                               rows,
                                               hidden,
                                               quant_group,
                                               groups_per_row,
                                               row_domain,
                                               scratch_elements,
                                               0.0f,
                                               needs_global_maximum};
  const long long tasks = full_row_route ? rows : row_domains * groups_per_row;
  const threading::RangeFn quant_kernel =
      full_row_route ? &canonical_norm_quant_rows<LayerNorm>
                     : &canonical_norm_quant_groups<LayerNorm>;
  if (needs_global_maximum) {
    threading::parallel_ranges_impl(tasks, 1, quant_kernel, &context);
    if (invalid.load(std::memory_order_relaxed)) {
      return Status::kInvalidArgument;
    }
    float maximum = 0.0f;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      maximum = std::max(maximum, worker_maximum[worker]);
    }
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
  threading::parallel_ranges_impl(tasks, 1, quant_kernel, &context);
  if (invalid.load(std::memory_order_relaxed)) {
    return Status::kInvalidArgument;
  }
  return validate_canonical_quant_tensor(*output);
}

}  // namespace

Status rms_norm_add_quant_int8(const float* x, const float* residual,
                               const float* weight, std::int8_t* codes,
                               float* residual_out, float* scales,
                               long long rows, long long hidden, float eps,
                               long long group_size) {
  return norm_add_quant_dynamic<false>(
      x, residual, weight, nullptr, codes, residual_out, scales, rows, hidden,
      eps, group_size, false, Float8Format::kE4M3FN);
}

Status layer_norm_add_quant_int8(const float* x, const float* residual,
                                 const float* weight, const float* bias,
                                 std::int8_t* codes, float* residual_out,
                                 float* scales, long long rows,
                                 long long hidden, float eps,
                                 long long group_size) {
  return norm_add_quant_dynamic<true>(x, residual, weight, bias, codes,
                                      residual_out, scales, rows, hidden, eps,
                                      group_size, false, Float8Format::kE4M3FN);
}

Status rms_norm_add_quant_float8(const float* x, const float* residual,
                                 const float* weight, std::uint8_t* codes,
                                 float* residual_out, float* scales,
                                 long long rows, long long hidden, float eps,
                                 long long group_size, bool power_of_two_scale,
                                 Float8Format format) {
  return norm_add_quant_dynamic<false>(x, residual, weight, nullptr, codes,
                                       residual_out, scales, rows, hidden, eps,
                                       group_size, power_of_two_scale, format);
}

Status layer_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out, float* scales,
    long long rows, long long hidden, float eps, long long group_size,
    bool power_of_two_scale, Float8Format format) {
  return norm_add_quant_dynamic<true>(x, residual, weight, bias, codes,
                                      residual_out, scales, rows, hidden, eps,
                                      group_size, power_of_two_scale, format);
}

Status rms_norm_add_quant_float8_static(const float* x, const float* residual,
                                        const float* weight,
                                        std::uint8_t* codes,
                                        float* residual_out, long long rows,
                                        long long hidden, float scale,
                                        float eps, Float8Format format) {
  return norm_add_quant_float8_static_impl<false>(x, residual, weight, nullptr,
                                                  codes, residual_out, rows,
                                                  hidden, scale, eps, format);
}

Status layer_norm_add_quant_float8_static(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out, long long rows,
    long long hidden, float scale, float eps, Float8Format format) {
  return norm_add_quant_float8_static_impl<true>(x, residual, weight, bias,
                                                 codes, residual_out, rows,
                                                 hidden, scale, eps, format);
}

Status rms_norm_add_quantized_storage(
    FloatStorageInput x, FloatStorageInput residual, FloatStorageInput weight,
    FloatStorageOutput residual_out, CanonicalQuantLayout output_layout,
    long long output_group_size, CanonicalQuantTensor* output, long long rows,
    long long hidden, float eps, bool scale_2d) {
  return norm_add_quantized_storage_impl<false>(
      x, residual, weight, {}, residual_out, output_layout, output_group_size,
      output, rows, hidden, eps, scale_2d);
}

Status layer_norm_add_quantized_storage(
    FloatStorageInput x, FloatStorageInput residual, FloatStorageInput weight,
    FloatStorageInput bias, FloatStorageOutput residual_out,
    CanonicalQuantLayout output_layout, long long output_group_size,
    CanonicalQuantTensor* output, long long rows, long long hidden, float eps,
    bool scale_2d) {
  return norm_add_quantized_storage_impl<true>(
      x, residual, weight, bias, residual_out, output_layout, output_group_size,
      output, rows, hidden, eps, scale_2d);
}

}  // namespace quixicore_cpu
