#include "quixicore_cpu/quantization.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/validation.h"
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
    float64x2_t squares0 = vdupq_n_f64(0.0);
    float64x2_t squares1 = vdupq_n_f64(0.0);
    long long item = 0;
    for (; item + 3 < hidden; item += 4) {
      const float32x4_t value =
          vaddq_f32(vld1q_f32(x + item), vld1q_f32(residual + item));
      vst1q_f32(residual_out + item, value);
      const float64x2_t low = vcvt_f64_f32(vget_low_f32(value));
      const float64x2_t high = vcvt_f64_f32(vget_high_f32(value));
      squares0 = vfmaq_f64(squares0, low, low);
      squares1 = vfmaq_f64(squares1, high, high);
    }
    sumsq = vaddvq_f64(vaddq_f64(squares0, squares1));
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
    return normalized * weight[item] +
           (bias != nullptr ? bias[item] : 0.0f);
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
Status norm_add_quant_dynamic(
    const float* x, const float* residual, const float* weight,
    const float* bias, Code* codes, float* residual_out, float* scales,
    long long rows, long long hidden, float eps, long long group_size,
    bool power_of_two_scale, Float8Format format) {
  if (!valid_dynamic_shape(rows, hidden, &group_size, eps)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, residual, weight, codes, residual_out, scales)) {
    return Status::kInvalidArgument;
  }

  const long long groups = hidden / group_size;
  const float maximum_code = std::is_same_v<Code, std::int8_t>
                                 ? 127.0f
                                 : (format == Float8Format::kE4M3FN ? 448.0f
                                                                    : 57344.0f);
  const std::size_t threads = static_cast<std::size_t>(num_threads());
  if (static_cast<std::size_t>(group_size) >
      std::numeric_limits<std::size_t>::max() / threads) {
    return Status::kInvalidShape;
  }
  detail::WorkspaceFrame workspace;
  float* scratch = workspace.allocate<float>(
      threads * static_cast<std::size_t>(group_size));
  if (scratch == nullptr) return Status::kOutOfMemory;
  std::atomic<bool> invalid{false};
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end,
                                          int worker) {
    float* group_values = scratch +
                          static_cast<long long>(worker) * group_size;
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
          float32x4_t vector_maximum = vdupq_n_f32(0.0f);
          long long item = 0;
          const float32x4_t inverse_norm = vdupq_n_f32(norm.inverse);
          for (; item + 3 < group_size; item += 4) {
            const long long column = group_base + item;
            const float32x4_t value = vmulq_f32(
                vmulq_f32(vld1q_f32(residual_row + column), inverse_norm),
                vld1q_f32(weight + column));
            vst1q_f32(group_values + item, value);
            vector_maximum =
                vmaxq_f32(vector_maximum, vabsq_f32(value));
          }
          maximum = vmaxvq_f32(vector_maximum);
          group_invalid = !std::isfinite(maximum);
          for (; item < group_size; ++item) {
            const long long column = group_base + item;
            const float value = (residual_row[column] * norm.inverse) *
                                weight[column];
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
            scale = std::exp2(std::ceil(std::log2(
                std::max(maximum, 1e-10f) / maximum_code)));
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
          for (; item + 3 < group_size; item += 4) {
            int32x4_t quantized = vcvtnq_s32_f32(
                vmulq_f32(vld1q_f32(group_values + item), inverse_scale));
            quantized = vmaxq_s32(lower, vminq_s32(upper, quantized));
            const int16x4_t narrowed = vqmovn_s32(quantized);
            const int8x8_t bytes =
                vqmovn_s16(vcombine_s16(narrowed, vdup_n_s16(0)));
            const std::uint32_t word = vget_lane_u32(
                vreinterpret_u32_s8(bytes), 0);
            std::memcpy(codes + row_base + group_base + item, &word,
                        sizeof(word));
          }
          for (; item < group_size; ++item) {
            codes[row_base + group_base + item] =
                static_cast<std::int8_t>(std::clamp(
                    static_cast<int>(
                        std::nearbyint(group_values[item] * inverse)),
                    -127, 127));
          }
        } else
#endif
        {
          for (long long item = 0; item < group_size; ++item) {
            const long long column = group_base + item;
            const float value = group_values[item];
            if constexpr (std::is_same_v<Code, std::int8_t>) {
              codes[row_base + column] = static_cast<std::int8_t>(std::clamp(
                  static_cast<int>(std::nearbyint(value * inverse)), -127,
                  127));
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
Status norm_add_quant_float8_static_impl(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out,
    long long rows, long long hidden, float scale, float eps,
    Float8Format format) {
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
        const float value = normalized_at<LayerNorm>(
            residual_row, weight, bias, item, norm);
        if (!std::isfinite(value)) invalid.store(true, std::memory_order_relaxed);
        codes[base + item] = float8_encode(value * inverse_scale, format);
      }
    }
  });
  return invalid.load(std::memory_order_relaxed) ? Status::kInvalidArgument
                                                 : Status::kOk;
}

}  // namespace

Status rms_norm_add_quant_int8(
    const float* x, const float* residual, const float* weight,
    std::int8_t* codes, float* residual_out, float* scales, long long rows,
    long long hidden, float eps, long long group_size) {
  return norm_add_quant_dynamic<false>(
      x, residual, weight, nullptr, codes, residual_out, scales, rows, hidden,
      eps, group_size, false, Float8Format::kE4M3FN);
}

Status layer_norm_add_quant_int8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::int8_t* codes, float* residual_out,
    float* scales, long long rows, long long hidden, float eps,
    long long group_size) {
  return norm_add_quant_dynamic<true>(
      x, residual, weight, bias, codes, residual_out, scales, rows, hidden,
      eps, group_size, false, Float8Format::kE4M3FN);
}

Status rms_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    std::uint8_t* codes, float* residual_out, float* scales, long long rows,
    long long hidden, float eps, long long group_size,
    bool power_of_two_scale, Float8Format format) {
  return norm_add_quant_dynamic<false>(
      x, residual, weight, nullptr, codes, residual_out, scales, rows, hidden,
      eps, group_size, power_of_two_scale, format);
}

Status layer_norm_add_quant_float8(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out,
    float* scales, long long rows, long long hidden, float eps,
    long long group_size, bool power_of_two_scale, Float8Format format) {
  return norm_add_quant_dynamic<true>(
      x, residual, weight, bias, codes, residual_out, scales, rows, hidden,
      eps, group_size, power_of_two_scale, format);
}

Status rms_norm_add_quant_float8_static(
    const float* x, const float* residual, const float* weight,
    std::uint8_t* codes, float* residual_out, long long rows,
    long long hidden, float scale, float eps, Float8Format format) {
  return norm_add_quant_float8_static_impl<false>(
      x, residual, weight, nullptr, codes, residual_out, rows, hidden, scale,
      eps, format);
}

Status layer_norm_add_quant_float8_static(
    const float* x, const float* residual, const float* weight,
    const float* bias, std::uint8_t* codes, float* residual_out,
    long long rows, long long hidden, float scale, float eps,
    Float8Format format) {
  return norm_add_quant_float8_static_impl<true>(
      x, residual, weight, bias, codes, residual_out, rows, hidden, scale, eps,
      format);
}

}  // namespace quixicore_cpu
