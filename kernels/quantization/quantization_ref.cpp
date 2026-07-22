#include "quixicore_cpu/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool quant_shape(long long rows, long long dim, long long* group_size) {
  if (!detail::valid_product({rows, dim})) return false;
  if (*group_size == 0) *group_size = dim;
  return *group_size > 0 && dim % *group_size == 0;
}

float fp8_positive(std::uint8_t code, Float8Format format) {
  if (format == Float8Format::kE4M3FN) {
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -9);
    return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                      exponent - 7);
  }
  const int exponent = (code >> 2) & 31;
  const int mantissa = code & 3;
  if (exponent == 31) {
    return mantissa == 0 ? std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::quiet_NaN();
  }
  if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -16);
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 4.0f,
                    exponent - 15);
}

std::uint8_t fp8_max_finite(Float8Format format) {
  return format == Float8Format::kE4M3FN ? 0x7e : 0x7b;
}

float fp8_max_value(Float8Format format) {
  return format == Float8Format::kE4M3FN ? 448.0f : 57344.0f;
}

int rounded_int(float value) {
  return static_cast<int>(std::nearbyint(value));
}

bool all_finite(const float* values, long long count) {
  for (long long i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

float half_at(const std::uint8_t* bytes) {
  std::uint16_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return fp16_to_fp32(value);
}

void put_half(std::uint8_t* bytes, float value) {
  const std::uint16_t bits = fp32_to_fp16(value);
  std::memcpy(bytes, &bits, sizeof(bits));
}

float fused_gate(float x, float gate, bool oai_mode, float alpha,
                 float limit) {
  if (oai_mode) {
    x = std::min(x, limit);
    gate = std::clamp(gate, -limit, limit);
    return (x / (1.0f + std::exp(-alpha * x))) * (1.0f + gate);
  }
  return (x / (1.0f + std::exp(-x))) * gate;
}

}  // namespace

float float8_decode(std::uint8_t code, Float8Format format) {
  const float value = fp8_positive(code & 0x7f, format);
  return code & 0x80 ? -value : value;
}

std::uint8_t float8_encode(float value, Float8Format format) {
  if (std::isnan(value)) {
    return format == Float8Format::kE4M3FN ? 0x7f : 0x7d;
  }
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0f) return negative ? 0x80 : 0;
  if (!std::isfinite(magnitude) || magnitude >= fp8_max_value(format)) {
    return fp8_max_finite(format) | (negative ? 0x80 : 0);
  }
  const int limit = format == Float8Format::kE4M3FN ? 0x7e : 0x7b;
  std::uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (int code = 0; code <= limit; ++code) {
    const float candidate = fp8_positive(static_cast<std::uint8_t>(code), format);
    if (!std::isfinite(candidate)) continue;
    const float error = std::fabs(candidate - magnitude);
    if (error < best_error ||
        (error == best_error && (best & 1u) != 0 && (code & 1) == 0)) {
      best_error = error;
      best = static_cast<std::uint8_t>(code);
    }
  }
  return best | (negative ? 0x80 : 0);
}

Status quantize_int8(const float* x, std::int8_t* codes, float* scales,
                     long long rows, long long dim, long long group_size) {
  if (!quant_shape(rows, dim, &group_size)) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, codes, scales)) return Status::kInvalidArgument;
  if (!all_finite(x, rows * dim)) return Status::kInvalidArgument;
  const long long groups = dim / group_size;
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long group = 0; group < groups; ++group) {
        const long long base = row * dim + group * group_size;
        float maximum = 0.0f;
        for (long long i = 0; i < group_size; ++i) {
          maximum = std::max(maximum, std::fabs(x[base + i]));
        }
        const float scale = maximum / 127.0f;
        scales[row * groups + group] = scale;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (long long i = 0; i < group_size; ++i) {
          codes[base + i] = static_cast<std::int8_t>(
              std::clamp(rounded_int(x[base + i] * inverse), -127, 127));
        }
      }
    }
  });
  return Status::kOk;
}

Status dequantize_int8(const std::int8_t* codes, const float* scales,
                       float* out, long long rows, long long dim,
                       long long group_size) {
  if (!quant_shape(rows, dim, &group_size)) return Status::kInvalidShape;
  if (!detail::all_nonnull(codes, scales, out)) return Status::kInvalidArgument;
  const long long groups = dim / group_size;
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long group = 0; group < groups; ++group) {
        const long long base = row * dim + group * group_size;
        const float scale = scales[row * groups + group];
        for (long long i = 0; i < group_size; ++i) {
          out[base + i] = scale * static_cast<float>(codes[base + i]);
        }
      }
    }
  });
  return Status::kOk;
}

Status quantize_int8_asymmetric(const float* x, std::int8_t* codes,
                                float* scales, int* zero_points,
                                long long rows, long long dim) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, codes, scales, zero_points)) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(x, rows * dim)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* input = x + row * dim;
      const auto bounds = std::minmax_element(input, input + dim);
      const float range = *bounds.second - *bounds.first;
      const float scale = range > 0.0f
                              ? range / 255.0f
                              : std::max(std::fabs(*bounds.first) / 127.0f, 1e-7f);
      const int zero = rounded_int(-128.0f - *bounds.first / scale);
      scales[row] = scale;
      zero_points[row] = zero;
      for (long long i = 0; i < dim; ++i) {
        codes[row * dim + i] = static_cast<std::int8_t>(
            std::clamp(rounded_int(input[i] / scale) + zero, -128, 127));
      }
    }
  });
  return Status::kOk;
}

Status dequantize_int8_asymmetric(const std::int8_t* codes,
                                  const float* scales,
                                  const int* zero_points, float* out,
                                  long long rows, long long dim) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(codes, scales, zero_points, out)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long i = 0; i < dim; ++i) {
        out[row * dim + i] = scales[row] *
            (static_cast<int>(codes[row * dim + i]) - zero_points[row]);
      }
    }
  });
  return Status::kOk;
}

Status quantize_int4_group(const float* x, std::uint8_t* packed,
                           float* scales, long long rows, long long dim,
                           long long group_size) {
  if (!detail::valid_product({rows, dim, group_size}) ||
      dim % group_size != 0 || dim % 2 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, packed, scales)) return Status::kInvalidArgument;
  if (!all_finite(x, rows * dim)) return Status::kInvalidArgument;
  const long long groups = dim / group_size;
  std::fill_n(packed, rows * dim / 2, std::uint8_t{0});
  for (long long row = 0; row < rows; ++row) {
    for (long long group = 0; group < groups; ++group) {
      const long long base = row * dim + group * group_size;
      float maximum = 0.0f;
      for (long long item = 0; item < group_size; ++item) {
        maximum = std::max(maximum, std::fabs(x[base + item]));
      }
      const float scale = maximum / 7.0f;
      scales[row * groups + group] = scale;
      for (long long item = 0; item < group_size; ++item) {
        const int code = scale > 0.0f
                             ? std::clamp(static_cast<int>(std::nearbyint(
                                              x[base + item] / scale)),
                                          -8, 7)
                             : 0;
        const long long flat = base + item;
        packed[flat / 2] |= static_cast<std::uint8_t>(
            (static_cast<unsigned>(code) & 0x0Fu) << (4 * (flat & 1)));
      }
    }
  }
  return Status::kOk;
}

Status dequantize_int4_group(const std::uint8_t* packed,
                             const float* scales, float* out,
                             long long rows, long long dim,
                             long long group_size) {
  if (!detail::valid_product({rows, dim, group_size}) ||
      dim % group_size != 0 || dim % 2 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed, scales, out)) {
    return Status::kInvalidArgument;
  }
  const long long groups = dim / group_size;
  for (long long row = 0; row < rows; ++row) {
    for (long long item = 0; item < dim; ++item) {
      const long long flat = row * dim + item;
      int code = (packed[flat / 2] >> (4 * (flat & 1))) & 0x0F;
      if (code & 8) code -= 16;
      out[flat] = code * scales[row * groups + item / group_size];
    }
  }
  return Status::kOk;
}

Status quantize_float8(const float* x, std::uint8_t* codes, float* scales,
                       long long rows, long long dim, long long group_size,
                       Float8Format format, bool power_of_two_scale) {
  if (!quant_shape(rows, dim, &group_size)) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, codes, scales)) return Status::kInvalidArgument;
  if (!all_finite(x, rows * dim)) return Status::kInvalidArgument;
  const long long groups = dim / group_size;
  const float maximum_code = fp8_max_value(format);
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long group = 0; group < groups; ++group) {
        const long long base = row * dim + group * group_size;
        float maximum = 0.0f;
        for (long long i = 0; i < group_size; ++i) {
          maximum = std::max(maximum, std::fabs(x[base + i]));
        }
        float scale = maximum / maximum_code;
        if (power_of_two_scale && maximum > 0.0f) {
          scale = std::exp2(std::ceil(std::log2(
              std::max(maximum, 1e-10f) / maximum_code)));
        }
        scales[row * groups + group] = scale;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (long long i = 0; i < group_size; ++i) {
          codes[base + i] = float8_encode(x[base + i] * inverse, format);
        }
      }
    }
  });
  return Status::kOk;
}

Status dequantize_float8(const std::uint8_t* codes, const float* scales,
                         float* out, long long rows, long long dim,
                         long long group_size, Float8Format format) {
  if (!quant_shape(rows, dim, &group_size)) return Status::kInvalidShape;
  if (!detail::all_nonnull(codes, scales, out)) return Status::kInvalidArgument;
  const long long groups = dim / group_size;
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long group = 0; group < groups; ++group) {
        const long long base = row * dim + group * group_size;
        const float scale = scales[row * groups + group];
        for (long long i = 0; i < group_size; ++i) {
          out[base + i] = scale * float8_decode(codes[base + i], format);
        }
      }
    }
  });
  return Status::kOk;
}

Status fake_quant_int8(const float* x, float* out, std::int8_t* codes,
                       float* scales, long long rows, long long dim) {
  Status status = quantize_int8(x, codes, scales, rows, dim, dim);
  return status == Status::kOk
             ? dequantize_int8(codes, scales, out, rows, dim, dim)
             : status;
}

Status silu_mul_fake_quant_int8(const float* x, const float* gate, float* out,
                                std::int8_t* codes, float* scales,
                                long long rows, long long dim, bool oai_mode,
                                float alpha, float limit) {
  if (!detail::valid_product({rows, dim}) || !std::isfinite(alpha) ||
      alpha <= 0.0f || !std::isfinite(limit) || limit < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, gate, out, codes, scales)) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(x, rows * dim) || !all_finite(gate, rows * dim)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 8, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      float maximum = 0.0f;
      for (long long i = 0; i < dim; ++i) {
        maximum = std::max(maximum, std::fabs(fused_gate(
            x[row * dim + i], gate[row * dim + i], oai_mode, alpha, limit)));
      }
      const float scale = maximum / 127.0f;
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      scales[row] = scale;
      for (long long i = 0; i < dim; ++i) {
        const long long index = row * dim + i;
        const int code = std::clamp(rounded_int(fused_gate(
            x[index], gate[index], oai_mode, alpha, limit) * inverse),
                                    -127, 127);
        codes[index] = static_cast<std::int8_t>(code);
        out[index] = scale * code;
      }
    }
  });
  return Status::kOk;
}

Status silu_mul_quant_int8(const float* x, const float* gate,
                           std::int8_t* codes, float* scales, long long rows,
                           long long dim, bool oai_mode, float alpha,
                           float limit) {
  if (!detail::valid_product({rows, dim}) || !std::isfinite(alpha) ||
      alpha <= 0.0f || !std::isfinite(limit) || limit < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, gate, codes, scales)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> activated(static_cast<std::size_t>(rows * dim));
  for (long long i = 0; i < rows * dim; ++i) {
    activated[i] = fused_gate(x[i], gate[i], oai_mode, alpha, limit);
  }
  return quantize_int8(activated.data(), codes, scales, rows, dim, dim);
}

Status silu_mul_quant_float8(const float* x, const float* gate,
                             std::uint8_t* codes, float* scales,
                             long long rows, long long dim,
                             long long group_size, bool power_of_two_scale,
                             bool oai_mode, float alpha, float limit,
                             Float8Format format) {
  if (!detail::valid_product({rows, dim}) || !std::isfinite(alpha) ||
      alpha <= 0.0f || !std::isfinite(limit) || limit < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, gate, codes, scales)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> activated(static_cast<std::size_t>(rows * dim));
  for (long long i = 0; i < rows * dim; ++i) {
    activated[i] = fused_gate(x[i], gate[i], oai_mode, alpha, limit);
  }
  return quantize_float8(activated.data(), codes, scales, rows, dim,
                         group_size, format, power_of_two_scale);
}

Status fake_quant_float8(const float* x, float* out, std::uint8_t* codes,
                         float* scale, long long count, Float8Format format) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  Status status = quantize_float8(x, codes, scale, 1, count, count, format);
  return status == Status::kOk
             ? dequantize_float8(codes, scale, out, 1, count, count, format)
             : status;
}

Status ternary_pack(const float* weights, std::uint8_t* packed,
                    float* dequantized, long long rows, long long k,
                    long long group_k) {
  if (!detail::valid_product({rows, k}) || group_k <= 0 || group_k % 32 != 0 ||
      k % group_k != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(weights, packed, dequantized)) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(weights, rows * k)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long group = 0; group < k / group_k; ++group) {
        const long long group_base = row * k + group * group_k;
        double sum = 0.0;
        for (long long i = 0; i < group_k; ++i) {
          sum += std::fabs(weights[group_base + i]);
        }
        const float scale = std::max(static_cast<float>(sum / group_k), 1e-5f);
        const float rounded_scale = fp16_to_fp32(fp32_to_fp16(scale));
        for (long long block = 0; block < group_k / 32; ++block) {
          std::uint8_t* destination = packed +
              ((row * (k / 32)) + group * (group_k / 32) + block) * 10;
          put_half(destination, scale);
          for (int byte = 0; byte < 8; ++byte) {
            std::uint8_t code = 0;
            for (int lane = 0; lane < 4; ++lane) {
              const long long local = block * 32 + byte * 4 + lane;
              const int q = std::clamp(rounded_int(
                  weights[group_base + local] / scale), -1, 1);
              code |= static_cast<std::uint8_t>((q + 1) << (2 * lane));
              dequantized[group_base + local] = rounded_scale * q;
            }
            destination[2 + byte] = code;
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status ternary_unpack(const std::uint8_t* packed, float* weights,
                      long long rows, long long k) {
  if (!detail::valid_product({rows, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed, weights)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    for (long long block = 0; block < k / 32; ++block) {
      const std::uint8_t* source = packed + (row * (k / 32) + block) * 10;
      const float scale = half_at(source);
      for (int i = 0; i < 32; ++i) {
        const int code = (source[2 + i / 4] >> (2 * (i % 4))) & 3;
        weights[row * k + block * 32 + i] = scale * (code - 1);
      }
    }
  }
  return Status::kOk;
}

Status ternary_stats(const std::uint8_t* packed, std::uint32_t* counts,
                     long long rows, long long k) {
  if (!detail::valid_product({rows, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed, counts)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    std::uint32_t local[3] = {};
    for (long long block = 0; block < k / 32; ++block) {
      const std::uint8_t* source = packed + (row * (k / 32) + block) * 10 + 2;
      for (int i = 0; i < 32; ++i) {
        const int code = (source[i / 4] >> (2 * (i % 4))) & 3;
        if (code < 3) ++local[code];
      }
    }
    std::copy_n(local, 3, counts + row * 3);
  }
  return Status::kOk;
}

Status ternary_code_flip_count(const std::uint8_t* a, const std::uint8_t* b,
                               std::uint32_t* flips, long long rows,
                               long long k) {
  if (!detail::valid_product({rows, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, flips)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    std::uint32_t count = 0;
    for (long long block = 0; block < k / 32; ++block) {
      const long long base = (row * (k / 32) + block) * 10 + 2;
      for (int byte = 0; byte < 8; ++byte) {
        const std::uint8_t difference = a[base + byte] ^ b[base + byte];
        for (int lane = 0; lane < 4; ++lane) {
          count += ((difference >> (2 * lane)) & 3) != 0;
        }
      }
    }
    flips[row] = count;
  }
  return Status::kOk;
}

Status tq2_0_pack(const float* weights, std::uint8_t* packed,
                  float* dequantized, long long rows, long long k) {
  if (!detail::valid_product({rows, k}) || k % 256 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(weights, packed, dequantized)) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(weights, rows * k)) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long block = 0; block < k / 256; ++block) {
        const long long base = row * k + block * 256;
        float maximum = 0.0f;
        for (int i = 0; i < 256; ++i) {
          maximum = std::max(maximum, std::fabs(weights[base + i]));
        }
        const float inverse = maximum > 0.0f ? 1.0f / maximum : 0.0f;
        const float rounded_scale = fp16_to_fp32(fp32_to_fp16(maximum));
        std::uint8_t* destination =
            packed + (row * (k / 256) + block) * 66;
        for (int half = 0; half < 2; ++half) {
          for (int lane = 0; lane < 32; ++lane) {
            std::uint8_t byte = 0;
            for (int group = 0; group < 4; ++group) {
              const int local = 128 * half + 32 * group + lane;
              const float value = weights[base + local] * inverse;
              const int q = value < 0.0f
                                ? -static_cast<int>(std::floor(-value + 0.5f))
                                : static_cast<int>(std::floor(value + 0.5f));
              byte |= static_cast<std::uint8_t>((q + 1) << (2 * group));
              dequantized[base + local] = rounded_scale * q;
            }
            destination[32 * half + lane] = byte;
          }
        }
        put_half(destination + 64, maximum);
      }
    }
  });
  return Status::kOk;
}

Status tq2_0_unpack(const std::uint8_t* packed, float* weights,
                    long long rows, long long k) {
  if (!detail::valid_product({rows, k}) || k % 256 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed, weights)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    for (long long block = 0; block < k / 256; ++block) {
      const std::uint8_t* source =
          packed + (row * (k / 256) + block) * 66;
      const float scale = half_at(source + 64);
      for (int i = 0; i < 256; ++i) {
        const int half = i / 128;
        const int rest = i % 128;
        const int group = rest / 32;
        const int lane = rest % 32;
        const int code = (source[32 * half + lane] >> (2 * group)) & 3;
        weights[row * k + block * 256 + i] = scale * (code - 1);
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
