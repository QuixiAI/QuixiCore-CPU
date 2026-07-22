#include "quixicore_cpu/quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common/validation.h"
#include "quixicore_cpu/qgemm.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

constexpr float kE2M1[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

bool all_finite(const float* values, long long count) {
  for (long long i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

float local_fp8_scale(std::uint8_t code) {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (int item = 0; item < 256; ++item) {
      values[static_cast<std::size_t>(item)] = float8_decode(
          static_cast<std::uint8_t>(item), Float8Format::kE4M3FN);
    }
    return values;
  }();
  return table[code];
}

float packed_fp4_at(const std::uint8_t* packed, long long row,
                    long long dim, long long column) {
  const std::uint8_t byte = packed[row * (dim / 2) + column / 2];
  const std::uint8_t code = static_cast<std::uint8_t>(
      (byte >> (4 * (column & 1))) & 0x0f);
  return fp4_e2m1_decode(code);
}

}  // namespace

float fp4_e2m1_decode(std::uint8_t code) {
  const float magnitude = kE2M1[code & 7u];
  return code & 8u ? -magnitude : magnitude;
}

std::uint8_t fp4_e2m1_encode(float value) {
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  if (std::isnan(magnitude)) return 0;
  if (!std::isfinite(magnitude) || magnitude >= kE2M1[7]) {
    return static_cast<std::uint8_t>(7u | (negative ? 8u : 0u));
  }
  int best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (int code = 0; code < 8; ++code) {
    const float error = std::fabs(kE2M1[code] - magnitude);
    if (error < best_error ||
        (error == best_error && (best & 1) != 0 && (code & 1) == 0)) {
      best = code;
      best_error = error;
    }
  }
  return static_cast<std::uint8_t>(best | (negative ? 8 : 0));
}

Status fp32_to_fp4x2(const float* x, std::uint8_t* packed,
                     long long count) {
  if (count <= 0 || count % 2 != 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, packed)) return Status::kInvalidArgument;
  if (!all_finite(x, count)) return Status::kInvalidArgument;
  threading::parallel_ranges(count / 2, 4096,
                             [&](long long begin, long long end, int) {
    for (long long pair = begin; pair < end; ++pair) {
      const std::uint8_t low = fp4_e2m1_encode(x[2 * pair]);
      const std::uint8_t high = fp4_e2m1_encode(x[2 * pair + 1]);
      packed[pair] = static_cast<std::uint8_t>(low | (high << 4));
    }
  });
  return Status::kOk;
}

Status fp4x2_to_fp32(const std::uint8_t* packed, float* out,
                     long long count) {
  if (count <= 0 || count % 2 != 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(count / 2, 4096,
                             [&](long long begin, long long end, int) {
    for (long long pair = begin; pair < end; ++pair) {
      out[2 * pair] = fp4_e2m1_decode(packed[pair] & 0x0f);
      out[2 * pair + 1] = fp4_e2m1_decode(packed[pair] >> 4);
    }
  });
  return Status::kOk;
}

Status mxfp8_quantize(const float* x, std::uint8_t* codes, float* scales,
                      long long rows, long long dim) {
  if (!detail::valid_product({rows, dim}) || dim % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, codes, scales)) return Status::kInvalidArgument;
  if (!all_finite(x, rows * dim)) return Status::kInvalidArgument;
  const long long groups = dim / 32;
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long group = 0; group < groups; ++group) {
        const long long base = row * dim + group * 32;
        float maximum = 0.0f;
        for (long long item = 0; item < 32; ++item) {
          maximum = std::max(maximum, std::fabs(x[base + item]));
        }
        const float requested = std::max(maximum / 448.0f, 1e-12f);
        const float scale = std::exp2(std::ceil(std::log2(requested)));
        scales[row * groups + group] = scale;
        const float inverse = 1.0f / scale;
        for (long long item = 0; item < 32; ++item) {
          codes[base + item] = float8_encode(
              x[base + item] * inverse, Float8Format::kE4M3FN);
        }
      }
    }
  });
  return Status::kOk;
}

Status nvfp4_quantize(const float* x, std::uint8_t* packed,
                      std::uint8_t* scale_codes, float* global_scale,
                      long long rows, long long dim, bool scale_2d) {
  if (!detail::valid_product({rows, dim}) || dim % 16 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, packed, scale_codes, global_scale)) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(x, rows * dim)) return Status::kInvalidArgument;

  float tensor_maximum = 0.0f;
  for (long long i = 0; i < rows * dim; ++i) {
    tensor_maximum = std::max(tensor_maximum, std::fabs(x[i]));
  }
  *global_scale = tensor_maximum / (6.0f * 448.0f);
  const long long blocks = dim / 16;
  const long long row_group = scale_2d ? 16 : 1;

  threading::parallel_ranges((rows + row_group - 1) / row_group, 1,
                             [&](long long begin, long long end, int) {
    for (long long group = begin; group < end; ++group) {
      const long long row_begin = group * row_group;
      const long long row_end = std::min(rows, row_begin + row_group);
      for (long long block = 0; block < blocks; ++block) {
        float block_maximum = 0.0f;
        for (long long row = row_begin; row < row_end; ++row) {
          const long long base = row * dim + block * 16;
          for (long long item = 0; item < 16; ++item) {
            block_maximum =
                std::max(block_maximum, std::fabs(x[base + item]));
          }
        }
        const float requested_local =
            *global_scale > 0.0f
                ? block_maximum / (6.0f * *global_scale)
                : 0.0f;
        const std::uint8_t scale_code =
            float8_encode(requested_local, Float8Format::kE4M3FN);
        const float scale = *global_scale * local_fp8_scale(scale_code);
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (long long row = row_begin; row < row_end; ++row) {
          scale_codes[row * blocks + block] = scale_code;
          const long long input_base = row * dim + block * 16;
          const long long packed_base = row * (dim / 2) + block * 8;
          for (long long pair = 0; pair < 8; ++pair) {
            const std::uint8_t low =
                fp4_e2m1_encode(x[input_base + 2 * pair] * inverse);
            const std::uint8_t high =
                fp4_e2m1_encode(x[input_base + 2 * pair + 1] * inverse);
            packed[packed_base + pair] =
                static_cast<std::uint8_t>(low | (high << 4));
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status mxfp8_gemm(const std::uint8_t* a_codes, const float* a_scales,
                  const std::uint8_t* b_codes, const float* b_scales,
                  float* y, long long m, long long n, long long k) {
  if (!detail::valid_product({m, n, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a_codes, a_scales, b_codes, b_scales, y)) {
    return Status::kInvalidArgument;
  }
  const long long groups = k / 32;
  for (long long i = 0; i < m * groups; ++i) {
    if (!std::isfinite(a_scales[i]) || a_scales[i] <= 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  for (long long i = 0; i < n * groups; ++i) {
    if (!std::isfinite(b_scales[i]) || b_scales[i] <= 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(m * n, 16,
                             [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const long long row = output / n;
      const long long column = output % n;
      double sum = 0.0;
      for (long long group = 0; group < groups; ++group) {
        double block_sum = 0.0;
        const long long a_base = row * k + group * 32;
        const long long b_base = column * k + group * 32;
        for (long long item = 0; item < 32; ++item) {
          block_sum += local_fp8_scale(a_codes[a_base + item]) *
                       local_fp8_scale(b_codes[b_base + item]);
        }
        sum += block_sum * a_scales[row * groups + group] *
               b_scales[column * groups + group];
      }
      y[output] = static_cast<float>(sum);
    }
  });
  return Status::kOk;
}

Status nvfp4_gemm(const std::uint8_t* a_packed,
                  const std::uint8_t* a_scale_codes, float a_global_scale,
                  const std::uint8_t* b_packed,
                  const std::uint8_t* b_scale_codes, float b_global_scale,
                  float* y, long long m, long long n, long long k) {
  if (!detail::valid_product({m, n, k}) || k % 16 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a_packed, a_scale_codes, b_packed,
                           b_scale_codes, y) ||
      !std::isfinite(a_global_scale) || !std::isfinite(b_global_scale) ||
      a_global_scale < 0.0f || b_global_scale < 0.0f) {
    return Status::kInvalidArgument;
  }
  const long long blocks = k / 16;
  for (long long i = 0; i < m * blocks; ++i) {
    if (!std::isfinite(local_fp8_scale(a_scale_codes[i]))) {
      return Status::kInvalidArgument;
    }
  }
  for (long long i = 0; i < n * blocks; ++i) {
    if (!std::isfinite(local_fp8_scale(b_scale_codes[i]))) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(m * n, 16,
                             [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const long long row = output / n;
      const long long column = output % n;
      double sum = 0.0;
      for (long long block = 0; block < blocks; ++block) {
        double block_sum = 0.0;
        for (long long item = 0; item < 16; ++item) {
          const long long input = block * 16 + item;
          block_sum += packed_fp4_at(a_packed, row, k, input) *
                       packed_fp4_at(b_packed, column, k, input);
        }
        const float a_scale = local_fp8_scale(
            a_scale_codes[row * blocks + block]);
        const float b_scale = local_fp8_scale(
            b_scale_codes[column * blocks + block]);
        sum += block_sum * a_scale * b_scale;
      }
      y[output] = static_cast<float>(
          sum * static_cast<double>(a_global_scale) * b_global_scale);
    }
  });
  return Status::kOk;
}

Status mxfp4_gemv(const std::uint8_t* packed_weights,
                  const std::uint8_t* scale_codes, const float* x, float* y,
                  long long n, long long k) {
  if (!detail::valid_product({n, k}) || k % 32 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, scale_codes, x, y)) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(x, k)) return Status::kInvalidArgument;
  const long long blocks = k / 32;
  threading::parallel_ranges(n, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      double sum = 0.0;
      for (long long block = 0; block < blocks; ++block) {
        const std::uint8_t scale_code = scale_codes[row * blocks + block];
        const float scale = scale_code == 0
                                ? std::ldexp(1.0f, -127)
                                : std::ldexp(
                                      1.0f,
                                      static_cast<int>(scale_code) - 127);
        double block_sum = 0.0;
        for (long long item = 0; item < 32; ++item) {
          const long long input = block * 32 + item;
          block_sum += packed_fp4_at(packed_weights, row, k, input) * x[input];
        }
        sum += block_sum * scale;
      }
      y[row] = static_cast<float>(sum);
    }
  });
  return Status::kOk;
}

Status nvfp4_gemv(const std::uint8_t* packed_weights,
                  const std::uint8_t* scale_codes, float global_scale,
                  const float* x, float* y, long long n, long long k) {
  if (!detail::valid_product({n, k}) || k % 16 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, scale_codes, x, y) ||
      !std::isfinite(global_scale) || global_scale < 0.0f) {
    return Status::kInvalidArgument;
  }
  if (!all_finite(x, k)) return Status::kInvalidArgument;
  const long long blocks = k / 16;
  for (long long i = 0; i < n * blocks; ++i) {
    if (!std::isfinite(local_fp8_scale(scale_codes[i]))) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(n, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      double sum = 0.0;
      for (long long block = 0; block < blocks; ++block) {
        double block_sum = 0.0;
        for (long long item = 0; item < 16; ++item) {
          const long long input = block * 16 + item;
          block_sum += packed_fp4_at(packed_weights, row, k, input) * x[input];
        }
        sum += block_sum * local_fp8_scale(scale_codes[row * blocks + block]);
      }
      y[row] = static_cast<float>(sum * global_scale);
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
