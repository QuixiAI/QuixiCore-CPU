#include "quixicore_cpu/lowbit.h"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "kernels/quantization/iq_tables.h"
#include "kernels/quantization/lowbit.h"
#include "quixicore_cpu/cpu_features.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

struct Layout {
  long long row_bytes = 0;
  long long groups = 0;
  long long group_size = 0;
};

bool checked_size(long long a, long long b, std::size_t* out) {
  if (a < 0 || b < 0) return false;
  const auto ua = static_cast<std::size_t>(a);
  const auto ub = static_cast<std::size_t>(b);
  if (ua != 0 && ub > std::numeric_limits<std::size_t>::max() / ua) {
    return false;
  }
  *out = ua * ub;
  return true;
}

bool layout_for(LowBitFormat format, long long dim, long long group_size,
                Layout* layout) {
  if (dim <= 0 || layout == nullptr) return false;
  switch (format) {
    case LowBitFormat::kInt2Row:
      layout->row_bytes = (dim + 3) / 4;
      layout->groups = 1;
      layout->group_size = dim;
      return true;
    case LowBitFormat::kInt3Group64:
      layout->groups = (dim + quant::kInt3Group - 1) / quant::kInt3Group;
      if (layout->groups > LLONG_MAX / quant::kInt3GroupBytes) return false;
      layout->row_bytes = layout->groups * quant::kInt3GroupBytes;
      layout->group_size = quant::kInt3Group;
      return true;
    case LowBitFormat::kInt4Row:
      layout->row_bytes = (dim + 1) / 2;
      layout->groups = 1;
      layout->group_size = dim;
      return true;
    case LowBitFormat::kInt4Group:
      if (group_size <= 0) return false;
      layout->row_bytes = (dim + 1) / 2;
      layout->groups = (dim + group_size - 1) / group_size;
      layout->group_size = group_size;
      return true;
  }
  return false;
}

bool finite_values(const float* values, long long count) {
  for (long long i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

int decode_code(LowBitFormat format, const std::uint8_t* row,
                long long input) {
  if (format == LowBitFormat::kInt2Row) {
    return int((row[input >> 2] >> (2 * (input & 3))) & 3) - 2;
  }
  if (format == LowBitFormat::kInt3Group64) {
    const long long group = input / quant::kInt3Group;
    const int local = static_cast<int>(input % quant::kInt3Group);
    const std::uint8_t* low =
        row + group * quant::kInt3GroupBytes;
    const std::uint8_t* high = low + 16;
    const unsigned code =
        ((low[local >> 2] >> (2 * (local & 3))) & 3u) |
        (((high[local >> 3] >> (local & 7)) & 1u) << 2);
    return static_cast<int>(code) - 4;
  }
  return int((row[input >> 1] >> (4 * (input & 1))) & 15) - 8;
}

float scale_for(const Layout& layout, const float* row_scales,
                long long input) {
  return row_scales[input / layout.group_size];
}

std::uint32_t load_u32(const std::uint8_t* bytes) {
  std::uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

float e8_value(const std::uint8_t* block, int column) {
  const int sub = column / 32;
  const int local = column & 31;
  const int sign_group = local / 8;
  const int sign_element = local & 7;
  const int grid_element = local & 3;
  const std::uint32_t word = load_u32(block + 64 + 4 * sub);
  const std::uint32_t signs = (word >> (7 * sign_group)) & 0x7fu;
  int negative = 0;
  if (sign_element < 7) {
    negative = (signs >> sign_element) & 1u;
  } else {
    negative = std::popcount(signs) & 1;
  }
  const int grid_slot = local / 4;
  const std::uint32_t grid =
      quant::iq_tables::iq3xxs_grid[block[sub * 8 + grid_slot]];
  const float magnitude =
      0.5f * float((grid >> (8 * grid_element)) & 255u);
  std::uint16_t half = 0;
  std::memcpy(&half, block + 96, sizeof(half));
  const float scale = fp16_to_fp32(half) *
                      (0.5f + float((word >> 28) & 15u)) * 0.5f;
  const float value = magnitude * scale;
  return negative ? -value : value;
}

float e8_grid_magnitude(int index, int element) {
  const std::uint32_t grid = quant::iq_tables::iq3xxs_grid[index];
  return 0.5f * static_cast<float>((grid >> (8 * element)) & 255u);
}

std::uint8_t e8_nearest_grid(const float* magnitudes) {
  int best = 0;
  float best_score = std::numeric_limits<float>::infinity();
  for (int index = 0; index < 256; ++index) {
    float norm = 0.0f;
    float dot = 0.0f;
    for (int element = 0; element < 4; ++element) {
      const float grid = e8_grid_magnitude(index, element);
      norm += grid * grid;
      dot += magnitudes[element] * grid;
    }
    const float score = norm - 2.0f * dot;
    if (score < best_score) {
      best_score = score;
      best = index;
    }
  }
  return static_cast<std::uint8_t>(best);
}

void e8_sign_bits(std::uint8_t* bits, long long n) {
  std::uint64_t state = 417u + static_cast<std::uint64_t>(n);
  for (long long i = 0; i < (n + 7) / 8; ++i) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    bits[i] = static_cast<std::uint8_t>(
        (state * UINT64_C(2685821657736338717)) >> 56);
  }
}

long long rotation_block(long long remaining) {
  const auto value = static_cast<unsigned long long>(remaining);
  long long block = static_cast<long long>(value & (~value + 1ULL));
  while (block > 32768) block >>= 1;
  return block;
}

struct Variant {
  const char* name;
  quant::LowBitGemmFn fn;
};

const Variant& lowbit_variant(LowBitFormat format) {
  static const Variant reference{"ref", &quant::lowbit_gemm_ref};
#if defined(QUIXICORE_CPU_HAVE_LOWBIT_AVX512)
  static const Variant avx512{"avx512", &quant::lowbit_gemm_avx512};
  if ((format == LowBitFormat::kInt4Row ||
       format == LowBitFormat::kInt4Group) &&
      cpu_features().avx512f) {
    return avx512;
  }
#endif
#if defined(QUIXICORE_CPU_HAVE_LOWBIT_AVX2)
  static const Variant avx2{"avx2", &quant::lowbit_gemm_avx2};
  if (cpu_features().avx2) return avx2;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  static const Variant neon{"neon", &quant::lowbit_gemm_neon};
  if (cpu_features().neon) return neon;
#endif
  (void)format;
  return reference;
}

Status validate_lowbit(LowBitFormat format, const std::uint8_t* packed,
                       const float* scales, long long rows, long long dim,
                       long long group_size, Layout* layout) {
  if (!detail::valid_product({rows, dim}) ||
      !detail::all_nonnull(packed, scales) ||
      !layout_for(format, dim, group_size, layout)) {
    return rows <= 0 || dim <= 0 || !layout_for(format, dim, group_size, layout)
               ? Status::kInvalidShape
               : Status::kInvalidArgument;
  }
  std::size_t ignored = 0;
  if (!checked_size(rows, layout->row_bytes, &ignored) ||
      !checked_size(rows, layout->groups, &ignored)) {
    return Status::kInvalidShape;
  }
  if (!finite_values(scales, rows * layout->groups)) {
    return Status::kInvalidArgument;
  }
  return Status::kOk;
}

}  // namespace

namespace quant {

void lowbit_gemm_ref(LowBitFormat format, const std::uint8_t* packed,
                     const float* scales, const float* x, float* y,
                     long long m, long long n, long long k,
                     long long group_size) {
  Layout layout;
  (void)layout_for(format, k, group_size, &layout);
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::uint8_t* weight = packed + output * layout.row_bytes;
      const float* row_scales = scales + output * layout.groups;
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        float sum = 0.0f;
        for (long long group = 0; group < layout.groups; ++group) {
          const long long first = group * layout.group_size;
          const long long last = std::min(k, first + layout.group_size);
          float partial = 0.0f;
          for (long long inner = first; inner < last; ++inner) {
            partial += input[inner] *
                       static_cast<float>(decode_code(format, weight, inner));
          }
          sum += partial * row_scales[group];
        }
        y[row * n + output] = sum;
      }
    }
  });
}

}  // namespace quant

Status lowbit_packed_size(LowBitFormat format, long long rows, long long dim,
                          long long group_size, std::size_t* weight_bytes,
                          std::size_t* scale_count) {
  if (rows <= 0 || dim <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(weight_bytes, scale_count)) {
    return Status::kInvalidArgument;
  }
  Layout layout;
  if (!layout_for(format, dim, group_size, &layout) ||
      !checked_size(rows, layout.row_bytes, weight_bytes) ||
      !checked_size(rows, layout.groups, scale_count)) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

Status lowbit_pack(LowBitFormat format, const float* weights,
                   std::uint8_t* packed, float* scales, long long rows,
                   long long dim, long long group_size) {
  Layout layout;
  if (!detail::valid_product({rows, dim}) ||
      !layout_for(format, dim, group_size, &layout)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(weights, packed, scales) ||
      !finite_values(weights, rows * dim)) {
    return Status::kInvalidArgument;
  }
  std::size_t weight_bytes = 0;
  std::size_t scale_count = 0;
  if (lowbit_packed_size(format, rows, dim, group_size, &weight_bytes,
                         &scale_count) != Status::kOk) {
    return Status::kInvalidShape;
  }
  (void)scale_count;
  std::fill_n(packed, weight_bytes, std::uint8_t{0});
  threading::parallel_ranges(rows, 8,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* source = weights + row * dim;
      std::uint8_t* destination = packed + row * layout.row_bytes;
      float* row_scales = scales + row * layout.groups;
      for (long long group = 0; group < layout.groups; ++group) {
        const long long first = group * layout.group_size;
        const long long last = std::min(dim, first + layout.group_size);
        float maximum = 0.0f;
        for (long long input = first; input < last; ++input) {
          maximum = std::max(maximum, std::fabs(source[input]));
        }
        const float divisor = format == LowBitFormat::kInt2Row
                                  ? 1.0f
                                  : (format == LowBitFormat::kInt3Group64
                                         ? 3.0f
                                         : 7.0f);
        const float scale = std::max(maximum / divisor, 1e-8f);
        row_scales[group] = scale;
        for (long long input = first; input < last; ++input) {
          int code = static_cast<int>(std::nearbyint(source[input] / scale));
          if (format == LowBitFormat::kInt2Row) {
            code = std::clamp(code, -2, 1) + 2;
            destination[input >> 2] |= static_cast<std::uint8_t>(
                code << (2 * (input & 3)));
          } else if (format == LowBitFormat::kInt3Group64) {
            code = std::clamp(code, -4, 3) + 4;
            const int local = static_cast<int>(input - first);
            std::uint8_t* low =
                destination + group * quant::kInt3GroupBytes;
            std::uint8_t* high = low + 16;
            low[local >> 2] |= static_cast<std::uint8_t>(
                (code & 3) << (2 * (local & 3)));
            high[local >> 3] |= static_cast<std::uint8_t>(
                ((code >> 2) & 1) << (local & 7));
          } else {
            code = std::clamp(code, -8, 7) + 8;
            destination[input >> 1] |= static_cast<std::uint8_t>(
                code << (4 * (input & 1)));
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status lowbit_unpack(LowBitFormat format, const std::uint8_t* packed,
                     const float* scales, float* weights, long long rows,
                     long long dim, long long group_size) {
  Layout layout;
  const Status status =
      validate_lowbit(format, packed, scales, rows, dim, group_size, &layout);
  if (status != Status::kOk) return status;
  if (weights == nullptr) return Status::kInvalidArgument;
  threading::parallel_ranges(rows, 8,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const std::uint8_t* source = packed + row * layout.row_bytes;
      const float* row_scales = scales + row * layout.groups;
      for (long long input = 0; input < dim; ++input) {
        weights[row * dim + input] =
            static_cast<float>(decode_code(format, source, input)) *
            scale_for(layout, row_scales, input);
      }
    }
  });
  return Status::kOk;
}

Status lowbit_gemm(LowBitFormat format, const std::uint8_t* packed_weights,
                   const float* weight_scales, const float* x, float* y,
                   long long m, long long n, long long k,
                   long long group_size) {
  Layout layout;
  const Status status = validate_lowbit(format, packed_weights, weight_scales,
                                        n, k, group_size, &layout);
  if (status != Status::kOk) return status;
  if (!detail::valid_product({m, k}) || !detail::valid_product({m, n})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y) || !finite_values(x, m * k)) {
    return Status::kInvalidArgument;
  }
  lowbit_variant(format).fn(format, packed_weights, weight_scales, x, y, m, n,
                            k, group_size);
  return Status::kOk;
}

Status lowbit_gemm_pair(
    LowBitFormat format, const std::uint8_t* packed_gate,
    const float* gate_scales, const std::uint8_t* packed_up,
    const float* up_scales, const float* x, float* gate, float* up,
    long long m, long long n, long long k, long long group_size) {
  if (format != LowBitFormat::kInt4Row &&
      format != LowBitFormat::kInt4Group) {
    return Status::kUnsupportedFormat;
  }
  Layout layout;
  Status status = validate_lowbit(format, packed_gate, gate_scales, n, k,
                                  group_size, &layout);
  if (status != Status::kOk) return status;
  status = validate_lowbit(format, packed_up, up_scales, n, k, group_size,
                           &layout);
  if (status != Status::kOk) return status;
  if (!detail::valid_product({m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, gate, up) || !finite_values(x, m * k)) {
    return Status::kInvalidArgument;
  }
  // A single thread-pool dispatch is the important fusion. This portable
  // inner loop is the exact fallback used when a format-specific paired SIMD
  // route is unavailable.
  threading::parallel_ranges(2 * n, 8,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const bool is_up = item >= n;
      const long long output = is_up ? item - n : item;
      const std::uint8_t* weights = is_up ? packed_up : packed_gate;
      const float* scales = is_up ? up_scales : gate_scales;
      float* destination = is_up ? up : gate;
      const std::uint8_t* weight = weights + output * layout.row_bytes;
      const float* row_scales = scales + output * layout.groups;
      for (long long row = 0; row < m; ++row) {
        float sum = 0.0f;
        for (long long group = 0; group < layout.groups; ++group) {
          const long long first = group * layout.group_size;
          const long long last = std::min(k, first + layout.group_size);
          float partial = 0.0f;
          for (long long inner = first; inner < last; ++inner) {
            partial += x[row * k + inner] *
                       static_cast<float>(decode_code(format, weight, inner));
          }
          sum += partial * row_scales[group];
        }
        destination[row * n + output] = sum;
      }
    }
  });
  return Status::kOk;
}

Status lowbit_axpy_row(LowBitFormat format,
                       const std::uint8_t* packed_weights,
                       const float* weight_scales, long long row,
                       float coefficient, float* out, long long n,
                       long long k, long long group_size) {
  Layout layout;
  const Status status = validate_lowbit(format, packed_weights, weight_scales,
                                        n, k, group_size, &layout);
  if (status != Status::kOk) return status;
  if (row < 0 || row >= n || !std::isfinite(coefficient)) {
    return Status::kInvalidArgument;
  }
  if (out == nullptr) return Status::kInvalidArgument;
  const std::uint8_t* weights = packed_weights + row * layout.row_bytes;
  const float* scales = weight_scales + row * layout.groups;
  for (long long input = 0; input < k; ++input) {
    out[input] += coefficient * scale_for(layout, scales, input) *
                  static_cast<float>(decode_code(format, weights, input));
  }
  return Status::kOk;
}

Status lowbit_gemv_rows(LowBitFormat format,
                        const std::uint8_t* packed_weights,
                        const float* weight_scales, const float* x,
                        const int* rows, float* y, long long row_count,
                        long long n, long long k, long long group_size) {
  Layout layout;
  const Status status = validate_lowbit(format, packed_weights, weight_scales,
                                        n, k, group_size, &layout);
  if (status != Status::kOk) return status;
  if (!detail::valid_product({row_count, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, rows, y) || !finite_values(x, k)) {
    return Status::kInvalidArgument;
  }
  for (long long i = 0; i < row_count; ++i) {
    if (rows[i] < 0 || rows[i] >= n) return Status::kInvalidArgument;
  }
  threading::parallel_ranges(row_count, 8,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = rows[item];
      const std::uint8_t* weights =
          packed_weights + row * layout.row_bytes;
      const float* scales = weight_scales + row * layout.groups;
      float sum = 0.0f;
      for (long long group = 0; group < layout.groups; ++group) {
        const long long first = group * layout.group_size;
        const long long last = std::min(k, first + layout.group_size);
        float partial = 0.0f;
        for (long long inner = first; inner < last; ++inner) {
          partial += x[inner] *
                     static_cast<float>(decode_code(format, weights, inner));
        }
        sum += partial * scales[group];
      }
      y[item] = sum;
    }
  });
  return Status::kOk;
}

const char* lowbit_gemm_variant(LowBitFormat format) {
  return lowbit_variant(format).name;
}

Status e8iq3_packed_size(long long rows, long long dim,
                         std::size_t* weight_bytes) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (weight_bytes == nullptr) return Status::kInvalidArgument;
  const long long blocks = (dim + quant::kE8Block - 1) / quant::kE8Block;
  std::size_t row_bytes = 0;
  if (!checked_size(blocks, quant::kE8BlockBytes, &row_bytes) ||
      !checked_size(rows, static_cast<long long>(row_bytes), weight_bytes)) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

Status e8iq3_pack(const float* weights, std::uint8_t* packed,
                  long long rows, long long dim) {
  std::size_t packed_size = 0;
  const Status status = e8iq3_packed_size(rows, dim, &packed_size);
  if (status != Status::kOk || dim % quant::kE8Block != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(weights, packed) ||
      !finite_values(weights, rows * dim)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(packed, packed_size, std::uint8_t{0});
  const long long blocks = dim / quant::kE8Block;
  const long long row_bytes = blocks * quant::kE8BlockBytes;
  threading::parallel_ranges(rows, 1,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* row_weights = weights + row * dim;
      std::uint8_t* row_packed = packed + row * row_bytes;
      for (long long block_index = 0; block_index < blocks; ++block_index) {
        const float* source =
            row_weights + block_index * quant::kE8Block;
        std::uint8_t* block =
            row_packed + block_index * quant::kE8BlockBytes;
        float magnitudes[quant::kE8Block];
        bool negative[quant::kE8Block];
        float square_sum = 0.0f;
        for (int input = 0; input < quant::kE8Block; ++input) {
          magnitudes[input] = std::fabs(source[input]);
          negative[input] = source[input] < 0.0f;
          square_sum += magnitudes[input] * magnitudes[input];
        }
        // Store seven signs per group of eight. When parity is odd, flip the
        // least-magnitude sign so the implied eighth sign closes parity at the
        // minimum reconstruction cost.
        for (int group = 0; group < quant::kE8Block / 8; ++group) {
          const int first = group * 8;
          int negatives = 0;
          int minimum = first;
          for (int element = 0; element < 8; ++element) {
            negatives += negative[first + element] ? 1 : 0;
            if (magnitudes[first + element] < magnitudes[minimum]) {
              minimum = first + element;
            }
          }
          if ((negatives & 1) != 0) negative[minimum] = !negative[minimum];
        }

        float super_scale =
            std::sqrt(square_sum / static_cast<float>(quant::kE8Block)) /
                20.0f +
            1e-12f;
        const std::uint16_t scale_half = fp32_to_fp16(super_scale);
        std::memcpy(block + 96, &scale_half, sizeof(scale_half));
        super_scale = fp16_to_fp32(scale_half);

        for (int sub = 0; sub < quant::kE8Block / 32; ++sub) {
          float best_error = std::numeric_limits<float>::infinity();
          std::uint8_t best_indices[8] = {};
          int best_code = 0;
          for (int code = 0; code < 16; ++code) {
            const float sub_scale =
                super_scale * (0.5f + static_cast<float>(code)) * 0.5f;
            const float inverse =
                1.0f / std::max(sub_scale, 1e-20f);
            float error = 0.0f;
            std::uint8_t indices[8];
            for (int group = 0; group < 8; ++group) {
              float normalized[4];
              for (int element = 0; element < 4; ++element) {
                normalized[element] =
                    magnitudes[sub * 32 + group * 4 + element] * inverse;
              }
              indices[group] = e8_nearest_grid(normalized);
              for (int element = 0; element < 4; ++element) {
                const float reconstructed =
                    e8_grid_magnitude(indices[group], element) * sub_scale;
                const float delta = reconstructed -
                    magnitudes[sub * 32 + group * 4 + element];
                error += delta * delta;
              }
            }
            if (error < best_error) {
              best_error = error;
              best_code = code;
              std::copy_n(indices, 8, best_indices);
            }
          }
          std::copy_n(best_indices, 8, block + sub * 8);
          std::uint32_t word = static_cast<std::uint32_t>(best_code) << 28;
          for (int sign_group = 0; sign_group < 4; ++sign_group) {
            std::uint32_t seven = 0;
            for (int element = 0; element < 7; ++element) {
              if (negative[sub * 32 + sign_group * 8 + element]) {
                seven |= UINT32_C(1) << element;
              }
            }
            word |= seven << (7 * sign_group);
          }
          std::memcpy(block + 64 + 4 * sub, &word, sizeof(word));
        }
      }
    }
  });
  return Status::kOk;
}

Status e8iq3_unpack(const std::uint8_t* packed, float* weights,
                    long long rows, long long dim) {
  std::size_t ignored = 0;
  const Status status = e8iq3_packed_size(rows, dim, &ignored);
  if (status != Status::kOk) return status;
  if (!detail::all_nonnull(packed, weights)) return Status::kInvalidArgument;
  const long long blocks = (dim + quant::kE8Block - 1) / quant::kE8Block;
  const long long row_bytes = blocks * quant::kE8BlockBytes;
  threading::parallel_ranges(rows, 8,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const std::uint8_t* source = packed + row * row_bytes;
      for (long long input = 0; input < dim; ++input) {
        weights[row * dim + input] =
            e8_value(source + (input / quant::kE8Block) *
                                  quant::kE8BlockBytes,
                     static_cast<int>(input % quant::kE8Block));
      }
    }
  });
  return Status::kOk;
}

Status e8iq3_gemm(const std::uint8_t* packed_weights, const float* x,
                  float* y, long long m, long long n, long long k) {
  std::size_t ignored = 0;
  const Status status = e8iq3_packed_size(n, k, &ignored);
  if (status != Status::kOk || !detail::valid_product({m, k}) ||
      !detail::valid_product({m, n})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, x, y) ||
      !finite_values(x, m * k)) {
    return Status::kInvalidArgument;
  }
  const long long blocks = (k + quant::kE8Block - 1) / quant::kE8Block;
  const long long row_bytes = blocks * quant::kE8BlockBytes;
  threading::parallel_ranges(n, 8, [&](long long begin, long long end, int) {
    for (long long output = begin; output < end; ++output) {
      const std::uint8_t* weights =
          packed_weights + output * row_bytes;
      for (long long row = 0; row < m; ++row) {
        const float* input = x + row * k;
        float sum = 0.0f;
        for (long long block = 0; block < blocks; ++block) {
          const long long first = block * quant::kE8Block;
          const long long last = std::min(k, first + quant::kE8Block);
          const std::uint8_t* source =
              weights + block * quant::kE8BlockBytes;
          for (long long inner = first; inner < last; ++inner) {
            sum += input[inner] *
                   e8_value(source, static_cast<int>(inner - first));
          }
        }
        y[row * n + output] = sum;
      }
    }
  });
  return Status::kOk;
}

Status e8iq3_rotate(const float* x, float* y, long long rows, long long dim,
                    bool inverse) {
  if (!detail::valid_product({rows, dim})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, y) || !finite_values(x, rows * dim)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 1,
                             [&](long long begin, long long end, int) {
    std::vector<float> buffer(static_cast<std::size_t>(dim));
    std::vector<std::uint8_t> signs(4096);
    for (long long row = begin; row < end; ++row) {
      std::copy_n(x + row * dim, dim, buffer.data());
      long long offset = 0;
      while (offset < dim) {
        const long long block = rotation_block(dim - offset);
        e8_sign_bits(signs.data(), block);
        if (!inverse) {
          for (long long i = 0; i < block; ++i) {
            if ((signs[static_cast<std::size_t>(i >> 3)] >> (i & 7)) & 1) {
              buffer[static_cast<std::size_t>(offset + i)] *= -1.0f;
            }
          }
        }
        for (long long stride = 1; stride < block; stride <<= 1) {
          for (long long base = 0; base < block; base += 2 * stride) {
            for (long long i = 0; i < stride; ++i) {
              const std::size_t a =
                  static_cast<std::size_t>(offset + base + i);
              const std::size_t b = static_cast<std::size_t>(
                  offset + base + stride + i);
              const float first = buffer[a];
              const float second = buffer[b];
              buffer[a] = first + second;
              buffer[b] = first - second;
            }
          }
        }
        const float normalization =
            1.0f / std::sqrt(static_cast<float>(block));
        for (long long i = 0; i < block; ++i) {
          float& value = buffer[static_cast<std::size_t>(offset + i)];
          value *= normalization;
          if (inverse &&
              ((signs[static_cast<std::size_t>(i >> 3)] >> (i & 7)) & 1)) {
            value *= -1.0f;
          }
        }
        offset += block;
      }
      std::copy_n(buffer.data(), dim, y + row * dim);
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
