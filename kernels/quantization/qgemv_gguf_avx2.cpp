#include "kernels/quantization/gguf_ref.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernels/common/fp16.h"
#include "kernels/quantization/iq_tables.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

float horizontal(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

float dot_i8x16_f32(__m128i weights, const float* input) {
  const __m256i low = _mm256_cvtepi8_epi32(weights);
  const __m256i high = _mm256_cvtepi8_epi32(_mm_srli_si128(weights, 8));
  const __m256 sum0 = _mm256_mul_ps(_mm256_cvtepi32_ps(low),
                                    _mm256_loadu_ps(input));
  const __m256 sum1 = _mm256_mul_ps(_mm256_cvtepi32_ps(high),
                                    _mm256_loadu_ps(input + 8));
  return horizontal(_mm256_add_ps(sum0, sum1));
}

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

float sum_f32x16(const float* input) {
  return horizontal(_mm256_add_ps(_mm256_loadu_ps(input),
                                  _mm256_loadu_ps(input + 8)));
}

void scale_min_k4(int index, const std::uint8_t* scales, int* scale,
                  int* minimum) {
  if (index < 4) {
    *scale = scales[index] & 63;
    *minimum = scales[index + 4] & 63;
  } else {
    *scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
    *minimum = (scales[index + 4] >> 4) |
               ((scales[index] >> 6) << 4);
  }
}

bool direct_block_dot(QuantFormat format, const std::uint8_t* block,
                      const float* input, float* result) {
  float total = 0.0f;
  if (format == QuantFormat::kQ4_1) {
    const __m128i codes =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 4));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i low = _mm_and_si128(codes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
    *result = half_at(block) *
                  (dot_i8x16_f32(low, input) +
                   dot_i8x16_f32(high, input + 16)) +
              half_at(block + 2) *
                  (sum_f32x16(input) + sum_f32x16(input + 16));
    return true;
  }
  if (format == QuantFormat::kQ5_0 || format == QuantFormat::kQ5_1) {
    const bool affine = format == QuantFormat::kQ5_1;
    const std::uint32_t high_bits =
        std::uint32_t(u16_at(block + (affine ? 4 : 2))) |
        (std::uint32_t(u16_at(block + (affine ? 6 : 4))) << 16);
    const std::uint8_t* low = block + (affine ? 8 : 6);
    alignas(16) std::int8_t q0[16], q1[16];
    for (int lane = 0; lane < 16; ++lane) {
      int v0 = (low[lane] & 15) | (((high_bits >> lane) & 1) << 4);
      int v1 = (low[lane] >> 4) |
               (((high_bits >> (16 + lane)) & 1) << 4);
      q0[lane] = static_cast<std::int8_t>(affine ? v0 : v0 - 16);
      q1[lane] = static_cast<std::int8_t>(affine ? v1 : v1 - 16);
    }
    total = half_at(block) *
            (dot_i8x16_f32(
                 _mm_load_si128(reinterpret_cast<const __m128i*>(q0)), input) +
             dot_i8x16_f32(
                 _mm_load_si128(reinterpret_cast<const __m128i*>(q1)),
                 input + 16));
    if (affine) {
      total += half_at(block + 2) *
               (sum_f32x16(input) + sum_f32x16(input + 16));
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kQ2_K) {
    const std::uint8_t* scales = block;
    const std::uint8_t* quants = block + 16;
    const float scale_base = half_at(block + 80);
    const float minimum_base = half_at(block + 82);
    alignas(16) std::int8_t decoded[16];
    for (int chunk = 0; chunk < 2; ++chunk) {
      for (int scale_index = 0; scale_index < 4; ++scale_index) {
        for (int sub = 0; sub < 2; ++sub) {
          const int index = chunk * 8 + scale_index * 2 + sub;
          for (int lane = 0; lane < 16; ++lane) {
            decoded[lane] = static_cast<std::int8_t>(
                (quants[chunk * 32 + sub * 16 + lane] >>
                 (2 * scale_index)) & 3);
          }
          const float* local_input =
              input + chunk * 128 + scale_index * 32 + sub * 16;
          total += scale_base * static_cast<float>(scales[index] & 15) *
                   dot_i8x16_f32(
                       _mm_load_si128(
                           reinterpret_cast<const __m128i*>(decoded)),
                       local_input);
          total -= minimum_base * static_cast<float>(scales[index] >> 4) *
                   sum_f32x16(local_input);
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
    alignas(16) std::int8_t decoded[16];
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
          for (int lane = 0; lane < 16; ++lane) {
            const int low =
                (quants[chunk * 32 + sub * 16 + lane] >>
                 (2 * scale_index)) & 3;
            const int high =
                (high_mask[sub * 16 + lane] &
                 (1 << (chunk * 4 + scale_index))) != 0;
            decoded[lane] =
                static_cast<std::int8_t>((low | (high << 2)) - 4);
          }
          total += scale_base * static_cast<float>(local_scale - 32) *
                   dot_i8x16_f32(
                       _mm_load_si128(
                           reinterpret_cast<const __m128i*>(decoded)),
                       input + chunk * 128 + scale_index * 32 + sub * 16);
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
    alignas(32) std::int8_t decoded[32];
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
        decoded[lane] = static_cast<std::int8_t>(quant);
      }
      const float* local_input = input + 32 * sub;
      const float quant_dot =
          dot_i8x16_f32(
              _mm_load_si128(reinterpret_cast<const __m128i*>(decoded)),
              local_input) +
          dot_i8x16_f32(
              _mm_load_si128(reinterpret_cast<const __m128i*>(decoded + 16)),
              local_input + 16);
      total += scale_base * static_cast<float>(scale) * quant_dot;
      total -= minimum_base * static_cast<float>(minimum) *
               (sum_f32x16(local_input) + sum_f32x16(local_input + 16));
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kQ6_K) {
    const std::uint8_t* low = block;
    const std::uint8_t* high = block + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
    const float scale_base = half_at(block + 208);
    alignas(16) std::int8_t decoded[16];
    for (int chunk = 0; chunk < 2; ++chunk) {
      for (int group = 0; group < 4; ++group) {
        for (int half = 0; half < 2; ++half) {
          for (int lane = 0; lane < 16; ++lane) {
            const int full_lane = 16 * half + lane;
            const int low_byte =
                low[chunk * 64 + full_lane + 32 * (group & 1)];
            const int nibble =
                group & 2 ? (low_byte >> 4) : (low_byte & 15);
            const int high_bits =
                (high[chunk * 32 + full_lane] >> (2 * group)) & 3;
            decoded[lane] = static_cast<std::int8_t>(
                (nibble | (high_bits << 4)) - 32);
          }
          const int scale_index = chunk * 8 + half + group * 2;
          total += scale_base * static_cast<float>(scales[scale_index]) *
                   dot_i8x16_f32(
                       _mm_load_si128(
                           reinterpret_cast<const __m128i*>(decoded)),
                       input + chunk * 128 + group * 32 + half * 16);
        }
      }
    }
    *result = total;
    return true;
  }
  if (format == QuantFormat::kIQ4_NL || format == QuantFormat::kIQ4_XS) {
    static constexpr std::int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1,    13,   25,  38,  53,  69,  89,  113};
    alignas(32) std::int8_t decoded[32];
    const int groups = format == QuantFormat::kIQ4_NL ? 1 : 8;
    const std::uint8_t* quants =
        block + (format == QuantFormat::kIQ4_NL ? 2 : 8);
    const float scale_base = half_at(block);
    const std::uint16_t scales_high =
        format == QuantFormat::kIQ4_XS ? u16_at(block + 2) : 0;
    for (int group = 0; group < groups; ++group) {
      for (int lane = 0; lane < 16; ++lane) {
        const std::uint8_t code = quants[16 * group + lane];
        decoded[lane] = values[code & 15];
        decoded[16 + lane] = values[code >> 4];
      }
      float scale = scale_base;
      if (format == QuantFormat::kIQ4_XS) {
        const int low_scale =
            (block[4 + group / 2] >> (4 * (group & 1))) & 15;
        const int high_scale = (scales_high >> (2 * group)) & 3;
        scale *= static_cast<float>((low_scale | (high_scale << 4)) - 32);
      }
      const float* local_input = input + 32 * group;
      total += scale *
               (dot_i8x16_f32(
                    _mm_load_si128(
                        reinterpret_cast<const __m128i*>(decoded)),
                    local_input) +
                dot_i8x16_f32(
                    _mm_load_si128(
                        reinterpret_cast<const __m128i*>(decoded + 16)),
                    local_input + 16));
    }
    *result = total;
    return true;
  }
  return false;
}

void q4_rows(const BlockQ4_0* packed, const float* x, float* y,
             long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i offset = _mm_set1_epi8(8);
    for (long long row = begin; row < end; ++row) {
      float total = 0.0f;
      const BlockQ4_0* row_weights = packed + row * blocks;
      for (long long block = 0; block < blocks; ++block) {
        const __m128i codes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(row_weights[block].qs));
        const __m128i low = _mm_sub_epi8(_mm_and_si128(codes, mask), offset);
        const __m128i high =
            _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(codes, 4), mask), offset);
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

void gguf_gemv_avx2(QuantFormat format, const void* packed, const float* x,
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
      double total = 0.0;
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
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int column = 0;
        for (; column + 15 < block_size; column += 16) {
          acc0 = _mm256_fmadd_ps(_mm256_load_ps(decoded + column),
                                 _mm256_loadu_ps(input + column), acc0);
          acc1 = _mm256_fmadd_ps(_mm256_load_ps(decoded + column + 8),
                                 _mm256_loadu_ps(input + column + 8), acc1);
        }
        float block_total = horizontal(_mm256_add_ps(acc0, acc1));
        for (; column < block_size; ++column) {
          block_total += decoded[column] * input[column];
        }
        total += block_total;
      }
      y[row] = static_cast<float>(total);
    }
  });
}

}  // namespace quixicore_cpu::quant

#endif
