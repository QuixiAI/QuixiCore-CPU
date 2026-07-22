#include "quixicore_cpu/qgemv.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "kernels/quantization/gguf_ref.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float dot_decoded(const float* weights, const float* x, long long count) {
#if defined(__aarch64__) || defined(_M_ARM64)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  long long item = 0;
  for (; item + 7 < count; item += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(weights + item),
                     vld1q_f32(x + item));
    acc1 = vfmaq_f32(acc1, vld1q_f32(weights + item + 4),
                     vld1q_f32(x + item + 4));
  }
  float total = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
  float total = 0.0f;
  long long item = 0;
#endif
  for (; item < count; ++item) total += weights[item] * x[item];
  return total;
}

#if defined(__aarch64__) || defined(_M_ARM64)
float dot_i8x16_f32(int8x16_t weights, const float* input) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(weights));
  const int16x8_t hi = vmovl_s8(vget_high_s8(weights));
  float32x4_t sum = vmulq_f32(
      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(input));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                  vld1q_f32(input + 4));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                  vld1q_f32(input + 8));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                  vld1q_f32(input + 12));
  return vaddvq_f32(sum);
}
#endif

float q4_0_row_dot(const quant::BlockQ4_0* row, const float* x,
                   long long blocks) {
  float total = 0.0f;
  for (long long block = 0; block < blocks; ++block) {
    const float* input = x + block * quant::kQ4_0BlockSize;
#if defined(__aarch64__) || defined(_M_ARM64)
    const uint8x16_t codes = vld1q_u8(row[block].qs);
    const uint8x16_t mask = vdupq_n_u8(15);
    const uint8x16_t offset = vdupq_n_u8(8);
    const int8x16_t low = vreinterpretq_s8_u8(
        vsubq_u8(vandq_u8(codes, mask), offset));
    const int8x16_t high = vreinterpretq_s8_u8(
        vsubq_u8(vshrq_n_u8(codes, 4), offset));
    const float dot =
        dot_i8x16_f32(low, input) + dot_i8x16_f32(high, input + 16);
#else
    float dot = 0.0f;
    for (int item = 0; item < 16; ++item) {
      dot += static_cast<float>((row[block].qs[item] & 15) - 8) * input[item];
      dot += static_cast<float>((row[block].qs[item] >> 4) - 8) *
             input[item + 16];
    }
#endif
    total += fp16_to_fp32(row[block].d) * dot;
  }
  return total;
}

float q8_0_row_dot(const quant::BlockQ8_0* row, const float* x,
                   long long blocks) {
  float total = 0.0f;
  for (long long block = 0; block < blocks; ++block) {
    const float* input = x + block * quant::kQ8_0BlockSize;
#if defined(__aarch64__) || defined(_M_ARM64)
    const float dot0 = dot_i8x16_f32(vld1q_s8(row[block].qs), input);
    const float dot1 = dot_i8x16_f32(vld1q_s8(row[block].qs + 16), input + 16);
    total += fp16_to_fp32(row[block].d) * (dot0 + dot1);
#else
    float dot = 0.0f;
    for (int item = 0; item < quant::kQ8_0BlockSize; ++item) {
      dot += static_cast<float>(row[block].qs[item]) * input[item];
    }
    total += fp16_to_fp32(row[block].d) * dot;
#endif
  }
  return total;
}

float packed_row_dot(QuantFormat format, const std::uint8_t* row,
                     const float* x, long long blocks, long long block_size,
                     std::size_t block_bytes) {
  if (format == QuantFormat::kQ4_0) {
    return q4_0_row_dot(reinterpret_cast<const quant::BlockQ4_0*>(row), x,
                        blocks);
  }
  if (format == QuantFormat::kQ8_0) {
    return q8_0_row_dot(reinterpret_cast<const quant::BlockQ8_0*>(row), x,
                        blocks);
  }
  alignas(64) float decoded[256];
  float total = 0.0f;
  for (long long block = 0; block < blocks; ++block) {
    quant::gguf_dequant_block_ref(format, row + block * block_bytes, decoded);
    total += dot_decoded(decoded, x + block * block_size, block_size);
  }
  return total;
}

bool fused_layout(QuantFormat format, long long n, long long k,
                  long long* blocks, long long* block_size,
                  std::size_t* block_bytes, std::size_t* row_bytes) {
  std::size_t total_bytes = 0;
  if (qgemv_packed_size(format, n, k, &total_bytes) != Status::kOk ||
      !quant::gguf_format_info(format, block_size, block_bytes)) {
    return false;
  }
  *blocks = k / *block_size;
  *row_bytes = total_bytes / static_cast<std::size_t>(n);
  return true;
}

float activate(float value, bool gelu_tanh) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return gelu_tanh
             ? 0.5f * value *
                   (1.0f + std::tanh(kSqrt2OverPi *
                                     (value + 0.044715f * value * value * value)))
             : value / (1.0f + std::exp(-value));
}

}  // namespace

Status qgemv_up_gate(QuantFormat format, const void* packed_up,
                     const void* packed_gate, const float* x, float* up,
                     float* gate, long long n, long long k) {
  if (!detail::valid_product({n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_up, packed_gate, x, up, gate)) {
    return Status::kInvalidArgument;
  }
  long long blocks = 0;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  std::size_t row_bytes = 0;
  if (!fused_layout(format, n, k, &blocks, &block_size, &block_bytes,
                    &row_bytes)) {
    return qgemv_variant(format)[0] == 'u' ? Status::kUnsupportedFormat
                                           : Status::kInvalidShape;
  }
  if (format == QuantFormat::kQ4_0) {
    const auto* up_blocks =
        static_cast<const quant::BlockQ4_0*>(packed_up);
    const auto* gate_blocks =
        static_cast<const quant::BlockQ4_0*>(packed_gate);
    threading::parallel_ranges(n, 24,
                               [&](long long begin, long long end, int) {
      for (long long row = begin; row < end; ++row) {
        up[row] = q4_0_row_dot(up_blocks + row * blocks, x, blocks);
        gate[row] = q4_0_row_dot(gate_blocks + row * blocks, x, blocks);
      }
    });
    return Status::kOk;
  }
  const auto* up_bytes = static_cast<const std::uint8_t*>(packed_up);
  const auto* gate_bytes = static_cast<const std::uint8_t*>(packed_gate);
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      up[row] = packed_row_dot(format, up_bytes + row * row_bytes, x, blocks,
                               block_size, block_bytes);
      gate[row] = packed_row_dot(format, gate_bytes + row * row_bytes, x,
                                 blocks, block_size, block_bytes);
    }
  });
  return Status::kOk;
}

Status qgemv_up_gate_activation(QuantFormat format, const void* packed_up,
                                const void* packed_gate, const float* x,
                                float* out, long long n, long long k,
                                bool gelu_tanh) {
  if (!detail::valid_product({n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_up, packed_gate, x, out)) {
    return Status::kInvalidArgument;
  }
  long long blocks = 0;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  std::size_t row_bytes = 0;
  if (!fused_layout(format, n, k, &blocks, &block_size, &block_bytes,
                    &row_bytes)) {
    return qgemv_variant(format)[0] == 'u' ? Status::kUnsupportedFormat
                                           : Status::kInvalidShape;
  }
  if (format == QuantFormat::kQ4_0) {
    const auto* up_blocks =
        static_cast<const quant::BlockQ4_0*>(packed_up);
    const auto* gate_blocks =
        static_cast<const quant::BlockQ4_0*>(packed_gate);
    threading::parallel_ranges(n, 24,
                               [&](long long begin, long long end, int) {
      for (long long row = begin; row < end; ++row) {
        const float up =
            q4_0_row_dot(up_blocks + row * blocks, x, blocks);
        const float gate =
            q4_0_row_dot(gate_blocks + row * blocks, x, blocks);
        out[row] = activate(gate, gelu_tanh) * up;
      }
    });
    return Status::kOk;
  }
  const auto* up_bytes = static_cast<const std::uint8_t*>(packed_up);
  const auto* gate_bytes = static_cast<const std::uint8_t*>(packed_gate);
  threading::parallel_ranges(n, 24, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float up = packed_row_dot(format, up_bytes + row * row_bytes, x,
                                      blocks, block_size, block_bytes);
      const float gate = packed_row_dot(format,
                                        gate_bytes + row * row_bytes, x,
                                        blocks, block_size, block_bytes);
      out[row] = activate(gate, gelu_tanh) * up;
    }
  });
  return Status::kOk;
}

Status qgemv_qkv(QuantFormat format, const void* packed_q,
                 const void* packed_k, const void* packed_v, const float* x,
                 float* q, float* k_out, float* v_out, long long query_dim,
                 long long kv_dim, long long input_dim) {
  if (!detail::valid_product({query_dim, kv_dim, input_dim})) {
    return Status::kInvalidShape;
  }
  if (kv_dim > (std::numeric_limits<long long>::max() - query_dim) / 2) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_q, packed_k, packed_v, x, q, k_out, v_out)) {
    return Status::kInvalidArgument;
  }

  long long blocks = 0;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  std::size_t q_row_bytes = 0;
  if (!fused_layout(format, query_dim, input_dim, &blocks, &block_size,
                    &block_bytes, &q_row_bytes)) {
    return qgemv_variant(format)[0] == 'u' ? Status::kUnsupportedFormat
                                           : Status::kInvalidShape;
  }
  long long kv_blocks = 0;
  long long kv_block_size = 0;
  std::size_t kv_block_bytes = 0;
  std::size_t kv_row_bytes = 0;
  if (!fused_layout(format, kv_dim, input_dim, &kv_blocks, &kv_block_size,
                    &kv_block_bytes, &kv_row_bytes)) {
    return Status::kInvalidShape;
  }
  const auto* q_bytes = static_cast<const std::uint8_t*>(packed_q);
  const auto* k_bytes = static_cast<const std::uint8_t*>(packed_k);
  const auto* v_bytes = static_cast<const std::uint8_t*>(packed_v);
  const long long total_rows = query_dim + 2 * kv_dim;
  threading::parallel_ranges(total_rows, 24,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const bool query = item < query_dim;
      const bool key = !query && item < query_dim + kv_dim;
      const long long row = query ? item
                                  : (key ? item - query_dim
                                         : item - query_dim - kv_dim);
      const std::uint8_t* matrix =
          query ? q_bytes : (key ? k_bytes : v_bytes);
      float* output = query ? q : (key ? k_out : v_out);
      const std::size_t stride = query ? q_row_bytes : kv_row_bytes;
      output[row] = packed_row_dot(format, matrix + row * stride, x, blocks,
                                   block_size, block_bytes);
    }
  });
  return Status::kOk;
}

Status qgemv_qkv_rope_kv(
    QuantFormat format, const void* packed_q, const void* packed_k,
    const void* packed_v, const float* x, const float* cosine,
    const float* sine, float* q, float* key_cache, float* value_cache,
    long long query_heads, long long kv_heads, long long head_dim,
    long long input_dim, long long slots, long long max_position, int position,
    int slot) {
  if (!detail::valid_product({query_heads, head_dim, input_dim}) ||
      !detail::valid_product({slots, kv_heads, head_dim}) ||
      head_dim % 2 != 0 || position < 0 || position >= max_position ||
      slot < -1 || slot >= slots) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_q, packed_k, packed_v, x, cosine, sine, q,
                           key_cache, value_cache)) {
    return Status::kInvalidArgument;
  }

  const long long query_dim = query_heads * head_dim;
  const long long kv_dim = kv_heads * head_dim;
  long long blocks = 0;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  std::size_t q_row_bytes = 0;
  if (!fused_layout(format, query_dim, input_dim, &blocks, &block_size,
                    &block_bytes, &q_row_bytes)) {
    return qgemv_variant(format)[0] == 'u' ? Status::kUnsupportedFormat
                                           : Status::kInvalidShape;
  }
  long long kv_blocks = 0;
  long long kv_block_size = 0;
  std::size_t kv_block_bytes = 0;
  std::size_t kv_row_bytes = 0;
  if (!fused_layout(format, kv_dim, input_dim, &kv_blocks, &kv_block_size,
                    &kv_block_bytes, &kv_row_bytes) ||
      kv_blocks != blocks || kv_block_size != block_size ||
      kv_block_bytes != block_bytes) {
    return Status::kInvalidShape;
  }

  const auto* q_bytes = static_cast<const std::uint8_t*>(packed_q);
  const auto* k_bytes = static_cast<const std::uint8_t*>(packed_k);
  const auto* v_bytes = static_cast<const std::uint8_t*>(packed_v);
  const long long half = head_dim / 2;
  const long long query_pairs = query_heads * half;
  const long long key_pairs = slot >= 0 ? kv_heads * half : 0;
  const long long value_rows = slot >= 0 ? kv_dim : 0;
  if (query_pairs > std::numeric_limits<long long>::max() - key_pairs ||
      query_pairs + key_pairs >
          std::numeric_limits<long long>::max() - value_rows) {
    return Status::kInvalidShape;
  }
  const long long total_items = query_pairs + key_pairs + value_rows;
  const float* cos_row = cosine + static_cast<long long>(position) * half;
  const float* sin_row = sine + static_cast<long long>(position) * half;
  const long long cache_base = static_cast<long long>(slot) * kv_dim;

  threading::parallel_ranges(total_items, 24,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      if (item < query_pairs) {
        const long long head = item / half;
        const long long dim = item - head * half;
        const long long row0 = head * head_dim + dim;
        const long long row1 = row0 + half;
        const float first = packed_row_dot(
            format, q_bytes + row0 * q_row_bytes, x, blocks, block_size,
            block_bytes);
        const float second = packed_row_dot(
            format, q_bytes + row1 * q_row_bytes, x, blocks, block_size,
            block_bytes);
        q[row0] = first * cos_row[dim] - second * sin_row[dim];
        q[row1] = second * cos_row[dim] + first * sin_row[dim];
        continue;
      }

      const long long kv_item = item - query_pairs;
      if (kv_item < key_pairs) {
        const long long head = kv_item / half;
        const long long dim = kv_item - head * half;
        const long long row0 = head * head_dim + dim;
        const long long row1 = row0 + half;
        const float first = packed_row_dot(
            format, k_bytes + row0 * kv_row_bytes, x, blocks, block_size,
            block_bytes);
        const float second = packed_row_dot(
            format, k_bytes + row1 * kv_row_bytes, x, blocks, block_size,
            block_bytes);
        key_cache[cache_base + row0] =
            first * cos_row[dim] - second * sin_row[dim];
        key_cache[cache_base + row1] =
            second * cos_row[dim] + first * sin_row[dim];
        continue;
      }

      const long long row = kv_item - key_pairs;
      value_cache[cache_base + row] = packed_row_dot(
          format, v_bytes + row * kv_row_bytes, x, blocks, block_size,
          block_bytes);
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
