#include "quixicore_cpu/qgemm.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/threading.h"
#include "src/memory/workspace_internal.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

void q4_argmax_multi_dot(const quant::BlockQ4_0* weights,
                         const float* hidden_states, long long rows,
                         long long hidden, long long blocks, float* totals) {
  std::fill_n(totals, rows, 0.0f);
  for (long long block = 0; block < blocks; ++block) {
    const float scale = fp16_to_fp32(weights[block].d);
#if defined(__aarch64__) || defined(_M_ARM64)
    const uint8x16_t codes = vld1q_u8(weights[block].qs);
    const uint8x16_t mask = vdupq_n_u8(15);
    const uint8x16_t offset = vdupq_n_u8(8);
    const int8x16_t low = vreinterpretq_s8_u8(
        vsubq_u8(vandq_u8(codes, mask), offset));
    const int8x16_t high = vreinterpretq_s8_u8(
        vsubq_u8(vshrq_n_u8(codes, 4), offset));
    const int16x8_t low16 = vmovl_s8(vget_low_s8(low));
    const int16x8_t low16_high = vmovl_s8(vget_high_s8(low));
    const int16x8_t high16 = vmovl_s8(vget_low_s8(high));
    const int16x8_t high16_high = vmovl_s8(vget_high_s8(high));
    const float32x4_t weight0 =
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(low16)));
    const float32x4_t weight1 =
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(low16)));
    const float32x4_t weight2 =
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(low16_high)));
    const float32x4_t weight3 =
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(low16_high)));
    const float32x4_t weight4 =
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(high16)));
    const float32x4_t weight5 =
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(high16)));
    const float32x4_t weight6 =
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(high16_high)));
    const float32x4_t weight7 =
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(high16_high)));
    for (long long row = 0; row < rows; ++row) {
      const float* values = hidden_states + row * hidden +
                            block * quant::kQ4_0BlockSize;
      float32x4_t low_sum = vmulq_f32(weight0, vld1q_f32(values));
      low_sum = vfmaq_f32(low_sum, weight1, vld1q_f32(values + 4));
      low_sum = vfmaq_f32(low_sum, weight2, vld1q_f32(values + 8));
      low_sum = vfmaq_f32(low_sum, weight3, vld1q_f32(values + 12));
      float32x4_t high_sum = vmulq_f32(weight4, vld1q_f32(values + 16));
      high_sum = vfmaq_f32(high_sum, weight5, vld1q_f32(values + 20));
      high_sum = vfmaq_f32(high_sum, weight6, vld1q_f32(values + 24));
      high_sum = vfmaq_f32(high_sum, weight7, vld1q_f32(values + 28));
      totals[row] += scale *
                     (vaddvq_f32(low_sum) + vaddvq_f32(high_sum));
    }
#else
    for (long long row = 0; row < rows; ++row) {
      const float* values = hidden_states + row * hidden +
                            block * quant::kQ4_0BlockSize;
      float dot = 0.0f;
      for (int item = 0; item < 16; ++item) {
        dot += static_cast<float>((weights[block].qs[item] & 15) - 8) *
               values[item];
        dot += static_cast<float>((weights[block].qs[item] >> 4) - 8) *
               values[item + 16];
      }
      totals[row] += scale * dot;
    }
#endif
  }
}

Status q4_argmax(const quant::BlockQ4_0* packed, const float* hidden_states,
                 int* token_ids, long long rows, long long vocab,
                 long long hidden) {
  const std::size_t workers = static_cast<std::size_t>(num_threads());
  detail::WorkspaceFrame workspace;
  float* best_logits = workspace.allocate<float>(workers * rows);
  int* best_ids = workspace.allocate<int>(workers * rows);
  if (best_logits == nullptr || best_ids == nullptr) {
    return Status::kOutOfMemory;
  }
  std::fill_n(best_logits, workers * rows,
              -std::numeric_limits<float>::infinity());
  std::fill_n(best_ids, workers * rows, INT_MAX);
  const long long blocks = hidden / quant::kQ4_0BlockSize;
  threading::parallel_ranges(vocab, 32,
                             [&](long long begin, long long end, int worker) {
    thread_local std::vector<float> token_logits;
    token_logits.resize(static_cast<std::size_t>(rows));
    float* worker_logits = best_logits + static_cast<long long>(worker) * rows;
    int* worker_ids = best_ids + static_cast<long long>(worker) * rows;
    for (long long token = begin; token < end; ++token) {
      const quant::BlockQ4_0* weight_row = packed + token * blocks;
      q4_argmax_multi_dot(weight_row, hidden_states, rows, hidden, blocks,
                          token_logits.data());
      for (long long row = 0; row < rows; ++row) {
        const float value = token_logits[static_cast<std::size_t>(row)];
        if (value > worker_logits[row] ||
            (value == worker_logits[row] && token < worker_ids[row])) {
          worker_logits[row] = value;
          worker_ids[row] = static_cast<int>(token);
        }
      }
    }
  });
  for (long long row = 0; row < rows; ++row) {
    float best = -std::numeric_limits<float>::infinity();
    int best_id = INT_MAX;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      const std::size_t index = worker * static_cast<std::size_t>(rows) + row;
      if (best_logits[index] > best ||
          (best_logits[index] == best && best_ids[index] < best_id)) {
        best = best_logits[index];
        best_id = best_ids[index];
      }
    }
    token_ids[row] = best_id;
  }
  return Status::kOk;
}

}  // namespace

Status qgemm(QuantFormat format, const void* packed_weights, const float* x,
             float* y, long long m, long long n, long long k) {
  if (!detail::valid_product({m, k}) || !detail::valid_product({m, n})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, x, y)) {
    return Status::kInvalidArgument;
  }
  std::size_t packed_size = 0;
  const Status packed_status =
      qgemv_packed_size(format, n, k, &packed_size);
  if (packed_status != Status::kOk) {
    return packed_status;
  }
  (void)packed_size;
  for (long long row = 0; row < m; ++row) {
    const Status status =
        qgemv(format, packed_weights, x + row * k, y + row * n, n, k);
    if (status != Status::kOk) {
      return status;
    }
  }
  return Status::kOk;
}

Status quantized_lm_head_argmax(QuantFormat format, const void* packed_weights,
                                const float* hidden_states, int* token_ids,
                                long long rows, long long vocab,
                                long long hidden) {
  if (!detail::valid_product({rows, hidden}) ||
      !detail::valid_product({vocab, hidden}) || vocab > INT_MAX) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, hidden_states, token_ids)) {
    return Status::kInvalidArgument;
  }
  std::size_t packed_bytes = 0;
  const Status packed_status =
      qgemv_packed_size(format, vocab, hidden, &packed_bytes);
  if (packed_status != Status::kOk) return packed_status;
  if (format == QuantFormat::kQ4_0) {
    return q4_argmax(static_cast<const quant::BlockQ4_0*>(packed_weights),
                     hidden_states, token_ids, rows, vocab, hidden);
  }
  const std::size_t row_bytes = packed_bytes / static_cast<std::size_t>(vocab);
  constexpr long long kVocabularyTile = 4096;
  detail::WorkspaceFrame workspace;
  float* logits = workspace.allocate<float>(kVocabularyTile);
  if (logits == nullptr) return Status::kOutOfMemory;
  const auto* packed = static_cast<const std::uint8_t*>(packed_weights);
  for (long long row = 0; row < rows; ++row) {
    int best = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (long long token_base = 0; token_base < vocab;
         token_base += kVocabularyTile) {
      const long long tile =
          std::min(kVocabularyTile, vocab - token_base);
      const Status status = qgemv(
          format, packed + static_cast<std::size_t>(token_base) * row_bytes,
          hidden_states + row * hidden, logits, tile, hidden);
      if (status != Status::kOk) return status;
      for (long long local = 0; local < tile; ++local) {
        if (logits[local] > best_logit) {
          best_logit = logits[local];
          best = static_cast<int>(token_base + local);
        }
      }
    }
    token_ids[row] = best;
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
