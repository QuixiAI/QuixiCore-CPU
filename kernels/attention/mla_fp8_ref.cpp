#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common/validation.h"
#include "quixicore_cpu/quantization.h"

namespace quixicore_cpu {
namespace {

float mla_code(const std::uint8_t* codes, const float* scales,
               long long vector, long long item, long long width,
               long long group_size, Float8Format format) {
  const long long groups = (width + group_size - 1) / group_size;
  return float8_decode(codes[vector * width + item], format) *
         scales[vector * groups + item / group_size];
}

template <bool Sparse>
Status mla_fp8_impl(
    const float* q, const std::uint8_t* codes, const float* scales,
    const int* block_table, const int* lengths_or_indices,
    const int* topk_lengths, float* out, long long cache_blocks,
    long long batch, long long heads, long long width, long long page_size,
    long long max_blocks, long long max_topk, long long group_size,
    Float8Format format, float score_scale) {
  if (!detail::valid_product({cache_blocks, batch, heads, width, page_size,
                              max_blocks, group_size}) ||
      width % group_size != 0 || (Sparse && max_topk <= 0) ||
      !std::isfinite(score_scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, codes, scales, block_table,
                           lengths_or_indices, out) ||
      (Sparse && topk_lengths == nullptr)) {
    return Status::kInvalidArgument;
  }
  const float factor = score_scale == 0.0f
                           ? 1.0f / std::sqrt(static_cast<float>(width))
                           : score_scale;
  for (long long request = 0; request < batch; ++request) {
    const int count = Sparse ? topk_lengths[request] : lengths_or_indices[request];
    const long long capacity = Sparse ? max_topk : max_blocks * page_size;
    if (count < 0 || count > capacity) return Status::kInvalidArgument;
    for (long long head = 0; head < heads; ++head) {
      const float* query = q + (request * heads + head) * width;
      float* destination = out + (request * heads + head) * width;
      std::fill_n(destination, width, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (int item_index = 0; item_index < count; ++item_index) {
        const int position =
            Sparse ? lengths_or_indices[request * max_topk + item_index]
                   : item_index;
        if (position < 0 || position >= max_blocks * page_size) {
          return Status::kInvalidArgument;
        }
        const int block =
            block_table[request * max_blocks + position / page_size];
        if (block < 0 || block >= cache_blocks) return Status::kInvalidArgument;
        const long long vector =
            static_cast<long long>(block) * page_size + position % page_size;
        double score = 0.0;
        for (long long dim = 0; dim < width; ++dim) {
          score += query[dim] *
                   mla_code(codes, scales, vector, dim, width, group_size,
                            format);
        }
        score *= factor;
        if (score > maximum) {
          const double old_weight = std::exp(maximum - score);
          denominator = denominator * old_weight + 1.0;
          for (long long dim = 0; dim < width; ++dim) {
            destination[dim] = static_cast<float>(
                destination[dim] * old_weight +
                mla_code(codes, scales, vector, dim, width, group_size,
                         format));
          }
          maximum = score;
        } else {
          const double weight = std::exp(score - maximum);
          denominator += weight;
          for (long long dim = 0; dim < width; ++dim) {
            destination[dim] += static_cast<float>(
                mla_code(codes, scales, vector, dim, width, group_size,
                         format) * weight);
          }
        }
      }
      if (denominator > 0.0) {
        for (long long dim = 0; dim < width; ++dim) {
          destination[dim] =
              static_cast<float>(destination[dim] / denominator);
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace

Status mla_kv_insert_fp8(
    const float* kv, const int* slot_mapping, std::uint8_t* code_cache,
    float* scale_cache, long long tokens, long long slots, long long width,
    long long group_size, Float8Format format, bool power_of_two_scale) {
  if (!detail::valid_product({tokens, slots, width, group_size})) {
    return Status::kInvalidShape;
  }
  if (width % group_size != 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(kv, slot_mapping, code_cache, scale_cache)) {
    return Status::kInvalidArgument;
  }
  const long long groups = width / group_size;
  for (long long token = 0; token < tokens; ++token) {
    const int slot = slot_mapping[token];
    if (slot < 0) continue;
    if (slot >= slots) return Status::kInvalidArgument;
    Status status = quantize_float8(
        kv + token * width,
        code_cache + static_cast<long long>(slot) * width,
        scale_cache + static_cast<long long>(slot) * groups, 1, width,
        group_size, format, power_of_two_scale);
    if (status != Status::kOk) return status;
  }
  return Status::kOk;
}

Status mla_decode_fp8(
    const float* q, const std::uint8_t* code_cache,
    const float* scale_cache, const int* block_table,
    const int* context_lengths, float* out, long long cache_blocks,
    long long batch, long long heads, long long width, long long page_size,
    long long max_blocks, long long group_size, Float8Format format,
    float scale) {
  return mla_fp8_impl<false>(
      q, code_cache, scale_cache, block_table, context_lengths, nullptr, out,
      cache_blocks, batch, heads, width, page_size, max_blocks, 1, group_size,
      format, scale);
}

Status mla_decode_fp8_sparse(
    const float* q, const std::uint8_t* code_cache,
    const float* scale_cache, const int* block_table,
    const int* token_indices, const int* topk_lengths, float* out,
    long long cache_blocks, long long batch, long long heads,
    long long width, long long page_size, long long max_blocks,
    long long max_topk, long long group_size, Float8Format format,
    float scale) {
  return mla_fp8_impl<true>(
      q, code_cache, scale_cache, block_table, token_indices, topk_lengths,
      out, cache_blocks, batch, heads, width, page_size, max_blocks, max_topk,
      group_size, format, scale);
}

}  // namespace quixicore_cpu
