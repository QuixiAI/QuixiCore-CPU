#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "kernels/quantization/gguf_ref.h"
#include "quixicore_cpu/qgemv.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

struct PackedMatrix {
  QuantFormat format;
  const std::uint8_t* bytes;
  long long block_size;
  std::size_t block_bytes;
  std::size_t row_bytes;

  float at(long long row, long long column) const {
    const long long block = column / block_size;
    return quant::gguf_dequant_element(
        format, bytes + static_cast<std::size_t>(row) * row_bytes +
                    static_cast<std::size_t>(block) * block_bytes,
        static_cast<int>(column % block_size));
  }
};

template <bool Sparse>
Status mla_absorb_impl(
    QuantFormat format, const void* packed_kv_b, const float* q,
    const float* latent_cache, const float* rope_cache,
    const int* block_table, const int* lengths_or_indices,
    const int* topk_lengths, float* out, long long cache_blocks,
    long long batch, long long heads, long long latent_dim,
    long long nope_dim, long long rope_dim, long long value_dim,
    long long page_size, long long max_blocks, long long max_topk,
    float scale) {
  if (!detail::valid_product({cache_blocks, batch, heads, latent_dim,
                              nope_dim, value_dim, page_size, max_blocks}) ||
      rope_dim < 0 ||
      rope_dim > std::numeric_limits<long long>::max() - nope_dim ||
      value_dim > std::numeric_limits<long long>::max() - nope_dim ||
      (Sparse && max_topk <= 0) ||
      !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  const long long query_dim = nope_dim + rope_dim;
  const long long weight_width = nope_dim + value_dim;
  if (!detail::valid_product({heads, query_dim}) ||
      !detail::valid_product({heads, weight_width})) {
    return Status::kInvalidShape;
  }
  const long long weight_rows = heads * weight_width;
  if (!detail::all_nonnull(packed_kv_b, q, latent_cache, block_table,
                           lengths_or_indices, out) ||
      (rope_dim != 0 && rope_cache == nullptr) ||
      (Sparse && topk_lengths == nullptr)) {
    return Status::kInvalidArgument;
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  if (latent_dim % block_size != 0) return Status::kInvalidShape;
  std::size_t packed_size = 0;
  const Status packed_status =
      qgemv_packed_size(format, weight_rows, latent_dim, &packed_size);
  if (packed_status != Status::kOk) return packed_status;
  (void)packed_size;

  for (long long request = 0; request < batch; ++request) {
    const int count = Sparse ? topk_lengths[request]
                             : lengths_or_indices[request];
    const long long capacity = Sparse ? max_topk : max_blocks * page_size;
    if (count <= 0 || count > capacity) return Status::kInvalidArgument;
    for (int item = 0; item < count; ++item) {
      const int position =
          Sparse ? lengths_or_indices[request * max_topk + item] : item;
      if (position < 0 || position >= max_blocks * page_size) {
        return Status::kInvalidArgument;
      }
      const int block =
          block_table[request * max_blocks + position / page_size];
      if (block < 0 || block >= cache_blocks) {
        return Status::kInvalidArgument;
      }
    }
  }

  const PackedMatrix matrix{
      format,
      static_cast<const std::uint8_t*>(packed_kv_b),
      block_size,
      block_bytes,
      static_cast<std::size_t>(latent_dim / block_size) * block_bytes};
  const float score_scale =
      scale == 0.0f
          ? 1.0f / std::sqrt(static_cast<float>(query_dim))
          : scale;
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<float> query_latent;
    thread_local std::vector<double> scores;
    thread_local std::vector<float> mixed_latent;
    query_latent.resize(static_cast<std::size_t>(latent_dim));
    mixed_latent.resize(static_cast<std::size_t>(latent_dim));
    for (long long item = begin; item < end; ++item) {
      const long long request = item / heads;
      const long long head = item % heads;
      const int count = Sparse ? topk_lengths[request]
                               : lengths_or_indices[request];
      scores.resize(static_cast<std::size_t>(count));
      std::fill(query_latent.begin(), query_latent.end(), 0.0f);
      const float* query = q + item * query_dim;
      const long long row_base = head * (nope_dim + value_dim);
      for (long long dim = 0; dim < nope_dim; ++dim) {
        const float coefficient = query[dim];
        for (long long latent = 0; latent < latent_dim; ++latent) {
          query_latent[static_cast<std::size_t>(latent)] +=
              coefficient * matrix.at(row_base + dim, latent);
        }
      }

      double maximum = -std::numeric_limits<double>::infinity();
      for (int selected = 0; selected < count; ++selected) {
        const int position =
            Sparse ? lengths_or_indices[request * max_topk + selected]
                   : selected;
        const int block =
            block_table[request * max_blocks + position / page_size];
        const long long cache_row =
            static_cast<long long>(block) * page_size + position % page_size;
        const float* latent = latent_cache + cache_row * latent_dim;
        double score = 0.0;
        for (long long dim = 0; dim < latent_dim; ++dim) {
          score += static_cast<double>(query_latent[dim]) * latent[dim];
        }
        if (rope_dim != 0) {
          const float* rope = rope_cache + cache_row * rope_dim;
          for (long long dim = 0; dim < rope_dim; ++dim) {
            score += static_cast<double>(query[nope_dim + dim]) * rope[dim];
          }
        }
        score *= score_scale;
        scores[static_cast<std::size_t>(selected)] = score;
        maximum = std::max(maximum, score);
      }
      double denominator = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      std::fill(mixed_latent.begin(), mixed_latent.end(), 0.0f);
      for (int selected = 0; selected < count; ++selected) {
        const int position =
            Sparse ? lengths_or_indices[request * max_topk + selected]
                   : selected;
        const int block =
            block_table[request * max_blocks + position / page_size];
        const long long cache_row =
            static_cast<long long>(block) * page_size + position % page_size;
        const float* latent = latent_cache + cache_row * latent_dim;
        const float probability = static_cast<float>(
            scores[static_cast<std::size_t>(selected)] / denominator);
        for (long long dim = 0; dim < latent_dim; ++dim) {
          mixed_latent[static_cast<std::size_t>(dim)] +=
              probability * latent[dim];
        }
      }
      float* destination = out + item * value_dim;
      for (long long value = 0; value < value_dim; ++value) {
        float sum = 0.0f;
        const long long weight_row = row_base + nope_dim + value;
        for (long long latent = 0; latent < latent_dim; ++latent) {
          sum += matrix.at(weight_row, latent) * mixed_latent[latent];
        }
        destination[value] = sum;
      }
    }
  });
  return Status::kOk;
}

}  // namespace

Status quantized_mla_decode_absorbed(
    QuantFormat format, const void* packed_kv_b, const float* q,
    const float* latent_cache, const float* rope_cache,
    const int* block_table, const int* context_lengths, float* out,
    long long cache_blocks, long long batch, long long heads,
    long long latent_dim, long long nope_dim, long long rope_dim,
    long long value_dim, long long page_size, long long max_blocks,
    float scale) {
  return mla_absorb_impl<false>(
      format, packed_kv_b, q, latent_cache, rope_cache, block_table,
      context_lengths, nullptr, out, cache_blocks, batch, heads, latent_dim,
      nope_dim, rope_dim, value_dim, page_size, max_blocks, 1, scale);
}

Status quantized_mla_decode_absorbed_sparse(
    QuantFormat format, const void* packed_kv_b, const float* q,
    const float* latent_cache, const float* rope_cache,
    const int* block_table, const int* token_indices,
    const int* topk_lengths, float* out, long long cache_blocks,
    long long batch, long long heads, long long latent_dim,
    long long nope_dim, long long rope_dim, long long value_dim,
    long long page_size, long long max_blocks, long long max_topk,
    float scale) {
  return mla_absorb_impl<true>(
      format, packed_kv_b, q, latent_cache, rope_cache, block_table,
      token_indices, topk_lengths, out, cache_blocks, batch, heads, latent_dim,
      nope_dim, rope_dim, value_dim, page_size, max_blocks, max_topk, scale);
}

}  // namespace quixicore_cpu
