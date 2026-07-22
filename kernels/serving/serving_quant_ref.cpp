#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kernels/common/validation.h"
#include "kernels/quantization/gguf_ref.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool packed_row_info(QuantFormat format, long long dim, long long* block_size,
                     std::size_t* block_bytes, std::size_t* row_bytes) {
  if (!quant::gguf_format_info(format, block_size, block_bytes) || dim <= 0 ||
      dim % *block_size != 0) {
    return false;
  }
  const std::size_t blocks = static_cast<std::size_t>(dim / *block_size);
  if (blocks > static_cast<std::size_t>(-1) / *block_bytes) return false;
  *row_bytes = blocks * *block_bytes;
  return true;
}

float packed_value(const std::uint8_t* row, QuantFormat format,
                   long long column, long long block_size,
                   std::size_t block_bytes) {
  const long long block = column / block_size;
  return quant::gguf_dequant_element(
      format, row + static_cast<std::size_t>(block) * block_bytes,
      static_cast<int>(column % block_size));
}

}  // namespace

Status quantized_embedding(const void* table, const int* ids,
                           const float* add, float* out, long long vocab,
                           long long count, long long dim,
                           QuantFormat quant_format, float scale,
                           bool use_add) {
  if (!detail::valid_product({vocab, count, dim}) || !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(table, ids, out) || (use_add && add == nullptr)) {
    return Status::kInvalidArgument;
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  std::size_t row_bytes = 0;
  if (!packed_row_info(quant_format, dim, &block_size, &block_bytes,
                       &row_bytes)) {
    return Status::kUnsupportedFormat;
  }
  for (long long i = 0; i < count; ++i) {
    if (ids[i] < 0 || ids[i] >= vocab) return Status::kInvalidArgument;
  }
  const auto* bytes = static_cast<const std::uint8_t*>(table);
  threading::parallel_ranges(count, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const std::uint8_t* row = bytes +
          static_cast<std::size_t>(ids[item]) * row_bytes;
      for (long long column = 0; column < dim; ++column) {
        const long long index = item * dim + column;
        out[index] = scale * packed_value(row, quant_format, column,
                                          block_size, block_bytes) +
                     (use_add ? add[index] : 0.0f);
      }
    }
  });
  return Status::kOk;
}

Status quantized_embedding_bag(const void* table, const int* ids,
                               const long long* offsets,
                               const float* sample_weights, float* out,
                               long long vocab, long long id_count,
                               long long bags, long long dim,
                               QuantFormat quant_format, float scale,
                               bool use_weights, bool mean_mode) {
  if (!detail::valid_product({vocab, id_count, bags, dim}) ||
      !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(table, ids, offsets, out) ||
      (use_weights && sample_weights == nullptr)) {
    return Status::kInvalidArgument;
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  std::size_t row_bytes = 0;
  if (!packed_row_info(quant_format, dim, &block_size, &block_bytes,
                       &row_bytes)) {
    return Status::kUnsupportedFormat;
  }
  if (offsets[0] != 0 || offsets[bags] != id_count) {
    return Status::kInvalidArgument;
  }
  for (long long bag = 0; bag < bags; ++bag) {
    if (offsets[bag] > offsets[bag + 1]) return Status::kInvalidArgument;
  }
  for (long long i = 0; i < id_count; ++i) {
    if (ids[i] < 0 || ids[i] >= vocab ||
        (use_weights && !std::isfinite(sample_weights[i]))) {
      return Status::kInvalidArgument;
    }
  }
  const auto* bytes = static_cast<const std::uint8_t*>(table);
  threading::parallel_ranges(bags, 8, [&](long long begin, long long end, int) {
    for (long long bag = begin; bag < end; ++bag) {
      const long long start = offsets[bag];
      const long long stop = offsets[bag + 1];
      const float divisor = mean_mode && stop > start
                                ? 1.0f / static_cast<float>(stop - start)
                                : 1.0f;
      for (long long column = 0; column < dim; ++column) {
        double sum = 0.0;
        for (long long item = start; item < stop; ++item) {
          const std::uint8_t* row = bytes +
              static_cast<std::size_t>(ids[item]) * row_bytes;
          sum += packed_value(row, quant_format, column, block_size,
                              block_bytes) *
                 (use_weights ? sample_weights[item] : 1.0f);
        }
        out[bag * dim + column] = static_cast<float>(scale * divisor * sum);
      }
    }
  });
  return Status::kOk;
}

Status indexer_k_quant_and_cache(const float* keys, const int* slot_mapping,
                                 std::uint8_t* code_cache,
                                 float* scale_cache, long long tokens,
                                 long long slots, long long head_dim,
                                 long long quant_block_size,
                                 bool power_of_two_scale) {
  if (!detail::valid_product({tokens, slots, head_dim}) ||
      quant_block_size <= 0 || head_dim % quant_block_size != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(keys, slot_mapping, code_cache, scale_cache)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (slot_mapping[token] >= slots) return Status::kInvalidArgument;
  }
  const long long groups = head_dim / quant_block_size;
  for (long long token = 0; token < tokens; ++token) {
    const int slot = slot_mapping[token];
    if (slot < 0) continue;
    const Status status = quantize_float8(
        keys + token * head_dim, code_cache + slot * head_dim,
        scale_cache + slot * groups, 1, head_dim, quant_block_size,
        Float8Format::kE4M3FN, power_of_two_scale);
    if (status != Status::kOk) return status;
  }
  return Status::kOk;
}

Status indexer_k_gather(const std::uint8_t* code_cache,
                        const float* scale_cache, const int* slots_to_gather,
                        float* out, long long cache_slots, long long count,
                        long long head_dim, long long quant_block_size) {
  if (!detail::valid_product({cache_slots, count, head_dim}) ||
      quant_block_size <= 0 || head_dim % quant_block_size != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(code_cache, scale_cache, slots_to_gather, out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < count; ++row) {
    if (slots_to_gather[row] < 0 || slots_to_gather[row] >= cache_slots) {
      return Status::kInvalidArgument;
    }
  }
  const long long groups = head_dim / quant_block_size;
  threading::parallel_ranges(count, 32,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const long long slot = slots_to_gather[row];
      (void)dequantize_float8(code_cache + slot * head_dim,
                              scale_cache + slot * groups,
                              out + row * head_dim, 1, head_dim,
                              quant_block_size, Float8Format::kE4M3FN);
    }
  });
  return Status::kOk;
}

Status minference_block_mask(const int* vertical_indices,
                             const int* slash_offsets,
                             const int* context_lengths, int* block_mask,
                             long long batch, long long heads,
                             long long vertical_width,
                             long long slash_width, long long max_blocks,
                             long long block_size, long long vertical_top_k,
                             long long slash_top_k,
                             long long last_n_blocks) {
  if (!detail::valid_product({batch, heads, max_blocks, block_size}) ||
      vertical_width < 0 || slash_width < 0 || vertical_top_k < 0 ||
      slash_top_k < 0 || last_n_blocks < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(vertical_indices, slash_offsets, context_lengths,
                           block_mask)) {
    return Status::kInvalidArgument;
  }
  for (long long item = 0; item < batch * heads; ++item) {
    const long long request = item / heads;
    const int context = context_lengths[request];
    if (context < 0) return Status::kInvalidArgument;
    int* mask = block_mask + item * max_blocks;
    std::fill_n(mask, max_blocks, 0);
    if (context == 0) continue;
    const long long blocks = std::min(
        (static_cast<long long>(context) + block_size - 1) / block_size,
        max_blocks);
    const long long recent = std::min(last_n_blocks, blocks);
    for (long long i = 0; i < recent; ++i) mask[blocks - 1 - i] = 1;
    const long long vertical_count = std::min(vertical_top_k, vertical_width);
    for (long long i = 0; i < vertical_count; ++i) {
      const int column = vertical_indices[item * vertical_width + i];
      if (column >= 0 && column < context) mask[column / block_size] = 1;
    }
    const long long slash_count = std::min(slash_top_k, slash_width);
    for (long long i = 0; i < slash_count; ++i) {
      const int offset = slash_offsets[item * slash_width + i];
      if (offset >= 0 && offset < context) {
        mask[(context - 1 - offset) / block_size] = 1;
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
