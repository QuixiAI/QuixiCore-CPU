#include "quixicore_cpu/ops.h"

#include <algorithm>

#include "kernels/common/validation.h"

namespace quixicore_cpu {

Status varlen_build_worklist(const int* cumulative_lengths,
                             int* padded_offsets, int* sequence_lengths,
                             int* tile_sequence, int* tile_local_start,
                             int* tile_count, long long batch,
                             long long max_tiles, int tile_rows) {
  if (batch <= 0 || max_tiles <= 0 || tile_rows <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(cumulative_lengths, padded_offsets,
                           sequence_lengths, tile_sequence,
                           tile_local_start, tile_count)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_lengths[0] != 0) return Status::kInvalidArgument;
  padded_offsets[0] = 0;
  long long tiles = 0;
  for (long long request = 0; request < batch; ++request) {
    const int length = cumulative_lengths[request + 1] -
                       cumulative_lengths[request];
    if (length < 0) return Status::kInvalidArgument;
    sequence_lengths[request] = length;
    const int padded = (length + tile_rows - 1) / tile_rows * tile_rows;
    padded_offsets[request + 1] = padded_offsets[request] + padded;
    for (int local = 0; local < length; local += tile_rows) {
      if (tiles >= max_tiles) return Status::kInvalidShape;
      tile_sequence[tiles] = static_cast<int>(request);
      tile_local_start[tiles] = local;
      ++tiles;
    }
  }
  std::fill(tile_sequence + tiles, tile_sequence + max_tiles, -1);
  std::fill(tile_local_start + tiles, tile_local_start + max_tiles, -1);
  *tile_count = static_cast<int>(tiles);
  return Status::kOk;
}

Status varlen_pad_q(const float* packed, const int* cumulative_lengths,
                    const int* padded_offsets, float* padded_head_major,
                    long long batch, long long heads, long long head_dim,
                    long long total_tokens, long long total_padded) {
  if (!detail::valid_product({batch, heads, head_dim, total_tokens,
                              total_padded})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed, cumulative_lengths, padded_offsets,
                           padded_head_major)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_lengths[0] != 0 || cumulative_lengths[batch] != total_tokens ||
      padded_offsets[0] != 0 || padded_offsets[batch] != total_padded) {
    return Status::kInvalidArgument;
  }
  std::fill_n(padded_head_major, heads * total_padded * head_dim, 0.0f);
  for (long long request = 0; request < batch; ++request) {
    const int length = cumulative_lengths[request + 1] -
                       cumulative_lengths[request];
    if (length < 0 || padded_offsets[request + 1] -
                            padded_offsets[request] < length) {
      return Status::kInvalidArgument;
    }
    for (int local = 0; local < length; ++local) {
      const long long token = cumulative_lengths[request] + local;
      const long long padded = padded_offsets[request] + local;
      for (long long head = 0; head < heads; ++head) {
        std::copy_n(packed + (token * heads + head) * head_dim, head_dim,
                    padded_head_major +
                        (head * total_padded + padded) * head_dim);
      }
    }
  }
  return Status::kOk;
}

Status varlen_regather_o(const float* padded_head_major,
                         const int* cumulative_lengths,
                         const int* padded_offsets, float* packed,
                         long long batch, long long heads,
                         long long head_dim, long long total_tokens,
                         long long total_padded) {
  if (!detail::valid_product({batch, heads, head_dim, total_tokens,
                              total_padded})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(padded_head_major, cumulative_lengths,
                           padded_offsets, packed)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_lengths[0] != 0 || cumulative_lengths[batch] != total_tokens ||
      padded_offsets[0] != 0 || padded_offsets[batch] != total_padded) {
    return Status::kInvalidArgument;
  }
  for (long long request = 0; request < batch; ++request) {
    const int length = cumulative_lengths[request + 1] -
                       cumulative_lengths[request];
    if (length < 0) return Status::kInvalidArgument;
    for (int local = 0; local < length; ++local) {
      const long long token = cumulative_lengths[request] + local;
      const long long padded = padded_offsets[request] + local;
      for (long long head = 0; head < heads; ++head) {
        std::copy_n(padded_head_major +
                        (head * total_padded + padded) * head_dim,
                    head_dim, packed + (token * heads + head) * head_dim);
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
