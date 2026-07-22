#include "quixicore_cpu/qgemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/common/validation.h"
#include "src/memory/workspace_internal.h"

namespace quixicore_cpu {

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
