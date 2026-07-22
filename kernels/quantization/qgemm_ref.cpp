#include "quixicore_cpu/qgemm.h"

#include <cmath>
#include <vector>

#include "kernels/common/validation.h"

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
  std::vector<float> logits(static_cast<std::size_t>(vocab));
  for (long long row = 0; row < rows; ++row) {
    const Status status =
        qgemv(format, packed_weights, hidden_states + row * hidden,
              logits.data(), vocab, hidden);
    if (status != Status::kOk) {
      return status;
    }
    int best = 0;
    for (long long token = 1; token < vocab; ++token) {
      if (logits[static_cast<std::size_t>(token)] >
          logits[static_cast<std::size_t>(best)]) {
        best = static_cast<int>(token);
      }
    }
    token_ids[row] = best;
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
