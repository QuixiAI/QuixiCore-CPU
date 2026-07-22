#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status embedding_backward(const int* ids, const float* grad_out,
                          float* grad_table, long long vocab, long long count,
                          long long dim) {
  if (!detail::valid_product({vocab, dim}) ||
      !detail::valid_product({count, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(ids, grad_out, grad_table)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_table, vocab * dim, 0.0f);
  // Serial token order gives deterministic scatter-add behavior for repeated
  // ids and is the CPU oracle for atomic/sorted sibling variants.
  for (long long token = 0; token < count; ++token) {
    const int id = ids[token];
    if (id < 0 || id >= vocab) continue;
    for (long long feature = 0; feature < dim; ++feature) {
      grad_table[static_cast<long long>(id) * dim + feature] +=
          grad_out[token * dim + feature];
    }
  }
  return Status::kOk;
}

Status embedding_backward_sorted(const int* sorted_ids,
                                  const int* permutation,
                                  const float* grad_out, float* grad_table,
                                  long long vocab, long long count,
                                  long long dim, float scale) {
  if (!detail::valid_product({vocab, dim}) ||
      !detail::valid_product({count, dim}) || !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(sorted_ids, permutation, grad_out, grad_table)) {
    return Status::kInvalidArgument;
  }
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(count), 0);
  for (long long token = 0; token < count; ++token) {
    if (token > 0 && sorted_ids[token] < sorted_ids[token - 1]) {
      return Status::kInvalidArgument;
    }
    if (permutation[token] < 0 || permutation[token] >= count ||
        seen[static_cast<std::size_t>(permutation[token])] != 0) {
      return Status::kInvalidArgument;
    }
    seen[static_cast<std::size_t>(permutation[token])] = 1;
  }
  std::fill_n(grad_table, vocab * dim, 0.0f);
  for (long long token = 0; token < count; ++token) {
    const int id = sorted_ids[token];
    if (id < 0 || id >= vocab) continue;
    const long long source = permutation[token];
    for (long long feature = 0; feature < dim; ++feature) {
      grad_table[static_cast<long long>(id) * dim + feature] +=
          scale * grad_out[source * dim + feature];
    }
  }
  return Status::kOk;
}

Status build_multimodal_source_map(
    const int* span_offsets, const int* span_lengths,
    const int* modal_starts, int* source_map, long long spans,
    long long text_tokens) {
  if (spans < 0 || text_tokens <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(span_offsets, span_lengths, modal_starts,
                           source_map)) {
    return Status::kInvalidArgument;
  }
  for (long long span = 0; span < spans; ++span) {
    if (span_offsets[span] < 0 || span_lengths[span] < 0 ||
        modal_starts[span] < 0 || span_offsets[span] > text_tokens ||
        span_lengths[span] > text_tokens - span_offsets[span]) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(text_tokens, 16,
                             [&](long long begin, long long end, int) {
    for (long long token = begin; token < end; ++token) {
      int source = -1;
      for (long long span = 0; span < spans; ++span) {
        const long long offset = token - span_offsets[span];
        if (offset >= 0 && offset < span_lengths[span]) {
          source = static_cast<int>(modal_starts[span] + offset);
          break;
        }
      }
      source_map[token] = source;
    }
  });
  return Status::kOk;
}

Status merge_multimodal_spans(const float* text, const float* modal,
                              const int* source_map, float* out,
                              long long text_tokens, long long modal_tokens,
                              long long dim) {
  if (!detail::valid_product({text_tokens, dim}) || modal_tokens < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(text, modal, source_map, out)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < text_tokens; ++token) {
    if (source_map[token] < -1 || source_map[token] >= modal_tokens) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(text_tokens, 8,
                             [&](long long begin, long long end, int) {
    for (long long token = begin; token < end; ++token) {
      const float* source = source_map[token] < 0
                                ? text + token * dim
                                : modal +
                                      static_cast<long long>(source_map[token]) *
                                          dim;
      std::copy_n(source, dim, out + token * dim);
    }
  });
  return Status::kOk;
}

Status mean_pool_rms_l2(const float* x, const float* weight,
                        const int* lengths, float* out, long long batch,
                        long long sequence, long long hidden, float eps) {
  if (!detail::valid_product({batch, sequence, hidden}) ||
      !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight, out)) return Status::kInvalidArgument;
  if (lengths != nullptr) {
    for (long long item = 0; item < batch; ++item) {
      if (lengths[item] <= 0 || lengths[item] > sequence) {
        return Status::kInvalidArgument;
      }
    }
  }
  threading::parallel_ranges(batch, 1,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long length = lengths == nullptr ? sequence : lengths[item];
      float* destination = out + item * hidden;
      double mean_square = 0.0;
      for (long long feature = 0; feature < hidden; ++feature) {
        double pooled = 0.0;
        for (long long token = 0; token < length; ++token) {
          pooled += x[(item * sequence + token) * hidden + feature];
        }
        destination[feature] = static_cast<float>(pooled / length);
        mean_square += double(destination[feature]) * destination[feature];
      }
      const double rms_inverse =
          1.0 / std::sqrt(mean_square / hidden + eps);
      double l2_square = 0.0;
      for (long long feature = 0; feature < hidden; ++feature) {
        destination[feature] = static_cast<float>(
            destination[feature] * rms_inverse * weight[feature]);
        l2_square += double(destination[feature]) * destination[feature];
      }
      const double l2_inverse =
          1.0 / std::sqrt(l2_square + 1.0e-12);
      for (long long feature = 0; feature < hidden; ++feature) {
        destination[feature] =
            static_cast<float>(destination[feature] * l2_inverse);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
