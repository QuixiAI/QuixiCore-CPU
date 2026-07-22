#include "quixicore_cpu/ops.h"

#include <algorithm>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status embedding_lookup(const float* table, const int* ids, float* out,
                        long long vocab, long long count, long long dim) {
  if (!detail::valid_product({vocab, dim}) ||
      !detail::valid_product({count, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(table, ids, out)) {
    return Status::kInvalidArgument;
  }
  for (long long item = 0; item < count; ++item) {
    if (ids[item] < 0 || ids[item] >= vocab) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(count, 8,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      std::copy_n(table + static_cast<long long>(ids[item]) * dim, dim,
                  out + item * dim);
    }
  });
  return Status::kOk;
}

Status kv_cache_scatter(float* cache, const float* src, const int* slots,
                        long long max_slots, long long count,
                        long long row_width) {
  if (!detail::valid_product({max_slots, row_width}) ||
      !detail::valid_product({count, row_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(cache, src, slots)) {
    return Status::kInvalidArgument;
  }
  for (long long item = 0; item < count; ++item) {
    if (slots[item] >= max_slots) {
      return Status::kInvalidArgument;
    }
  }
  // Keep scatter serial: duplicate slots have deterministic last-writer-wins
  // semantics, matching the sibling serving paths.
  for (long long item = 0; item < count; ++item) {
    if (slots[item] < 0) {
      continue;
    }
    std::copy_n(src + item * row_width, row_width,
                cache + static_cast<long long>(slots[item]) * row_width);
  }
  return Status::kOk;
}

Status kv_cache_gather(const float* cache, const int* indices, float* out,
                       long long max_slots, long long count,
                       long long row_width) {
  if (!detail::valid_product({max_slots, row_width}) ||
      !detail::valid_product({count, row_width})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(cache, indices, out)) {
    return Status::kInvalidArgument;
  }
  for (long long item = 0; item < count; ++item) {
    if (indices[item] < 0 || indices[item] >= max_slots) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(count, 8,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      std::copy_n(cache + static_cast<long long>(indices[item]) * row_width,
                  row_width, out + item * row_width);
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
