#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status moe_route_topk(const float* router_logits, int* expert_ids,
                      float* expert_weights, long long tokens,
                      long long experts, int top_k) {
  if (!detail::valid_product({tokens, experts}) || experts > INT_MAX ||
      top_k <= 0 || top_k > experts) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(router_logits, expert_ids, expert_weights)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    const float* row = router_logits + token * experts;
    std::vector<int> ids(static_cast<std::size_t>(experts));
    std::iota(ids.begin(), ids.end(), 0);
    for (long long expert = 0; expert < experts; ++expert) {
      if (std::isnan(row[expert]) ||
          row[expert] == std::numeric_limits<float>::infinity()) {
        return Status::kInvalidArgument;
      }
    }
    std::partial_sort(ids.begin(), ids.begin() + top_k, ids.end(),
                      [&](int lhs, int rhs) {
      if (row[lhs] == row[rhs]) {
        return lhs < rhs;
      }
      return row[lhs] > row[rhs];
    });
    const float maximum = row[ids.front()];
    double sum = 0.0;
    for (int rank = 0; rank < top_k; ++rank) {
      const long long output = token * top_k + rank;
      expert_ids[output] = ids[static_cast<std::size_t>(rank)];
      const double value = std::isfinite(maximum)
                               ? std::exp(static_cast<double>(
                                     row[expert_ids[output]] - maximum))
                               : (rank == 0 ? 1.0 : 0.0);
      expert_weights[output] = static_cast<float>(value);
      sum += value;
    }
    const float inverse = static_cast<float>(1.0 / sum);
    for (int rank = 0; rank < top_k; ++rank) {
      expert_weights[token * top_k + rank] *= inverse;
    }
  }
  return Status::kOk;
}

Status grouped_gemm(const float* a, const float* b, float* c,
                    long long groups, long long m, long long n, long long k) {
  if (!detail::valid_product({groups, m, k}) ||
      !detail::valid_product({groups, k, n}) ||
      !detail::valid_product({groups, m, n})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, c)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(groups * m, 1,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long group = item / m;
      const long long row = item % m;
      const float* a_row = a + (group * m + row) * k;
      const float* b_group = b + group * k * n;
      float* c_row = c + (group * m + row) * n;
      for (long long column = 0; column < n; ++column) {
        double sum = 0.0;
        for (long long inner = 0; inner < k; ++inner) {
          sum += static_cast<double>(a_row[inner]) *
                 b_group[inner * n + column];
        }
        c_row[column] = static_cast<float>(sum);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
