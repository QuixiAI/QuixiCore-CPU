#include "quixicore_cpu/ops.h"

#include <cmath>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status linear_attention(const float* q, const float* k, const float* v,
                        float* out, long long heads, long long sequence,
                        long long dim, float eps) {
  if (!detail::valid_product({heads, sequence, dim}) ||
      !detail::valid_product({dim, dim}) || !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(heads, 1,
                             [&](long long begin, long long end, int) {
    std::vector<double> kv(static_cast<std::size_t>(dim * dim));
    std::vector<double> z(static_cast<std::size_t>(dim));
    for (long long head = begin; head < end; ++head) {
      std::fill(kv.begin(), kv.end(), 0.0);
      std::fill(z.begin(), z.end(), 0.0);
      const long long head_offset = head * sequence * dim;
      for (long long token = 0; token < sequence; ++token) {
        const float* key = k + head_offset + token * dim;
        const float* value = v + head_offset + token * dim;
        for (long long i = 0; i < dim; ++i) {
          z[static_cast<std::size_t>(i)] += key[i];
          for (long long j = 0; j < dim; ++j) {
            kv[static_cast<std::size_t>(i * dim + j)] +=
                static_cast<double>(key[i]) * value[j];
          }
        }
      }
      for (long long token = 0; token < sequence; ++token) {
        const float* query = q + head_offset + token * dim;
        float* destination = out + head_offset + token * dim;
        double denominator = eps;
        for (long long i = 0; i < dim; ++i) {
          denominator += static_cast<double>(query[i]) *
                         z[static_cast<std::size_t>(i)];
        }
        for (long long j = 0; j < dim; ++j) {
          double numerator = 0.0;
          for (long long i = 0; i < dim; ++i) {
            numerator += static_cast<double>(query[i]) *
                         kv[static_cast<std::size_t>(i * dim + j)];
          }
          destination[j] = static_cast<float>(numerator / denominator);
        }
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
