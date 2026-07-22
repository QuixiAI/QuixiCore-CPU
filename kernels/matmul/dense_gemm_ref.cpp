#include "quixicore_cpu/ops.h"

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status dense_gemm(const float* a, const float* b, float* c, long long m,
                  long long n, long long k) {
  if (!detail::valid_product({m, k}) || !detail::valid_product({k, n}) ||
      !detail::valid_product({m, n})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(a, b, c)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(m, 1, [&](long long r0, long long r1, int) {
    for (long long row = r0; row < r1; ++row) {
      for (long long column = 0; column < n; ++column) {
        double sum = 0.0;
        for (long long inner = 0; inner < k; ++inner) {
          sum += static_cast<double>(a[row * k + inner]) *
                 b[inner * n + column];
        }
        c[row * n + column] = static_cast<float>(sum);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
