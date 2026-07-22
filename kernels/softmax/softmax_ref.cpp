#include "quixicore_cpu/ops.h"

#include <cmath>
#include <limits>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status softmax(const float* x, float* y, long long rows, long long dim) {
  if (!detail::valid_product({rows, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 8, [&](long long r0, long long r1, int) {
    for (long long row = r0; row < r1; ++row) {
      const float* in = x + row * dim;
      float* out = y + row * dim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (long long i = 0; i < dim; ++i) {
        maximum = std::fmax(maximum, in[i]);
      }
      double sum = 0.0;
      for (long long i = 0; i < dim; ++i) {
        out[i] = std::exp(in[i] - maximum);
        sum += out[i];
      }
      const float inverse = static_cast<float>(1.0 / sum);
      for (long long i = 0; i < dim; ++i) {
        out[i] *= inverse;
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
