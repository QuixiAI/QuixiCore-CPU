#include "quixicore_cpu/ops.h"

#include <algorithm>

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
  constexpr long long kRowsPerTile = 4;
  constexpr long long kColumnsPerTile = 32;
  const long long row_tiles = (m + kRowsPerTile - 1) / kRowsPerTile;
  const long long column_tiles = (n + kColumnsPerTile - 1) / kColumnsPerTile;
  threading::parallel_ranges(
      row_tiles * column_tiles, 1,
      [&](long long begin, long long end, int) {
    alignas(64) double accumulators[kRowsPerTile * kColumnsPerTile];
    for (long long task = begin; task < end; ++task) {
      const long long row_begin = (task / column_tiles) * kRowsPerTile;
      const long long column_begin = (task % column_tiles) * kColumnsPerTile;
      const long long rows = std::min(kRowsPerTile, m - row_begin);
      const long long columns = std::min(kColumnsPerTile, n - column_begin);
      std::fill_n(accumulators, rows * kColumnsPerTile, 0.0);
      for (long long inner = 0; inner < k; ++inner) {
        const float* b_row = b + inner * n + column_begin;
        for (long long row = 0; row < rows; ++row) {
          const double av = static_cast<double>(
              a[(row_begin + row) * k + inner]);
          double* sums = accumulators + row * kColumnsPerTile;
          for (long long column = 0; column < columns; ++column) {
            sums[column] += av * static_cast<double>(b_row[column]);
          }
        }
      }
      for (long long row = 0; row < rows; ++row) {
        for (long long column = 0; column < columns; ++column) {
          c[(row_begin + row) * n + column_begin + column] =
              static_cast<float>(
                  accumulators[row * kColumnsPerTile + column]);
        }
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
