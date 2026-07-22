#include "quixicore_cpu/collectives.h"

#include <algorithm>
#include <vector>

#include "kernels/common/validation.h"

namespace quixicore_cpu {

Status all_reduce_sum(const float* input, float* output, long long world,
                      long long count) {
  if (!detail::valid_product({world, count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  std::vector<float> sums(static_cast<std::size_t>(count), 0.0f);
  for (long long rank = 0; rank < world; ++rank) {
    for (long long item = 0; item < count; ++item) {
      sums[item] += input[rank * count + item];
    }
  }
  for (long long rank = 0; rank < world; ++rank) {
    std::copy(sums.begin(), sums.end(), output + rank * count);
  }
  return Status::kOk;
}

Status all_gather(const float* input, float* output, long long world,
                  long long count) {
  if (!detail::valid_product({world, count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  std::vector<float> gathered(input, input + world * count);
  for (long long rank = 0; rank < world; ++rank) {
    std::copy(gathered.begin(), gathered.end(),
              output + rank * world * count);
  }
  return Status::kOk;
}

Status reduce_scatter_sum(const float* input, float* output, long long world,
                          long long count) {
  if (!detail::valid_product({world, world, count})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  std::vector<float> result(static_cast<std::size_t>(world * count), 0.0f);
  for (long long source = 0; source < world; ++source) {
    for (long long destination = 0; destination < world; ++destination) {
      for (long long item = 0; item < count; ++item) {
        result[destination * count + item] +=
            input[(source * world + destination) * count + item];
      }
    }
  }
  std::copy(result.begin(), result.end(), output);
  return Status::kOk;
}

Status all_to_all(const float* input, float* output, long long world,
                  long long count) {
  if (!detail::valid_product({world, world, count})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  std::vector<float> source(input, input + world * world * count);
  for (long long source_rank = 0; source_rank < world; ++source_rank) {
    for (long long destination = 0; destination < world; ++destination) {
      std::copy_n(source.data() +
                      (source_rank * world + destination) * count,
                  count,
                  output + (destination * world + source_rank) * count);
    }
  }
  return Status::kOk;
}

Status broadcast(const float* input, float* output, long long world,
                 long long count, int root) {
  if (!detail::valid_product({world, count}) || root < 0 || root >= world) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  std::vector<float> root_values(input + static_cast<long long>(root) * count,
                                 input + (static_cast<long long>(root) + 1) * count);
  for (long long rank = 0; rank < world; ++rank) {
    std::copy(root_values.begin(), root_values.end(), output + rank * count);
  }
  return Status::kOk;
}

Status reduce_sum(const float* input, float* output, long long world,
                  long long count, int root) {
  if (!detail::valid_product({world, count}) || root < 0 || root >= world) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, output)) return Status::kInvalidArgument;
  std::vector<float> sums(static_cast<std::size_t>(count), 0.0f);
  for (long long rank = 0; rank < world; ++rank) {
    for (long long item = 0; item < count; ++item) {
      sums[item] += input[rank * count + item];
    }
  }
  std::copy(sums.begin(), sums.end(), output + static_cast<long long>(root) * count);
  return Status::kOk;
}

Status gemm_all_reduce(const float* a, const float* b, float* output,
                       long long world, long long m, long long n,
                       long long k) {
  if (!detail::valid_product({world, m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(a, b, output)) return Status::kInvalidArgument;
  std::vector<float> reduced(static_cast<std::size_t>(m * n), 0.0f);
  for (long long rank = 0; rank < world; ++rank) {
    for (long long row = 0; row < m; ++row) {
      for (long long column = 0; column < n; ++column) {
        double sum = 0.0;
        for (long long inner = 0; inner < k; ++inner) {
          sum += a[(rank * m + row) * k + inner] *
                 b[(rank * k + inner) * n + column];
        }
        reduced[row * n + column] += static_cast<float>(sum);
      }
    }
  }
  for (long long rank = 0; rank < world; ++rank) {
    std::copy(reduced.begin(), reduced.end(), output + rank * m * n);
  }
  return Status::kOk;
}

Status all_gather_gemm(const float* a, const float* b, float* output,
                       long long world, long long m, long long n,
                       long long k) {
  if (!detail::valid_product({world, m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(a, b, output)) return Status::kInvalidArgument;
  for (long long destination = 0; destination < world; ++destination) {
    for (long long source = 0; source < world; ++source) {
      for (long long row = 0; row < m; ++row) {
        for (long long column = 0; column < n; ++column) {
          double sum = 0.0;
          for (long long inner = 0; inner < k; ++inner) {
            sum += a[(source * m + row) * k + inner] *
                   b[(destination * k + inner) * n + column];
          }
          output[((destination * world + source) * m + row) * n + column] =
              static_cast<float>(sum);
        }
      }
    }
  }
  return Status::kOk;
}

Status gemm_reduce_scatter(const float* a, const float* b, float* output,
                           long long world, long long m, long long n,
                           long long k) {
  if (!detail::valid_product({world, m, n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(a, b, output)) return Status::kInvalidArgument;
  std::fill_n(output, world * m * n, 0.0f);
  for (long long source = 0; source < world; ++source) {
    for (long long destination = 0; destination < world; ++destination) {
      for (long long row = 0; row < m; ++row) {
        for (long long column = 0; column < n; ++column) {
          double sum = 0.0;
          const long long global_row = destination * m + row;
          for (long long inner = 0; inner < k; ++inner) {
            sum += a[(source * world * m + global_row) * k + inner] *
                   b[(source * k + inner) * n + column];
          }
          output[(destination * m + row) * n + column] +=
              static_cast<float>(sum);
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
