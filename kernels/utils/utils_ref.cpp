#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

double uniform01(std::uint32_t seed, std::uint64_t index) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^ index ^
                    0xD1B54A32D192ED03ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) *
         (1.0 / 9007199254740992.0);
}

}  // namespace

Status dropout(const float* x, float* y, long long count, float probability,
               std::uint32_t seed) {
  if (!detail::valid_product({count}) || !std::isfinite(probability) ||
      probability < 0.0f || probability >= 1.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  const float scale = 1.0f / (1.0f - probability);
  for (long long i = 0; i < count; ++i) {
    y[i] = uniform01(seed, static_cast<std::uint64_t>(i)) < probability
               ? 0.0f
               : x[i] * scale;
  }
  return Status::kOk;
}

Status dropout_backward(const float* grad_out, float* grad_in,
                        long long count, float probability,
                        std::uint32_t seed) {
  return dropout(grad_out, grad_in, count, probability, seed);
}

Status cross_entropy(const float* logits, const int* target, float* loss,
                     long long rows, long long vocab) {
  if (!detail::valid_product({rows, vocab})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, target, loss)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 8,
                             [&](long long begin, long long end, int) {
    for (long long row_index = begin; row_index < end; ++row_index) {
      if (target[row_index] < 0 || target[row_index] >= vocab) {
        loss[row_index] = std::numeric_limits<float>::quiet_NaN();
        continue;
      }
      const float* row = logits + row_index * vocab;
      float maximum = -std::numeric_limits<float>::infinity();
      for (long long token = 0; token < vocab; ++token) {
        maximum = std::max(maximum, row[token]);
      }
      double sum = 0.0;
      for (long long token = 0; token < vocab; ++token) {
        sum += std::exp(static_cast<double>(row[token] - maximum));
      }
      loss[row_index] = static_cast<float>(
          static_cast<double>(maximum) + std::log(sum) -
          row[target[row_index]]);
    }
  });
  for (long long row = 0; row < rows; ++row) {
    if (!std::isfinite(loss[row])) {
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

Status hadamard(const float* x, float* y, long long rows, long long dim) {
  if (!detail::valid_product({rows, dim}) ||
      (static_cast<unsigned long long>(dim) &
       (static_cast<unsigned long long>(dim) - 1ULL)) != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 1,
                             [&](long long begin, long long end, int) {
    std::vector<float> buffer(static_cast<std::size_t>(dim));
    for (long long row = begin; row < end; ++row) {
      std::copy_n(x + row * dim, dim, buffer.begin());
      for (long long stride = 1; stride < dim; stride *= 2) {
        for (long long block = 0; block < dim; block += 2 * stride) {
          for (long long i = 0; i < stride; ++i) {
            const float first = buffer[static_cast<std::size_t>(block + i)];
            const float second =
                buffer[static_cast<std::size_t>(block + stride + i)];
            buffer[static_cast<std::size_t>(block + i)] = first + second;
            buffer[static_cast<std::size_t>(block + stride + i)] =
                first - second;
          }
        }
      }
      std::copy(buffer.begin(), buffer.end(), y + row * dim);
    }
  });
  return Status::kOk;
}

Status fwht_rotate(const float* x, const float* sign, float* y,
                   long long rows, long long dim, bool inverse) {
  if (!detail::valid_product({rows, dim}) ||
      (static_cast<unsigned long long>(dim) &
       (static_cast<unsigned long long>(dim) - 1ULL)) != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, sign, y)) return Status::kInvalidArgument;
  for (long long column = 0; column < dim; ++column) {
    if (!std::isfinite(sign[column])) return Status::kInvalidArgument;
  }
  const float normalization =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)));
  threading::parallel_ranges(rows, 1,
                             [&](long long begin, long long end, int) {
    std::vector<float> buffer(static_cast<std::size_t>(dim));
    for (long long row = begin; row < end; ++row) {
      for (long long column = 0; column < dim; ++column) {
        buffer[static_cast<std::size_t>(column)] =
            x[row * dim + column] * (inverse ? 1.0f : sign[column]);
      }
      for (long long stride = 1; stride < dim; stride *= 2) {
        for (long long block = 0; block < dim; block += 2 * stride) {
          for (long long i = 0; i < stride; ++i) {
            const float first = buffer[static_cast<std::size_t>(block + i)];
            const float second =
                buffer[static_cast<std::size_t>(block + stride + i)];
            buffer[static_cast<std::size_t>(block + i)] = first + second;
            buffer[static_cast<std::size_t>(block + stride + i)] =
                first - second;
          }
        }
      }
      for (long long column = 0; column < dim; ++column) {
        y[row * dim + column] =
            buffer[static_cast<std::size_t>(column)] * normalization *
            (inverse ? sign[column] : 1.0f);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
