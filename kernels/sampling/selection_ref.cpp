#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float kth_descending(float* values, long long count, long long target) {
  long long left = 0;
  long long right = count - 1;
  while (left < right) {
    const long long middle = left + (right - left) / 2;
    const float a = values[left];
    const float b = values[middle];
    const float c = values[right];
    const float pivot = std::max(std::min(a, b), std::min(std::max(a, b), c));
    long long low = left;
    long long high = right;
    while (low <= high) {
      while (values[low] > pivot) ++low;
      while (values[high] < pivot) --high;
      if (low <= high) {
        std::swap(values[low], values[high]);
        ++low;
        --high;
      }
    }
    if (target <= high) {
      right = high;
    } else if (target >= low) {
      left = low;
    } else {
      break;
    }
  }
  return values[target];
}

}  // namespace

Status threshold_topk_indices(const float* scores, int* indices,
                              long long rows, long long width, int k) {
  if (!detail::valid_product({rows, width}) || k <= 0 || k > width) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(scores, indices)) return Status::kInvalidArgument;
  for (long long item = 0; item < rows * width; ++item) {
    if (!std::isfinite(scores[item])) return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    thread_local std::vector<float> scratch;
    scratch.resize(static_cast<std::size_t>(width));
    for (long long row = begin; row < end; ++row) {
      const float* source = scores + row * width;
      std::copy_n(source, width, scratch.data());
      const float threshold = kth_descending(scratch.data(), width, k - 1);
      int selected = 0;
      for (long long column = 0; column < width && selected < k; ++column) {
        if (source[column] > threshold) {
          indices[row * k + selected++] = static_cast<int>(column);
        }
      }
      for (long long column = 0; column < width && selected < k; ++column) {
        if (source[column] == threshold) {
          indices[row * k + selected++] = static_cast<int>(column);
        }
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
