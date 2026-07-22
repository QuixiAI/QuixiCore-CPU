#include "quixicore_cpu/qgemv.h"

#include <cmath>
#include <cstdint>

#include "kernels/common/validation.h"
#include "kernels/quantization/gguf_ref.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {

Status qgemv_rows(QuantFormat format, const void* packed, const float* x,
                  const int* row_ids, float* y, long long row_count,
                  long long n, long long k) {
  if (!detail::valid_product({row_count, n, k})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed, x, row_ids, y)) {
    return Status::kInvalidArgument;
  }
  std::size_t ignored = 0;
  const Status status = qgemv_packed_size(format, n, k, &ignored);
  if (status != Status::kOk) return status;
  for (long long item = 0; item < row_count; ++item) {
    if (row_ids[item] < 0 || row_ids[item] >= n) {
      return Status::kInvalidArgument;
    }
  }
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  const long long blocks_per_row = k / block_size;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks_per_row) * block_bytes;
  const auto* bytes = static_cast<const std::uint8_t*>(packed);
  threading::parallel_ranges(row_count, 16,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const std::uint8_t* row = bytes + row_ids[item] * row_bytes;
      double sum = 0.0;
      for (long long block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* source = row + block * block_bytes;
        for (int column = 0; column < block_size; ++column) {
          sum += quant::gguf_dequant_element(format, source, column) *
                 x[block * block_size + column];
        }
      }
      y[item] = static_cast<float>(sum);
    }
  });
  return Status::kOk;
}

Status qgemv_axpy_row(QuantFormat format, const void* packed, long long row,
                      float coefficient, float* out, long long n,
                      long long k) {
  if (!detail::valid_product({n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed, out) || !std::isfinite(coefficient) ||
      row < 0 || row >= n) {
    return Status::kInvalidArgument;
  }
  std::size_t ignored = 0;
  const Status status = qgemv_packed_size(format, n, k, &ignored);
  if (status != Status::kOk) return status;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  const long long blocks_per_row = k / block_size;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks_per_row) * block_bytes;
  const auto* source_row = static_cast<const std::uint8_t*>(packed) +
                           static_cast<std::size_t>(row) * row_bytes;
  threading::parallel_ranges(k, 4096,
                             [&](long long begin, long long end, int) {
    for (long long input = begin; input < end; ++input) {
      const long long block = input / block_size;
      const int column = static_cast<int>(input % block_size);
      out[input] += coefficient * quant::gguf_dequant_element(
                                        format,
                                        source_row + block * block_bytes,
                                        column);
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
