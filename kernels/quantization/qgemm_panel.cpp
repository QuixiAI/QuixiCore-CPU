#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "quixicore_cpu/qgemm.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "kernels/quantization/gguf_ref.h"
#include "kernels/quantization/qgemv.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/threading.h"
#include "src/memory/workspace_internal.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

void accumulate_rows(float* sums, const float* values, float weight,
                     long long rows) {
  long long row = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  const float32x4_t broadcast = vdupq_n_f32(weight);
  for (; row + 3 < rows; row += 4) {
    vst1q_f32(sums + row, vfmaq_f32(vld1q_f32(sums + row), broadcast,
                                    vld1q_f32(values + row)));
  }
#endif
  for (; row < rows; ++row) sums[row] += weight * values[row];
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

struct TransposeContext {
  const float* source;
  float* destination;
  long long rows;
  long long columns;
};

void transpose_columns(void* opaque, long long begin, long long end, int) {
  const auto& context = *static_cast<const TransposeContext*>(opaque);
  for (long long column = begin; column < end; ++column) {
    float* destination = context.destination + column * context.rows;
    for (long long row = 0; row < context.rows; ++row) {
      destination[row] = context.source[row * context.columns + column];
    }
  }
}

struct PanelContext {
  const std::uint8_t* panel;
  const float* transposed_x;
  float* output;
  float* scratch;
  long long m;
  long long n;
  long long row_tile;
  long long block_size;
  long long blocks_per_row;
  std::size_t block_bytes;
  QuantFormat format;
};

const std::uint8_t* panel_block(const PanelContext& context, long long panel,
                                long long block, long long lane) {
  const std::size_t index = static_cast<std::size_t>(
      (panel * context.blocks_per_row + block) * context.row_tile + lane);
  return context.panel + index * context.block_bytes;
}

void q4_panel_rows(void* opaque, long long begin, long long end, int worker) {
  const auto& context = *static_cast<const PanelContext*>(opaque);
  const long long lane_stride = context.m;
  const long long tile_elements = context.row_tile * context.m;
  float* accumulator =
      context.scratch + static_cast<long long>(worker) * tile_elements * 2;
  float* block_sum = accumulator + tile_elements;
  for (long long panel = begin; panel < end; ++panel) {
    const long long first_row = panel * context.row_tile;
    const long long lanes = std::min(context.row_tile, context.n - first_row);
    std::fill_n(accumulator, lanes * lane_stride, 0.0f);
    for (long long block = 0; block < context.blocks_per_row; ++block) {
      for (long long lane = 0; lane < lanes; ++lane) {
        const auto* weights = reinterpret_cast<const quant::BlockQ4_0*>(
            panel_block(context, panel, block, lane));
        float* sums = block_sum + lane * lane_stride;
        std::fill_n(sums, context.m, 0.0f);
        for (int element = 0; element < 16; ++element) {
          const float low =
              static_cast<float>((weights->qs[element] & 0x0F) - 8);
          const float high =
              static_cast<float>((weights->qs[element] >> 4) - 8);
          const float* x_low =
              context.transposed_x +
              (block * quant::kQ4_0BlockSize + element) * context.m;
          const float* x_high =
              context.transposed_x +
              (block * quant::kQ4_0BlockSize + element + 16) * context.m;
          for (long long row = 0; row < context.m; ++row) {
            sums[row] += low * x_low[row];
            sums[row] += high * x_high[row];
          }
        }
        const float scale = fp16_to_fp32(weights->d);
        float* accumulated = accumulator + lane * lane_stride;
        for (long long row = 0; row < context.m; ++row) {
          accumulated[row] += scale * sums[row];
        }
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long output = first_row + lane;
      const float* accumulated = accumulator + lane * lane_stride;
      for (long long row = 0; row < context.m; ++row) {
        context.output[row * context.n + output] = accumulated[row];
      }
    }
  }
}

void q8_panel_rows(void* opaque, long long begin, long long end, int worker) {
  const auto& context = *static_cast<const PanelContext*>(opaque);
  const long long lane_stride = context.m;
  const long long tile_elements = context.row_tile * context.m;
  float* accumulator =
      context.scratch + static_cast<long long>(worker) * tile_elements * 2;
  float* block_sum = accumulator + tile_elements;
  for (long long panel = begin; panel < end; ++panel) {
    const long long first_row = panel * context.row_tile;
    const long long lanes = std::min(context.row_tile, context.n - first_row);
    std::fill_n(accumulator, lanes * lane_stride, 0.0f);
    for (long long block = 0; block < context.blocks_per_row; ++block) {
      for (long long lane = 0; lane < lanes; ++lane) {
        const auto* weights = reinterpret_cast<const quant::BlockQ8_0*>(
            panel_block(context, panel, block, lane));
        float* sums = block_sum + lane * lane_stride;
        std::fill_n(sums, context.m, 0.0f);
        for (int element = 0; element < quant::kQ8_0BlockSize; ++element) {
          const float weight = static_cast<float>(weights->qs[element]);
          const float* values =
              context.transposed_x +
              (block * quant::kQ8_0BlockSize + element) * context.m;
          accumulate_rows(sums, values, weight, context.m);
        }
        const float scale = fp16_to_fp32(weights->d);
        float* accumulated = accumulator + lane * lane_stride;
        accumulate_rows(accumulated, sums, scale, context.m);
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long output = first_row + lane;
      const float* accumulated = accumulator + lane * lane_stride;
      for (long long row = 0; row < context.m; ++row) {
        context.output[row * context.n + output] = accumulated[row];
      }
    }
  }
}

void generic_panel_rows(void* opaque, long long begin, long long end,
                        int worker) {
  const auto& context = *static_cast<const PanelContext*>(opaque);
  const long long lane_stride = context.m;
  const long long tile_elements = context.row_tile * context.m;
  float* accumulator =
      context.scratch + static_cast<long long>(worker) * tile_elements * 2;
  float* block_sum = accumulator + tile_elements;
  alignas(64) float decoded[256];
  for (long long panel = begin; panel < end; ++panel) {
    const long long first_row = panel * context.row_tile;
    const long long lanes = std::min(context.row_tile, context.n - first_row);
    std::fill_n(accumulator, lanes * lane_stride, 0.0f);
    for (long long block = 0; block < context.blocks_per_row; ++block) {
      for (long long lane = 0; lane < lanes; ++lane) {
        quant::gguf_dequant_block_ref(
            context.format, panel_block(context, panel, block, lane), decoded);
        float* sums = block_sum + lane * lane_stride;
        std::fill_n(sums, context.m, 0.0f);
        for (long long element = 0; element < context.block_size; ++element) {
          const float weight = decoded[element];
          const float* values =
              context.transposed_x +
              (block * context.block_size + element) * context.m;
          accumulate_rows(sums, values, weight, context.m);
        }
        float* accumulated = accumulator + lane * lane_stride;
        accumulate_rows(accumulated, sums, 1.0f, context.m);
      }
    }
    for (long long lane = 0; lane < lanes; ++lane) {
      const long long output = first_row + lane;
      const float* accumulated = accumulator + lane * lane_stride;
      for (long long row = 0; row < context.m; ++row) {
        context.output[row * context.n + output] = accumulated[row];
      }
    }
  }
}

}  // namespace

Status qgemm_prepacked(const CpuPackedWeights& weights, const float* x,
                       float* y, long long m, Workspace* workspace) {
  if (!weights.ready()) return Status::kInvalidArgument;
  const CpuPackedWeightsInfo info = weights.info();
  if (!detail::valid_product({m, info.rows, info.columns})) {
    return Status::kInvalidShape;
  }
  if (info.has_canonical_layout) {
    return qgemm_prepacked_storage(
        weights, {x, FloatStorageType::kF32, m * info.columns},
        {y, FloatStorageType::kF32, m * info.rows}, m, workspace);
  }
  if (!detail::all_nonnull(x, y, weights.contract_data(),
                           weights.panel_data())) {
    return Status::kInvalidArgument;
  }
  if (m == 1) {
    return qgemv(info.format, weights.contract_data(), x, y, info.rows,
                 info.columns);
  }
  if (m < 64 &&
      (info.format == QuantFormat::kQ4_0 || info.format == QuantFormat::kQ8_0 ||
       info.format == QuantFormat::kIQ4_NL ||
       info.format == QuantFormat::kIQ4_XS)) {
    return qgemm(info.format, weights.contract_data(), x, y, m, info.rows,
                 info.columns);
  }

  std::size_t transposed_elements = 0;
  std::size_t tile_elements = 0;
  std::size_t worker_elements = 0;
  std::size_t total_elements = 0;
  if (!checked_mul(static_cast<std::size_t>(m),
                   static_cast<std::size_t>(info.columns),
                   &transposed_elements) ||
      !checked_mul(static_cast<std::size_t>(info.row_tile),
                   static_cast<std::size_t>(m), &tile_elements) ||
      !checked_mul(tile_elements, 2 * static_cast<std::size_t>(num_threads()),
                   &worker_elements) ||
      worker_elements >
          std::numeric_limits<std::size_t>::max() - transposed_elements) {
    return Status::kInvalidShape;
  }
  total_elements = transposed_elements + worker_elements;

  detail::WorkspaceFrame frame(workspace);
  float* storage = frame.allocate<float>(total_elements);
  if (storage == nullptr) return Status::kOutOfMemory;
  float* transposed = storage;
  float* scratch = storage + transposed_elements;

  TransposeContext transpose{x, transposed, m, info.columns};
  threading::parallel_ranges_impl(info.columns, 128, transpose_columns,
                                  &transpose);

  PanelContext context{static_cast<const std::uint8_t*>(weights.panel_data()),
                       transposed,
                       y,
                       scratch,
                       m,
                       info.rows,
                       info.row_tile,
                       info.block_size,
                       info.blocks_per_row,
                       info.block_bytes,
                       info.format};
  const long long panels =
      info.rows / info.row_tile + (info.rows % info.row_tile != 0 ? 1 : 0);
  auto kernel = generic_panel_rows;
  if (info.format == QuantFormat::kQ4_0) kernel = q4_panel_rows;
  if (info.format == QuantFormat::kQ8_0) kernel = q8_panel_rows;
  threading::parallel_ranges_impl(panels, 1, kernel, &context);
  return Status::kOk;
}

}  // namespace quixicore_cpu
