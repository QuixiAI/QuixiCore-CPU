#pragma once

#include <cstddef>

#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/quant_contract.h"
#include "quixicore_cpu/status.h"

namespace quixicore_cpu::quant {

enum class QuantEpilogueKind { kNone, kBias, kRelu, kSilu, kGelu };

struct QuantBlockDotParams {
  const void *packed_weight_block = nullptr;
  const void *activation_block = nullptr;
  float weight_scale = 1.0f;
  float activation_scale = 1.0f;
  float weight_zero_point = 0.0f;
  float activation_zero_point = 0.0f;
  long long elements = 0;
};

struct QuantTileParams {
  const void *packed_weights = nullptr;
  const void *activations = nullptr;
  const void *scales = nullptr;
  const void *zero_points = nullptr;
  const int *act_order = nullptr;
  float *output = nullptr;
  long long m = 0;
  long long n = 0;
  long long k = 0;
  long long output_stride = 0;
  QuantEpilogueKind epilogue = QuantEpilogueKind::kNone;
  const float *bias = nullptr;
};

struct QuantPairedProjectionParams {
  QuantTileParams first;
  QuantTileParams second;
  bool fuse_swiglu = false;
};

struct QuantSelectedRowsParams {
  QuantTileParams projection;
  const int *row_ids = nullptr;
  long long row_count = 0;
};

using QuantBlockDotFn = Status (*)(const QuantBlockDotParams &, float *sum);
using QuantTileFn = Status (*)(const QuantTileParams &);
using QuantPairedProjectionFn = Status (*)(const QuantPairedProjectionParams &);
using QuantSelectedRowsFn = Status (*)(const QuantSelectedRowsParams &);

struct QuantMicrokernelSet {
  const char *name = nullptr;
  CanonicalQuantLayout layout = CanonicalQuantLayout::kInt4Symmetric;
  CpuPreparedIsa required_isa = CpuPreparedIsa::kPortable;
  long long row_tile = 1;
  long long column_tile = 1;
  long long k_tile = 1;
  QuantBlockDotFn block_dot_f32 = nullptr;
  QuantBlockDotFn block_dot_f16 = nullptr;
  QuantBlockDotFn block_dot_bf16 = nullptr;
  QuantBlockDotFn block_dot_quantized = nullptr;
  QuantTileFn tile = nullptr;
  QuantPairedProjectionFn paired_projection = nullptr;
  QuantSelectedRowsFn selected_rows = nullptr;
};

} // namespace quixicore_cpu::quant
