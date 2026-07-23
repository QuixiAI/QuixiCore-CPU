#include "quixicore_cpu/quant_contract.h"

#include <cmath>
#include <limits>

namespace quixicore_cpu {
namespace {

constexpr CanonicalQuantDescriptor kDescriptors[] = {
    {1, CanonicalQuantLayout::kInt4Symmetric, "int4_symmetric", 4, 0, 0,
     QuantPackingOrder::kLowKFirstNibble, QuantScaleMode::kGroup,
     QuantScaleEncoding::kFP32, QuantZeroPointMode::kNone, 0, false, true,
     false, false, false},
    {1, CanonicalQuantLayout::kUInt4Affine, "uint4_affine", 4, 0, 0,
     QuantPackingOrder::kLowKFirstNibble, QuantScaleMode::kGroup,
     QuantScaleEncoding::kFP32, QuantZeroPointMode::kFractional, 0, false, true,
     false, false, false},
    {1, CanonicalQuantLayout::kInt8Symmetric, "int8_symmetric", 8, 0, 0,
     QuantPackingOrder::kBytePerElement, QuantScaleMode::kRow,
     QuantScaleEncoding::kFP32, QuantZeroPointMode::kNone, 0, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kInt8Affine, "int8_affine", 8, 0, 0,
     QuantPackingOrder::kBytePerElement, QuantScaleMode::kRow,
     QuantScaleEncoding::kFP32, QuantZeroPointMode::kInteger, 0, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kFP8E4M3FN, "fp8_e4m3fn", 8, 0, 0,
     QuantPackingOrder::kBytePerElement, QuantScaleMode::kTensor,
     QuantScaleEncoding::kFP32, QuantZeroPointMode::kNone, 0, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kFP8E5M2, "fp8_e5m2", 8, 0, 0,
     QuantPackingOrder::kBytePerElement, QuantScaleMode::kTensor,
     QuantScaleEncoding::kFP32, QuantZeroPointMode::kNone, 0, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kFP4E2M1, "fp4_e2m1", 4, 0, 0,
     QuantPackingOrder::kLowKFirstNibble, QuantScaleMode::kBlock,
     QuantScaleEncoding::kFP16, QuantZeroPointMode::kNone, 0, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kMXFP8E4M3E8M0, "mxfp8_e4m3_e8m0", 8, 32, 33,
     QuantPackingOrder::kSplitCodesAndMetadata, QuantScaleMode::kMicroscaleK32,
     QuantScaleEncoding::kE8M0, QuantZeroPointMode::kNone, 32, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kMXFP4E2M1E8M0, "mxfp4_e2m1_e8m0", 4, 32, 17,
     QuantPackingOrder::kSplitCodesAndMetadata, QuantScaleMode::kMicroscaleK32,
     QuantScaleEncoding::kE8M0, QuantZeroPointMode::kNone, 32, false, false,
     false, false, false},
    {1, CanonicalQuantLayout::kNVFP4E2M1E4M3, "nvfp4_e2m1_e4m3", 4, 16, 0,
     QuantPackingOrder::kSplitCodesAndMetadata, QuantScaleMode::kNVFP4K16,
     QuantScaleEncoding::kE4M3FN, QuantZeroPointMode::kNone, 16, true, false,
     false, false, false},
    {1, CanonicalQuantLayout::kBitNetTernary, "bitnet_ternary", 2, 32, 10,
     QuantPackingOrder::kLowKFirstTwoBits, QuantScaleMode::kBlock,
     QuantScaleEncoding::kFP16, QuantZeroPointMode::kNone, 32, false, false,
     true, false, false},
    {1, CanonicalQuantLayout::kTurboQuantKey, "turboquant_key", 0, 32, 0,
     QuantPackingOrder::kLeastSignificantBitStream,
     QuantScaleMode::kTurboQuantK32, QuantScaleEncoding::kFP16,
     QuantZeroPointMode::kTurboQuantOffset, 32, false, false, false, false,
     true},
    {1, CanonicalQuantLayout::kTurboQuantValue, "turboquant_value", 0, 32, 0,
     QuantPackingOrder::kLeastSignificantBitStream,
     QuantScaleMode::kTurboQuantK32, QuantScaleEncoding::kFP16,
     QuantZeroPointMode::kNone, 32, false, false, false, true, true},
};

const CanonicalQuantDescriptor *find_descriptor(CanonicalQuantLayout layout) {
  for (const auto &descriptor : kDescriptors) {
    if (descriptor.layout == layout)
      return &descriptor;
  }
  return nullptr;
}

bool valid_shape(long long rows, long long columns) {
  if (rows <= 0 || columns <= 0)
    return false;
  return static_cast<unsigned long long>(rows) <=
         std::numeric_limits<std::size_t>::max() /
             static_cast<unsigned long long>(columns);
}

} // namespace

Status canonical_quant_descriptor(CanonicalQuantLayout layout,
                                  CanonicalQuantDescriptor *descriptor) {
  if (descriptor == nullptr)
    return Status::kInvalidArgument;
  const auto *found = find_descriptor(layout);
  if (found == nullptr)
    return Status::kUnsupportedFormat;
  *descriptor = *found;
  return Status::kOk;
}

const char *canonical_quant_layout_name(CanonicalQuantLayout layout) {
  const auto *descriptor = find_descriptor(layout);
  return descriptor == nullptr ? "unknown" : descriptor->name;
}

Status validate_quant_metadata(const QuantTensorMetadata &metadata) {
  const auto *descriptor = find_descriptor(metadata.layout);
  if (descriptor == nullptr)
    return Status::kUnsupportedFormat;
  if (metadata.contract_version != descriptor->contract_version ||
      !valid_shape(metadata.logical_rows, metadata.logical_columns)) {
    return Status::kInvalidShape;
  }
  if (!std::isfinite(metadata.global_scale) || metadata.global_scale < 0.0f ||
      metadata.scale_domain_rows <= 0) {
    return Status::kInvalidArgument;
  }
  if (descriptor->block_elements != 0 &&
      metadata.logical_columns % descriptor->block_elements != 0) {
    return Status::kInvalidShape;
  }
  if (metadata.group_size < 0 ||
      (metadata.group_size > 0 &&
       metadata.logical_columns % metadata.group_size != 0)) {
    return Status::kInvalidShape;
  }
  if (metadata.act_order_count != 0 &&
      metadata.act_order_count !=
          static_cast<std::size_t>(metadata.logical_columns)) {
    return Status::kInvalidShape;
  }
  if (metadata.group_index_count != 0 &&
      metadata.group_index_count !=
          static_cast<std::size_t>(metadata.logical_columns)) {
    return Status::kInvalidShape;
  }
  if (metadata.group_index_count != 0 &&
      metadata.layout != CanonicalQuantLayout::kInt4Symmetric &&
      metadata.layout != CanonicalQuantLayout::kUInt4Affine) {
    return Status::kInvalidArgument;
  }
  if (metadata.act_order_count != 0 && !descriptor->supports_act_order) {
    return Status::kInvalidArgument;
  }
  if (metadata.sparse_mask_bytes != 0 &&
      metadata.sparse_mask_bytes <
          static_cast<std::size_t>((metadata.logical_columns + 7) / 8)) {
    return Status::kInvalidShape;
  }
  if (metadata.sparse_mask_bytes != 0 && !descriptor->supports_sparse_mask) {
    return Status::kInvalidArgument;
  }
  if (descriptor->requires_centroid_table) {
    if (metadata.value_bits < 2 || metadata.value_bits > 8 ||
        metadata.centroid_count !=
            (std::size_t{1} << static_cast<unsigned>(metadata.value_bits))) {
      return Status::kInvalidShape;
    }
  } else if (metadata.centroid_count != 0) {
    return Status::kInvalidArgument;
  }
  if (descriptor->is_cache_layout) {
    const int bits = metadata.layout == CanonicalQuantLayout::kTurboQuantKey
                         ? metadata.key_bits
                         : metadata.value_bits;
    if (bits < 2 || bits > 8 ||
        (metadata.logical_columns != 64 && metadata.logical_columns != 128 &&
         metadata.logical_columns != 256)) {
      return Status::kInvalidShape;
    }
  }
  if (metadata.scale_2d) {
    if (metadata.layout != CanonicalQuantLayout::kNVFP4E2M1E4M3 ||
        metadata.scale_domain_rows > 16) {
      return Status::kInvalidArgument;
    }
  }
  if (descriptor->has_global_scale &&
      metadata.scale_mode != QuantScaleMode::kNVFP4K16) {
    return Status::kInvalidArgument;
  }
  std::size_t expected_scales = 0;
  const auto rows = static_cast<std::size_t>(metadata.logical_rows);
  const auto columns = static_cast<std::size_t>(metadata.logical_columns);
  switch (metadata.scale_mode) {
  case QuantScaleMode::kNone:
    break;
  case QuantScaleMode::kTensor:
    expected_scales = 1;
    break;
  case QuantScaleMode::kRow:
  case QuantScaleMode::kChannel:
    expected_scales = rows;
    break;
  case QuantScaleMode::kGroup:
    if (metadata.group_size <= 0)
      return Status::kInvalidShape;
    expected_scales =
        rows * (columns / static_cast<std::size_t>(metadata.group_size));
    break;
  case QuantScaleMode::kBlock:
    if (metadata.scale_count == 0)
      return Status::kInvalidShape;
    expected_scales = metadata.scale_count;
    break;
  case QuantScaleMode::kMicroscaleK32:
  case QuantScaleMode::kTurboQuantK32:
    expected_scales = rows * (columns / 32);
    break;
  case QuantScaleMode::kNVFP4K16:
    expected_scales = rows * (columns / 16);
    break;
  }
  if (metadata.scale_count != expected_scales)
    return Status::kInvalidShape;
  if ((metadata.scale_mode == QuantScaleMode::kNone) !=
      (metadata.scale_encoding == QuantScaleEncoding::kNone)) {
    return Status::kInvalidArgument;
  }
  const std::size_t expected_zero_points =
      metadata.zero_point_mode == QuantZeroPointMode::kNone ? 0
                                                            : expected_scales;
  if (metadata.zero_point_count != expected_zero_points) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

} // namespace quixicore_cpu
