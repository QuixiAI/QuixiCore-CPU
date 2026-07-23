#pragma once

#include <cstddef>
#include <cstdint>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Stable logical layouts described by the umbrella format specifications.
// CPU-private panel/LUT layouts deliberately do not appear in this enum.
enum class CanonicalQuantLayout {
  kInt4Symmetric,
  kUInt4Affine,
  kInt8Symmetric,
  kInt8Affine,
  kFP8E4M3FN,
  kFP8E5M2,
  kFP4E2M1,
  kMXFP8E4M3E8M0,
  kMXFP4E2M1E8M0,
  kNVFP4E2M1E4M3,
  kBitNetTernary,
  kTurboQuantKey,
  kTurboQuantValue,
};

enum class QuantScaleMode {
  kNone,
  kTensor,
  kRow,
  kChannel,
  kGroup,
  kBlock,
  kMicroscaleK32,
  kNVFP4K16,
  kTurboQuantK32,
};

enum class QuantScaleEncoding { kNone, kFP16, kFP32, kE8M0, kE4M3FN };

enum class QuantZeroPointMode {
  kNone,
  kInteger,
  kFractional,
  kTurboQuantOffset,
};

enum class QuantPackingOrder {
  kBytePerElement,
  kLowKFirstNibble,
  kLowKFirstTwoBits,
  kLeastSignificantBitStream,
  kSplitCodesAndMetadata,
};

struct CanonicalQuantDescriptor {
  std::uint32_t contract_version = 1;
  CanonicalQuantLayout layout = CanonicalQuantLayout::kInt4Symmetric;
  const char *name = nullptr;
  std::uint8_t element_bits = 0;
  std::uint16_t block_elements = 0;   // zero means runtime group metadata
  std::size_t packed_block_bytes = 0; // zero means split logical fields
  QuantPackingOrder packing = QuantPackingOrder::kBytePerElement;
  QuantScaleMode default_scale_mode = QuantScaleMode::kNone;
  QuantScaleEncoding scale_encoding = QuantScaleEncoding::kNone;
  QuantZeroPointMode zero_point_mode = QuantZeroPointMode::kNone;
  std::uint16_t scale_group_elements = 0;
  bool has_global_scale = false;
  bool supports_act_order = false;
  bool supports_sparse_mask = false;
  bool requires_centroid_table = false;
  bool is_cache_layout = false;
};

// Runtime metadata accompanies canonical bytes. Pointer ownership remains with
// the caller/container; counts make every optional field explicit.
struct QuantTensorMetadata {
  std::uint32_t contract_version = 1;
  CanonicalQuantLayout layout = CanonicalQuantLayout::kInt4Symmetric;
  long long logical_rows = 0;
  long long logical_columns = 0;
  long long group_size = 0;
  QuantScaleMode scale_mode = QuantScaleMode::kNone;
  QuantScaleEncoding scale_encoding = QuantScaleEncoding::kNone;
  QuantZeroPointMode zero_point_mode = QuantZeroPointMode::kNone;
  std::size_t scale_count = 0;
  std::size_t zero_point_count = 0;
  // Optional packed-K -> scale-group map used by GPTQ act-order checkpoints.
  // This is distinct from act_order, which maps packed-K -> logical-K.
  std::size_t group_index_count = 0;
  std::size_t act_order_count = 0;
  std::size_t sparse_mask_bytes = 0;
  std::size_t centroid_count = 0;
  float global_scale = 1.0f;
  int key_bits = 0;
  int value_bits = 0;
  bool key_signed = false;
  bool scale_2d = false;
  int scale_domain_rows = 1;
};

Status canonical_quant_descriptor(CanonicalQuantLayout layout,
                                  CanonicalQuantDescriptor *descriptor);
Status validate_quant_metadata(const QuantTensorMetadata &metadata);
const char *canonical_quant_layout_name(CanonicalQuantLayout layout);

} // namespace quixicore_cpu
