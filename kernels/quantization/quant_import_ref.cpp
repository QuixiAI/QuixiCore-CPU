#include "quixicore_cpu/quant_import.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "quixicore_cpu/quantization.h"

namespace quixicore_cpu {
namespace {

constexpr std::array<int, 8> kAwqReverseOrder = {0, 4, 1, 5, 2, 6, 3, 7};

bool checked_product(long long lhs, long long rhs, std::size_t* result) {
  if (result == nullptr || lhs <= 0 || rhs <= 0) return false;
  const auto left = static_cast<unsigned long long>(lhs);
  const auto right = static_cast<unsigned long long>(rhs);
  if (left > std::numeric_limits<std::size_t>::max() / right) return false;
  *result = static_cast<std::size_t>(left * right);
  return true;
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool finite_values(const float* values, std::size_t count) {
  if (count != 0 && values == nullptr) return false;
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) return false;
  }
  return true;
}

bool finite_nonnegative_values(const float* values, std::size_t count) {
  if (!finite_values(values, count)) return false;
  for (std::size_t index = 0; index < count; ++index) {
    if (values[index] < 0.0f) return false;
  }
  return true;
}

std::uint8_t nibble_at(const std::vector<std::uint8_t>& bytes,
                       std::size_t element) {
  return static_cast<std::uint8_t>(
      (bytes[element / 2] >> (4 * (element & 1))) & 0x0f);
}

void put_nibble(std::vector<std::uint8_t>* bytes, std::size_t element,
                std::uint8_t code) {
  const int shift = static_cast<int>(4 * (element & 1));
  const auto mask = static_cast<std::uint8_t>(0x0fU << shift);
  (*bytes)[element / 2] = static_cast<std::uint8_t>(
      ((*bytes)[element / 2] & ~mask) | ((code & 0x0fU) << shift));
}

float fp8_maximum(Float8Format format) {
  return format == Float8Format::kE4M3FN ? 448.0f : 57344.0f;
}

std::uint8_t e8m0_encode_scale(float requested) {
  if (!(requested > 0.0f)) return 0;
  const int exponent = static_cast<int>(std::ceil(std::log2(requested)));
  return static_cast<std::uint8_t>(std::clamp(exponent + 127, 0, 254));
}

float e8m0_decode_scale(std::uint8_t code) {
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

std::size_t group_for_column(const CanonicalQuantTensor& tensor,
                             std::size_t column) {
  if (!tensor.group_index.empty()) {
    return static_cast<std::size_t>(tensor.group_index[column]);
  }
  const std::size_t group_size =
      static_cast<std::size_t>(tensor.metadata.group_size);
  return group_size == 0 ? 0 : column / group_size;
}

Status expected_data_bytes(const QuantTensorMetadata& metadata,
                           std::size_t* bytes) {
  std::size_t elements = 0;
  if (!checked_product(metadata.logical_rows, metadata.logical_columns,
                       &elements)) {
    return Status::kInvalidShape;
  }
  switch (metadata.layout) {
    case CanonicalQuantLayout::kInt4Symmetric:
    case CanonicalQuantLayout::kUInt4Affine:
    case CanonicalQuantLayout::kFP4E2M1:
      if ((metadata.logical_columns & 1) != 0) return Status::kInvalidShape;
      *bytes = elements / 2;
      return Status::kOk;
    case CanonicalQuantLayout::kInt8Symmetric:
    case CanonicalQuantLayout::kInt8Affine:
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
      *bytes = elements;
      return Status::kOk;
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
      return checked_mul(elements / 32, 33, bytes) ? Status::kOk
                                                   : Status::kInvalidShape;
    case CanonicalQuantLayout::kMXFP4E2M1E8M0:
      return checked_mul(elements / 32, 17, bytes) ? Status::kOk
                                                   : Status::kInvalidShape;
    case CanonicalQuantLayout::kNVFP4E2M1E4M3:
      *bytes = elements / 2;
      return Status::kOk;
    case CanonicalQuantLayout::kBitNetTernary:
      return checked_mul(elements / 32, 10, bytes) ? Status::kOk
                                                   : Status::kInvalidShape;
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return Status::kUnsupportedFormat;
  }
  return Status::kUnsupportedFormat;
}

bool valid_permutation(const std::vector<int>& order, std::size_t count) {
  if (order.empty()) return true;
  if (order.size() != count) return false;
  std::vector<std::uint8_t> seen(count, 0);
  for (int value : order) {
    if (value < 0 || static_cast<std::size_t>(value) >= count ||
        seen[static_cast<std::size_t>(value)] != 0) {
      return false;
    }
    seen[static_cast<std::size_t>(value)] = 1;
  }
  return true;
}

template <class T>
void copy_optional(const T* source, std::size_t count,
                   std::vector<T>* destination) {
  if (count == 0) {
    destination->clear();
  } else {
    destination->assign(source, source + count);
  }
}

Status stage_float_input(FloatStorageInput input, std::size_t count,
                         std::vector<float>* storage, const float** values) {
  if (input.data == nullptr || input.count <= 0 ||
      static_cast<std::size_t>(input.count) != count) {
    return Status::kInvalidArgument;
  }
  if (input.type == FloatStorageType::kF32) {
    *values = static_cast<const float*>(input.data);
    return finite_values(*values, count) ? Status::kOk
                                         : Status::kInvalidArgument;
  }
  storage->resize(count);
  const Status status = float_storage_to_f32(
      input.type, input.data, storage->data(), input.count);
  if (status != Status::kOk) return status;
  *values = storage->data();
  return finite_values(*values, count) ? Status::kOk
                                       : Status::kInvalidArgument;
}

void initialize_metadata(CanonicalQuantTensor* tensor,
                         CanonicalQuantLayout layout, long long rows,
                         long long columns, long long group_size) {
  *tensor = CanonicalQuantTensor{};
  tensor->metadata.layout = layout;
  tensor->metadata.logical_rows = rows;
  tensor->metadata.logical_columns = columns;
  tensor->metadata.group_size = group_size;
  tensor->provenance = QuantImportProvenance::kCanonical;
}

Status quantize_grouped_fp(const float* input, long long rows,
                           long long columns, long long group_size,
                           CanonicalQuantLayout layout,
                           QuantScaleMode requested_mode,
                           int scale_domain_rows,
                           CanonicalQuantTensor* tensor) {
  const bool fp4 = layout == CanonicalQuantLayout::kFP4E2M1;
  QuantScaleMode mode = requested_mode;
  if (mode == QuantScaleMode::kNone) {
    mode = group_size == 0
               ? QuantScaleMode::kTensor
               : (fp4 ? QuantScaleMode::kBlock : QuantScaleMode::kGroup);
  }
  long long k_group = group_size;
  int row_group = 1;
  if (mode == QuantScaleMode::kTensor) {
    k_group = columns;
    row_group = static_cast<int>(rows);
  } else if (mode == QuantScaleMode::kRow ||
             mode == QuantScaleMode::kChannel) {
    k_group = columns;
  } else if (mode == QuantScaleMode::kBlock) {
    row_group = scale_domain_rows;
  } else if (mode != QuantScaleMode::kGroup) {
    return Status::kInvalidArgument;
  }
  if (k_group <= 0 || columns % k_group != 0 || row_group <= 0 ||
      (fp4 && (columns & 1) != 0)) {
    return Status::kInvalidShape;
  }
  const long long scale_rows = (rows + row_group - 1) / row_group;
  const long long groups_per_scale_row = columns / k_group;
  const long long groups = scale_rows * groups_per_scale_row;
  const std::size_t elements =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
  tensor->data.assign(fp4 ? elements / 2 : elements, 0);
  tensor->scales.resize(static_cast<std::size_t>(groups));
  tensor->metadata.scale_mode = mode;
  tensor->metadata.scale_encoding = fp4 ? QuantScaleEncoding::kFP16
                                        : QuantScaleEncoding::kFP32;
  tensor->metadata.scale_count = static_cast<std::size_t>(groups);
  tensor->metadata.group_size =
      mode == QuantScaleMode::kGroup || mode == QuantScaleMode::kBlock
          ? k_group
          : 0;
  tensor->metadata.scale_domain_rows = row_group;

  const auto format = layout == CanonicalQuantLayout::kFP8E5M2
                          ? Float8Format::kE5M2
                          : Float8Format::kE4M3FN;
  const float maximum_code = fp4 ? 6.0f : fp8_maximum(format);
  for (long long scale_row = 0; scale_row < scale_rows; ++scale_row) {
    const long long row_begin = scale_row * row_group;
    const long long row_end = std::min(rows, row_begin + row_group);
    for (long long k_block = 0; k_block < groups_per_scale_row; ++k_block) {
      const long long column_begin = k_block * k_group;
      float maximum = 0.0f;
      for (long long row = row_begin; row < row_end; ++row) {
        const std::size_t base = static_cast<std::size_t>(row * columns +
                                                         column_begin);
        for (long long item = 0; item < k_group; ++item) {
          maximum = std::max(
              maximum, std::fabs(input[base + static_cast<std::size_t>(item)]));
        }
      }
      const float scale = maximum == 0.0f ? 0.0f : maximum / maximum_code;
      const std::size_t scale_index = static_cast<std::size_t>(
          scale_row * groups_per_scale_row + k_block);
      tensor->scales[scale_index] =
          fp4 ? f16_to_float(float_to_f16(scale)) : scale;
      const float stored_scale = tensor->scales[scale_index];
      const float inverse = stored_scale > 0.0f ? 1.0f / stored_scale : 0.0f;
      for (long long row = row_begin; row < row_end; ++row) {
        const std::size_t base = static_cast<std::size_t>(row * columns +
                                                         column_begin);
        for (long long item = 0; item < k_group; ++item) {
          const std::size_t element = base + static_cast<std::size_t>(item);
          const auto code = fp4 ? fp4_e2m1_encode(input[element] * inverse)
                                : float8_encode(input[element] * inverse,
                                                format);
          if (fp4) {
            put_nibble(&tensor->data, element, code);
          } else {
            tensor->data[element] = code;
          }
        }
      }
    }
  }
  return Status::kOk;
}

std::size_t tensor_scale_index(const CanonicalQuantTensor& tensor,
                               std::size_t row, std::size_t column) {
  const std::size_t columns =
      static_cast<std::size_t>(tensor.metadata.logical_columns);
  switch (tensor.metadata.scale_mode) {
    case QuantScaleMode::kTensor:
      return 0;
    case QuantScaleMode::kRow:
    case QuantScaleMode::kChannel:
      return row;
    case QuantScaleMode::kBlock: {
      const std::size_t group_size =
          static_cast<std::size_t>(tensor.metadata.group_size);
      const std::size_t groups_per_scale_row = columns / group_size;
      return (row / static_cast<std::size_t>(
                        tensor.metadata.scale_domain_rows)) *
                 groups_per_scale_row +
             column / group_size;
    }
    default: {
      const std::size_t group = group_for_column(tensor, column);
      const std::size_t rows =
          static_cast<std::size_t>(tensor.metadata.logical_rows);
      return row * (tensor.metadata.scale_count / rows) + group;
    }
  }
}

Status quantize_u4_affine(const float* input, long long rows,
                          long long columns, long long group_size,
                          CanonicalQuantTensor* tensor) {
  if (group_size <= 0 || columns % group_size != 0 ||
      (columns & 1) != 0) {
    return Status::kInvalidShape;
  }
  const std::size_t elements =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
  const std::size_t groups = static_cast<std::size_t>(rows) *
                             static_cast<std::size_t>(columns / group_size);
  tensor->data.assign(elements / 2, 0);
  tensor->scales.resize(groups);
  tensor->zero_points.resize(groups);
  tensor->metadata.scale_mode = QuantScaleMode::kGroup;
  tensor->metadata.scale_encoding = QuantScaleEncoding::kFP32;
  tensor->metadata.zero_point_mode = QuantZeroPointMode::kFractional;
  tensor->metadata.scale_count = groups;
  tensor->metadata.zero_point_count = groups;
  for (std::size_t group = 0; group < groups; ++group) {
    const std::size_t begin = group * static_cast<std::size_t>(group_size);
    const std::size_t end = begin + static_cast<std::size_t>(group_size);
    float minimum = input[begin];
    float maximum = input[begin];
    for (std::size_t item = begin + 1; item < end; ++item) {
      minimum = std::min(minimum, input[item]);
      maximum = std::max(maximum, input[item]);
    }
    float scale = (maximum - minimum) / 15.0f;
    float zero = 0.0f;
    if (scale > 0.0f) {
      zero = -minimum / scale;
    } else if (maximum > 0.0f) {
      scale = maximum / 15.0f;
    } else if (minimum < 0.0f) {
      scale = -minimum / 15.0f;
      zero = 15.0f;
    }
    tensor->scales[group] = scale;
    tensor->zero_points[group] = zero;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (std::size_t item = begin; item < end; ++item) {
      const int code = std::clamp(
          static_cast<int>(std::nearbyint(input[item] * inverse + zero)), 0,
          15);
      put_nibble(&tensor->data, item, static_cast<std::uint8_t>(code));
    }
  }
  return Status::kOk;
}

Status quantize_mx(const float* input, long long rows, long long columns,
                   bool fp4, CanonicalQuantTensor* tensor) {
  if (columns % 32 != 0) return Status::kInvalidShape;
  const std::size_t blocks = static_cast<std::size_t>(rows) *
                             static_cast<std::size_t>(columns / 32);
  const std::size_t block_bytes = fp4 ? 17 : 33;
  tensor->data.assign(blocks * block_bytes, 0);
  tensor->metadata.group_size = 32;
  tensor->metadata.scale_mode = QuantScaleMode::kMicroscaleK32;
  tensor->metadata.scale_encoding = QuantScaleEncoding::kE8M0;
  tensor->metadata.scale_count = blocks;
  const float maximum_code = fp4 ? 6.0f : 448.0f;
  for (std::size_t block = 0; block < blocks; ++block) {
    const std::size_t input_base = block * 32;
    float maximum = 0.0f;
    for (std::size_t item = 0; item < 32; ++item) {
      maximum = std::max(maximum, std::fabs(input[input_base + item]));
    }
    const std::uint8_t scale_code =
        e8m0_encode_scale(maximum / maximum_code);
    const float scale = maximum == 0.0f ? 0.0f
                                        : e8m0_decode_scale(scale_code);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    std::uint8_t* destination = tensor->data.data() + block * block_bytes;
    destination[0] = scale_code;
    if (fp4) {
      for (std::size_t pair = 0; pair < 16; ++pair) {
        const auto low = fp4_e2m1_encode(input[input_base + 2 * pair] * inverse);
        const auto high =
            fp4_e2m1_encode(input[input_base + 2 * pair + 1] * inverse);
        destination[1 + pair] =
            static_cast<std::uint8_t>(low | (high << 4));
      }
    } else {
      for (std::size_t item = 0; item < 32; ++item) {
        destination[1 + item] = float8_encode(
            input[input_base + item] * inverse, Float8Format::kE4M3FN);
      }
    }
  }
  return Status::kOk;
}

}  // namespace

Status validate_canonical_quant_tensor(const CanonicalQuantTensor& tensor) {
  Status status = validate_quant_metadata(tensor.metadata);
  if (status != Status::kOk) return status;
  const auto mode = tensor.metadata.scale_mode;
  const auto encoding = tensor.metadata.scale_encoding;
  switch (tensor.metadata.layout) {
    case CanonicalQuantLayout::kInt4Symmetric:
    case CanonicalQuantLayout::kUInt4Affine:
      if ((mode != QuantScaleMode::kRow && mode != QuantScaleMode::kChannel &&
           mode != QuantScaleMode::kGroup) ||
          encoding != QuantScaleEncoding::kFP32)
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kInt8Symmetric:
    case CanonicalQuantLayout::kInt8Affine:
      if ((mode != QuantScaleMode::kRow && mode != QuantScaleMode::kChannel &&
           mode != QuantScaleMode::kGroup) ||
          encoding != QuantScaleEncoding::kFP32)
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
      if (mode != QuantScaleMode::kNone &&
          (encoding != QuantScaleEncoding::kFP32 ||
           (mode != QuantScaleMode::kTensor && mode != QuantScaleMode::kRow &&
            mode != QuantScaleMode::kChannel &&
            mode != QuantScaleMode::kGroup &&
            mode != QuantScaleMode::kBlock)))
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kFP4E2M1:
      if ((encoding != QuantScaleEncoding::kFP16 &&
           encoding != QuantScaleEncoding::kFP32) ||
          (mode != QuantScaleMode::kTensor && mode != QuantScaleMode::kRow &&
           mode != QuantScaleMode::kChannel &&
           mode != QuantScaleMode::kGroup &&
           mode != QuantScaleMode::kBlock))
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
    case CanonicalQuantLayout::kMXFP4E2M1E8M0:
      if (mode != QuantScaleMode::kMicroscaleK32 ||
          encoding != QuantScaleEncoding::kE8M0)
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kNVFP4E2M1E4M3:
      if (mode != QuantScaleMode::kNVFP4K16 ||
          encoding != QuantScaleEncoding::kE4M3FN)
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kBitNetTernary:
      if (mode != QuantScaleMode::kBlock ||
          encoding != QuantScaleEncoding::kFP16)
        return Status::kInvalidArgument;
      break;
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      break;
  }
  std::size_t bytes = 0;
  status = expected_data_bytes(tensor.metadata, &bytes);
  if (status != Status::kOk || tensor.data.size() != bytes) {
    return status == Status::kOk ? Status::kInvalidShape : status;
  }
  const bool embedded_scales =
      tensor.metadata.layout == CanonicalQuantLayout::kMXFP8E4M3E8M0 ||
      tensor.metadata.layout == CanonicalQuantLayout::kMXFP4E2M1E8M0 ||
      tensor.metadata.layout == CanonicalQuantLayout::kBitNetTernary;
  const bool split_scales =
      tensor.metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3;
  if (embedded_scales) {
    if (!tensor.scales.empty() || !tensor.scale_codes.empty())
      return Status::kInvalidShape;
  } else if (split_scales) {
    if (!tensor.scales.empty() ||
        tensor.scale_codes.size() != tensor.metadata.scale_count)
      return Status::kInvalidShape;
  } else if (tensor.scales.size() != tensor.metadata.scale_count ||
             !tensor.scale_codes.empty()) {
    return Status::kInvalidShape;
  }
  if (tensor.zero_points.size() != tensor.metadata.zero_point_count ||
      tensor.group_index.size() != tensor.metadata.group_index_count ||
      tensor.act_order.size() != tensor.metadata.act_order_count ||
      !finite_nonnegative_values(tensor.scales.data(), tensor.scales.size()) ||
      !finite_values(tensor.zero_points.data(), tensor.zero_points.size()) ||
      !finite_values(tensor.activation_scales.data(),
                     tensor.activation_scales.size())) {
    return Status::kInvalidShape;
  }
  const std::size_t columns =
      static_cast<std::size_t>(tensor.metadata.logical_columns);
  if (!valid_permutation(tensor.act_order, columns))
    return Status::kInvalidArgument;
  if (!tensor.group_index.empty()) {
    const std::size_t groups = tensor.metadata.scale_count /
                               static_cast<std::size_t>(
                                   tensor.metadata.logical_rows);
    for (int group : tensor.group_index) {
      if (group < 0 || static_cast<std::size_t>(group) >= groups)
        return Status::kInvalidArgument;
    }
  }
  if (!tensor.activation_zero_points.empty() &&
      tensor.activation_zero_points.size() !=
          tensor.activation_scales.size()) {
    return Status::kInvalidShape;
  }
  if (!tensor.row_sums.empty() &&
      tensor.row_sums.size() !=
          static_cast<std::size_t>(tensor.metadata.logical_rows)) {
    return Status::kInvalidShape;
  }
  if (tensor.metadata.layout ==
      CanonicalQuantLayout::kMXFP8E4M3E8M0) {
    for (std::size_t block = 0; block < tensor.data.size(); block += 33) {
      if (tensor.data[block] == 0xff) return Status::kInvalidArgument;
    }
  }
  if (tensor.metadata.layout ==
      CanonicalQuantLayout::kMXFP4E2M1E8M0) {
    for (std::size_t block = 0; block < tensor.data.size(); block += 17) {
      if (tensor.data[block] == 0xff) return Status::kInvalidArgument;
    }
  }
  if (tensor.metadata.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
    for (std::uint8_t code : tensor.scale_codes) {
      if (code > 0x7e) return Status::kInvalidArgument;
    }
  }
  if (tensor.metadata.layout == CanonicalQuantLayout::kBitNetTernary) {
    for (std::size_t block = 0; block < tensor.data.size(); block += 10) {
      const std::uint16_t scale_bits = static_cast<std::uint16_t>(
          tensor.data[block] | (static_cast<std::uint16_t>(
                                    tensor.data[block + 1])
                                << 8));
      const float scale = f16_to_float(scale_bits);
      if (!std::isfinite(scale) || scale < 0.0f)
        return Status::kInvalidArgument;
      for (std::size_t byte = 0; byte < 8; ++byte) {
        const std::uint8_t codes = tensor.data[block + 2 + byte];
        for (int lane = 0; lane < 4; ++lane) {
          if (((codes >> (2 * lane)) & 3U) == 3U)
            return Status::kInvalidArgument;
        }
      }
    }
  }
  return Status::kOk;
}

Status quantize_canonical(FloatStorageInput input, long long rows,
                          long long columns, CanonicalQuantLayout layout,
                          long long group_size, CanonicalQuantTensor* tensor,
                          bool scale_2d) {
  if (tensor == nullptr) return Status::kInvalidArgument;
  std::size_t elements = 0;
  if (!checked_product(rows, columns, &elements)) return Status::kInvalidShape;
  try {
    std::vector<float> storage;
    const float* values = nullptr;
    Status status = stage_float_input(input, elements, &storage, &values);
    if (status != Status::kOk) return status;
    CanonicalQuantTensor candidate;
    initialize_metadata(&candidate, layout, rows, columns, group_size);
    switch (layout) {
      case CanonicalQuantLayout::kInt4Symmetric: {
        if (group_size <= 0 || columns % group_size != 0 ||
            (columns & 1) != 0)
          return Status::kInvalidShape;
        candidate.data.resize(elements / 2);
        candidate.scales.resize(static_cast<std::size_t>(rows) *
                                static_cast<std::size_t>(columns / group_size));
        status = quantize_int4_group(values, candidate.data.data(),
                                     candidate.scales.data(), rows, columns,
                                     group_size);
        candidate.metadata.scale_mode = QuantScaleMode::kGroup;
        candidate.metadata.scale_encoding = QuantScaleEncoding::kFP32;
        candidate.metadata.scale_count = candidate.scales.size();
        break;
      }
      case CanonicalQuantLayout::kUInt4Affine:
        status = quantize_u4_affine(values, rows, columns, group_size,
                                    &candidate);
        break;
      case CanonicalQuantLayout::kInt8Symmetric: {
        const long long effective = group_size == 0 ? columns : group_size;
        if (effective <= 0 || columns % effective != 0)
          return Status::kInvalidShape;
        candidate.data.resize(elements);
        candidate.scales.resize(static_cast<std::size_t>(rows) *
                                static_cast<std::size_t>(columns / effective));
        status = quantize_int8(values,
                               reinterpret_cast<std::int8_t*>(
                                   candidate.data.data()),
                               candidate.scales.data(), rows, columns,
                               effective);
        candidate.metadata.group_size = effective;
        candidate.metadata.scale_mode = effective == columns
                                            ? QuantScaleMode::kRow
                                            : QuantScaleMode::kGroup;
        candidate.metadata.scale_encoding = QuantScaleEncoding::kFP32;
        candidate.metadata.scale_count = candidate.scales.size();
        break;
      }
      case CanonicalQuantLayout::kInt8Affine: {
        if (group_size != 0 && group_size != columns)
          return Status::kInvalidShape;
        candidate.data.resize(elements);
        candidate.scales.resize(static_cast<std::size_t>(rows));
        std::vector<int> zeros(static_cast<std::size_t>(rows));
        status = quantize_int8_asymmetric(
            values, reinterpret_cast<std::int8_t*>(candidate.data.data()),
            candidate.scales.data(), zeros.data(), rows, columns);
        candidate.zero_points.assign(zeros.begin(), zeros.end());
        candidate.metadata.group_size = columns;
        candidate.metadata.scale_mode = QuantScaleMode::kRow;
        candidate.metadata.scale_encoding = QuantScaleEncoding::kFP32;
        candidate.metadata.zero_point_mode = QuantZeroPointMode::kInteger;
        candidate.metadata.scale_count = candidate.scales.size();
        candidate.metadata.zero_point_count = candidate.zero_points.size();
        break;
      }
      case CanonicalQuantLayout::kFP8E4M3FN:
      case CanonicalQuantLayout::kFP8E5M2:
      case CanonicalQuantLayout::kFP4E2M1:
        status = quantize_grouped_fp(values, rows, columns, group_size, layout,
                                     QuantScaleMode::kNone, 1, &candidate);
        break;
      case CanonicalQuantLayout::kMXFP8E4M3E8M0:
        status = quantize_mx(values, rows, columns, false, &candidate);
        break;
      case CanonicalQuantLayout::kMXFP4E2M1E8M0:
        status = quantize_mx(values, rows, columns, true, &candidate);
        break;
      case CanonicalQuantLayout::kNVFP4E2M1E4M3: {
        if (columns % 16 != 0) return Status::kInvalidShape;
        candidate.data.resize(elements / 2);
        candidate.scale_codes.resize(static_cast<std::size_t>(rows) *
                                     static_cast<std::size_t>(columns / 16));
        status = nvfp4_quantize(values, candidate.data.data(),
                                candidate.scale_codes.data(),
                                &candidate.metadata.global_scale, rows,
                                columns, scale_2d);
        candidate.metadata.group_size = 16;
        candidate.metadata.scale_mode = QuantScaleMode::kNVFP4K16;
        candidate.metadata.scale_encoding = QuantScaleEncoding::kE4M3FN;
        candidate.metadata.scale_count = candidate.scale_codes.size();
        candidate.metadata.scale_2d = scale_2d;
        candidate.metadata.scale_domain_rows = scale_2d ? 16 : 1;
        break;
      }
      case CanonicalQuantLayout::kBitNetTernary: {
        const long long effective = group_size == 0 ? 32 : group_size;
        if (columns % effective != 0 || effective % 32 != 0)
          return Status::kInvalidShape;
        candidate.data.resize(elements / 32 * 10);
        std::vector<float> ignored(elements);
        status = ternary_pack(values, candidate.data.data(), ignored.data(),
                              rows, columns, effective);
        candidate.metadata.group_size = effective;
        candidate.metadata.scale_mode = QuantScaleMode::kBlock;
        candidate.metadata.scale_encoding = QuantScaleEncoding::kFP16;
        candidate.metadata.scale_count = elements / 32;
        break;
      }
      case CanonicalQuantLayout::kTurboQuantKey:
      case CanonicalQuantLayout::kTurboQuantValue:
        return Status::kUnsupportedFormat;
    }
    if (status != Status::kOk) return status;
    status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status quantize_canonical_fp_scaled(
    FloatStorageInput input, long long rows, long long columns,
    CanonicalQuantLayout layout, QuantScaleMode scale_mode,
    long long group_size, int scale_domain_rows,
    CanonicalQuantTensor* tensor) {
  if (tensor == nullptr) return Status::kInvalidArgument;
  if (layout != CanonicalQuantLayout::kFP8E4M3FN &&
      layout != CanonicalQuantLayout::kFP8E5M2 &&
      layout != CanonicalQuantLayout::kFP4E2M1) {
    return Status::kUnsupportedFormat;
  }
  std::size_t elements = 0;
  if (!checked_product(rows, columns, &elements)) return Status::kInvalidShape;
  try {
    std::vector<float> storage;
    const float* values = nullptr;
    Status status = stage_float_input(input, elements, &storage, &values);
    if (status != Status::kOk) return status;
    CanonicalQuantTensor candidate;
    initialize_metadata(&candidate, layout, rows, columns, group_size);
    status = quantize_grouped_fp(values, rows, columns, group_size, layout,
                                 scale_mode, scale_domain_rows, &candidate);
    if (status != Status::kOk) return status;
    status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status dequantize_canonical(const CanonicalQuantTensor& tensor, float* output,
                            long long output_count) {
  if (output == nullptr) return Status::kInvalidArgument;
  Status status = validate_canonical_quant_tensor(tensor);
  if (status != Status::kOk) return status;
  std::size_t elements = 0;
  if (!checked_product(tensor.metadata.logical_rows,
                       tensor.metadata.logical_columns, &elements) ||
      output_count <= 0 || static_cast<std::size_t>(output_count) != elements) {
    return Status::kInvalidShape;
  }
  const std::size_t rows =
      static_cast<std::size_t>(tensor.metadata.logical_rows);
  const std::size_t columns =
      static_cast<std::size_t>(tensor.metadata.logical_columns);
  switch (tensor.metadata.layout) {
    case CanonicalQuantLayout::kInt4Symmetric:
    case CanonicalQuantLayout::kUInt4Affine:
    case CanonicalQuantLayout::kFP4E2M1: {
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
          const std::size_t element = row * columns + column;
          const std::uint8_t code = nibble_at(tensor.data, element);
          const std::size_t scale_index =
              tensor_scale_index(tensor, row, column);
          if (tensor.metadata.layout == CanonicalQuantLayout::kInt4Symmetric) {
            const int signed_code = code >= 8 ? code - 16 : code;
            output[element] = tensor.scales[scale_index] * signed_code;
          } else if (tensor.metadata.layout ==
                     CanonicalQuantLayout::kUInt4Affine) {
            output[element] = tensor.scales[scale_index] *
                              (static_cast<float>(code) -
                               tensor.zero_points[scale_index]);
          } else {
            output[element] = tensor.scales[scale_index] *
                              fp4_e2m1_decode(code);
          }
        }
      }
      return Status::kOk;
    }
    case CanonicalQuantLayout::kInt8Symmetric:
    case CanonicalQuantLayout::kInt8Affine:
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2: {
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
          const std::size_t element = row * columns + column;
          const std::size_t scale_index =
              tensor_scale_index(tensor, row, column);
          if (tensor.metadata.layout ==
                  CanonicalQuantLayout::kInt8Symmetric ||
              tensor.metadata.layout == CanonicalQuantLayout::kInt8Affine) {
            const int code = static_cast<std::int8_t>(tensor.data[element]);
            const float zero = tensor.zero_points.empty()
                                   ? 0.0f
                                   : tensor.zero_points[scale_index];
            output[element] = tensor.scales[scale_index] * (code - zero);
          } else {
            const auto format =
                tensor.metadata.layout == CanonicalQuantLayout::kFP8E4M3FN
                    ? Float8Format::kE4M3FN
                    : Float8Format::kE5M2;
            const float scale =
                tensor.scales.empty() ? 1.0f : tensor.scales[scale_index];
            output[element] = scale *
                              float8_decode(tensor.data[element], format);
          }
        }
      }
      return Status::kOk;
    }
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
    case CanonicalQuantLayout::kMXFP4E2M1E8M0: {
      const bool fp4 = tensor.metadata.layout ==
                       CanonicalQuantLayout::kMXFP4E2M1E8M0;
      const std::size_t block_bytes = fp4 ? 17 : 33;
      const std::size_t blocks = elements / 32;
      for (std::size_t block = 0; block < blocks; ++block) {
        const std::uint8_t* source = tensor.data.data() + block * block_bytes;
        const float scale = e8m0_decode_scale(source[0]);
        for (std::size_t item = 0; item < 32; ++item) {
          const std::uint8_t code =
              fp4 ? static_cast<std::uint8_t>(
                        (source[1 + item / 2] >> (4 * (item & 1))) & 0x0f)
                  : source[1 + item];
          output[block * 32 + item] =
              scale * (fp4 ? fp4_e2m1_decode(code)
                           : float8_decode(code, Float8Format::kE4M3FN));
        }
      }
      return Status::kOk;
    }
    case CanonicalQuantLayout::kNVFP4E2M1E4M3: {
      const std::size_t blocks_per_row = columns / 16;
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
          const float scale = tensor.metadata.global_scale *
                              float8_decode(
                                  tensor.scale_codes[row * blocks_per_row +
                                                     block],
                                  Float8Format::kE4M3FN);
          for (std::size_t item = 0; item < 16; ++item) {
            const std::size_t column = block * 16 + item;
            const std::size_t element = row * columns + column;
            output[element] = scale * fp4_e2m1_decode(
                                          nibble_at(tensor.data, element));
          }
        }
      }
      return Status::kOk;
    }
    case CanonicalQuantLayout::kBitNetTernary:
      return ternary_unpack(tensor.data.data(), output,
                            tensor.metadata.logical_rows,
                            tensor.metadata.logical_columns);
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return Status::kUnsupportedFormat;
  }
  return Status::kUnsupportedFormat;
}

Status import_awq_u4(const AwqU4Source& source,
                     CanonicalQuantTensor* tensor) {
  if (tensor == nullptr || source.qweight == nullptr ||
      source.qzeros == nullptr || source.scales == nullptr ||
      source.input_features <= 0 || source.output_features <= 0 ||
      source.group_size <= 0 ||
      source.input_features % source.group_size != 0 ||
      source.output_features % 8 != 0 || source.input_features % 2 != 0) {
    return Status::kInvalidArgument;
  }
  const std::size_t k = static_cast<std::size_t>(source.input_features);
  const std::size_t n = static_cast<std::size_t>(source.output_features);
  const std::size_t groups = k / static_cast<std::size_t>(source.group_size);
  if (source.qweight_words != k * (n / 8) ||
      source.qzero_words != groups * (n / 8) ||
      source.scale_count != groups * n ||
      !finite_values(source.scales, source.scale_count)) {
    return Status::kInvalidShape;
  }
  try {
    CanonicalQuantTensor candidate;
    initialize_metadata(&candidate, CanonicalQuantLayout::kUInt4Affine,
                        source.output_features, source.input_features,
                        source.group_size);
    candidate.provenance = QuantImportProvenance::kAWQ;
    candidate.data.assign(n * k / 2, 0);
    candidate.scales.resize(n * groups);
    candidate.zero_points.resize(n * groups);
    for (std::size_t output = 0; output < n; ++output) {
      const std::size_t word_column = output / 8;
      const int physical_lane = kAwqReverseOrder[output & 7];
      for (std::size_t input = 0; input < k; ++input) {
        const std::uint32_t word =
            source.qweight[input * (n / 8) + word_column];
        const std::uint8_t code = static_cast<std::uint8_t>(
            (word >> (4 * physical_lane)) & 0x0fU);
        put_nibble(&candidate.data, output * k + input, code);
      }
      for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t source_index = group * n + output;
        const std::uint32_t word =
            source.qzeros[group * (n / 8) + word_column];
        const std::uint8_t zero = static_cast<std::uint8_t>(
            (word >> (4 * physical_lane)) & 0x0fU);
        candidate.scales[output * groups + group] =
            source.scales[source_index];
        candidate.zero_points[output * groups + group] = zero;
      }
    }
    candidate.metadata.scale_mode = QuantScaleMode::kGroup;
    candidate.metadata.scale_encoding = QuantScaleEncoding::kFP32;
    candidate.metadata.zero_point_mode = QuantZeroPointMode::kFractional;
    candidate.metadata.scale_count = candidate.scales.size();
    candidate.metadata.zero_point_count = candidate.zero_points.size();
    const Status status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status import_gptq_u4(const GptqU4Source& source,
                      CanonicalQuantTensor* tensor) {
  if (tensor == nullptr || source.qweight == nullptr ||
      source.qzeros == nullptr || source.scales == nullptr ||
      source.input_features <= 0 || source.output_features <= 0 ||
      source.input_features % 8 != 0 || source.output_features % 8 != 0 ||
      source.group_size <= 0 ||
      source.input_features % source.group_size != 0) {
    return Status::kInvalidArgument;
  }
  const std::size_t k = static_cast<std::size_t>(source.input_features);
  const std::size_t n = static_cast<std::size_t>(source.output_features);
  const std::size_t groups = k / static_cast<std::size_t>(source.group_size);
  if (source.qweight_words != (k / 8) * n ||
      source.qzero_words != groups * (n / 8) ||
      source.scale_count != groups * n ||
      (source.group_index_count != 0 &&
       (source.group_index == nullptr || source.group_index_count != k)) ||
      (source.act_order_count != 0 &&
       (source.act_order == nullptr || source.act_order_count != k)) ||
      !finite_values(source.scales, source.scale_count)) {
    return Status::kInvalidShape;
  }
  try {
    CanonicalQuantTensor candidate;
    initialize_metadata(
        &candidate, source.symmetric ? CanonicalQuantLayout::kInt4Symmetric
                                     : CanonicalQuantLayout::kUInt4Affine,
        source.output_features, source.input_features, source.group_size);
    candidate.provenance = QuantImportProvenance::kGPTQ;
    candidate.data.assign(n * k / 2, 0);
    candidate.scales.resize(n * groups);
    if (!source.symmetric) candidate.zero_points.resize(n * groups);
    for (std::size_t output = 0; output < n; ++output) {
      for (std::size_t input = 0; input < k; ++input) {
        const std::uint32_t word = source.qweight[(input / 8) * n + output];
        const int raw = static_cast<int>((word >> (4 * (input & 7))) & 0x0fU);
        int code = raw;
        if (source.symmetric) {
          const std::size_t zero_group =
              source.group_index_count == 0
                  ? input / static_cast<std::size_t>(source.group_size)
                  : static_cast<std::size_t>(source.group_index[input]);
          if (zero_group >= groups) return Status::kInvalidArgument;
          const std::uint32_t zero_word =
              source.qzeros[zero_group * (n / 8) + output / 8];
          int zero = static_cast<int>(
              (zero_word >> (4 * (output & 7))) & 0x0fU);
          if (!source.gptq_v2) zero = (zero + 1) & 0x0f;
          code = std::clamp(raw - zero, -8, 7) & 0x0f;
        }
        put_nibble(&candidate.data, output * k + input,
                   static_cast<std::uint8_t>(code));
      }
      for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t source_index = group * n + output;
        candidate.scales[output * groups + group] =
            source.scales[source_index];
        if (!source.symmetric) {
          const std::uint32_t word =
              source.qzeros[group * (n / 8) + output / 8];
          int zero =
              static_cast<int>((word >> (4 * (output & 7))) & 0x0fU);
          if (!source.gptq_v2) zero = (zero + 1) & 0x0f;
          candidate.zero_points[output * groups + group] =
              static_cast<float>(zero);
        }
      }
    }
    candidate.metadata.scale_mode = QuantScaleMode::kGroup;
    candidate.metadata.scale_encoding = QuantScaleEncoding::kFP32;
    candidate.metadata.scale_count = candidate.scales.size();
    if (!source.symmetric) {
      candidate.metadata.zero_point_mode = QuantZeroPointMode::kFractional;
      candidate.metadata.zero_point_count = candidate.zero_points.size();
    }
    if (source.group_index_count != 0) {
      candidate.group_index.assign(source.group_index,
                                   source.group_index + k);
      candidate.metadata.group_index_count = k;
    }
    if (source.act_order_count != 0) {
      candidate.act_order.assign(source.act_order, source.act_order + k);
      candidate.metadata.act_order_count = k;
    }
    const Status status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status import_autoround_gptq_u4(const GptqU4Source& source,
                                CanonicalQuantTensor* tensor) {
  Status status = import_gptq_u4(source, tensor);
  if (status == Status::kOk)
    tensor->provenance = QuantImportProvenance::kAutoRound;
  return status;
}

Status import_autoround_canonical(const AutoRoundCanonicalSource& source,
                                  CanonicalQuantTensor* tensor) {
  if (tensor == nullptr || (source.data_bytes != 0 && source.data == nullptr) ||
      (source.scale_code_count != 0 && source.scale_codes == nullptr) ||
      (source.scale_count != 0 && source.scales == nullptr) ||
      (source.zero_point_count != 0 && source.zero_points == nullptr) ||
      (source.group_index_count != 0 && source.group_index == nullptr) ||
      (source.act_order_count != 0 && source.act_order == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (source.metadata.scale_count != source.scale_count &&
      source.metadata.layout != CanonicalQuantLayout::kMXFP8E4M3E8M0 &&
      source.metadata.layout != CanonicalQuantLayout::kMXFP4E2M1E8M0 &&
      source.metadata.layout != CanonicalQuantLayout::kNVFP4E2M1E4M3 &&
      source.metadata.layout != CanonicalQuantLayout::kBitNetTernary) {
    return Status::kInvalidShape;
  }
  try {
    CanonicalQuantTensor candidate;
    candidate.metadata = source.metadata;
    copy_optional(source.data, source.data_bytes, &candidate.data);
    copy_optional(source.scale_codes, source.scale_code_count,
                  &candidate.scale_codes);
    copy_optional(source.scales, source.scale_count, &candidate.scales);
    copy_optional(source.zero_points, source.zero_point_count,
                  &candidate.zero_points);
    copy_optional(source.group_index, source.group_index_count,
                  &candidate.group_index);
    copy_optional(source.act_order, source.act_order_count,
                  &candidate.act_order);
    candidate.provenance = QuantImportProvenance::kAutoRound;
    const Status status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status import_smoothquant_w8a8(const SmoothQuantW8A8Source& source,
                               CanonicalQuantTensor* tensor) {
  std::size_t elements = 0;
  if (tensor == nullptr || source.weights == nullptr ||
      source.weight_scales == nullptr || source.activation_scales == nullptr ||
      !checked_product(source.rows, source.columns, &elements)) {
    return Status::kInvalidArgument;
  }
  if (source.weight_count != elements ||
      source.weight_scale_count != static_cast<std::size_t>(source.rows) ||
      source.activation_scale_count == 0 ||
      (source.activation_zero_point_count != 0 &&
       (source.activation_zero_points == nullptr ||
        source.activation_zero_point_count != source.activation_scale_count)) ||
      (source.row_sum_count != 0 &&
       (source.row_sums == nullptr ||
        source.row_sum_count != static_cast<std::size_t>(source.rows))) ||
      !finite_values(source.weight_scales, source.weight_scale_count) ||
      !finite_values(source.activation_scales,
                     source.activation_scale_count)) {
    return Status::kInvalidShape;
  }
  try {
    CanonicalQuantTensor candidate;
    initialize_metadata(&candidate, CanonicalQuantLayout::kInt8Symmetric,
                        source.rows, source.columns, source.columns);
    candidate.provenance = QuantImportProvenance::kSmoothQuant;
    const auto* weight_bytes =
        reinterpret_cast<const std::uint8_t*>(source.weights);
    candidate.data.assign(weight_bytes, weight_bytes + elements);
    candidate.scales.assign(source.weight_scales,
                            source.weight_scales + source.weight_scale_count);
    candidate.activation_scales.assign(
        source.activation_scales,
        source.activation_scales + source.activation_scale_count);
    if (source.activation_zero_point_count != 0) {
      candidate.activation_zero_points.assign(
          source.activation_zero_points,
          source.activation_zero_points + source.activation_zero_point_count);
    }
    candidate.row_sums.resize(static_cast<std::size_t>(source.rows));
    if (source.row_sum_count != 0) {
      std::copy_n(source.row_sums, source.row_sum_count,
                  candidate.row_sums.begin());
    } else {
      for (long long row = 0; row < source.rows; ++row) {
        std::int32_t sum = 0;
        for (long long column = 0; column < source.columns; ++column) {
          sum += source.weights[row * source.columns + column];
        }
        candidate.row_sums[static_cast<std::size_t>(row)] = sum;
      }
    }
    candidate.metadata.scale_mode = QuantScaleMode::kChannel;
    candidate.metadata.scale_encoding = QuantScaleEncoding::kFP32;
    candidate.metadata.scale_count = candidate.scales.size();
    const Status status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status import_bitnet_i2_s(const BitNetI2Source& source,
                          CanonicalQuantTensor* tensor) {
  std::size_t elements = 0;
  if (tensor == nullptr || source.packed == nullptr ||
      !checked_product(source.rows, source.columns, &elements)) {
    return Status::kInvalidArgument;
  }
  if (source.columns % 32 != 0 || source.group_size <= 0 ||
      source.group_size % 32 != 0 ||
      source.columns % source.group_size != 0 ||
      source.packed_bytes != elements / 32 * 10) {
    return Status::kInvalidShape;
  }
  try {
    CanonicalQuantTensor candidate;
    initialize_metadata(&candidate, CanonicalQuantLayout::kBitNetTernary,
                        source.rows, source.columns, source.group_size);
    candidate.data.assign(source.packed,
                          source.packed + source.packed_bytes);
    candidate.metadata.scale_mode = QuantScaleMode::kBlock;
    candidate.metadata.scale_encoding = QuantScaleEncoding::kFP16;
    candidate.metadata.scale_count = elements / 32;
    candidate.provenance = QuantImportProvenance::kBitNet;
    const Status status = validate_canonical_quant_tensor(candidate);
    if (status != Status::kOk) return status;
    *tensor = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

const char* quant_import_provenance_name(QuantImportProvenance provenance) {
  switch (provenance) {
    case QuantImportProvenance::kCanonical:
      return "canonical";
    case QuantImportProvenance::kAWQ:
      return "awq";
    case QuantImportProvenance::kGPTQ:
      return "gptq";
    case QuantImportProvenance::kAutoRound:
      return "autoround";
    case QuantImportProvenance::kSmoothQuant:
      return "smoothquant";
    case QuantImportProvenance::kBitNet:
      return "bitnet";
  }
  return "unknown";
}

}  // namespace quixicore_cpu
