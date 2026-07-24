#include "quixicore_cpu/packed_weights.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kernels/quantization/gguf_ref.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/quant_import.h"

namespace quixicore_cpu {
namespace {

class AlignedStorage {
 public:
  AlignedStorage() = default;
  explicit AlignedStorage(std::size_t bytes)
      : data_(bytes == 0 ? nullptr
                         : static_cast<std::byte*>(::operator new(
                               bytes, std::align_val_t(WorkspaceAlignment)))) {}
  ~AlignedStorage() { clear(); }
  AlignedStorage(AlignedStorage&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)) {}
  AlignedStorage& operator=(AlignedStorage&& other) noexcept {
    if (this == &other) return *this;
    clear();
    data_ = std::exchange(other.data_, nullptr);
    return *this;
  }
  AlignedStorage(const AlignedStorage&) = delete;
  AlignedStorage& operator=(const AlignedStorage&) = delete;

  std::byte* data() { return data_; }
  const std::byte* data() const { return data_; }

 private:
  static constexpr std::size_t WorkspaceAlignment = 64;
  void clear() noexcept {
    if (data_ != nullptr) {
      ::operator delete(data_, std::align_val_t(WorkspaceAlignment));
    }
    data_ = nullptr;
  }

  std::byte* data_ = nullptr;
};

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
  *result = lhs + rhs;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t* result) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  return checked_add(value, alignment - remainder, result);
}

bool canonical_block_geometry(const QuantTensorMetadata& metadata,
                              long long* block_elements,
                              std::size_t* block_bytes) {
  switch (metadata.layout) {
    case CanonicalQuantLayout::kInt4Symmetric:
    case CanonicalQuantLayout::kUInt4Affine:
    case CanonicalQuantLayout::kFP4E2M1:
      *block_elements = metadata.group_size > 0 ? metadata.group_size
                                                : metadata.logical_columns;
      *block_bytes = static_cast<std::size_t>(*block_elements / 2);
      return *block_elements > 0 && (*block_elements & 1) == 0;
    case CanonicalQuantLayout::kInt8Symmetric:
    case CanonicalQuantLayout::kInt8Affine:
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
      *block_elements = metadata.group_size > 0 ? metadata.group_size
                                                : metadata.logical_columns;
      *block_bytes = static_cast<std::size_t>(*block_elements);
      return *block_elements > 0;
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
      *block_elements = 32;
      *block_bytes = 33;
      return true;
    case CanonicalQuantLayout::kMXFP4E2M1E8M0:
      *block_elements = 32;
      *block_bytes = 17;
      return true;
    case CanonicalQuantLayout::kNVFP4E2M1E4M3:
      *block_elements = 16;
      *block_bytes = 8;
      return true;
    case CanonicalQuantLayout::kBitNetTernary:
      *block_elements = 32;
      *block_bytes = 10;
      return true;
    case CanonicalQuantLayout::kTurboQuantKey:
    case CanonicalQuantLayout::kTurboQuantValue:
      return false;
  }
  return false;
}

template <class T>
bool add_table_size(std::size_t count, std::size_t* total) {
  if (count == 0) return true;
  std::size_t offset = 0;
  std::size_t bytes = 0;
  return align_up(*total, alignof(T), &offset) &&
         checked_mul(count, sizeof(T), &bytes) &&
         checked_add(offset, bytes, total);
}

template <class T>
void copy_table(const std::vector<T>& table, std::byte* destination,
                std::size_t* cursor, std::size_t* recorded_offset) {
  if (table.empty()) return;
  std::size_t aligned = 0;
  (void)align_up(*cursor, alignof(T), &aligned);
  *recorded_offset = aligned;
  std::memcpy(destination + aligned, table.data(), table.size() * sizeof(T));
  *cursor = aligned + table.size() * sizeof(T);
}

long long row_tile(CpuPanelLayout layout) {
  switch (layout) {
    case CpuPanelLayout::kPortableRows1:
      return 1;
    case CpuPanelLayout::kNeonRows4:
      return 4;
    case CpuPanelLayout::kAvx2Rows8:
      return 8;
    case CpuPanelLayout::kAvx512Rows16:
      return 16;
    case CpuPanelLayout::kAuto:
      return 0;
  }
  return 0;
}

CpuPanelLayout automatic_layout() {
  const CpuFeatures& features = cpu_features();
  if (features.avx512f) return CpuPanelLayout::kAvx512Rows16;
  if (features.avx2) return CpuPanelLayout::kAvx2Rows8;
  if (features.neon) return CpuPanelLayout::kNeonRows4;
  return CpuPanelLayout::kPortableRows1;
}

CpuPreparedIsa required_isa(CpuPanelLayout layout) {
  switch (layout) {
    case CpuPanelLayout::kAuto:
    case CpuPanelLayout::kPortableRows1:
      return CpuPreparedIsa::kPortable;
    case CpuPanelLayout::kNeonRows4:
      return CpuPreparedIsa::kNeon;
    case CpuPanelLayout::kAvx2Rows8:
      return CpuPreparedIsa::kAvx2;
    case CpuPanelLayout::kAvx512Rows16:
      return CpuPreparedIsa::kAvx512;
  }
  return CpuPreparedIsa::kPortable;
}

}  // namespace

struct CpuPackedWeights::Impl {
  CpuPackedWeightsInfo info;
  std::vector<std::uint8_t> contract;
  AlignedStorage panel;
};

CpuPackedWeights::CpuPackedWeights() = default;
CpuPackedWeights::~CpuPackedWeights() = default;
CpuPackedWeights::CpuPackedWeights(CpuPackedWeights&& other) noexcept = default;
CpuPackedWeights& CpuPackedWeights::operator=(
    CpuPackedWeights&& other) noexcept = default;

Status CpuPackedWeights::prepare(QuantFormat format,
                                 const void* contract_packed, long long rows,
                                 long long columns, CpuPanelLayout layout) {
  const auto preparation_start = std::chrono::steady_clock::now();
  if (contract_packed == nullptr) return Status::kInvalidArgument;
  if (layout == CpuPanelLayout::kAuto) layout = automatic_layout();
  const long long tile = row_tile(layout);
  if (tile == 0) return Status::kInvalidArgument;

  std::size_t contract_bytes = 0;
  Status status = qgemv_packed_size(format, rows, columns, &contract_bytes);
  if (status != Status::kOk) return status;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  const long long blocks_per_row = columns / block_size;
  const long long panels = rows / tile + (rows % tile != 0 ? 1 : 0);
  std::size_t panel_blocks = 0;
  std::size_t panel_bytes = 0;
  if (!checked_mul(static_cast<std::size_t>(panels),
                   static_cast<std::size_t>(blocks_per_row), &panel_blocks) ||
      !checked_mul(panel_blocks, static_cast<std::size_t>(tile),
                   &panel_blocks) ||
      !checked_mul(panel_blocks, block_bytes, &panel_bytes)) {
    return Status::kInvalidShape;
  }

  try {
    auto candidate = std::make_unique<Impl>();
    candidate->info.format = format;
    candidate->info.layout = layout;
    candidate->info.rows = rows;
    candidate->info.columns = columns;
    candidate->info.row_tile = tile;
    candidate->info.block_size = block_size;
    candidate->info.blocks_per_row = blocks_per_row;
    candidate->info.block_bytes = block_bytes;
    candidate->info.contract_bytes = contract_bytes;
    candidate->info.panel_bytes = panel_bytes;
    candidate->info.prepared_version = 1;
    candidate->info.required_isa = required_isa(layout);
    candidate->info.column_tile = block_size;
    candidate->info.panel_alignment = 64;
    candidate->info.prepared_bytes = panel_bytes;
    candidate->info.memory_amplification =
        contract_bytes == 0 ? 0.0
                            : static_cast<double>(panel_bytes) /
                                  static_cast<double>(contract_bytes);
    const auto* source = static_cast<const std::uint8_t*>(contract_packed);
    candidate->contract.assign(source, source + contract_bytes);
    candidate->panel = AlignedStorage(panel_bytes);
    std::memset(candidate->panel.data(), 0, panel_bytes);
    auto* destination =
        reinterpret_cast<std::uint8_t*>(candidate->panel.data());
    for (long long panel = 0; panel < panels; ++panel) {
      for (long long block = 0; block < blocks_per_row; ++block) {
        for (long long lane = 0; lane < tile; ++lane) {
          const long long row = panel * tile + lane;
          if (row >= rows) continue;
          const std::size_t source_block =
              static_cast<std::size_t>(row * blocks_per_row + block);
          const std::size_t destination_block = static_cast<std::size_t>(
              (panel * blocks_per_row + block) * tile + lane);
          std::memcpy(destination + destination_block * block_bytes,
                      source + source_block * block_bytes, block_bytes);
        }
      }
    }
    candidate->info.preparation_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - preparation_start)
            .count());
    impl_ = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

Status CpuPackedWeights::prepare(const CanonicalQuantTensor& tensor,
                                 CpuPanelLayout layout) {
  const auto preparation_start = std::chrono::steady_clock::now();
  Status status = validate_canonical_quant_tensor(tensor);
  if (status != Status::kOk) return status;
  if (layout == CpuPanelLayout::kAuto) layout = automatic_layout();
  const long long tile = row_tile(layout);
  if (tile == 0) return Status::kInvalidArgument;

  long long block_elements = 0;
  std::size_t block_bytes = 0;
  if (!canonical_block_geometry(tensor.metadata, &block_elements,
                                &block_bytes) ||
      tensor.metadata.logical_columns % block_elements != 0) {
    return Status::kUnsupportedFormat;
  }
  const long long rows = tensor.metadata.logical_rows;
  const long long columns = tensor.metadata.logical_columns;
  const long long blocks_per_row = columns / block_elements;
  const long long panels = rows / tile + (rows % tile != 0 ? 1 : 0);
  std::size_t panel_blocks = 0;
  std::size_t panel_bytes = 0;
  if (!checked_mul(static_cast<std::size_t>(panels),
                   static_cast<std::size_t>(blocks_per_row), &panel_blocks) ||
      !checked_mul(panel_blocks, static_cast<std::size_t>(tile),
                   &panel_blocks) ||
      !checked_mul(panel_blocks, block_bytes, &panel_bytes)) {
    return Status::kInvalidShape;
  }

  std::size_t prepared_bytes = panel_bytes;
  if (!add_table_size<std::uint8_t>(tensor.scale_codes.size(),
                                    &prepared_bytes) ||
      !add_table_size<float>(tensor.scales.size(), &prepared_bytes) ||
      !add_table_size<float>(tensor.zero_points.size(), &prepared_bytes) ||
      !add_table_size<float>(tensor.activation_scales.size(),
                             &prepared_bytes) ||
      !add_table_size<int>(tensor.activation_zero_points.size(),
                           &prepared_bytes) ||
      !add_table_size<int>(tensor.group_index.size(), &prepared_bytes) ||
      !add_table_size<int>(tensor.act_order.size(), &prepared_bytes) ||
      !add_table_size<std::int32_t>(tensor.row_sums.size(), &prepared_bytes)) {
    return Status::kInvalidShape;
  }
  std::size_t logical_bytes = tensor.data.size();
  auto add_logical = [&](std::size_t count, std::size_t width) {
    std::size_t bytes = 0;
    return checked_mul(count, width, &bytes) &&
           checked_add(logical_bytes, bytes, &logical_bytes);
  };
  if (!add_logical(tensor.scale_codes.size(), sizeof(std::uint8_t)) ||
      !add_logical(tensor.scales.size(), sizeof(float)) ||
      !add_logical(tensor.zero_points.size(), sizeof(float)) ||
      !add_logical(tensor.activation_scales.size(), sizeof(float)) ||
      !add_logical(tensor.activation_zero_points.size(), sizeof(int)) ||
      !add_logical(tensor.group_index.size(), sizeof(int)) ||
      !add_logical(tensor.act_order.size(), sizeof(int)) ||
      !add_logical(tensor.row_sums.size(), sizeof(std::int32_t))) {
    return Status::kInvalidShape;
  }

  try {
    auto candidate = std::make_unique<Impl>();
    candidate->info.canonical_layout = tensor.metadata.layout;
    candidate->info.quant_metadata = tensor.metadata;
    candidate->info.has_canonical_layout = true;
    candidate->info.layout = layout;
    candidate->info.rows = rows;
    candidate->info.columns = columns;
    candidate->info.row_tile = tile;
    candidate->info.block_size = block_elements;
    candidate->info.blocks_per_row = blocks_per_row;
    candidate->info.block_bytes = block_bytes;
    candidate->info.contract_bytes = tensor.data.size();
    candidate->info.panel_bytes = panel_bytes;
    candidate->info.prepared_version = 2;
    candidate->info.required_isa = required_isa(layout);
    candidate->info.column_tile = block_elements;
    candidate->info.panel_alignment = 64;
    candidate->info.logical_bytes = logical_bytes;
    candidate->info.prepared_bytes = prepared_bytes;
    candidate->info.memory_amplification =
        logical_bytes == 0 ? 0.0
                           : static_cast<double>(prepared_bytes) /
                                 static_cast<double>(logical_bytes);
    candidate->contract = tensor.data;
    candidate->panel = AlignedStorage(prepared_bytes);
    std::memset(candidate->panel.data(), 0, prepared_bytes);
    auto* destination = candidate->panel.data();
    const auto* source = tensor.data.data();
    for (long long panel = 0; panel < panels; ++panel) {
      for (long long block = 0; block < blocks_per_row; ++block) {
        for (long long lane = 0; lane < tile; ++lane) {
          const long long row = panel * tile + lane;
          if (row >= rows) continue;
          const std::size_t source_block =
              static_cast<std::size_t>(row * blocks_per_row + block);
          const std::size_t destination_block = static_cast<std::size_t>(
              (panel * blocks_per_row + block) * tile + lane);
          std::memcpy(destination + destination_block * block_bytes,
                      source + source_block * block_bytes, block_bytes);
        }
      }
    }
    std::size_t cursor = panel_bytes;
    copy_table(tensor.scale_codes, destination, &cursor,
               &candidate->info.scale_code_table_offset);
    copy_table(tensor.scales, destination, &cursor,
               &candidate->info.scale_table_offset);
    copy_table(tensor.zero_points, destination, &cursor,
               &candidate->info.zero_point_table_offset);
    copy_table(tensor.activation_scales, destination, &cursor,
               &candidate->info.activation_scale_offset);
    copy_table(tensor.activation_zero_points, destination, &cursor,
               &candidate->info.activation_zero_point_offset);
    copy_table(tensor.group_index, destination, &cursor,
               &candidate->info.group_index_offset);
    copy_table(tensor.act_order, destination, &cursor,
               &candidate->info.act_order_offset);
    copy_table(tensor.row_sums, destination, &cursor,
               &candidate->info.row_sum_table_offset);
    candidate->info.preparation_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - preparation_start)
            .count());
    impl_ = std::move(candidate);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kOutOfMemory;
  }
}

void CpuPackedWeights::reset() noexcept { impl_.reset(); }
bool CpuPackedWeights::ready() const noexcept { return impl_ != nullptr; }

CpuPackedWeightsInfo CpuPackedWeights::info() const noexcept {
  return impl_ == nullptr ? CpuPackedWeightsInfo{} : impl_->info;
}

const void* CpuPackedWeights::contract_data() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->contract.data();
}

const void* CpuPackedWeights::panel_data() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->panel.data();
}

const char* cpu_panel_layout_name(CpuPanelLayout layout) {
  switch (layout) {
    case CpuPanelLayout::kAuto:
      return "auto";
    case CpuPanelLayout::kPortableRows1:
      return "portable_rows1";
    case CpuPanelLayout::kNeonRows4:
      return "neon_rows4";
    case CpuPanelLayout::kAvx2Rows8:
      return "avx2_rows8";
    case CpuPanelLayout::kAvx512Rows16:
      return "avx512_rows16";
  }
  return "unknown";
}

const char* cpu_prepared_isa_name(CpuPreparedIsa isa) {
  switch (isa) {
    case CpuPreparedIsa::kPortable:
      return "portable";
    case CpuPreparedIsa::kNeon:
      return "neon";
    case CpuPreparedIsa::kDotProd:
      return "dotprod";
    case CpuPreparedIsa::kI8MM:
      return "i8mm";
    case CpuPreparedIsa::kSVE:
      return "sve";
    case CpuPreparedIsa::kSME:
      return "sme";
    case CpuPreparedIsa::kAvx2:
      return "avx2";
    case CpuPreparedIsa::kAvx512:
      return "avx512";
    case CpuPreparedIsa::kAvx512VNNI:
      return "avx512_vnni";
    case CpuPreparedIsa::kAMX:
      return "amx";
  }
  return "unknown";
}

}  // namespace quixicore_cpu
