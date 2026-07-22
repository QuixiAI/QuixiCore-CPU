#include "quixicore_cpu/packed_weights.h"

#include <algorithm>
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

namespace quixicore_cpu {
namespace {

class AlignedStorage {
 public:
  AlignedStorage() = default;
  explicit AlignedStorage(std::size_t bytes)
      : data_(bytes == 0
                  ? nullptr
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

}  // namespace

struct CpuPackedWeights::Impl {
  CpuPackedWeightsInfo info;
  std::vector<std::uint8_t> contract;
  AlignedStorage panel;
};

CpuPackedWeights::CpuPackedWeights() = default;
CpuPackedWeights::~CpuPackedWeights() = default;
CpuPackedWeights::CpuPackedWeights(CpuPackedWeights&& other) noexcept = default;
CpuPackedWeights& CpuPackedWeights::operator=(CpuPackedWeights&& other) noexcept =
    default;

Status CpuPackedWeights::prepare(QuantFormat format,
                                 const void* contract_packed, long long rows,
                                 long long columns, CpuPanelLayout layout) {
  if (contract_packed == nullptr) return Status::kInvalidArgument;
  if (layout == CpuPanelLayout::kAuto) layout = automatic_layout();
  const long long tile = row_tile(layout);
  if (tile == 0) return Status::kInvalidArgument;

  std::size_t contract_bytes = 0;
  Status status =
      qgemv_packed_size(format, rows, columns, &contract_bytes);
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
    candidate->info = {format,
                       layout,
                       rows,
                       columns,
                       tile,
                       block_size,
                       blocks_per_row,
                       block_bytes,
                       contract_bytes,
                       panel_bytes};
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

}  // namespace quixicore_cpu
