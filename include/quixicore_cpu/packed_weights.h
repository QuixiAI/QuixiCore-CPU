#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quant_contract.h"

namespace quixicore_cpu {

// CPU-private row-panel layouts. These are never serialized and never replace
// the canonical QuantFormat bytes owned by CpuPackedWeights.
enum class CpuPanelLayout {
  kAuto,
  kPortableRows1,
  kNeonRows4,
  kAvx2Rows8,
  kAvx512Rows16,
};

enum class CpuPreparedIsa {
  kPortable,
  kNeon,
  kDotProd,
  kI8MM,
  kSVE,
  kSME,
  kAvx2,
  kAvx512,
  kAvx512VNNI,
  kAMX,
};

struct CpuPackedWeightsInfo {
  QuantFormat format = QuantFormat::kQ8_0;
  CanonicalQuantLayout canonical_layout =
      CanonicalQuantLayout::kInt4Symmetric;
  bool has_canonical_layout = false;
  CpuPanelLayout layout = CpuPanelLayout::kPortableRows1;
  long long rows = 0;
  long long columns = 0;
  long long row_tile = 0;
  long long block_size = 0;
  long long blocks_per_row = 0;
  std::size_t block_bytes = 0;
  std::size_t contract_bytes = 0;
  std::size_t panel_bytes = 0;
  std::uint32_t prepared_version = 0;
  CpuPreparedIsa required_isa = CpuPreparedIsa::kPortable;
  long long column_tile = 0;
  std::size_t panel_alignment = 0;
  std::size_t scale_table_offset = static_cast<std::size_t>(-1);
  std::size_t scale_code_table_offset = static_cast<std::size_t>(-1);
  std::size_t zero_point_table_offset = static_cast<std::size_t>(-1);
  std::size_t row_sum_table_offset = static_cast<std::size_t>(-1);
  std::size_t group_index_offset = static_cast<std::size_t>(-1);
  std::size_t act_order_offset = static_cast<std::size_t>(-1);
  std::size_t activation_scale_offset = static_cast<std::size_t>(-1);
  std::size_t activation_zero_point_offset = static_cast<std::size_t>(-1);
  std::size_t lut_offset = static_cast<std::size_t>(-1);
  std::size_t logical_bytes = 0;
  std::size_t prepared_bytes = 0;
  double memory_amplification = 0.0;
  std::uint64_t preparation_ns = 0;
};

// Owns canonical QuixiCore/GGUF bytes plus a 64-byte-aligned CPU row panel.
// The panel orders blocks as [row_panel, k_block, row_lane, block_bytes] and
// zero-pads the final row panel. Moving is cheap; copying is deliberately
// disabled so prepared weights have one clear lifetime.
class CpuPackedWeights {
 public:
  CpuPackedWeights();
  ~CpuPackedWeights();
  CpuPackedWeights(CpuPackedWeights&& other) noexcept;
  CpuPackedWeights& operator=(CpuPackedWeights&& other) noexcept;

  CpuPackedWeights(const CpuPackedWeights&) = delete;
  CpuPackedWeights& operator=(const CpuPackedWeights&) = delete;

  Status prepare(QuantFormat format, const void* contract_packed,
                 long long rows, long long columns,
                 CpuPanelLayout layout = CpuPanelLayout::kAuto);
  Status prepare(const struct CanonicalQuantTensor& tensor,
                 CpuPanelLayout layout = CpuPanelLayout::kAuto);
  void reset() noexcept;

  bool ready() const noexcept;
  CpuPackedWeightsInfo info() const noexcept;
  const void* contract_data() const noexcept;
  const void* panel_data() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

const char* cpu_panel_layout_name(CpuPanelLayout layout);
const char* cpu_prepared_isa_name(CpuPreparedIsa isa);

}  // namespace quixicore_cpu
