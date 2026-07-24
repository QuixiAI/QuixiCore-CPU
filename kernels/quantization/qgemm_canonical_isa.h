#pragma once

#include <cstdint>

#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/quant_import.h"

namespace quixicore_cpu::quant {

#if defined(QUIXICORE_CPU_HAVE_CANONICAL_GEMM_AVX2)

// Accumulate one decoded K chunk into an M x output-panel tile. The portable
// caller owns decoding and preserves the canonical packing contract; this
// helper only supplies the AVX2 row-panel microkernel.
void canonical_accumulate_tile_avx2(float* accumulators,
                                    const float* decoded_weights,
                                    const float* activations,
                                    long long activation_row_stride,
                                    long long first_column, long long items,
                                    long long rows, long long lanes);

// Accumulate one packed dual-quantized K block across adjacent output panels.
// Activations are unpacked once and reused for every panel in the group.
bool canonical_dual_panel_group_avx2(
    CanonicalQuantLayout weight_layout, CanonicalQuantLayout activation_layout,
    const CpuPackedWeightsInfo& info, const std::uint8_t* panel,
    const CanonicalQuantTensor& activation, long long activation_row,
    long long first_panel, long long panel_count, long long block,
    float* accumulators);

#endif

}  // namespace quixicore_cpu::quant
