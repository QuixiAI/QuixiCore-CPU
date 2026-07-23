#pragma once

#include <cstdint>

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu::quant {

bool iq_pack_supported(QuantFormat format);
void iq_pack_block_ref(QuantFormat format, const float* values,
                       const float* importance, std::uint8_t* block);

}  // namespace quixicore_cpu::quant
