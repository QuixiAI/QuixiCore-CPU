#pragma once

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu::quant {

// Canonical GGML/GGUF packers owned by the CPU backend. These produce the
// contract layout consumed directly by GGUF files; ISA-specific repacking is
// deliberately handled later by CpuPackedWeights.
bool gguf_pack_supported(QuantFormat format);
bool gguf_pack_ref(QuantFormat format, const float* weights, long long n,
                   long long k, void* packed,
                   const float* importance = nullptr);

}  // namespace quixicore_cpu::quant
