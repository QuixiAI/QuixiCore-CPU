#pragma once

#include <cstddef>
#include <cstdint>

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu::quant {

bool gguf_format_info(QuantFormat format, long long* block_size,
                      std::size_t* block_bytes);
void gguf_unpack_ref(QuantFormat format, const void* packed, long long n,
                     long long k, float* weights);
void gguf_gemv_ref(QuantFormat format, const void* packed, const float* x,
                   float* y, long long n, long long k);

// Internal element decoder used by packed embedding and LM-head composites.
float gguf_dequant_element(QuantFormat format, const std::uint8_t* block,
                           int column);

}  // namespace quixicore_cpu::quant
