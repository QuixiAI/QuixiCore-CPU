#pragma once

#include <cstddef>

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu::quant {

bool activation_format_info(QuantActivationFormat format,
                            long long* block_size,
                            std::size_t* block_bytes);

// All public shape and pointer checks happen in the dispatch layer. The pack
// routine still rejects non-finite inputs before any float-to-integer cast.
bool activation_pack_ref(QuantActivationFormat format, const float* input,
                         long long n, long long k, void* packed);
void activation_unpack_ref(QuantActivationFormat format, const void* packed,
                           long long n, long long k, float* output);

}  // namespace quixicore_cpu::quant
