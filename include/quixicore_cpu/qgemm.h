#pragma once

#include "quixicore_cpu/qgemv.h"

namespace quixicore_cpu {

// Quantized matrix multiply with packed W [N,K], activations X [M,K], and
// output Y [M,N]: Y = X @ dequantize(W)^T. Activations and output are f32.
Status qgemm(QuantFormat format, const void* packed_weights, const float* x,
             float* y, long long m, long long n, long long k);

// Fused quantized vocabulary projection + greedy selection. packed_weights is
// [vocab,hidden], hidden_states [rows,hidden], and token_ids [rows]. Lowest id
// wins ties.
Status quantized_lm_head_argmax(QuantFormat format, const void* packed_weights,
                                const float* hidden_states, int* token_ids,
                                long long rows, long long vocab,
                                long long hidden);

}  // namespace quixicore_cpu
