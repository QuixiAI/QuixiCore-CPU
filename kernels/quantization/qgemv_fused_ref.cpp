#include "quixicore_cpu/qgemv.h"

#include <cstddef>
#include <cmath>
#include <limits>

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "src/memory/workspace_internal.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {

void q4_0_gemv_pair_ref(const BlockQ4_0* packed_up,
                        const BlockQ4_0* packed_gate, const float* x,
                        float* up, float* gate, long long n, long long k) {
  const long long blocks = k / kQ4_0BlockSize;
  threading::parallel_ranges(2 * n, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const bool is_gate = item >= n;
      const long long row_index = is_gate ? item - n : item;
      const BlockQ4_0* row =
          (is_gate ? packed_gate : packed_up) + row_index * blocks;
      double sum = 0.0;
      for (long long block = 0; block < blocks; ++block) {
        const float scale = fp16_to_fp32(row[block].d);
        const float* input = x + block * kQ4_0BlockSize;
        for (int element = 0; element < 16; ++element) {
          sum += scale * float((row[block].qs[element] & 15) - 8) *
                 input[element];
          sum += scale * float((row[block].qs[element] >> 4) - 8) *
                 input[element + 16];
        }
      }
      (is_gate ? gate : up)[row_index] = static_cast<float>(sum);
    }
  });
}

}  // namespace quixicore_cpu::quant

namespace quixicore_cpu {

Status qgemv_up_gate(QuantFormat format, const void* packed_up,
                     const void* packed_gate, const float* x, float* up,
                     float* gate, long long n, long long k) {
  if (!detail::valid_product({n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_up, packed_gate, x, up, gate)) {
    return Status::kInvalidArgument;
  }
  if (format == QuantFormat::kQ4_0) {
    std::size_t ignored = 0;
    const Status status = qgemv_packed_size(format, n, k, &ignored);
    if (status != Status::kOk) return status;
    quant::q4_0_gemv_pair_ref(
        static_cast<const quant::BlockQ4_0*>(packed_up),
        static_cast<const quant::BlockQ4_0*>(packed_gate), x, up, gate, n, k);
    return Status::kOk;
  }
  Status status = qgemv(format, packed_up, x, up, n, k);
  return status == Status::kOk
             ? qgemv(format, packed_gate, x, gate, n, k)
             : status;
}

Status qgemv_up_gate_activation(QuantFormat format, const void* packed_up,
                                const void* packed_gate, const float* x,
                                float* out, long long n, long long k,
                                bool gelu_tanh) {
  if (!detail::valid_product({n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_up, packed_gate, x, out)) {
    return Status::kInvalidArgument;
  }
  if (static_cast<unsigned long long>(n) >
      static_cast<unsigned long long>(
          std::numeric_limits<std::size_t>::max() / 2)) {
    return Status::kInvalidShape;
  }
  detail::WorkspaceFrame workspace;
  float* temporary = workspace.allocate<float>(2 * static_cast<std::size_t>(n));
  if (temporary == nullptr) return Status::kOutOfMemory;
  float* up = temporary;
  float* gate = temporary + n;
  Status status =
      qgemv_up_gate(format, packed_up, packed_gate, x, up, gate, n, k);
  if (status != Status::kOk) return status;
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (long long item = 0; item < n; ++item) {
    const float value = gate[item];
    const float activated =
        gelu_tanh
            ? 0.5f * value *
                  (1.0f + std::tanh(kSqrt2OverPi *
                                    (value + 0.044715f * value * value * value)))
            : value / (1.0f + std::exp(-value));
    out[item] = activated * up[item];
  }
  return Status::kOk;
}

Status qgemv_qkv(QuantFormat format, const void* packed_q,
                 const void* packed_k, const void* packed_v, const float* x,
                 float* q, float* k_out, float* v_out, long long query_dim,
                 long long kv_dim, long long input_dim) {
  if (!detail::valid_product({query_dim, kv_dim, input_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_q, packed_k, packed_v, x, q, k_out, v_out)) {
    return Status::kInvalidArgument;
  }
  Status status = qgemv(format, packed_q, x, q, query_dim, input_dim);
  if (status != Status::kOk) return status;
  status = qgemv(format, packed_k, x, k_out, kv_dim, input_dim);
  return status == Status::kOk
             ? qgemv(format, packed_v, x, v_out, kv_dim, input_dim)
             : status;
}

}  // namespace quixicore_cpu
