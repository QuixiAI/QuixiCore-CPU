#include "quixicore_cpu/qgemv.h"

#include <cmath>
#include <vector>

#include "kernels/common/validation.h"

namespace quixicore_cpu {

Status qgemv_up_gate(QuantFormat format, const void* packed_up,
                     const void* packed_gate, const float* x, float* up,
                     float* gate, long long n, long long k) {
  if (!detail::valid_product({n, k})) return Status::kInvalidShape;
  if (!detail::all_nonnull(packed_up, packed_gate, x, up, gate)) {
    return Status::kInvalidArgument;
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
  std::vector<float> up(static_cast<std::size_t>(n));
  std::vector<float> gate(static_cast<std::size_t>(n));
  Status status = qgemv_up_gate(format, packed_up, packed_gate, x, up.data(),
                                gate.data(), n, k);
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
