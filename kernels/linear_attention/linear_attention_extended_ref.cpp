#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

Status validate_linear(const float* q, const float* k, const float* v,
                       float* out, long long batch, long long heads,
                       long long sequence, long long dim) {
  if (!detail::valid_product({batch, heads, sequence, dim}) ||
      !detail::valid_product({dim, dim})) {
    return Status::kInvalidShape;
  }
  return detail::all_nonnull(q, k, v, out) ? Status::kOk
                                            : Status::kInvalidArgument;
}

}  // namespace

Status linear_attention_unnormalized(const float* q, const float* k,
                                     const float* v, float* out,
                                     long long batch, long long heads,
                                     long long sequence, long long dim) {
  const Status status =
      validate_linear(q, k, v, out, batch, heads, sequence, dim);
  if (status != Status::kOk) return status;
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    std::vector<double> state(static_cast<std::size_t>(dim * dim));
    for (long long bh = begin; bh < end; ++bh) {
      std::fill(state.begin(), state.end(), 0.0);
      const long long offset = bh * sequence * dim;
      for (long long token = 0; token < sequence; ++token) {
        const float* kr = k + offset + token * dim;
        const float* vr = v + offset + token * dim;
        for (long long i = 0; i < dim; ++i) {
          for (long long j = 0; j < dim; ++j) {
            state[i * dim + j] += double(kr[i]) * vr[j];
          }
        }
      }
      for (long long token = 0; token < sequence; ++token) {
        const float* qr = q + offset + token * dim;
        float* destination = out + offset + token * dim;
        for (long long j = 0; j < dim; ++j) {
          double accumulator = 0.0;
          for (long long i = 0; i < dim; ++i) {
            accumulator += double(qr[i]) * state[i * dim + j];
          }
          destination[j] = static_cast<float>(accumulator);
        }
      }
    }
  });
  return Status::kOk;
}

Status causal_linear_attention(const float* q, const float* k, const float* v,
                               float* out, long long batch, long long heads,
                               long long sequence, long long dim) {
  const Status status =
      validate_linear(q, k, v, out, batch, heads, sequence, dim);
  if (status != Status::kOk) return status;
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    std::vector<double> state(static_cast<std::size_t>(dim * dim));
    for (long long bh = begin; bh < end; ++bh) {
      std::fill(state.begin(), state.end(), 0.0);
      const long long offset = bh * sequence * dim;
      for (long long token = 0; token < sequence; ++token) {
        const float* kr = k + offset + token * dim;
        const float* vr = v + offset + token * dim;
        const float* qr = q + offset + token * dim;
        float* destination = out + offset + token * dim;
        for (long long i = 0; i < dim; ++i) {
          for (long long j = 0; j < dim; ++j) {
            state[i * dim + j] += double(kr[i]) * vr[j];
          }
        }
        for (long long j = 0; j < dim; ++j) {
          double accumulator = 0.0;
          for (long long i = 0; i < dim; ++i) {
            accumulator += double(qr[i]) * state[i * dim + j];
          }
          destination[j] = static_cast<float>(accumulator);
        }
      }
    }
  });
  return Status::kOk;
}

Status decayed_linear_attention(const float* q, const float* k,
                                const float* v, const float* cumulative_log,
                                float* out, long long batch, long long heads,
                                long long sequence, long long dim) {
  const Status status =
      validate_linear(q, k, v, out, batch, heads, sequence, dim);
  if (status != Status::kOk) return status;
  if (cumulative_log == nullptr) return Status::kInvalidArgument;
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    std::vector<double> state(static_cast<std::size_t>(dim * dim));
    for (long long bh = begin; bh < end; ++bh) {
      std::fill(state.begin(), state.end(), 0.0);
      const long long offset = bh * sequence * dim;
      const float* decay = cumulative_log + bh * sequence;
      for (long long token = 0; token < sequence; ++token) {
        if (token > 0) {
          const double alpha = std::exp(
              static_cast<double>(decay[token] - decay[token - 1]));
          for (double& value : state) value *= alpha;
        }
        const float* kr = k + offset + token * dim;
        const float* vr = v + offset + token * dim;
        const float* qr = q + offset + token * dim;
        float* destination = out + offset + token * dim;
        for (long long i = 0; i < dim; ++i) {
          for (long long j = 0; j < dim; ++j) {
            state[i * dim + j] += double(kr[i]) * vr[j];
          }
        }
        for (long long j = 0; j < dim; ++j) {
          double accumulator = 0.0;
          for (long long i = 0; i < dim; ++i) {
            accumulator += double(qr[i]) * state[i * dim + j];
          }
          destination[j] = static_cast<float>(accumulator);
        }
      }
    }
  });
  return Status::kOk;
}

Status based_attention(const float* q, const float* k, const float* v,
                       float* out, long long batch, long long heads,
                       long long sequence, long long qk_dim,
                       long long value_dim) {
  if (!detail::valid_product({batch, heads, sequence, qk_dim}) ||
      !detail::valid_product({batch, heads, sequence, value_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) return Status::kInvalidArgument;
  const double scale = 1.0 / std::sqrt(static_cast<double>(qk_dim));
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long bh = begin; bh < end; ++bh) {
      const long long qk_offset = bh * sequence * qk_dim;
      const long long v_offset = bh * sequence * value_dim;
      for (long long query = 0; query < sequence; ++query) {
        float* destination = out + v_offset + query * value_dim;
        std::fill_n(destination, value_dim, 0.0f);
        for (long long key = 0; key <= query; ++key) {
          double dot = 0.0;
          for (long long d = 0; d < qk_dim; ++d) {
            dot += double(q[qk_offset + query * qk_dim + d]) *
                   k[qk_offset + key * qk_dim + d];
          }
          const double x = dot * scale;
          const double feature = 1.0 + x + 0.5 * x * x;
          for (long long d = 0; d < value_dim; ++d) {
            destination[d] +=
                static_cast<float>(feature * v[v_offset + key * value_dim + d]);
          }
        }
      }
    }
  });
  return Status::kOk;
}

Status hedgehog_attention(const float* q, const float* k, const float* v,
                          float* out, long long batch, long long heads,
                          long long sequence, long long dim) {
  const Status status =
      validate_linear(q, k, v, out, batch, heads, sequence, dim);
  if (status != Status::kOk) return status;
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    std::vector<double> kv(static_cast<std::size_t>(dim * dim));
    std::vector<double> phi_k(static_cast<std::size_t>(dim));
    for (long long bh = begin; bh < end; ++bh) {
      std::fill(kv.begin(), kv.end(), 0.0);
      const long long offset = bh * sequence * dim;
      for (long long token = 0; token < sequence; ++token) {
        const float* kr = k + offset + token * dim;
        const float* vr = v + offset + token * dim;
        const float maximum = *std::max_element(kr, kr + dim);
        for (long long i = 0; i < dim; ++i) {
          phi_k[i] = std::exp(static_cast<double>(kr[i] - maximum));
          for (long long j = 0; j < dim; ++j) {
            kv[i * dim + j] += phi_k[i] * vr[j];
          }
        }
      }
      for (long long token = 0; token < sequence; ++token) {
        const float* qr = q + offset + token * dim;
        const float maximum = *std::max_element(qr, qr + dim);
        float* destination = out + offset + token * dim;
        for (long long j = 0; j < dim; ++j) {
          double accumulator = 0.0;
          for (long long i = 0; i < dim; ++i) {
            accumulator += std::exp(static_cast<double>(qr[i] - maximum)) *
                           kv[i * dim + j];
          }
          destination[j] = static_cast<float>(accumulator);
        }
      }
    }
  });
  return Status::kOk;
}

Status gdn_recurrence(const float* q, const float* k, const float* v,
                      const float* gate, const float* beta, float* state_pool,
                      const int* cumulative_lengths, const int* slot_mapping,
                      float* out, long long requests, long long slots,
                      long long key_heads, long long value_heads,
                      long long key_dim, long long value_dim,
                      bool load_initial) {
  if (!detail::valid_product({requests}) || slots <= 0 || key_heads <= 0 ||
      value_heads <= 0 || value_heads % key_heads != 0 || key_dim <= 0 ||
      value_dim <= 0 ||
      !detail::valid_product({slots, value_heads, value_dim, key_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, gate, beta, state_pool,
                           cumulative_lengths, slot_mapping, out)) {
    return Status::kInvalidArgument;
  }
  if (cumulative_lengths[0] != 0) return Status::kInvalidArgument;
  for (long long request = 0; request < requests; ++request) {
    if (cumulative_lengths[request + 1] < cumulative_lengths[request] ||
        slot_mapping[request] < 0 || slot_mapping[request] >= slots) {
      return Status::kInvalidArgument;
    }
  }
  const long long head_group = value_heads / key_heads;
  for (long long request = 0; request < requests; ++request) {
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    for (long long value_head = 0; value_head < value_heads; ++value_head) {
      const long long key_head = value_head / head_group;
      for (long long dv = 0; dv < value_dim; ++dv) {
        float* state =
            state_pool +
            ((static_cast<long long>(slot_mapping[request]) * value_heads +
              value_head) *
                 value_dim +
             dv) *
                key_dim;
        if (!load_initial) std::fill_n(state, key_dim, 0.0f);
        for (long long token = start; token < end; ++token) {
          const float* kr =
              k + (token * key_heads + key_head) * key_dim;
          const float* qr =
              q + (token * key_heads + key_head) * key_dim;
          const float decay = gate[token * value_heads + value_head];
          double memory = 0.0;
          for (long long dk = 0; dk < key_dim; ++dk) {
            state[dk] *= decay;
            memory += double(kr[dk]) * state[dk];
          }
          const double correction =
              (v[(token * value_heads + value_head) * value_dim + dv] -
               memory) *
              beta[token * value_heads + value_head];
          double result = 0.0;
          for (long long dk = 0; dk < key_dim; ++dk) {
            state[dk] += static_cast<float>(kr[dk] * correction);
            result += double(qr[dk]) * state[dk];
          }
          out[(token * value_heads + value_head) * value_dim + dv] =
              static_cast<float>(result);
        }
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
