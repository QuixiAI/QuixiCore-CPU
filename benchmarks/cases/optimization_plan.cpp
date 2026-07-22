// Focused evidence for the optimization-plan kernels. Each target is the
// public API; every baseline is the previous materialized/direct algorithm.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"

namespace qcb {
namespace {

using quixicore_cpu::LmHeadSampling;
using quixicore_cpu::QuantFormat;
using quixicore_cpu::Status;

class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0x8CB92BA7u;
};

CheckResult check_arrays(const float* actual, const float* expected,
                         long long count, Tolerance tolerance) {
  CheckResult check;
  for (long long i = 0; i < count; ++i) {
    check_value(check, actual[i], expected[i], tolerance);
  }
  return check;
}

struct DenseBuffers {
  std::vector<float> a, b, target, reference;
  long long m = 0, n = 0, k = 0;
};

void dense_direct(DenseBuffers& b) {
  for (long long row = 0; row < b.m; ++row) {
    for (long long column = 0; column < b.n; ++column) {
      double sum = 0.0;
      for (long long inner = 0; inner < b.k; ++inner) {
        sum += static_cast<double>(b.a[row * b.k + inner]) *
               b.b[inner * b.n + column];
      }
      b.reference[row * b.n + column] = static_cast<float>(sum);
    }
  }
}

CaseDecl make_dense(long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "blocked_dense_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.notes = "4x32 blocked dense GEMM versus scalar row-column-inner loop";
  decl.flops = 2.0 * m * n * k;
  decl.make = [=]() {
    auto b = std::make_shared<DenseBuffers>();
    b->m = m; b->n = n; b->k = k;
    b->a.resize(static_cast<std::size_t>(m * k));
    b->b.resize(static_cast<std::size_t>(k * n));
    b->target.resize(static_cast<std::size_t>(m * n));
    b->reference.resize(static_cast<std::size_t>(m * n));
    Rng rng;
    for (float& value : b->a) value = rng.next();
    for (float& value : b->b) value = rng.next();
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::dense_gemm(b->a.data(), b->b.data(), b->target.data(),
                                    b->m, b->n, b->k) != Status::kOk) {
        throw std::runtime_error("dense_gemm failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("scalar_direct", [b]() {
      dense_direct(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::dense_gemm(b->a.data(), b->b.data(), b->target.data(),
                                    b->m, b->n, b->k) != Status::kOk) {
        throw std::runtime_error("dense_gemm failed");
      }
      dense_direct(*b);
      return check_arrays(b->target.data(), b->reference.data(), b->m * b->n,
                          kFp32Tolerance);
    };
    return body;
  };
  return decl;
}

struct ProjectionBuffers {
  std::vector<float> up_weights, gate_weights, x, target, up, gate, reference;
  std::vector<std::uint8_t> packed_up, packed_gate;
  long long n = 0, k = 0;
};

void decomposed_projection(ProjectionBuffers& b) {
  if (quixicore_cpu::qgemv(QuantFormat::kQ4_0, b.packed_up.data(), b.x.data(),
                           b.up.data(), b.n, b.k) != Status::kOk ||
      quixicore_cpu::qgemv(QuantFormat::kQ4_0, b.packed_gate.data(), b.x.data(),
                           b.gate.data(), b.n, b.k) != Status::kOk) {
    throw std::runtime_error("decomposed projection failed");
  }
  for (long long item = 0; item < b.n; ++item) {
    const float gate = b.gate[item];
    b.reference[item] = gate / (1.0f + std::exp(-gate)) * b.up[item];
  }
}

CaseDecl make_fused_projection(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "fused_swiglu_q4_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"n", n}, {"k", k}};
  decl.format = "q4_0";
  decl.notes = "SIMD gate+up+SwiGLU without projection intermediates";
  decl.flops = 4.0 * n * k;
  decl.make = [=]() {
    auto b = std::make_shared<ProjectionBuffers>();
    b->n = n; b->k = k;
    b->up_weights.resize(static_cast<std::size_t>(n * k));
    b->gate_weights.resize(static_cast<std::size_t>(n * k));
    b->x.resize(static_cast<std::size_t>(k));
    b->target.resize(static_cast<std::size_t>(n));
    b->up.resize(static_cast<std::size_t>(n));
    b->gate.resize(static_cast<std::size_t>(n));
    b->reference.resize(static_cast<std::size_t>(n));
    std::size_t bytes = 0;
    if (quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, n, k, &bytes) !=
        Status::kOk) {
      throw std::runtime_error("projection packed size failed");
    }
    b->packed_up.resize(bytes);
    b->packed_gate.resize(bytes);
    Rng rng;
    for (float& value : b->up_weights) value = 0.2f * rng.next();
    for (float& value : b->gate_weights) value = 0.2f * rng.next();
    for (float& value : b->x) value = rng.next();
    if (quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, b->up_weights.data(), n,
                                  k, b->packed_up.data()) != Status::kOk ||
        quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, b->gate_weights.data(), n,
                                  k, b->packed_gate.data()) != Status::kOk) {
      throw std::runtime_error("projection packing failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::qgemv_up_gate_activation(
              QuantFormat::kQ4_0, b->packed_up.data(), b->packed_gate.data(),
              b->x.data(), b->target.data(), b->n, b->k, false) !=
          Status::kOk) {
        throw std::runtime_error("fused projection failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("two_qgemv_plus_swiglu", [b]() {
      decomposed_projection(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::qgemv_up_gate_activation(
              QuantFormat::kQ4_0, b->packed_up.data(), b->packed_gate.data(),
              b->x.data(), b->target.data(), b->n, b->k, false) !=
          Status::kOk) {
        throw std::runtime_error("fused projection failed");
      }
      decomposed_projection(*b);
      return check_arrays(b->target.data(), b->reference.data(), b->n,
                          Tolerance{2e-4, 2e-4});
    };
    return body;
  };
  return decl;
}

struct QkvRopeBuffers {
  std::vector<float> q_weights, k_weights, v_weights, x, cosine, sine;
  std::vector<float> target_q, target_key, target_value;
  std::vector<float> raw_q, raw_k, raw_v, reference_q, reference_key,
      reference_value;
  std::vector<std::uint8_t> packed_q, packed_k, packed_v;
  long long query_heads = 0, kv_heads = 0, dim = 0, input = 0;
  long long slots = 0, max_position = 0;
  int position = 0, slot = 0;
};

void composed_qkv_rope_kv(QkvRopeBuffers& b) {
  const long long query_dim = b.query_heads * b.dim;
  const long long kv_dim = b.kv_heads * b.dim;
  if (quixicore_cpu::qgemv_qkv(
          QuantFormat::kQ4_0, b.packed_q.data(), b.packed_k.data(),
          b.packed_v.data(), b.x.data(), b.raw_q.data(), b.raw_k.data(),
          b.raw_v.data(), query_dim, kv_dim, b.input) != Status::kOk ||
      quixicore_cpu::rope_q_norm(
          b.raw_q.data(), b.cosine.data(), b.sine.data(), &b.position,
          nullptr, b.reference_q.data(), 1, b.query_heads, b.dim,
          b.max_position) != Status::kOk ||
      quixicore_cpu::rope_kv_insert(
          b.raw_k.data(), b.raw_v.data(), b.cosine.data(), b.sine.data(),
          &b.position, &b.slot, nullptr, b.reference_key.data(),
          b.reference_value.data(), 1, b.slots, b.kv_heads, b.dim,
          b.max_position) != Status::kOk) {
    throw std::runtime_error("composed QKV RoPE KV insert failed");
  }
}

CaseDecl make_qkv_rope_kv(long long query_heads, long long kv_heads,
                          long long dim, long long input) {
  const long long query_dim = query_heads * dim;
  const long long kv_dim = kv_heads * dim;
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "fused_qkv_rope_kv_q4_HQ" + std::to_string(query_heads) +
                 "_HKV" + std::to_string(kv_heads) + "_D" +
                 std::to_string(dim) + "_K" + std::to_string(input);
  decl.shape = {{"query_heads", query_heads}, {"kv_heads", kv_heads},
                {"head_dim", dim}, {"input", input}};
  decl.format = "q4_0";
  decl.notes = "one-region QKV projection, split-half RoPE, and KV write";
  decl.flops = 2.0 * (query_dim + 2 * kv_dim) * input;
  decl.make = [=]() {
    auto b = std::make_shared<QkvRopeBuffers>();
    b->query_heads = query_heads;
    b->kv_heads = kv_heads;
    b->dim = dim;
    b->input = input;
    b->slots = 4;
    b->max_position = 4;
    b->position = 2;
    b->slot = 1;
    b->q_weights.resize(static_cast<std::size_t>(query_dim * input));
    b->k_weights.resize(static_cast<std::size_t>(kv_dim * input));
    b->v_weights.resize(static_cast<std::size_t>(kv_dim * input));
    b->x.resize(static_cast<std::size_t>(input));
    b->cosine.resize(static_cast<std::size_t>(b->max_position * dim / 2));
    b->sine.resize(b->cosine.size());
    b->target_q.resize(static_cast<std::size_t>(query_dim));
    b->raw_q.resize(static_cast<std::size_t>(query_dim));
    b->raw_k.resize(static_cast<std::size_t>(kv_dim));
    b->raw_v.resize(static_cast<std::size_t>(kv_dim));
    b->reference_q.resize(static_cast<std::size_t>(query_dim));
    b->target_key.assign(static_cast<std::size_t>(b->slots * kv_dim), 0.0f);
    b->target_value.assign(static_cast<std::size_t>(b->slots * kv_dim), 0.0f);
    b->reference_key.assign(static_cast<std::size_t>(b->slots * kv_dim),
                            0.0f);
    b->reference_value.assign(static_cast<std::size_t>(b->slots * kv_dim),
                              0.0f);
    std::size_t q_bytes = 0;
    std::size_t kv_bytes = 0;
    if (quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, query_dim,
                                        input, &q_bytes) != Status::kOk ||
        quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, kv_dim, input,
                                        &kv_bytes) != Status::kOk) {
      throw std::runtime_error("QKV packed size failed");
    }
    b->packed_q.resize(q_bytes);
    b->packed_k.resize(kv_bytes);
    b->packed_v.resize(kv_bytes);
    Rng rng;
    for (float& value : b->q_weights) value = 0.1f * rng.next();
    for (float& value : b->k_weights) value = 0.1f * rng.next();
    for (float& value : b->v_weights) value = 0.1f * rng.next();
    for (float& value : b->x) value = rng.next();
    for (long long position = 0; position < b->max_position; ++position) {
      for (long long pair = 0; pair < dim / 2; ++pair) {
        const float angle = 0.01f * static_cast<float>((position + 1) *
                                                       (pair + 1));
        b->cosine[position * (dim / 2) + pair] = std::cos(angle);
        b->sine[position * (dim / 2) + pair] = std::sin(angle);
      }
    }
    if (quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, b->q_weights.data(),
                                  query_dim, input, b->packed_q.data()) !=
            Status::kOk ||
        quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, b->k_weights.data(),
                                  kv_dim, input, b->packed_k.data()) !=
            Status::kOk ||
        quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, b->v_weights.data(),
                                  kv_dim, input, b->packed_v.data()) !=
            Status::kOk) {
      throw std::runtime_error("QKV packing failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::qgemv_qkv_rope_kv(
              QuantFormat::kQ4_0, b->packed_q.data(), b->packed_k.data(),
              b->packed_v.data(), b->x.data(), b->cosine.data(),
              b->sine.data(), b->target_q.data(), b->target_key.data(),
              b->target_value.data(), b->query_heads, b->kv_heads, b->dim,
              b->input, b->slots, b->max_position, b->position, b->slot) !=
          Status::kOk) {
        throw std::runtime_error("fused QKV RoPE KV insert failed");
      }
      do_not_optimize(b->target_q.data());
      do_not_optimize(b->target_key.data());
      do_not_optimize(b->target_value.data());
    };
    body.baselines.emplace_back("qkv_then_rope_kv_write", [b]() {
      composed_qkv_rope_kv(*b);
      do_not_optimize(b->reference_q.data());
      do_not_optimize(b->reference_key.data());
      do_not_optimize(b->reference_value.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::qgemv_qkv_rope_kv(
              QuantFormat::kQ4_0, b->packed_q.data(), b->packed_k.data(),
              b->packed_v.data(), b->x.data(), b->cosine.data(),
              b->sine.data(), b->target_q.data(), b->target_key.data(),
              b->target_value.data(), b->query_heads, b->kv_heads, b->dim,
              b->input, b->slots, b->max_position, b->position, b->slot) !=
          Status::kOk) {
        throw std::runtime_error("fused QKV RoPE KV insert failed");
      }
      composed_qkv_rope_kv(*b);
      CheckResult check = check_arrays(
          b->target_q.data(), b->reference_q.data(),
          b->query_heads * b->dim,
          Tolerance{2e-4, 2e-4});
      for (long long item = 0;
           item < b->slots * b->kv_heads * b->dim; ++item) {
        check_value(check, b->target_key[item], b->reference_key[item],
                    Tolerance{2e-4, 2e-4});
        check_value(check, b->target_value[item], b->reference_value[item],
                    Tolerance{2e-4, 2e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct LmHeadBuffers {
  std::vector<float> weights, hidden, logits;
  std::vector<std::uint8_t> packed;
  std::vector<int> target, reference;
  long long rows = 0, vocab = 0, dim = 0;
};

void materialized_argmax(LmHeadBuffers& b) {
  if (quixicore_cpu::qgemm(QuantFormat::kQ4_0, b.packed.data(),
                           b.hidden.data(), b.logits.data(), b.rows, b.vocab,
                           b.dim) != Status::kOk) {
    throw std::runtime_error("materialized LM head failed");
  }
  for (long long row = 0; row < b.rows; ++row) {
    int best = 0;
    for (long long token = 1; token < b.vocab; ++token) {
      if (b.logits[row * b.vocab + token] >
          b.logits[row * b.vocab + best]) {
        best = static_cast<int>(token);
      }
    }
    b.reference[row] = best;
  }
}

CaseDecl make_lm_head(long long rows, long long vocab, long long dim) {
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "streaming_lm_head_R" + std::to_string(rows) + "_V" +
                 std::to_string(vocab) + "_H" + std::to_string(dim);
  decl.shape = {{"rows", rows}, {"vocab", vocab}, {"hidden", dim}};
  decl.format = "q4_0";
  decl.notes = "4K-token tiled projection plus streaming argmax";
  decl.flops = 2.0 * rows * vocab * dim;
  decl.make = [=]() {
    auto b = std::make_shared<LmHeadBuffers>();
    b->rows = rows; b->vocab = vocab; b->dim = dim;
    b->weights.resize(static_cast<std::size_t>(vocab * dim));
    b->hidden.resize(static_cast<std::size_t>(rows * dim));
    b->logits.resize(static_cast<std::size_t>(rows * vocab));
    b->target.resize(static_cast<std::size_t>(rows));
    b->reference.resize(static_cast<std::size_t>(rows));
    std::size_t bytes = 0;
    if (quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, vocab, dim,
                                        &bytes) != Status::kOk) {
      throw std::runtime_error("LM head packed size failed");
    }
    b->packed.resize(bytes);
    Rng rng;
    for (float& value : b->weights) value = 0.1f * rng.next();
    for (float& value : b->hidden) value = rng.next();
    if (quixicore_cpu::qgemv_pack(QuantFormat::kQ4_0, b->weights.data(), vocab,
                                  dim, b->packed.data()) != Status::kOk) {
      throw std::runtime_error("LM head packing failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::quantized_lm_head_argmax(
              QuantFormat::kQ4_0, b->packed.data(), b->hidden.data(),
              b->target.data(), b->rows, b->vocab, b->dim) != Status::kOk) {
        throw std::runtime_error("streaming LM head failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialized_logits", [b]() {
      materialized_argmax(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::quantized_lm_head_argmax(
              QuantFormat::kQ4_0, b->packed.data(), b->hidden.data(),
              b->target.data(), b->rows, b->vocab, b->dim) != Status::kOk) {
        throw std::runtime_error("streaming LM head failed");
      }
      materialized_argmax(*b);
      CheckResult check;
      check.passed = b->target == b->reference;
      return check;
    };
    return body;
  };
  return decl;
}

struct FftBuffers {
  std::vector<float> x, kernel, target, reference;
  long long batch = 0, heads = 0, length = 0;
};

void direct_convolution(FftBuffers& b) {
  for (long long bh = 0; bh < b.batch * b.heads; ++bh) {
    const long long head = bh % b.heads;
    for (long long target = 0; target < b.length; ++target) {
      double sum = 0.0;
      for (long long source = 0; source < b.length; ++source) {
        const long long index =
            (target - source + b.length) % b.length;
        sum += static_cast<double>(b.x[bh * b.length + source]) *
               b.kernel[head * b.length + index];
      }
      b.reference[bh * b.length + target] = static_cast<float>(sum);
    }
  }
}

CaseDecl make_fft(long long batch, long long heads, long long length) {
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "fft_convolution_B" + std::to_string(batch) + "_H" +
                 std::to_string(heads) + "_L" + std::to_string(length);
  decl.shape = {{"batch", batch}, {"heads", heads}, {"length", length}};
  decl.notes = "radix-2 circular FFT convolution versus O(L^2) direct oracle";
  decl.make = [=]() {
    auto b = std::make_shared<FftBuffers>();
    b->batch = batch; b->heads = heads; b->length = length;
    b->x.resize(static_cast<std::size_t>(batch * heads * length));
    b->kernel.resize(static_cast<std::size_t>(heads * length));
    b->target.resize(b->x.size());
    b->reference.resize(b->x.size());
    Rng rng;
    for (float& value : b->x) value = rng.next();
    for (float& value : b->kernel) value = 0.1f * rng.next();
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::fft_convolution(
              b->x.data(), b->kernel.data(), b->target.data(), b->batch,
              b->heads, b->length) != Status::kOk) {
        throw std::runtime_error("fft_convolution failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("direct_circular", [b]() {
      direct_convolution(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::fft_convolution(
              b->x.data(), b->kernel.data(), b->target.data(), b->batch,
              b->heads, b->length) != Status::kOk) {
        throw std::runtime_error("fft_convolution failed");
      }
      direct_convolution(*b);
      return check_arrays(b->target.data(), b->reference.data(),
                          b->batch * b->heads * b->length,
                          Tolerance{3e-4, 3e-4});
    };
    return body;
  };
  return decl;
}

struct MambaBuffers {
  std::vector<float> c, b, x, decay, target, reference;
  long long batch = 0, heads = 0, sequence = 0, dim = 0;
};

void direct_mamba(MambaBuffers& b) {
  for (long long bh = 0; bh < b.batch * b.heads; ++bh) {
    const long long offset = bh * b.sequence * b.dim;
    const long long decay_offset = bh * b.sequence;
    for (long long target = 0; target < b.sequence; ++target) {
      for (long long value = 0; value < b.dim; ++value) {
        double sum = 0.0;
        for (long long source = 0; source <= target; ++source) {
          double similarity = 0.0;
          for (long long key = 0; key < b.dim; ++key) {
            similarity +=
                static_cast<double>(b.c[offset + target * b.dim + key]) *
                b.b[offset + source * b.dim + key];
          }
          sum += similarity *
                 std::exp(static_cast<double>(
                     b.decay[decay_offset + target] -
                     b.decay[decay_offset + source])) *
                 b.x[offset + source * b.dim + value];
        }
        b.reference[offset + target * b.dim + value] =
            static_cast<float>(sum);
      }
    }
  }
}

CaseDecl make_mamba(long long batch, long long heads, long long sequence,
                    long long dim) {
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "recurrent_mamba_B" + std::to_string(batch) + "_H" +
                 std::to_string(heads) + "_L" + std::to_string(sequence) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"batch", batch}, {"heads", heads},
                {"sequence", sequence}, {"dim", dim}};
  decl.notes = "O(LD^2) recurrent SSD scan versus O(L^2D) source expansion";
  decl.make = [=]() {
    auto b = std::make_shared<MambaBuffers>();
    b->batch = batch; b->heads = heads; b->sequence = sequence; b->dim = dim;
    const long long elements = batch * heads * sequence * dim;
    b->c.resize(static_cast<std::size_t>(elements));
    b->b.resize(static_cast<std::size_t>(elements));
    b->x.resize(static_cast<std::size_t>(elements));
    b->decay.resize(static_cast<std::size_t>(batch * heads * sequence));
    b->target.resize(static_cast<std::size_t>(elements));
    b->reference.resize(static_cast<std::size_t>(elements));
    Rng rng;
    for (float& value : b->c) value = 0.2f * rng.next();
    for (float& value : b->b) value = 0.2f * rng.next();
    for (float& value : b->x) value = rng.next();
    for (long long bh = 0; bh < batch * heads; ++bh) {
      for (long long t = 0; t < sequence; ++t) {
        b->decay[bh * sequence + t] = -0.002f * static_cast<float>(t);
      }
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::mamba2(
              b->c.data(), b->b.data(), b->x.data(), b->decay.data(),
              b->target.data(), b->batch, b->heads, b->sequence, b->dim) !=
          Status::kOk) {
        throw std::runtime_error("mamba2 failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("source_expansion", [b]() {
      direct_mamba(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::mamba2(
              b->c.data(), b->b.data(), b->x.data(), b->decay.data(),
              b->target.data(), b->batch, b->heads, b->sequence, b->dim) !=
          Status::kOk) {
        throw std::runtime_error("mamba2 failed");
      }
      direct_mamba(*b);
      return check_arrays(b->target.data(), b->reference.data(),
                          b->batch * b->heads * b->sequence * b->dim,
                          Tolerance{3e-4, 4e-4});
    };
    return body;
  };
  return decl;
}

struct MoeBuffers {
  std::vector<float> weights, x, target, projection, reference;
  std::vector<std::uint8_t> packed;
  std::vector<int> expert_ids;
  long long rows = 0, experts = 0, input = 0, intermediate = 0;
  std::size_t expert_bytes = 0;
};

void materialized_moe(MoeBuffers& b) {
  if (quixicore_cpu::moe_grouped_qgemm(
          b.x.data(), b.packed.data(), b.expert_ids.data(), nullptr,
          b.projection.data(), b.rows, b.experts, b.input,
          2 * b.intermediate, QuantFormat::kQ4_0, false) != Status::kOk) {
    throw std::runtime_error("materialized MoE projection failed");
  }
  for (long long row = 0; row < b.rows; ++row) {
    for (long long item = 0; item < b.intermediate; ++item) {
      const float gate = b.projection[row * 2 * b.intermediate + item];
      const float up =
          b.projection[row * 2 * b.intermediate + b.intermediate + item];
      b.reference[row * b.intermediate + item] =
          gate / (1.0f + std::exp(-gate)) * up;
    }
  }
}

CaseDecl make_moe(long long rows, long long experts, long long input,
                  long long intermediate) {
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "tiled_moe_R" + std::to_string(rows) + "_E" +
                 std::to_string(experts) + "_K" + std::to_string(input) +
                 "_I" + std::to_string(intermediate);
  decl.shape = {{"rows", rows}, {"experts", experts}, {"input", input},
                {"intermediate", intermediate}};
  decl.format = "q4_0";
  decl.notes = "expert-shared gate/up decode with directly fused SwiGLU";
  decl.make = [=]() {
    auto b = std::make_shared<MoeBuffers>();
    b->rows = rows; b->experts = experts; b->input = input;
    b->intermediate = intermediate;
    const long long outputs = 2 * intermediate;
    b->weights.resize(
        static_cast<std::size_t>(experts * outputs * input));
    b->x.resize(static_cast<std::size_t>(rows * input));
    b->target.resize(static_cast<std::size_t>(rows * intermediate));
    b->reference.resize(b->target.size());
    b->projection.resize(static_cast<std::size_t>(rows * outputs));
    b->expert_ids.resize(static_cast<std::size_t>(rows));
    if (quixicore_cpu::qgemv_packed_size(
            QuantFormat::kQ4_0, outputs, input, &b->expert_bytes) !=
        Status::kOk) {
      throw std::runtime_error("MoE packed size failed");
    }
    b->packed.resize(static_cast<std::size_t>(experts) * b->expert_bytes);
    Rng rng;
    for (float& value : b->weights) value = 0.05f * rng.next();
    for (float& value : b->x) value = rng.next();
    const long long active = std::max(1LL, experts / 2);
    for (long long row = 0; row < rows; ++row) {
      b->expert_ids[row] = static_cast<int>(row % active);
    }
    for (long long expert = 0; expert < experts; ++expert) {
      if (quixicore_cpu::qgemv_pack(
              QuantFormat::kQ4_0,
              b->weights.data() + expert * outputs * input, outputs, input,
              b->packed.data() + expert * b->expert_bytes) != Status::kOk) {
        throw std::runtime_error("MoE packing failed");
      }
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::moe_grouped_qswiglu(
              b->x.data(), b->packed.data(), b->expert_ids.data(), nullptr,
              b->target.data(), b->rows, b->experts, b->input,
              b->intermediate, QuantFormat::kQ4_0, false) != Status::kOk) {
        throw std::runtime_error("fused MoE failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialized_2I", [b]() {
      materialized_moe(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::moe_grouped_qswiglu(
              b->x.data(), b->packed.data(), b->expert_ids.data(), nullptr,
              b->target.data(), b->rows, b->experts, b->input,
              b->intermediate, QuantFormat::kQ4_0, false) != Status::kOk) {
        throw std::runtime_error("fused MoE failed");
      }
      materialized_moe(*b);
      return check_arrays(b->target.data(), b->reference.data(),
                          b->rows * b->intermediate,
                          Tolerance{3e-4, 3e-4});
    };
    return body;
  };
  return decl;
}

struct AttentionBuffers {
  std::vector<float> q, key, value, target, reference, alibi, sinks;
  std::vector<int> block_table, context, block_mask;
  long long cache_blocks = 0, batch = 0, query_heads = 0, kv_heads = 0;
  long long dim = 0, page = 0, max_blocks = 0;
};

void materialized_paged_attention(AttentionBuffers& b) {
  const long long group = b.query_heads / b.kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(b.dim));
  for (long long request = 0; request < b.batch; ++request) {
    for (long long qhead = 0; qhead < b.query_heads; ++qhead) {
      const long long kvhead = qhead / group;
      const long long context = b.context[request];
      std::vector<float> scores(static_cast<std::size_t>(context));
      float maximum = b.sinks[qhead];
      const float* query =
          b.q.data() + (request * b.query_heads + qhead) * b.dim;
      for (long long position = 0; position < context; ++position) {
        const long long logical = position / b.page;
        if (b.block_mask[(request * b.query_heads + qhead) * b.max_blocks +
                         logical] == 0) {
          scores[position] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const int physical = b.block_table[request * b.max_blocks + logical];
        const long long base =
            ((static_cast<long long>(physical) * b.page + position % b.page) *
                 b.kv_heads +
             kvhead) *
            b.dim;
        double dot = 0.0;
        for (long long d = 0; d < b.dim; ++d) {
          dot += static_cast<double>(query[d]) * b.key[base + d];
        }
        scores[position] = static_cast<float>(
            dot * scale + b.alibi[qhead] * (position - (context - 1)));
        maximum = std::max(maximum, scores[position]);
      }
      double denominator = std::exp(b.sinks[qhead] - maximum);
      for (float score : scores) {
        if (score != -std::numeric_limits<float>::infinity()) {
          denominator += std::exp(score - maximum);
        }
      }
      float* output =
          b.reference.data() + (request * b.query_heads + qhead) * b.dim;
      std::fill_n(output, b.dim, 0.0f);
      for (long long position = 0; position < context; ++position) {
        if (scores[position] == -std::numeric_limits<float>::infinity()) {
          continue;
        }
        const int physical =
            b.block_table[request * b.max_blocks + position / b.page];
        const long long base =
            ((static_cast<long long>(physical) * b.page + position % b.page) *
                 b.kv_heads +
             kvhead) *
            b.dim;
        const double probability = std::exp(scores[position] - maximum) /
                                   denominator;
        for (long long d = 0; d < b.dim; ++d) {
          output[d] += static_cast<float>(probability * b.value[base + d]);
        }
      }
    }
  }
}

CaseDecl make_paged_attention(long long batch, long long query_heads,
                              long long kv_heads, long long context,
                              long long dim, long long page) {
  const long long max_blocks = (context + page - 1) / page;
  const long long cache_blocks = batch * max_blocks;
  CaseDecl decl;
  decl.kernel = "optimization_plan";
  decl.variant = "online_paged_B" + std::to_string(batch) + "_HQ" +
                 std::to_string(query_heads) + "_HKV" +
                 std::to_string(kv_heads) + "_S" + std::to_string(context) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"batch", batch}, {"query_heads", query_heads},
                {"kv_heads", kv_heads}, {"context", context}, {"dim", dim}};
  decl.notes = "online block softmax versus materialized score-buffer decode";
  decl.make = [=]() {
    auto b = std::make_shared<AttentionBuffers>();
    b->cache_blocks = cache_blocks; b->batch = batch;
    b->query_heads = query_heads; b->kv_heads = kv_heads; b->dim = dim;
    b->page = page; b->max_blocks = max_blocks;
    b->q.resize(static_cast<std::size_t>(batch * query_heads * dim));
    b->key.resize(
        static_cast<std::size_t>(cache_blocks * page * kv_heads * dim));
    b->value.resize(b->key.size());
    b->target.resize(b->q.size());
    b->reference.resize(b->q.size());
    b->alibi.resize(static_cast<std::size_t>(query_heads));
    b->sinks.resize(static_cast<std::size_t>(query_heads));
    b->block_table.resize(static_cast<std::size_t>(batch * max_blocks));
    b->context.assign(static_cast<std::size_t>(batch),
                      static_cast<int>(context));
    b->block_mask.assign(
        static_cast<std::size_t>(batch * query_heads * max_blocks), 1);
    Rng rng;
    for (float& value : b->q) value = rng.next();
    for (float& value : b->key) value = rng.next();
    for (float& value : b->value) value = rng.next();
    for (long long head = 0; head < query_heads; ++head) {
      b->alibi[head] = 0.001f * static_cast<float>(head);
      b->sinks[head] = -1.0f - 0.01f * static_cast<float>(head);
    }
    for (long long request = 0; request < batch; ++request) {
      for (long long block = 0; block < max_blocks; ++block) {
        b->block_table[request * max_blocks + block] =
            static_cast<int>(request * max_blocks + block);
      }
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::paged_attention_advanced(
              b->q.data(), b->key.data(), b->value.data(),
              b->block_table.data(), b->context.data(), b->block_mask.data(),
              b->alibi.data(), b->sinks.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks) != Status::kOk) {
        throw std::runtime_error("paged_attention_advanced failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialized_scores", [b]() {
      materialized_paged_attention(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::paged_attention_advanced(
              b->q.data(), b->key.data(), b->value.data(),
              b->block_table.data(), b->context.data(), b->block_mask.data(),
              b->alibi.data(), b->sinks.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks) != Status::kOk) {
        throw std::runtime_error("paged_attention_advanced failed");
      }
      materialized_paged_attention(*b);
      return check_arrays(b->target.data(), b->reference.data(),
                          b->batch * b->query_heads * b->dim,
                          Tolerance{2e-5, 2e-4});
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_optimization_plan_cases(const BuildCtx& ctx,
                                   std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_dense(2, 32, 32));
    out.push_back(make_fused_projection(64, 64));
    out.push_back(make_qkv_rope_kv(2, 1, 16, 64));
    out.push_back(make_lm_head(1, 256, 64));
    out.push_back(make_fft(1, 1, 64));
    out.push_back(make_mamba(1, 1, 16, 8));
    out.push_back(make_moe(8, 2, 32, 16));
    out.push_back(make_paged_attention(1, 2, 1, 32, 16, 16));
    return;
  }
  out.push_back(make_dense(16, 512, 512));
  out.push_back(make_fused_projection(1024, 1408));
  out.push_back(make_qkv_rope_kv(8, 2, 64, 1408));
  out.push_back(make_lm_head(2, 8192, 1024));
  out.push_back(make_fft(1, 2, 1024));
  out.push_back(make_mamba(1, 2, 128, 32));
  out.push_back(make_moe(64, 8, 256, 512));
  out.push_back(make_paged_attention(2, 8, 2, 512, 64, 16));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_dense(128, 1024, 1024));
    out.push_back(make_fft(1, 4, 4096));
    out.push_back(make_mamba(1, 4, 512, 64));
    out.push_back(make_moe(128, 16, 1024, 1024));
    out.push_back(make_paged_attention(4, 32, 8, 4096, 128, 16));
  }
}

}  // namespace qcb
