// Representative portable-reference cases for the sibling-parity kernel batch.
// Each target calls the public CPU API and carries an independent scalar
// baseline plus a correctness gate. These cases establish honest evidence for
// the new contract paths; they do not claim ISA-tuned performance.

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"

namespace qcb {
namespace {

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
  uint32_t state_ = 0xA341316Cu;
};

struct SoftmaxBuffers {
  AlignedBuffer<float> x;
  AlignedBuffer<float> y;
  long long rows;
  long long dim;
};

void softmax_scalar(const float* x, float* y, long long rows, long long dim) {
  for (long long row = 0; row < rows; ++row) {
    const float* in = x + row * dim;
    float* out = y + row * dim;
    float maximum = in[0];
    for (long long i = 1; i < dim; ++i) maximum = std::max(maximum, in[i]);
    double sum = 0.0;
    for (long long i = 0; i < dim; ++i) {
      out[i] = std::exp(in[i] - maximum);
      sum += out[i];
    }
    for (long long i = 0; i < dim; ++i) out[i] /= static_cast<float>(sum);
  }
}

CaseDecl make_softmax(long long rows, long long dim) {
  CaseDecl decl;
  decl.kernel = "softmax";
  decl.variant = "R" + std::to_string(rows) + "_H" + std::to_string(dim);
  decl.shape = {{"rows", rows}, {"hidden", dim}};
  decl.notes = "public portable f32 softmax";
  decl.bytes_moved = 8.0 * rows * dim;
  decl.make = [rows, dim]() {
    auto buffers = std::make_shared<SoftmaxBuffers>();
    buffers->rows = rows;
    buffers->dim = dim;
    buffers->x = aligned_alloc_array<float>(rows * dim);
    buffers->y = aligned_alloc_array<float>(rows * dim);
    Rng rng;
    for (long long i = 0; i < rows * dim; ++i) buffers->x.get()[i] = rng.next();
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::softmax(buffers->x.get(), buffers->y.get(),
                                 buffers->rows, buffers->dim) != Status::kOk) {
        throw std::runtime_error("softmax failed");
      }
      do_not_optimize(buffers->y.get());
    };
    body.baselines.emplace_back("scalar_ref", [buffers]() {
      softmax_scalar(buffers->x.get(), buffers->y.get(), buffers->rows,
                     buffers->dim);
      do_not_optimize(buffers->y.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::softmax(buffers->x.get(), buffers->y.get(),
                                 buffers->rows, buffers->dim) != Status::kOk) {
        throw std::runtime_error("softmax failed");
      }
      CheckResult check;
      for (long long row = 0; row < buffers->rows; ++row) {
        double sum = 0.0;
        for (long long i = 0; i < buffers->dim; ++i) {
          const float value = buffers->y.get()[row * buffers->dim + i];
          if (value < 0.0f) check.passed = false;
          sum += value;
        }
        check_value(check, sum, 1.0, kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct AttentionBuffers {
  AlignedBuffer<float> q;
  AlignedBuffer<float> k;
  AlignedBuffer<float> v;
  AlignedBuffer<float> out;
  AlignedBuffer<float> reference;
  long long heads;
  long long sequence;
  long long dim;
};

void attention_scalar(AttentionBuffers& buffers) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(buffers.dim));
  for (long long head = 0; head < buffers.heads; ++head) {
    for (long long query_position = 0; query_position < buffers.sequence;
         ++query_position) {
      const float* query = buffers.q.get() +
                           (head * buffers.sequence + query_position) *
                               buffers.dim;
      std::vector<double> scores(
          static_cast<std::size_t>(query_position + 1));
      double maximum = -INFINITY;
      for (long long key_position = 0; key_position <= query_position;
           ++key_position) {
        const float* key = buffers.k.get() +
                           (head * buffers.sequence + key_position) *
                               buffers.dim;
        double score = 0.0;
        for (long long d = 0; d < buffers.dim; ++d) score += query[d] * key[d];
        scores[static_cast<std::size_t>(key_position)] = score * scale;
        maximum = std::max(maximum, score * scale);
      }
      double denominator = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (long long d = 0; d < buffers.dim; ++d) {
        double value = 0.0;
        for (long long key_position = 0; key_position <= query_position;
             ++key_position) {
          value += scores[static_cast<std::size_t>(key_position)] *
                   buffers.v.get()[(head * buffers.sequence + key_position) *
                                       buffers.dim +
                                   d];
        }
        buffers.reference.get()[(head * buffers.sequence + query_position) *
                                    buffers.dim +
                                d] = static_cast<float>(value / denominator);
      }
    }
  }
}

CaseDecl make_attention(long long heads, long long sequence, long long dim) {
  CaseDecl decl;
  decl.kernel = "causal_attention";
  decl.variant = "H" + std::to_string(heads) + "_S" +
                 std::to_string(sequence) + "_D" + std::to_string(dim);
  decl.shape = {{"heads", heads}, {"sequence", sequence}, {"dim", dim}};
  decl.notes = "public portable f32 causal attention";
  decl.flops = 4.0 * heads * sequence * sequence * dim / 2.0;
  decl.make = [heads, sequence, dim]() {
    auto buffers = std::make_shared<AttentionBuffers>();
    buffers->heads = heads;
    buffers->sequence = sequence;
    buffers->dim = dim;
    const long long count = heads * sequence * dim;
    buffers->q = aligned_alloc_array<float>(count);
    buffers->k = aligned_alloc_array<float>(count);
    buffers->v = aligned_alloc_array<float>(count);
    buffers->out = aligned_alloc_array<float>(count);
    buffers->reference = aligned_alloc_array<float>(count);
    Rng rng;
    for (long long i = 0; i < count; ++i) {
      buffers->q.get()[i] = rng.next();
      buffers->k.get()[i] = rng.next();
      buffers->v.get()[i] = rng.next();
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::attention(
              buffers->q.get(), buffers->k.get(), buffers->v.get(),
              buffers->out.get(), buffers->heads, buffers->heads,
              buffers->sequence, buffers->sequence, buffers->dim, true) !=
          Status::kOk) {
        throw std::runtime_error("attention failed");
      }
      do_not_optimize(buffers->out.get());
    };
    body.baselines.emplace_back("materialized_scalar", [buffers]() {
      attention_scalar(*buffers);
      do_not_optimize(buffers->reference.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::attention(
              buffers->q.get(), buffers->k.get(), buffers->v.get(),
              buffers->out.get(), buffers->heads, buffers->heads,
              buffers->sequence, buffers->sequence, buffers->dim, true) !=
          Status::kOk) {
        throw std::runtime_error("attention failed");
      }
      attention_scalar(*buffers);
      CheckResult check;
      const long long count = buffers->heads * buffers->sequence * buffers->dim;
      for (long long i = 0; i < count; ++i) {
        check_value(check, buffers->out.get()[i], buffers->reference.get()[i],
                    kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct MoeBuffers {
  AlignedBuffer<float> logits;
  AlignedBuffer<int> ids;
  AlignedBuffer<float> weights;
  long long tokens;
  long long experts;
  int top_k;
};

CaseDecl make_moe(long long tokens, long long experts, int top_k) {
  CaseDecl decl;
  decl.kernel = "moe_routing";
  decl.variant = "T" + std::to_string(tokens) + "_E" +
                 std::to_string(experts) + "_K" + std::to_string(top_k);
  decl.shape = {{"tokens", tokens}, {"experts", experts}, {"top_k", top_k}};
  decl.notes = "public portable f32 top-k routing";
  decl.make = [tokens, experts, top_k]() {
    auto buffers = std::make_shared<MoeBuffers>();
    buffers->tokens = tokens;
    buffers->experts = experts;
    buffers->top_k = top_k;
    buffers->logits = aligned_alloc_array<float>(tokens * experts);
    buffers->ids = aligned_alloc_array<int>(tokens * top_k);
    buffers->weights = aligned_alloc_array<float>(tokens * top_k);
    Rng rng;
    for (long long i = 0; i < tokens * experts; ++i) {
      buffers->logits.get()[i] = rng.next();
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::moe_route_topk(
              buffers->logits.get(), buffers->ids.get(),
              buffers->weights.get(), buffers->tokens, buffers->experts,
              buffers->top_k) != Status::kOk) {
        throw std::runtime_error("moe_route_topk failed");
      }
      do_not_optimize(buffers->weights.get());
    };
    body.baselines.emplace_back("full_sort_scalar", [buffers]() {
      for (long long token = 0; token < buffers->tokens; ++token) {
        std::vector<int> ids(static_cast<std::size_t>(buffers->experts));
        std::iota(ids.begin(), ids.end(), 0);
        const float* row = buffers->logits.get() + token * buffers->experts;
        std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
          return row[lhs] == row[rhs] ? lhs < rhs : row[lhs] > row[rhs];
        });
        const float maximum = row[ids.front()];
        double sum = 0.0;
        for (int rank = 0; rank < buffers->top_k; ++rank) {
          const long long output = token * buffers->top_k + rank;
          buffers->ids.get()[output] = ids[rank];
          const float weight = std::exp(row[ids[rank]] - maximum);
          buffers->weights.get()[output] = weight;
          sum += weight;
        }
        for (int rank = 0; rank < buffers->top_k; ++rank) {
          buffers->weights.get()[token * buffers->top_k + rank] /=
              static_cast<float>(sum);
        }
      }
      do_not_optimize(buffers->weights.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::moe_route_topk(
              buffers->logits.get(), buffers->ids.get(),
              buffers->weights.get(), buffers->tokens, buffers->experts,
              buffers->top_k) != Status::kOk) {
        throw std::runtime_error("moe_route_topk failed");
      }
      CheckResult check;
      for (long long token = 0; token < buffers->tokens; ++token) {
        double sum = 0.0;
        float previous = INFINITY;
        for (int rank = 0; rank < buffers->top_k; ++rank) {
          const long long index = token * buffers->top_k + rank;
          const int id = buffers->ids.get()[index];
          if (id < 0 || id >= buffers->experts) {
            check.passed = false;
            continue;
          }
          const float logit =
              buffers->logits.get()[token * buffers->experts + id];
          if (logit > previous) {
            check.passed = false;
          }
          previous = logit;
          sum += buffers->weights.get()[index];
        }
        check_value(check, sum, 1.0, kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct ScanBuffers {
  AlignedBuffer<float> u, delta, a, b, c, d, y, reference;
  long long channels, sequence, state;
};

void scan_scalar(ScanBuffers& buffers, float* destination) {
  for (long long channel = 0; channel < buffers.channels; ++channel) {
    std::vector<double> hidden(static_cast<std::size_t>(buffers.state));
    for (long long token = 0; token < buffers.sequence; ++token) {
      const double input = buffers.u.get()[channel * buffers.sequence + token];
      const double step = buffers.delta.get()[channel * buffers.sequence + token];
      double output = buffers.d.get()[channel] * input;
      for (long long state = 0; state < buffers.state; ++state) {
        const long long cs = channel * buffers.state + state;
        const long long ts = token * buffers.state + state;
        hidden[static_cast<std::size_t>(state)] =
            std::exp(step * buffers.a.get()[cs]) * hidden[state] +
            step * buffers.b.get()[ts] * input;
        output += buffers.c.get()[ts] * hidden[state];
      }
      destination[channel * buffers.sequence + token] =
          static_cast<float>(output);
    }
  }
}

CaseDecl make_scan(long long channels, long long sequence, long long state) {
  CaseDecl decl;
  decl.kernel = "mamba_ssd";
  decl.variant = "C" + std::to_string(channels) + "_S" +
                 std::to_string(sequence) + "_N" + std::to_string(state);
  decl.shape = {{"channels", channels}, {"sequence", sequence}, {"state", state}};
  decl.notes = "public portable f32 selective scan";
  decl.make = [channels, sequence, state]() {
    auto buffers = std::make_shared<ScanBuffers>();
    buffers->channels = channels;
    buffers->sequence = sequence;
    buffers->state = state;
    buffers->u = aligned_alloc_array<float>(channels * sequence);
    buffers->delta = aligned_alloc_array<float>(channels * sequence);
    buffers->a = aligned_alloc_array<float>(channels * state);
    buffers->b = aligned_alloc_array<float>(sequence * state);
    buffers->c = aligned_alloc_array<float>(sequence * state);
    buffers->d = aligned_alloc_array<float>(channels);
    buffers->y = aligned_alloc_array<float>(channels * sequence);
    buffers->reference = aligned_alloc_array<float>(channels * sequence);
    Rng rng;
    for (long long i = 0; i < channels * sequence; ++i) {
      buffers->u.get()[i] = rng.next();
      buffers->delta.get()[i] = 0.01f + 0.02f * std::fabs(rng.next());
    }
    for (long long i = 0; i < channels * state; ++i) {
      buffers->a.get()[i] = -0.1f - std::fabs(rng.next());
    }
    for (long long i = 0; i < sequence * state; ++i) {
      buffers->b.get()[i] = rng.next();
      buffers->c.get()[i] = rng.next();
    }
    for (long long i = 0; i < channels; ++i) buffers->d.get()[i] = rng.next();
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::selective_scan(
              buffers->u.get(), buffers->delta.get(), buffers->a.get(),
              buffers->b.get(), buffers->c.get(), buffers->d.get(),
              buffers->y.get(), buffers->channels, buffers->sequence,
              buffers->state) != Status::kOk) {
        throw std::runtime_error("selective_scan failed");
      }
      do_not_optimize(buffers->y.get());
    };
    body.baselines.emplace_back("serial_scalar", [buffers]() {
      scan_scalar(*buffers, buffers->reference.get());
      do_not_optimize(buffers->reference.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::selective_scan(
              buffers->u.get(), buffers->delta.get(), buffers->a.get(),
              buffers->b.get(), buffers->c.get(), buffers->d.get(),
              buffers->y.get(), buffers->channels, buffers->sequence,
              buffers->state) != Status::kOk) {
        throw std::runtime_error("selective_scan failed");
      }
      scan_scalar(*buffers, buffers->reference.get());
      CheckResult check;
      for (long long i = 0; i < buffers->channels * buffers->sequence; ++i) {
        check_value(check, buffers->y.get()[i], buffers->reference.get()[i],
                    kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct QGemmBuffers {
  AlignedBuffer<float> weights, x, y, dequant;
  AlignedBuffer<std::uint8_t> packed;
  long long m, n, k;
};

CaseDecl make_qgemm(long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "quant_gemm";
  decl.variant = "M" + std::to_string(m) + "_N" + std::to_string(n) +
                 "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = "q8_0";
  decl.notes = "public q8_0 qgemm reference composition";
  decl.flops = 2.0 * m * n * k;
  decl.make = [m, n, k]() {
    auto buffers = std::make_shared<QGemmBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    buffers->weights = aligned_alloc_array<float>(n * k);
    buffers->x = aligned_alloc_array<float>(m * k);
    buffers->y = aligned_alloc_array<float>(m * n);
    buffers->dequant = aligned_alloc_array<float>(n * k);
    std::size_t packed_bytes = 0;
    if (quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, n, k,
                                         &packed_bytes) != Status::kOk) {
      throw std::runtime_error("qgemm packed size failed");
    }
    buffers->packed = aligned_alloc_array<std::uint8_t>(packed_bytes);
    Rng rng;
    for (long long i = 0; i < n * k; ++i) buffers->weights.get()[i] = rng.next();
    for (long long i = 0; i < m * k; ++i) buffers->x.get()[i] = rng.next();
    if (quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, buffers->weights.get(), n,
                                  k, buffers->packed.get()) != Status::kOk ||
        quixicore_cpu::qgemv_unpack(QuantFormat::kQ8_0, buffers->packed.get(),
                                    n, k, buffers->dequant.get()) !=
            Status::kOk) {
      throw std::runtime_error("qgemm pack failed");
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::qgemm(QuantFormat::kQ8_0, buffers->packed.get(),
                               buffers->x.get(), buffers->y.get(), buffers->m,
                               buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("qgemm failed");
      }
      do_not_optimize(buffers->y.get());
    };
    body.baselines.emplace_back("dequant_scalar", [buffers]() {
      for (long long row = 0; row < buffers->m; ++row) {
        for (long long column = 0; column < buffers->n; ++column) {
          float sum = 0.0f;
          for (long long inner = 0; inner < buffers->k; ++inner) {
            sum += buffers->x.get()[row * buffers->k + inner] *
                   buffers->dequant.get()[column * buffers->k + inner];
          }
          buffers->y.get()[row * buffers->n + column] = sum;
        }
      }
      do_not_optimize(buffers->y.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::qgemm(QuantFormat::kQ8_0, buffers->packed.get(),
                               buffers->x.get(), buffers->y.get(), buffers->m,
                               buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("qgemm failed");
      }
      CheckResult check;
      for (long long row = 0; row < buffers->m; ++row) {
        for (long long column = 0; column < buffers->n; ++column) {
          double sum = 0.0;
          for (long long inner = 0; inner < buffers->k; ++inner) {
            sum += static_cast<double>(
                       buffers->x.get()[row * buffers->k + inner]) *
                   buffers->dequant.get()[column * buffers->k + inner];
          }
          check_value(check, buffers->y.get()[row * buffers->n + column], sum,
                      Tolerance{1e-4, 1e-4});
        }
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_contract_ops_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_softmax(4, 256));
    out.push_back(make_attention(2, 16, 32));
    out.push_back(make_moe(32, 8, 2));
    out.push_back(make_scan(8, 32, 4));
    out.push_back(make_qgemm(2, 64, 64));
    return;
  }
  out.push_back(make_softmax(512, 4096));
  out.push_back(make_attention(8, 128, 64));
  out.push_back(make_moe(1024, 64, 4));
  out.push_back(make_scan(256, 512, 16));
  out.push_back(make_qgemm(16, 2048, 2048));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_attention(32, 512, 128));
    out.push_back(make_qgemm(128, 4096, 4096));
  }
}

}  // namespace qcb
