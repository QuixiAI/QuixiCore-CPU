// Focused evidence for the Metal-compatible GDN linear-attention helpers and
// split sigmoid/value fusion. Baselines are independent scalar compositions
// of the public mathematical contracts.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/ops.h"

namespace qcb {
namespace {

using quixicore_cpu::Status;

class GdnRng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0xBB67AE85u;
};

float sigmoid_reference(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
  const float exponential = std::exp(value);
  return exponential / (1.0f + exponential);
}

CheckResult compare(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    Tolerance tolerance = {2e-5, 2e-5}) {
  CheckResult check;
  if (actual.size() != expected.size()) {
    check.passed = false;
    return check;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    check_value(check, actual[index], expected[index], tolerance);
  }
  return check;
}

struct RecurBuffers {
  std::vector<float> q, k, v, decay, beta, state, out, state_out, reference,
      reference_state;
  std::vector<int> cumulative, slots;
  long long requests = 0, pool_slots = 0, key_heads = 0, value_heads = 0,
            key_dim = 0, value_dim = 0;
};

void recur_reference(RecurBuffers& b) {
  b.reference_state = b.state;
  const long long group = b.value_heads / b.key_heads;
  for (long long request = 0; request < b.requests; ++request) {
    for (long long vh = 0; vh < b.value_heads; ++vh) {
      const long long kh = vh / group;
      for (long long dv = 0; dv < b.value_dim; ++dv) {
        float* state = b.reference_state.data() +
                       ((static_cast<long long>(b.slots[request]) *
                              b.value_heads +
                          vh) *
                             b.value_dim +
                        dv) *
                           b.key_dim;
        for (long long token = b.cumulative[request];
             token < b.cumulative[request + 1]; ++token) {
          const float* key = b.k.data() + (token * b.key_heads + kh) * b.key_dim;
          const float* query =
              b.q.data() + (token * b.key_heads + kh) * b.key_dim;
          double memory = 0.0;
          for (long long dim = 0; dim < b.key_dim; ++dim) {
            state[dim] *= b.decay[token * b.value_heads + vh];
            memory += static_cast<double>(state[dim]) * key[dim];
          }
          const double correction =
              (b.v[(token * b.value_heads + vh) * b.value_dim + dv] - memory) *
              b.beta[token * b.value_heads + vh];
          double output = 0.0;
          for (long long dim = 0; dim < b.key_dim; ++dim) {
            state[dim] += static_cast<float>(key[dim] * correction);
            output += static_cast<double>(state[dim]) * query[dim];
          }
          b.reference[(token * b.value_heads + vh) * b.value_dim + dv] =
              static_cast<float>(output);
        }
      }
    }
  }
}

CaseDecl make_recur(long long requests, long long sequence, long long key_heads,
                    long long value_heads, long long key_dim,
                    long long value_dim) {
  CaseDecl decl;
  decl.kernel = "gdn";
  decl.variant = "recur_R" + std::to_string(requests) + "_L" +
                 std::to_string(sequence) + "_HK" +
                 std::to_string(key_heads) + "_HV" +
                 std::to_string(value_heads) + "_DK" +
                 std::to_string(key_dim) + "_DV" +
                 std::to_string(value_dim);
  decl.shape = {{"requests", requests}, {"sequence", sequence},
                {"key_heads", key_heads}, {"value_heads", value_heads},
                {"key_dim", key_dim}, {"value_dim", value_dim}};
  decl.notes = "functional recurrent GDN state update";
  decl.make = [=]() {
    auto b = std::make_shared<RecurBuffers>();
    b->requests = requests;
    b->pool_slots = requests + 2;
    b->key_heads = key_heads;
    b->value_heads = value_heads;
    b->key_dim = key_dim;
    b->value_dim = value_dim;
    const long long tokens = requests * sequence;
    b->q.resize(tokens * key_heads * key_dim);
    b->k.resize(b->q.size());
    b->v.resize(tokens * value_heads * value_dim);
    b->decay.resize(tokens * value_heads);
    b->beta.resize(b->decay.size());
    b->state.resize(b->pool_slots * value_heads * value_dim * key_dim);
    b->state_out.resize(b->state.size());
    b->reference_state.resize(b->state.size());
    b->out.resize(b->v.size());
    b->reference.resize(b->v.size());
    b->cumulative.resize(requests + 1);
    b->slots.resize(requests);
    GdnRng rng;
    for (float& x : b->q) x = 0.08f * rng.next();
    for (float& x : b->k) x = 0.08f * rng.next();
    for (float& x : b->v) x = 0.15f * rng.next();
    for (float& x : b->state) x = 0.01f * rng.next();
    for (std::size_t i = 0; i < b->decay.size(); ++i) {
      b->decay[i] = 0.94f + 0.04f * std::fabs(rng.next());
      b->beta[i] = 0.2f + 0.7f * std::fabs(rng.next());
    }
    for (long long r = 0; r <= requests; ++r) {
      b->cumulative[r] = static_cast<int>(r * sequence);
      if (r < requests) b->slots[r] = static_cast<int>(r + 1);
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::gdn_recur(
              b->q.data(), b->k.data(), b->v.data(), b->decay.data(),
              b->beta.data(), b->state.data(), b->cumulative.data(),
              b->slots.data(), b->out.data(), b->state_out.data(), b->requests,
              b->pool_slots, b->key_heads, b->value_heads, b->key_dim,
              b->value_dim, true) != Status::kOk) {
        throw std::runtime_error("gdn_recur failed");
      }
      do_not_optimize(b->out.data());
    };
    body.baselines.emplace_back("scalar_contract", [b]() {
      recur_reference(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      b->out.assign(b->out.size(), 0.0f);
      if (quixicore_cpu::gdn_recur(
              b->q.data(), b->k.data(), b->v.data(), b->decay.data(),
              b->beta.data(), b->state.data(), b->cumulative.data(),
              b->slots.data(), b->out.data(), b->state_out.data(), b->requests,
              b->pool_slots, b->key_heads, b->value_heads, b->key_dim,
              b->value_dim, true) != Status::kOk) {
        throw std::runtime_error("gdn_recur check failed");
      }
      recur_reference(*b);
      CheckResult result = compare(b->out, b->reference, {2e-5, 2e-5});
      const CheckResult state =
          compare(b->state_out, b->reference_state, {2e-5, 2e-5});
      result.passed = result.passed && state.passed;
      result.finite = result.finite && state.finite;
      result.max_abs_err = std::max(result.max_abs_err, state.max_abs_err);
      result.max_rel_err = std::max(result.max_rel_err, state.max_rel_err);
      return result;
    };
    return body;
  };
  return decl;
}

struct ConvBuffers {
  std::vector<float> x, weight, state, out, state_out, reference,
      reference_state;
  std::vector<int> cumulative, slots;
  long long requests = 0, pool_slots = 0, channels = 0, kernel_size = 0;
};

void conv_reference(ConvBuffers& b) {
  b.reference_state = b.state;
  const long long history_size = b.kernel_size - 1;
  for (long long request = 0; request < b.requests; ++request) {
    for (long long channel = 0; channel < b.channels; ++channel) {
      float history[7]{};
      float* state = b.reference_state.data() +
                     (static_cast<long long>(b.slots[request]) * b.channels +
                      channel) *
                         history_size;
      std::copy_n(state, history_size, history);
      for (long long token = b.cumulative[request];
           token < b.cumulative[request + 1]; ++token) {
        float value = b.x[token * b.channels + channel] *
                      b.weight[channel * b.kernel_size + b.kernel_size - 1];
        for (long long item = 0; item < history_size; ++item) {
          value += history[item] * b.weight[channel * b.kernel_size + item];
        }
        b.reference[token * b.channels + channel] =
            value * sigmoid_reference(value);
        std::move(history + 1, history + history_size, history);
        history[history_size - 1] = b.x[token * b.channels + channel];
      }
      std::copy_n(history, history_size, state);
    }
  }
}

CaseDecl make_conv(long long requests, long long sequence, long long channels,
                   long long kernel_size) {
  CaseDecl decl;
  decl.kernel = "gdn";
  decl.variant = "short_conv_R" + std::to_string(requests) + "_L" +
                 std::to_string(sequence) + "_C" + std::to_string(channels) +
                 "_K" + std::to_string(kernel_size);
  decl.shape = {{"requests", requests}, {"sequence", sequence},
                {"channels", channels}, {"kernel", kernel_size}};
  decl.notes = "functional depthwise causal convolution with SiLU";
  decl.make = [=]() {
    auto b = std::make_shared<ConvBuffers>();
    b->requests = requests;
    b->pool_slots = requests + 1;
    b->channels = channels;
    b->kernel_size = kernel_size;
    const long long tokens = requests * sequence;
    b->x.resize(tokens * channels);
    b->weight.resize(channels * kernel_size);
    b->state.resize(b->pool_slots * channels * (kernel_size - 1));
    b->state_out.resize(b->state.size());
    b->reference_state.resize(b->state.size());
    b->out.resize(b->x.size());
    b->reference.resize(b->x.size());
    b->cumulative.resize(requests + 1);
    b->slots.resize(requests);
    GdnRng rng;
    for (float& x : b->x) x = 0.5f * rng.next();
    for (float& x : b->weight) x = 0.2f * rng.next();
    for (float& x : b->state) x = 0.1f * rng.next();
    for (long long r = 0; r <= requests; ++r) {
      b->cumulative[r] = static_cast<int>(r * sequence);
      if (r < requests) b->slots[r] = static_cast<int>(r);
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::gdn_short_conv(
              b->x.data(), b->weight.data(), b->state.data(),
              b->cumulative.data(), b->slots.data(), b->out.data(),
              b->state_out.data(), b->requests, b->pool_slots, b->channels,
              b->kernel_size, true, true) != Status::kOk) {
        throw std::runtime_error("gdn_short_conv failed");
      }
      do_not_optimize(b->out.data());
    };
    body.baselines.emplace_back("scalar_contract", [b]() {
      conv_reference(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::gdn_short_conv(
              b->x.data(), b->weight.data(), b->state.data(),
              b->cumulative.data(), b->slots.data(), b->out.data(),
              b->state_out.data(), b->requests, b->pool_slots, b->channels,
              b->kernel_size, true, true) != Status::kOk) {
        throw std::runtime_error("gdn_short_conv check failed");
      }
      conv_reference(*b);
      return compare(b->out, b->reference);
    };
    return body;
  };
  return decl;
}

struct QkvBuffers {
  std::vector<float> mixed, q, k, v, ref_q, ref_k, ref_v;
  long long tokens = 0, key_heads = 0, value_heads = 0, key_dim = 0,
            value_dim = 0;
};

void qkv_reference(QkvBuffers& b) {
  const long long qk_width = b.key_heads * b.key_dim;
  const long long value_width = b.value_heads * b.value_dim;
  const long long channels = 2 * qk_width + value_width;
  for (long long token = 0; token < b.tokens; ++token) {
    const float* source = b.mixed.data() + token * channels;
    for (long long head = 0; head < b.key_heads; ++head) {
      for (int is_key = 0; is_key < 2; ++is_key) {
        const float* input = source + is_key * qk_width + head * b.key_dim;
        float squares = 0.0f;
        for (long long dim = 0; dim < b.key_dim; ++dim) {
          squares += input[dim] * input[dim];
        }
        float* output = (is_key ? b.ref_k : b.ref_q).data() +
                        (token * b.key_heads + head) * b.key_dim;
        const float base = is_key ? 1.0f / std::sqrt(float(b.key_dim))
                                  : 1.0f / float(b.key_dim);
        const float scale =
            base / std::sqrt(squares / float(b.key_dim) + 1e-6f);
        for (long long dim = 0; dim < b.key_dim; ++dim) {
          output[dim] = input[dim] * scale;
        }
      }
    }
    std::copy_n(source + 2 * qk_width, value_width,
                b.ref_v.data() + token * value_width);
  }
}

CaseDecl make_qkv(long long tokens, long long key_heads, long long value_heads,
                  long long key_dim, long long value_dim) {
  CaseDecl decl;
  decl.kernel = "gdn";
  decl.variant = "qkv_prepare_T" + std::to_string(tokens) + "_HK" +
                 std::to_string(key_heads) + "_HV" +
                 std::to_string(value_heads) + "_DK" +
                 std::to_string(key_dim) + "_DV" +
                 std::to_string(value_dim);
  decl.shape = {{"tokens", tokens}, {"key_heads", key_heads},
                {"value_heads", value_heads}, {"key_dim", key_dim},
                {"value_dim", value_dim}};
  decl.notes = "split mixed projection and RMS-scale Q/K";
  decl.make = [=]() {
    auto b = std::make_shared<QkvBuffers>();
    b->tokens = tokens;
    b->key_heads = key_heads;
    b->value_heads = value_heads;
    b->key_dim = key_dim;
    b->value_dim = value_dim;
    const long long qk_count = tokens * key_heads * key_dim;
    const long long v_count = tokens * value_heads * value_dim;
    b->mixed.resize(2 * qk_count + v_count);
    b->q.resize(qk_count);
    b->k.resize(qk_count);
    b->ref_q.resize(qk_count);
    b->ref_k.resize(qk_count);
    b->v.resize(v_count);
    b->ref_v.resize(v_count);
    GdnRng rng;
    for (float& x : b->mixed) x = rng.next();
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::gdn_qkv_prepare(
              b->mixed.data(), b->q.data(), b->k.data(), b->v.data(),
              b->tokens, b->key_heads, b->value_heads, b->key_dim,
              b->value_dim) != Status::kOk) {
        throw std::runtime_error("gdn_qkv_prepare failed");
      }
      do_not_optimize(b->q.data());
    };
    body.baselines.emplace_back("scalar_contract", [b]() {
      qkv_reference(*b);
      do_not_optimize(b->ref_q.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::gdn_qkv_prepare(
              b->mixed.data(), b->q.data(), b->k.data(), b->v.data(),
              b->tokens, b->key_heads, b->value_heads, b->key_dim,
              b->value_dim) != Status::kOk) {
        throw std::runtime_error("gdn_qkv_prepare check failed");
      }
      qkv_reference(*b);
      CheckResult result = compare(b->q, b->ref_q);
      const CheckResult k = compare(b->k, b->ref_k);
      const CheckResult v = compare(b->v, b->ref_v);
      result.passed = result.passed && k.passed && v.passed;
      result.max_abs_err = std::max({result.max_abs_err, k.max_abs_err,
                                    v.max_abs_err});
      result.max_rel_err = std::max({result.max_rel_err, k.max_rel_err,
                                    v.max_rel_err});
      return result;
    };
    return body;
  };
  return decl;
}

struct ElementBuffers {
  std::vector<float> a, b, c, d, out0, out1, ref0, ref1;
  long long rows = 0, width = 0;
};

CaseDecl make_gate(long long tokens, long long heads) {
  CaseDecl decl;
  decl.kernel = "gdn";
  decl.variant = "gate_beta_T" + std::to_string(tokens) + "_H" +
                 std::to_string(heads);
  decl.shape = {{"tokens", tokens}, {"heads", heads}};
  decl.notes = "fused decay and beta preparation";
  decl.make = [=]() {
    auto b = std::make_shared<ElementBuffers>();
    b->rows = tokens;
    b->width = heads;
    b->a.resize(tokens * heads);
    b->b.resize(tokens * heads);
    b->c.resize(heads);
    b->d.resize(heads);
    b->out0.resize(tokens * heads);
    b->out1.resize(tokens * heads);
    b->ref0.resize(tokens * heads);
    b->ref1.resize(tokens * heads);
    GdnRng rng;
    for (float& x : b->a) x = rng.next();
    for (float& x : b->b) x = rng.next();
    for (float& x : b->c) x = 2.0f * rng.next();
    for (float& x : b->d) x = rng.next();
    auto reference = [b]() {
      for (long long i = 0; i < b->rows * b->width; ++i) {
        const long long h = i % b->width;
        const float alpha = b->a[i] + b->d[h];
        const float softplus = alpha > 20.0f
                                   ? alpha
                                   : (alpha < -20.0f
                                          ? std::exp(alpha)
                                          : std::log1p(std::exp(alpha)));
        b->ref0[i] = std::exp(-std::exp(b->c[h]) * softplus);
        b->ref1[i] = sigmoid_reference(b->b[i]);
      }
    };
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::gdn_gate_beta(
              b->a.data(), b->b.data(), b->c.data(), b->d.data(),
              b->out0.data(), b->out1.data(), b->rows, b->width) !=
          Status::kOk) {
        throw std::runtime_error("gdn_gate_beta failed");
      }
      do_not_optimize(b->out0.data());
    };
    body.baselines.emplace_back("scalar_contract", [b, reference]() {
      reference();
      do_not_optimize(b->ref0.data());
    });
    body.check = [b, reference]() {
      if (quixicore_cpu::gdn_gate_beta(
              b->a.data(), b->b.data(), b->c.data(), b->d.data(),
              b->out0.data(), b->out1.data(), b->rows, b->width) !=
          Status::kOk) {
        throw std::runtime_error("gdn_gate_beta check failed");
      }
      reference();
      CheckResult result = compare(b->out0, b->ref0);
      const CheckResult beta = compare(b->out1, b->ref1);
      result.passed = result.passed && beta.passed;
      result.max_abs_err = std::max(result.max_abs_err, beta.max_abs_err);
      result.max_rel_err = std::max(result.max_rel_err, beta.max_rel_err);
      return result;
    };
    return body;
  };
  return decl;
}

CaseDecl make_norm(long long tokens, long long heads, long long dim) {
  CaseDecl decl;
  decl.kernel = "gdn";
  decl.variant = "gated_rmsnorm_T" + std::to_string(tokens) + "_H" +
                 std::to_string(heads) + "_D" + std::to_string(dim);
  decl.shape = {{"tokens", tokens}, {"heads", heads}, {"dim", dim}};
  decl.notes = "fused RMSNorm, weight, and SiLU gate";
  decl.make = [=]() {
    auto b = std::make_shared<ElementBuffers>();
    b->rows = tokens * heads;
    b->width = dim;
    b->a.resize(b->rows * dim);
    b->b.resize(b->rows * dim);
    b->c.resize(dim);
    b->out0.resize(b->rows * dim);
    b->ref0.resize(b->rows * dim);
    GdnRng rng;
    for (float& x : b->a) x = rng.next();
    for (float& x : b->b) x = rng.next();
    for (float& x : b->c) x = 0.75f + 0.5f * std::fabs(rng.next());
    auto reference = [b]() {
      for (long long row = 0; row < b->rows; ++row) {
        float squares = 0.0f;
        for (long long d = 0; d < b->width; ++d) {
          squares += b->a[row * b->width + d] * b->a[row * b->width + d];
        }
        const float inverse =
            1.0f / std::sqrt(squares / float(b->width) + 1e-6f);
        for (long long d = 0; d < b->width; ++d) {
          const long long i = row * b->width + d;
          b->ref0[i] = b->a[i] * inverse * b->c[d] * b->b[i] *
                       sigmoid_reference(b->b[i]);
        }
      }
    };
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::gdn_gated_rmsnorm(
              b->a.data(), b->b.data(), b->c.data(), b->out0.data(), b->rows,
              1, b->width) != Status::kOk) {
        throw std::runtime_error("gdn_gated_rmsnorm failed");
      }
      do_not_optimize(b->out0.data());
    };
    body.baselines.emplace_back("scalar_contract", [b, reference]() {
      reference();
      do_not_optimize(b->ref0.data());
    });
    body.check = [b, reference]() {
      if (quixicore_cpu::gdn_gated_rmsnorm(
              b->a.data(), b->b.data(), b->c.data(), b->out0.data(), b->rows,
              1, b->width) != Status::kOk) {
        throw std::runtime_error("gdn_gated_rmsnorm check failed");
      }
      reference();
      return compare(b->out0, b->ref0);
    };
    return body;
  };
  return decl;
}

CaseDecl make_sigmoid(long long count) {
  CaseDecl decl;
  decl.kernel = "gdn";
  decl.variant = "sigmoid_mul_N" + std::to_string(count);
  decl.shape = {{"count", count}};
  decl.notes = "split gate/value sigmoid multiplication";
  decl.make = [=]() {
    auto b = std::make_shared<ElementBuffers>();
    b->rows = count;
    b->a.resize(count);
    b->b.resize(count);
    b->out0.resize(count);
    b->ref0.resize(count);
    GdnRng rng;
    for (float& x : b->a) x = 6.0f * rng.next();
    for (float& x : b->b) x = rng.next();
    auto reference = [b]() {
      for (long long i = 0; i < b->rows; ++i) {
        b->ref0[i] = sigmoid_reference(b->a[i]) * b->b[i];
      }
    };
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::sigmoid_mul(b->a.data(), b->b.data(), b->out0.data(),
                                    b->rows) != Status::kOk) {
        throw std::runtime_error("sigmoid_mul failed");
      }
      do_not_optimize(b->out0.data());
    };
    body.baselines.emplace_back("scalar_contract", [b, reference]() {
      reference();
      do_not_optimize(b->ref0.data());
    });
    body.check = [b, reference]() {
      if (quixicore_cpu::sigmoid_mul(b->a.data(), b->b.data(), b->out0.data(),
                                    b->rows) != Status::kOk) {
        throw std::runtime_error("sigmoid_mul check failed");
      }
      reference();
      return compare(b->out0, b->ref0);
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_gdn_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_recur(1, 2, 1, 2, 64, 16));
    out.push_back(make_conv(1, 8, 64, 4));
    out.push_back(make_qkv(4, 1, 2, 64, 64));
    out.push_back(make_gate(32, 8));
    out.push_back(make_norm(4, 2, 64));
    out.push_back(make_sigmoid(1024));
    return;
  }
  const bool comprehensive = ctx.preset == Preset::kComprehensive;
  out.push_back(make_recur(comprehensive ? 8 : 4, comprehensive ? 32 : 16, 2,
                           8, comprehensive ? 128 : 64,
                           comprehensive ? 128 : 64));
  out.push_back(make_conv(comprehensive ? 16 : 8, comprehensive ? 128 : 64,
                          comprehensive ? 4096 : 1024, 4));
  out.push_back(make_qkv(comprehensive ? 1024 : 256, 4, 8, 128, 128));
  out.push_back(make_gate(comprehensive ? 8192 : 2048, 32));
  out.push_back(make_norm(comprehensive ? 1024 : 256, 16, 128));
  out.push_back(make_sigmoid(comprehensive ? 8 * 1024 * 1024
                                           : 1024 * 1024));
}

}  // namespace qcb
