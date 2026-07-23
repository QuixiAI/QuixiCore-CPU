// Focused performance evidence for llama.cpp semantic parity additions.

#include <algorithm>
#include <cmath>
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

float datum(long long index) {
  return std::sin(0.013f * static_cast<float>(index)) * .25f;
}

struct ConvBuffers {
  std::vector<float> input, weights, bias, output, reference;
  long long channels, outputs, side;
};

void conv_scalar(ConvBuffers& b) {
  for (long long oc = 0; oc < b.outputs; ++oc)
    for (long long y = 0; y < b.side; ++y)
      for (long long x = 0; x < b.side; ++x) {
        float sum = b.bias[oc];
        for (long long ic = 0; ic < b.channels; ++ic)
          for (long long ky = 0; ky < 3; ++ky) {
            const long long iy = y + ky - 1;
            if (iy < 0 || iy >= b.side) continue;
            for (long long kx = 0; kx < 3; ++kx) {
              const long long ix = x + kx - 1;
              if (ix < 0 || ix >= b.side) continue;
              sum += b.input[(ic * b.side + iy) * b.side + ix] *
                     b.weights[((oc * b.channels + ic) * 3 + ky) * 3 + kx];
            }
          }
        b.reference[(oc * b.side + y) * b.side + x] = sum;
      }
}

CaseDecl make_conv(long long channels, long long outputs, long long side) {
  CaseDecl decl;
  decl.kernel = "llama_parity";
  decl.variant = "conv2d_C" + std::to_string(channels) + "_O" +
                 std::to_string(outputs) + "_S" + std::to_string(side);
  decl.shape = {{"channels", channels}, {"outputs", outputs}, {"side", side}};
  decl.flops = 2.0 * channels * outputs * side * side * 9;
  decl.notes = "NCHW 3x3 f32 convolution; output-channel parallel";
  decl.make = [channels, outputs, side]() {
    auto b = std::make_shared<ConvBuffers>();
    b->channels = channels; b->outputs = outputs; b->side = side;
    b->input.resize(channels * side * side);
    b->weights.resize(outputs * channels * 9);
    b->bias.resize(outputs);
    b->output.resize(outputs * side * side);
    b->reference.resize(b->output.size());
    for (std::size_t i = 0; i < b->input.size(); ++i) b->input[i] = datum(i);
    for (std::size_t i = 0; i < b->weights.size(); ++i)
      b->weights[i] = datum(i + 17);
    for (long long i = 0; i < outputs; ++i) b->bias[i] = datum(i + 31);
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::conv2d(b->input.data(), b->weights.data(),
                                b->bias.data(), b->output.data(), 1,
                                b->channels, b->outputs, b->side, b->side,
                                3, 3, 1, 1, 1, 1) != Status::kOk)
        throw std::runtime_error("conv2d failed");
      do_not_optimize(b->output.data());
    };
    body.baselines.emplace_back("serial_direct", [b]() {
      conv_scalar(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      quixicore_cpu::conv2d(b->input.data(), b->weights.data(), b->bias.data(),
                            b->output.data(), 1, b->channels, b->outputs,
                            b->side, b->side, 3, 3, 1, 1, 1, 1);
      conv_scalar(*b);
      CheckResult check;
      for (std::size_t i = 0; i < b->output.size(); ++i)
        check_value(check, b->output[i], b->reference[i],
                    Tolerance{2e-4, 2e-5});
      return check;
    };
    return body;
  };
  return decl;
}

struct RecurrentBuffers {
  std::vector<float> key, value, query, gate, initial, output, final_state;
  long long sequences, tokens, heads, dim;
};

CaseDecl make_gla(long long sequences, long long tokens, long long heads,
                  long long dim) {
  CaseDecl decl;
  decl.kernel = "llama_parity";
  decl.variant = "gla_B" + std::to_string(sequences) + "_T" +
                 std::to_string(tokens) + "_H" + std::to_string(heads) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"sequences", sequences}, {"tokens", tokens},
                {"heads", heads}, {"dim", dim}};
  decl.flops = 6.0 * sequences * tokens * heads * dim * dim;
  decl.notes = "state-local recurrent GLA, parallel over sequence/head";
  decl.make = [sequences, tokens, heads, dim]() {
    auto b = std::make_shared<RecurrentBuffers>();
    b->sequences = sequences; b->tokens = tokens; b->heads = heads; b->dim = dim;
    const long long count = sequences * tokens * heads * dim;
    const long long states = sequences * heads * dim * dim;
    b->key.resize(count); b->value.resize(count); b->query.resize(count);
    b->gate.resize(count); b->initial.resize(states); b->output.resize(count);
    b->final_state.resize(states);
    for (long long i = 0; i < count; ++i) {
      b->key[i] = datum(i); b->value[i] = datum(i + 11);
      b->query[i] = datum(i + 23);
      b->gate[i] = .98f + datum(i + 37) * .01f;
    }
    for (long long i = 0; i < states; ++i) b->initial[i] = datum(i + 41);
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::gated_linear_attention(
              b->key.data(), b->value.data(), b->query.data(), b->gate.data(),
              b->initial.data(), b->output.data(), b->final_state.data(),
              b->sequences, b->tokens, b->heads, b->dim,
              1.0f / std::sqrt(static_cast<float>(b->dim))) != Status::kOk)
        throw std::runtime_error("GLA failed");
      do_not_optimize(b->output.data());
    };
    body.check = [b]() {
      CheckResult check;
      check.passed = quixicore_cpu::gated_linear_attention(
          b->key.data(), b->value.data(), b->query.data(), b->gate.data(),
          b->initial.data(), b->output.data(), b->final_state.data(),
          b->sequences, b->tokens, b->heads, b->dim,
          1.0f / std::sqrt(static_cast<float>(b->dim))) == Status::kOk;
      for (float value : b->output) if (!std::isfinite(value)) {
        check.passed = false; check.finite = false; break;
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct DsvBuffers {
  std::vector<float> mixes, scale, base, comb;
  long long tokens;
};

CaseDecl make_dsv(long long tokens) {
  CaseDecl decl;
  decl.kernel = "llama_parity";
  decl.variant = "dsv4_comb_T" + std::to_string(tokens);
  decl.shape = {{"tokens", tokens}};
  decl.notes = "DSV4 four-way softmax plus Sinkhorn normalization";
  decl.make = [tokens]() {
    auto b = std::make_shared<DsvBuffers>();
    b->tokens = tokens; b->mixes.resize(tokens * 24); b->scale = {1, 1, .5f};
    b->base.resize(24); b->comb.resize(tokens * 16);
    for (std::size_t i = 0; i < b->mixes.size(); ++i) b->mixes[i] = datum(i);
    for (int i = 0; i < 24; ++i) b->base[i] = datum(i + 5);
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::dsv4_hc_comb(b->mixes.data(), b->scale.data(),
                                      b->base.data(), b->comb.data(),
                                      b->tokens, 1e-6f, 4) != Status::kOk)
        throw std::runtime_error("DSV4 comb failed");
      do_not_optimize(b->comb.data());
    };
    body.check = [b]() {
      CheckResult check;
      check.passed = quixicore_cpu::dsv4_hc_comb(
          b->mixes.data(), b->scale.data(), b->base.data(), b->comb.data(),
          b->tokens, 1e-6f, 4) == Status::kOk;
      for (long long token = 0; token < b->tokens; ++token)
        for (int destination = 0; destination < 4; ++destination) {
          double sum = 0;
          for (int source = 0; source < 4; ++source)
            sum += b->comb[token * 16 + source * 4 + destination];
          check_value(check, sum, 1.0, Tolerance{2e-5, 2e-5});
        }
      return check;
    };
    return body;
  };
  return decl;
}

struct UnaryBuffers {
  std::vector<float> input, output, reference;
};

CaseDecl make_unary(long long count) {
  CaseDecl decl;
  decl.kernel = "llama_parity";
  decl.variant = "unary_softplus_N" + std::to_string(count);
  decl.shape = {{"count", count}};
  decl.bytes_moved = 2.0 * count * sizeof(float);
  decl.notes = "complete-unary dispatcher, stable softplus, element parallel";
  decl.make = [count]() {
    auto b = std::make_shared<UnaryBuffers>();
    b->input.resize(count); b->output.resize(count); b->reference.resize(count);
    for (long long i = 0; i < count; ++i) {
      b->input[i] = 24.0f * datum(i);
    }
    CaseBody body;
    body.target = [b, count]() {
      if (quixicore_cpu::unary(b->input.data(), b->output.data(), count,
                               quixicore_cpu::UnaryOp::kSoftplus) !=
          Status::kOk) {
        throw std::runtime_error("unary softplus failed");
      }
      do_not_optimize(b->output.data());
    };
    body.baselines.emplace_back("serial_direct", [b, count]() {
      for (long long i = 0; i < count; ++i) {
        const float x = b->input[i];
        b->reference[i] = x > 20.0f ? x : std::log(1.0f + std::exp(x));
      }
      do_not_optimize(b->reference.data());
    });
    body.check = [b, count]() {
      quixicore_cpu::unary(b->input.data(), b->output.data(), count,
                           quixicore_cpu::UnaryOp::kSoftplus);
      CheckResult check;
      for (long long i = 0; i < count; ++i) {
        const float x = b->input[i];
        const float expected =
            x > 20.0f ? x : std::log(1.0f + std::exp(x));
        check_value(check, b->output[i], expected, Tolerance{0.0, 0.0});
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct GluBuffers {
  std::vector<float> gate, value, output, reference;
};

CaseDecl make_glu_oai(long long count) {
  CaseDecl decl;
  decl.kernel = "llama_parity";
  decl.variant = "swiglu_oai_N" + std::to_string(count);
  decl.shape = {{"count", count}};
  decl.bytes_moved = 3.0 * count * sizeof(float);
  decl.notes = "clamped OpenAI SwiGLU, element parallel";
  decl.make = [count]() {
    auto b = std::make_shared<GluBuffers>();
    b->gate.resize(count); b->value.resize(count); b->output.resize(count);
    b->reference.resize(count);
    for (long long i = 0; i < count; ++i) {
      b->gate[i] = 40.0f * datum(i);
      b->value[i] = 40.0f * datum(i + 19);
    }
    CaseBody body;
    body.target = [b, count]() {
      if (quixicore_cpu::swiglu_oai(b->gate.data(), b->value.data(),
                                    b->output.data(), 1, count, 1.5f, 7.0f) !=
          Status::kOk) {
        throw std::runtime_error("OpenAI SwiGLU failed");
      }
      do_not_optimize(b->output.data());
    };
    body.baselines.emplace_back("serial_direct", [b, count]() {
      for (long long i = 0; i < count; ++i) {
        const float x = std::min(b->gate[i], 7.0f);
        const float v = std::clamp(b->value[i], -7.0f, 7.0f);
        b->reference[i] = x / (1.0f + std::exp(-1.5f * x)) * (v + 1.0f);
      }
      do_not_optimize(b->reference.data());
    });
    body.check = [b, count]() {
      quixicore_cpu::swiglu_oai(b->gate.data(), b->value.data(),
                                b->output.data(), 1, count, 1.5f, 7.0f);
      CheckResult check;
      for (long long i = 0; i < count; ++i) {
        const float x = std::min(b->gate[i], 7.0f);
        const float v = std::clamp(b->value[i], -7.0f, 7.0f);
        const float expected =
            x / (1.0f + std::exp(-1.5f * x)) * (v + 1.0f);
        check_value(check, b->output[i], expected, Tolerance{2e-6, 2e-6});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_llama_parity_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_conv(8, 16, 16));
    out.push_back(make_gla(2, 16, 4, 16));
    out.push_back(make_dsv(256));
    out.push_back(make_unary(1LL << 16));
    out.push_back(make_glu_oai(1LL << 16));
  } else if (ctx.preset == Preset::kQuick) {
    out.push_back(make_conv(16, 32, 32));
    out.push_back(make_gla(4, 64, 8, 32));
    out.push_back(make_dsv(4096));
    out.push_back(make_unary(1LL << 20));
    out.push_back(make_glu_oai(1LL << 20));
  } else {
    out.push_back(make_conv(32, 64, 64));
    out.push_back(make_gla(8, 256, 16, 64));
    out.push_back(make_dsv(65536));
    out.push_back(make_unary(1LL << 24));
    out.push_back(make_glu_oai(1LL << 24));
  }
}

}  // namespace qcb
