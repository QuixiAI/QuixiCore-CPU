#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ":"      \
                << __LINE__ << '\n';                                      \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

using quixicore_cpu::Status;

bool close(float actual, double expected, double atol = 1e-6,
           double rtol = 1e-5) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::fabs(static_cast<double>(actual) - expected) <=
             atol + rtol * std::fabs(expected);
}

std::vector<double> attention_oracle(const std::vector<float>& q,
                                     const std::vector<float>& k,
                                     const std::vector<float>& v,
                                     long long query_heads,
                                     long long kv_heads, long long query_length,
                                     long long kv_length, long long dim,
                                     bool causal) {
  std::vector<double> out(
      static_cast<std::size_t>(query_heads * query_length * dim));
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  for (long long head = 0; head < query_heads; ++head) {
    const long long kv_head = head / (query_heads / kv_heads);
    for (long long query_position = 0; query_position < query_length;
         ++query_position) {
      std::vector<double> scores(static_cast<std::size_t>(kv_length),
                                 -std::numeric_limits<double>::infinity());
      double maximum = -std::numeric_limits<double>::infinity();
      const long long limit = query_position + kv_length - query_length;
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > limit) {
          continue;
        }
        double score = 0.0;
        for (long long d = 0; d < dim; ++d) {
          score += q[static_cast<std::size_t>(
                       (head * query_length + query_position) * dim + d)] *
                   k[static_cast<std::size_t>(
                       (kv_head * kv_length + key_position) * dim + d)];
        }
        scores[static_cast<std::size_t>(key_position)] = score * scale;
        maximum = std::max(maximum, score * scale);
      }
      double sum = 0.0;
      for (double& score : scores) {
        score = std::isfinite(score) ? std::exp(score - maximum) : 0.0;
        sum += score;
      }
      for (long long d = 0; d < dim; ++d) {
        double value = 0.0;
        for (long long key_position = 0; key_position < kv_length;
             ++key_position) {
          value += scores[static_cast<std::size_t>(key_position)] *
                   v[static_cast<std::size_t>(
                       (kv_head * kv_length + key_position) * dim + d)];
        }
        out[static_cast<std::size_t>(
            (head * query_length + query_position) * dim + d)] = value / sum;
      }
    }
  }
  return out;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;

  float scalar = 0.0f;
  REQUIRE(gelu(nullptr, &scalar, 1) == Status::kInvalidArgument);
  REQUIRE(softmax(&scalar, &scalar, 0, 1) == Status::kInvalidShape);
  REQUIRE(attention(&scalar, &scalar, &scalar, &scalar, 3, 2, 1, 1, 1,
                    false) == Status::kInvalidShape);

  // Activations and backward derivatives.
  {
    const std::vector<float> x = {-2.0f, -0.5f, 0.0f, 1.5f};
    std::vector<float> y(x.size());
    REQUIRE(gelu(x.data(), y.data(), x.size(), GeluApprox::kErf) ==
            Status::kOk);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double expected =
          0.5 * x[i] * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
      REQUIRE(close(y[i], expected));
    }
    REQUIRE(gelu(x.data(), y.data(), x.size(), GeluApprox::kTanh) ==
            Status::kOk);
    for (float value : y) {
      REQUIRE(std::isfinite(value));
    }
    const std::vector<float> grad(x.size(), 0.75f);
    std::vector<float> backward(x.size());
    REQUIRE(gelu_backward(grad.data(), x.data(), backward.data(), x.size()) ==
            Status::kOk);
    constexpr double kStep = 1e-3;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const auto exact = [](double value) {
        return 0.5 * value *
               (1.0 + std::erf(value / std::sqrt(2.0)));
      };
      const double derivative =
          (exact(x[i] + kStep) - exact(x[i] - kStep)) / (2.0 * kStep);
      REQUIRE(close(backward[i], 0.75 * derivative, 2e-5, 2e-4));
    }
    REQUIRE(silu(x.data(), y.data(), x.size()) == Status::kOk);
    for (std::size_t i = 0; i < x.size(); ++i) {
      REQUIRE(close(y[i], x[i] / (1.0 + std::exp(-x[i]))));
    }
    REQUIRE(silu_backward(grad.data(), x.data(), backward.data(), x.size()) ==
            Status::kOk);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double probability = 1.0 / (1.0 + std::exp(-x[i]));
      REQUIRE(close(backward[i],
                    grad[i] * probability *
                        (1.0 + x[i] * (1.0 - probability))));
    }
  }

  // The complete 22-value llama unary selector, including constrained XiELU.
  {
    const std::vector<float> x = {-2.5f, -0.25f, 0.0f, 0.75f, 3.5f};
    std::vector<float> y(x.size());
    const std::vector<UnaryOp> modes = {
        UnaryOp::kAbs,       UnaryOp::kSign,       UnaryOp::kNegate,
        UnaryOp::kStep,      UnaryOp::kTanh,       UnaryOp::kElu,
        UnaryOp::kRelu,      UnaryOp::kSigmoid,    UnaryOp::kGelu,
        UnaryOp::kGeluQuick, UnaryOp::kSilu,       UnaryOp::kHardSwish,
        UnaryOp::kHardSigmoid, UnaryOp::kExp,      UnaryOp::kExpm1,
        UnaryOp::kSoftplus,  UnaryOp::kGeluErf,    UnaryOp::kFloor,
        UnaryOp::kCeil,      UnaryOp::kRound,      UnaryOp::kTrunc};
    for (UnaryOp mode : modes) {
      REQUIRE(unary(x.data(), y.data(), x.size(), mode) == Status::kOk);
      for (std::size_t i = 0; i < x.size(); ++i) {
        const double value = x[i];
        double expected = 0.0;
        switch (mode) {
          case UnaryOp::kAbs: expected = std::fabs(value); break;
          case UnaryOp::kSign:
            expected = value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
            break;
          case UnaryOp::kNegate: expected = -value; break;
          case UnaryOp::kStep: expected = value > 0.0 ? 1.0 : 0.0; break;
          case UnaryOp::kTanh: expected = std::tanh(value); break;
          case UnaryOp::kElu:
            expected = value > 0.0 ? value : std::expm1(value); break;
          case UnaryOp::kRelu: expected = std::max(0.0, value); break;
          case UnaryOp::kSigmoid:
            expected = 1.0 / (1.0 + std::exp(-value)); break;
          case UnaryOp::kGelu: {
            constexpr double kC = 0.7978845608028654;
            expected = 0.5 * value *
                       (1.0 + std::tanh(kC * value *
                                                (1.0 + 0.044715 * value * value)));
            break;
          }
          case UnaryOp::kGeluQuick:
            expected = value / (1.0 + std::exp(-1.702 * value)); break;
          case UnaryOp::kSilu:
            expected = value / (1.0 + std::exp(-value)); break;
          case UnaryOp::kHardSwish:
            expected = value * std::clamp((value + 3.0) / 6.0, 0.0, 1.0);
            break;
          case UnaryOp::kHardSigmoid:
            expected = std::clamp((value + 3.0) / 6.0, 0.0, 1.0); break;
          case UnaryOp::kExp: expected = std::exp(value); break;
          case UnaryOp::kExpm1: expected = std::exp(value) - 1.0; break;
          case UnaryOp::kSoftplus:
            expected = value > 20.0 ? value : std::log(1.0 + std::exp(value));
            break;
          case UnaryOp::kGeluErf:
            expected = 0.5 * value *
                       (1.0 + std::erf(value / std::sqrt(2.0)));
            break;
          case UnaryOp::kFloor: expected = std::floor(value); break;
          case UnaryOp::kCeil: expected = std::ceil(value); break;
          case UnaryOp::kRound: expected = std::round(value); break;
          case UnaryOp::kTrunc: expected = std::trunc(value); break;
          case UnaryOp::kXiElu: break;
        }
        REQUIRE(close(y[i], expected, 2e-6, 2e-5));
      }
    }
    const XiEluParams params{-0.4f, 0.3f, 0.2f, -0.1f};
    REQUIRE(unary(x.data(), y.data(), x.size(), UnaryOp::kXiElu, params) ==
            Status::kOk);
    const double alpha_n = params.beta +
                           std::log(1.0 + std::exp(params.alpha_n));
    const double alpha_p = std::log(1.0 + std::exp(params.alpha_p));
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double expected =
          x[i] > 0.0f
              ? alpha_p * x[i] * x[i] + params.beta * x[i]
              : (std::expm1(std::min(x[i], params.eps)) - x[i]) * alpha_n +
                    params.beta * x[i];
      REQUIRE(close(y[i], expected, 2e-6, 2e-5));
    }
    REQUIRE(unary(x.data(), y.data(), x.size(), static_cast<UnaryOp>(99)) ==
            Status::kInvalidArgument);
  }

  // All GLU modes share gate/value split semantics.
  {
    const float x[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float y[2] = {};
    REQUIRE(glu(x, y, 1, 2, GluMode::kSwiGlu) == Status::kOk);
    REQUIRE(close(y[0], 0.0));
    REQUIRE(close(y[1], 3.0 / (1.0 + std::exp(-1.0))));
    REQUIRE(glu(x, y, 1, 2, GluMode::kReGlu) == Status::kOk);
    REQUIRE(close(y[0], 0.0));
    REQUIRE(close(y[1], 3.0));
    REQUIRE(glu(x, y, 1, 2, GluMode::kGlu) == Status::kOk);
    REQUIRE(close(y[0], 1.0));
    REQUIRE(close(y[1], 3.0 / (1.0 + std::exp(-1.0))));
    REQUIRE(glu(x, y, 1, 2, GluMode::kGeGlu) == Status::kOk);
    const double geglu =
        0.5 * (1.0 + std::tanh(0.7978845608028654 * 1.044715));
    REQUIRE(close(y[1], 3.0 * geglu));
    REQUIRE(glu(x, y, 1, 2, GluMode::kGeGluErf) == Status::kOk);
    REQUIRE(close(y[1], 3.0 * 0.5 * (1.0 + std::erf(1.0 / std::sqrt(2.0)))));
    REQUIRE(glu(x, y, 1, 2, GluMode::kGeGluQuick) == Status::kOk);
    REQUIRE(close(y[1], 3.0 / (1.0 + std::exp(-1.702))));
    REQUIRE(glu(x, y, 1, 2, static_cast<GluMode>(99)) ==
            Status::kInvalidArgument);

    const float gate[] = {-3.0f, 2.0f};
    const float value[] = {-4.0f, 3.0f};
    REQUIRE(swiglu_oai(gate, value, y, 1, 2, 1.5f, 2.0f) == Status::kOk);
    REQUIRE(close(y[0], -3.0 / (1.0 + std::exp(4.5)) * -1.0));
    REQUIRE(close(y[1], 2.0 / (1.0 + std::exp(-3.0)) * 3.0));
    REQUIRE(swiglu_oai(gate, value, y, 1, 2, 1.5f, -1.0f) ==
            Status::kInvalidArgument);
  }

  // Stable softmax and LayerNorm.
  {
    const float x[] = {1000.0f, 1001.0f, 999.0f, -2.0f, -2.0f, -2.0f};
    float y[6] = {};
    REQUIRE(softmax(x, y, 2, 3) == Status::kOk);
    for (int row = 0; row < 2; ++row) {
      REQUIRE(close(y[row * 3] + y[row * 3 + 1] + y[row * 3 + 2], 1.0));
    }
    REQUIRE(y[1] > y[0] && y[0] > y[2]);
    REQUIRE(close(y[3], 1.0 / 3.0));

    const float weight[] = {1.0f, 2.0f, 0.5f};
    const float bias[] = {0.25f, -0.5f, 1.0f};
    REQUIRE(layer_norm(x, weight, bias, y, 2, 3, 1e-5f) == Status::kOk);
    for (int row = 0; row < 2; ++row) {
      double mean = 0.0;
      for (int i = 0; i < 3; ++i) mean += x[row * 3 + i];
      mean /= 3.0;
      double variance = 0.0;
      for (int i = 0; i < 3; ++i) {
        const double delta = x[row * 3 + i] - mean;
        variance += delta * delta;
      }
      variance /= 3.0;
      for (int i = 0; i < 3; ++i) {
        const double expected =
            (x[row * 3 + i] - mean) / std::sqrt(variance + 1e-5) *
                weight[i] +
            bias[i];
        REQUIRE(close(y[row * 3 + i], expected));
      }
    }
  }

  // Dense and grouped GEMM.
  {
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float b[] = {1, -1, 2, 0, 3, 1};
    float c[4] = {};
    REQUIRE(dense_gemm(a, b, c, 2, 2, 3) == Status::kOk);
    REQUIRE(close(c[0], 14.0) && close(c[1], 2.0));
    REQUIRE(close(c[2], 32.0) && close(c[3], 2.0));

    const float ga[] = {1, 2, 3, 4};
    const float gb[] = {2, 0, 0, 2, 1, 1, 1, -1};
    float gc[4] = {};
    REQUIRE(grouped_gemm(ga, gb, gc, 2, 1, 2, 2) == Status::kOk);
    REQUIRE(close(gc[0], 2.0) && close(gc[1], 4.0));
    REQUIRE(close(gc[2], 7.0) && close(gc[3], -1.0));
  }

  // RoPE identity at position zero and dense GQA attention.
  {
    const std::vector<float> x = {1, 2, 3, 4, -1, -2, -3, -4};
    std::vector<float> y(x.size());
    REQUIRE(rope(x.data(), y.data(), 2, 1, 4, 10000.0f, 0) == Status::kOk);
    for (int i = 0; i < 4; ++i) REQUIRE(close(y[i], x[i]));
    REQUIRE(!std::equal(y.begin() + 4, y.end(), x.begin() + 4));

    const std::vector<float> q = {1, 0, 0, 1, 1, 1, -1, 1};
    const std::vector<float> k = {1, 0, 0, 1};
    const std::vector<float> v = {2, 1, -1, 3};
    std::vector<float> out(q.size());
    REQUIRE(attention(q.data(), k.data(), v.data(), out.data(), 2, 1, 2, 2,
                      2, true) == Status::kOk);
    const auto oracle = attention_oracle(q, k, v, 2, 1, 2, 2, 2, true);
    for (std::size_t i = 0; i < out.size(); ++i) {
      REQUIRE(close(out[i], oracle[i], 2e-6, 2e-5));
    }
  }

  // Paged attention and MLA decode follow their declared cache layouts.
  {
    const float q[] = {1, 0};
    const float key_cache[] = {1, 0, 0, 1, -1, 0, 0, -1};
    const float value_cache[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int table[] = {1, 0};
    const int context[] = {3};
    float out[2] = {};
    REQUIRE(paged_attention(q, key_cache, value_cache, table, context, out, 2,
                            1, 1, 1, 2, 2, 2) == Status::kOk);
    const double s0 = std::exp(-1.0 / std::sqrt(2.0));
    const double s1 = std::exp(0.0);
    const double s2 = std::exp(1.0 / std::sqrt(2.0));
    const double denominator = s0 + s1 + s2;
    REQUIRE(close(out[0], (5 * s0 + 7 * s1 + 1 * s2) / denominator));
    REQUIRE(close(out[1], (6 * s0 + 8 * s1 + 2 * s2) / denominator));
    const int invalid_table[] = {2, 0};
    REQUIRE(paged_attention(q, key_cache, value_cache, invalid_table, context,
                            out, 2, 1, 1, 1, 2, 2, 2) ==
            Status::kInvalidArgument);

    const float mla_q[] = {1, 0, 1};
    const float mla_cache[] = {1, 10, 0, 3, 20, 1};
    const int mla_table[] = {0};
    const int mla_context[] = {2};
    float mla_out[2] = {};
    REQUIRE(mla_decode(mla_q, mla_cache, mla_table, mla_context, mla_out, 1, 1,
                       1, 2, 1, 2, 1) == Status::kOk);
    const double w0 = 1.0;
    const double w1 = std::exp(3.0 / std::sqrt(3.0));
    REQUIRE(close(mla_out[0], (1 * w0 + 3 * w1) / (w0 + w1)));
    REQUIRE(close(mla_out[1], (10 * w0 + 20 * w1) / (w0 + w1)));
  }

  // Sampling, beam search, and speculative verification.
  {
    const float logits[] = {1, 3, 3, -1, 4, 0, 2, 1};
    int out[2] = {};
    REQUIRE(argmax_sample(logits, out, 2, 4) == Status::kOk);
    REQUIRE(out[0] == 1 && out[1] == 0);
    int repeat[2] = {};
    REQUIRE(sample_categorical(logits, out, 2, 4, 0.7f, 123) == Status::kOk);
    REQUIRE(sample_categorical(logits, repeat, 2, 4, 0.7f, 123) ==
            Status::kOk);
    REQUIRE(out[0] == repeat[0] && out[1] == repeat[1]);
    REQUIRE(top_k_sample(logits, out, 2, 4, 2, 1.0f, 5) == Status::kOk);
    REQUIRE((out[0] == 1 || out[0] == 2));
    REQUIRE((out[1] == 0 || out[1] == 2));
    REQUIRE(top_p_sample(logits, out, 2, 4, 0.01f, 1.0f, 9) == Status::kOk);
    REQUIRE(out[0] == 1 && out[1] == 0);
    REQUIRE(min_p_sample(logits, out, 2, 4, 1.0f, 1.0f, 9) == Status::kOk);
    REQUIRE((out[0] == 1 || out[0] == 2) && out[1] == 0);

    const float beam_logits[] = {3, 1, 0, 2, 2, 0};
    const float cumulative[] = {0, -2};
    int tokens[2] = {};
    int parents[2] = {};
    float scores[2] = {};
    REQUIRE(beam_search_step(beam_logits, cumulative, tokens, parents, scores,
                             1, 2, 3) == Status::kOk);
    REQUIRE(tokens[0] == 0 && parents[0] == 0);
    REQUIRE(scores[0] >= scores[1]);

    const int draft_tokens[] = {0, 1};
    const float draft_probs[] = {0.8f, 0.2f, 0.1f, 0.9f};
    const float target_probs[] = {0.9f, 0.1f, 0.2f, 0.8f, 0.3f, 0.7f};
    const int bonus[] = {1};
    const float uniforms[] = {0.1f, 0.1f};
    int verified[3] = {};
    int accepted = 0;
    REQUIRE(speculative_verify(draft_tokens, draft_probs, target_probs, bonus,
                               uniforms, verified, &accepted, 1, 2, 2, 42) ==
            Status::kOk);
    REQUIRE(accepted == 2);
    REQUIRE(verified[0] == 0 && verified[1] == 1 && verified[2] == 1);
  }

  // Embedding and KV cache movement.
  {
    const float table[] = {1, 2, 3, 4, 5, 6};
    const int ids[] = {2, 0};
    float embedded[4] = {};
    REQUIRE(embedding_lookup(table, ids, embedded, 3, 2, 2) == Status::kOk);
    REQUIRE(embedded[0] == 5 && embedded[1] == 6 && embedded[2] == 1 &&
            embedded[3] == 2);
    float cache[8] = {};
    const int slots[] = {3, 1};
    REQUIRE(kv_cache_scatter(cache, embedded, slots, 4, 2, 2) == Status::kOk);
    const int gather_ids[] = {1, 3};
    float gathered[4] = {};
    REQUIRE(kv_cache_gather(cache, gather_ids, gathered, 4, 2, 2) ==
            Status::kOk);
    REQUIRE(gathered[0] == 1 && gathered[1] == 2 && gathered[2] == 5 &&
            gathered[3] == 6);
  }

  // Utility kernels.
  {
    const float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dropped[8] = {};
    float dropped_repeat[8] = {};
    REQUIRE(dropout(x, dropped, 8, 0.25f, 77) == Status::kOk);
    REQUIRE(dropout(x, dropped_repeat, 8, 0.25f, 77) == Status::kOk);
    REQUIRE(std::equal(dropped, dropped + 8, dropped_repeat));
    for (int i = 0; i < 8; ++i) {
      REQUIRE(dropped[i] == 0.0f || close(dropped[i], x[i] / 0.75));
    }
    const float logits[] = {1, 2, 3, -1, 0, 1};
    const int target[] = {2, 0};
    float loss[2] = {};
    REQUIRE(cross_entropy(logits, target, loss, 2, 3) == Status::kOk);
    REQUIRE(close(loss[0], std::log(std::exp(1.0) + std::exp(2.0) +
                                    std::exp(3.0)) -
                               3.0));
    const float hx[] = {1, 2, 3, 4};
    float hy[4] = {};
    REQUIRE(hadamard(hx, hy, 1, 4) == Status::kOk);
    REQUIRE(hy[0] == 10 && hy[1] == -2 && hy[2] == -4 && hy[3] == 0);
  }

  // MoE routing, linear attention, and selective scan.
  {
    const float router[] = {1, 3, 2, 0};
    int ids[2] = {};
    float weights[2] = {};
    REQUIRE(moe_route_topk(router, ids, weights, 1, 4, 2) == Status::kOk);
    REQUIRE(ids[0] == 1 && ids[1] == 2);
    REQUIRE(close(weights[0] + weights[1], 1.0));
    REQUIRE(close(weights[0], std::exp(3.0) / (std::exp(3.0) + std::exp(2.0))));

    const float q[] = {1, 2};
    const float k[] = {3, 4};
    const float v[] = {5, 6};
    float linear_out[2] = {};
    REQUIRE(linear_attention(q, k, v, linear_out, 1, 2, 1, 1e-6f) ==
            Status::kOk);
    const double kv = 3.0 * 5.0 + 4.0 * 6.0;
    const double z = 7.0;
    REQUIRE(close(linear_out[0], kv / (z + 1e-6)));
    REQUIRE(close(linear_out[1], 2.0 * kv / (2.0 * z + 1e-6)));

    const float u[] = {1, 2, 3};
    const float delta[] = {0.5f, 0.5f, 0.5f};
    const float a[] = {-1};
    const float b[] = {2, 2, 2};
    const float c[] = {3, 3, 3};
    const float d[] = {0.25f};
    float scan[3] = {};
    REQUIRE(selective_scan(u, delta, a, b, c, d, scan, 1, 3, 1) ==
            Status::kOk);
    double hidden = 0.0;
    for (int i = 0; i < 3; ++i) {
      hidden = std::exp(-0.5) * hidden + 0.5 * 2.0 * u[i];
      REQUIRE(close(scan[i], 3.0 * hidden + 0.25 * u[i]));
    }
  }

  // AdamW and q8_0 matrix/LM-head composition.
  {
    float parameters[] = {1.0f, -2.0f};
    const float gradients[] = {0.5f, -0.25f};
    float first[] = {0, 0};
    float second[] = {0, 0};
    REQUIRE(adamw(parameters, gradients, first, second, 2, 0.1f, 0.9f,
                  0.99f, 1e-8f, 0.01f, 1) == Status::kOk);
    REQUIRE(close(first[0], 0.05));
    REQUIRE(close(second[0], 0.0025));
    REQUIRE(close(parameters[0], 1.0 - 0.1 * (1.0 + 0.01), 1e-6, 1e-5));

    constexpr long long kRows = 3;
    constexpr long long kHidden = 32;
    std::vector<float> weights(static_cast<std::size_t>(kRows * kHidden));
    for (long long row = 0; row < kRows; ++row) {
      for (long long i = 0; i < kHidden; ++i) {
        weights[static_cast<std::size_t>(row * kHidden + i)] =
            static_cast<float>((row + 1) * (i - 16)) / 32.0f;
      }
    }
    std::vector<float> x(static_cast<std::size_t>(2 * kHidden));
    for (long long i = 0; i < kHidden; ++i) {
      x[static_cast<std::size_t>(i)] = static_cast<float>(i + 1) / 32.0f;
      x[static_cast<std::size_t>(kHidden + i)] = -x[static_cast<std::size_t>(i)];
    }
    for (const QuantFormat format : {QuantFormat::kQ8_0,
                                     QuantFormat::kQ4_0}) {
      std::size_t packed_size = 0;
      REQUIRE(qgemv_packed_size(format, kRows, kHidden, &packed_size) ==
              Status::kOk);
      std::vector<std::uint8_t> packed(packed_size);
      REQUIRE(qgemv_pack(format, weights.data(), kRows, kHidden,
                         packed.data()) == Status::kOk);
      float matrix_out[6] = {};
      REQUIRE(qgemm(format, packed.data(), x.data(), matrix_out, 2, kRows,
                    kHidden) == Status::kOk);
      float vector_out[3] = {};
      REQUIRE(qgemv(format, packed.data(), x.data(), vector_out, kRows,
                    kHidden) == Status::kOk);
      for (int i = 0; i < 3; ++i) REQUIRE(matrix_out[i] == vector_out[i]);
      int tokens[2] = {};
      REQUIRE(quantized_lm_head_argmax(format, packed.data(), x.data(), tokens,
                                       2, kRows, kHidden) == Status::kOk);
      for (int row = 0; row < 2; ++row) {
        int best = 0;
        for (int token = 1; token < 3; ++token) {
          if (matrix_out[row * 3 + token] > matrix_out[row * 3 + best]) {
            best = token;
          }
        }
        REQUIRE(tokens[row] == best);
      }
    }
  }

  return 0;
}
