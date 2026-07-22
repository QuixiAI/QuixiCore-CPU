#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quantization.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool require(bool value, const char* message) {
  if (!value) std::cerr << "FAIL: " << message << '\n';
  return value;
}

bool close(float lhs, float rhs, float tolerance = 1e-4f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;
  constexpr long long kM = 2, kN = 4, kK = 32;
  std::vector<float> weights(kN * kK, 0.0f);
  for (long long output = 0; output < kN; ++output) {
    for (long long input = 0; input < kK; ++input) {
      weights[output * kK + input] =
          static_cast<float>((output + 1) * (input % 5 - 2)) / 8.0f;
    }
  }
  std::size_t bytes = 0;
  ok &= require(qgemv_packed_size(QuantFormat::kQ8_0, kN, kK, &bytes) ==
                    Status::kOk,
                "q8 packed size");
  std::vector<std::uint8_t> packed(bytes);
  std::vector<float> unpacked(kN * kK), x(kM * kK);
  for (long long i = 0; i < kM * kK; ++i) x[i] = (i % 7 - 3) * 0.125f;
  ok &= require(qgemv_pack(QuantFormat::kQ8_0, weights.data(), kN, kK,
                           packed.data()) == Status::kOk &&
                    qgemv_unpack(QuantFormat::kQ8_0, packed.data(), kN, kK,
                                 unpacked.data()) == Status::kOk,
                "q8 projection pack");
  std::vector<float> y(kM * kN), expected(kM * kN);
  ok &= require(qgemm(QuantFormat::kQ8_0, packed.data(), x.data(), y.data(),
                      kM, kN, kK) == Status::kOk,
                "qgemm");
  for (long long row = 0; row < kM; ++row) {
    for (long long output = 0; output < kN; ++output) {
      double sum = 0.0;
      for (long long input = 0; input < kK; ++input) {
        sum += x[row * kK + input] * unpacked[output * kK + input];
      }
      expected[row * kN + output] = static_cast<float>(sum);
      ok &= require(close(y[row * kN + output], expected[row * kN + output]),
                    "qgemm value");
    }
  }
  const std::vector<float> bias = {-1.0f, 0.0f, 0.5f, 1.0f};
  ok &= require(qgemm_epilogue(QuantFormat::kQ8_0, packed.data(), x.data(),
                               bias.data(), y.data(), kM, kN, kK,
                               LinearActivation::kSilu) == Status::kOk,
                "qgemm epilogue");
  ok &= require(std::isfinite(y[0]), "qgemm epilogue value");

  std::vector<float> grad_y(kM * kN, 1.0f), grad_x(kM * kK);
  ok &= require(qgemm_backward_input(QuantFormat::kQ8_0, packed.data(),
                                     grad_y.data(), grad_x.data(), kM, kN,
                                     kK) == Status::kOk,
                "qgemm backward input");
  for (long long input = 0; input < kK; ++input) {
    float sum = 0.0f;
    for (long long output = 0; output < kN; ++output) {
      sum += unpacked[output * kK + input];
    }
    ok &= require(close(grad_x[input], sum), "qgemm backward value");
  }

  const std::int8_t int_weights[] = {1, 2, -1, 3, -2, 1, 0, 2};
  const std::int8_t int_x[] = {2, -1, 3, 1};
  const float weight_scale[] = {0.5f, 0.25f};
  const float activation_scale[] = {2.0f};
  float int_y[2] = {};
  ok &= require(int8_gemm(int_weights, int_x, weight_scale,
                          activation_scale, nullptr, nullptr, int_y, 1, 2, 4,
                          false) == Status::kOk &&
                    close(int_y[0], 0.0f) && close(int_y[1], -1.5f),
                "int8 gemm");

  std::vector<std::uint8_t> fp8_weights(8), fp8_x(4);
  for (int i = 0; i < 8; ++i) {
    fp8_weights[i] = float8_encode(static_cast<float>(int_weights[i]),
                                   Float8Format::kE4M3FN);
  }
  for (int i = 0; i < 4; ++i) {
    fp8_x[i] = float8_encode(static_cast<float>(int_x[i]),
                             Float8Format::kE4M3FN);
  }
  const float ones[] = {1.0f, 1.0f};
  ok &= require(fp8_scaled_gemm(fp8_weights.data(), fp8_x.data(), ones,
                                ones, int_y, 1, 2, 4,
                                Float8Format::kE4M3FN) == Status::kOk &&
                    close(int_y[0], 0.0f) && close(int_y[1], -3.0f),
                "fp8 scaled gemm");

  std::vector<float> ternary_weights(2 * kK);
  for (long long input = 0; input < kK; ++input) {
    ternary_weights[input] = input % 3 == 0 ? -1.0f : 1.0f;
    ternary_weights[kK + input] = input % 2 == 0 ? 0.0f : 1.0f;
  }
  std::vector<std::uint8_t> bitnet_packed(20);
  std::vector<float> ternary_dequant(2 * kK);
  ok &= require(ternary_pack(ternary_weights.data(), bitnet_packed.data(),
                             ternary_dequant.data(), 2, kK, kK) ==
                        Status::kOk,
                "BitNet pack");
  std::vector<std::int8_t> activation_codes(kK);
  float dynamic_scale = 0.0f;
  ok &= require(quantize_int8(x.data(), activation_codes.data(),
                              &dynamic_scale, 1, kK, kK) == Status::kOk,
                "BitNet activation quantization");
  float bitnet_direct[2], bitnet_fused[2];
  ok &= require(bitnet_int8_gemm(
                        bitnet_packed.data(), activation_codes.data(),
                        &dynamic_scale, bitnet_direct, 1, 2, kK) ==
                        Status::kOk &&
                    bitnet_fused_gemm(bitnet_packed.data(), x.data(),
                                      bitnet_fused, 1, 2, kK) == Status::kOk &&
                    close(bitnet_direct[0], bitnet_fused[0]) &&
                    close(bitnet_direct[1], bitnet_fused[1]),
                "BitNet direct and fused GEMM");

  int tokens[kM] = {};
  ok &= require(quantized_lm_head_sample(
                        QuantFormat::kQ8_0, packed.data(), x.data(),
                        bias.data(), tokens, kM, kN, kK,
                        LmHeadSampling::kArgmax, 1, 1.0f, 1.0f, 7) ==
                        Status::kOk,
                "quantized lm head argmax");
  std::uint8_t mask[kM] = {0xF0, 0xF0};
  int top_tokens[kM * 2] = {};
  float log_probs[kM * 2] = {};
  ok &= require(quantized_lm_head_masked_topk(
                        QuantFormat::kQ8_0, packed.data(), x.data(),
                        bias.data(), mask, top_tokens, log_probs, kM, kN, kK,
                        2) == Status::kOk &&
                    top_tokens[0] != top_tokens[1],
                "quantized lm head masked topk");
  const int candidates[] = {0, 2, 1, 3};
  const long long offsets[] = {0, 2, 4};
  ok &= require(quantized_lm_head_candidates(
                        QuantFormat::kQ8_0, packed.data(), x.data(),
                        bias.data(), candidates, offsets, top_tokens,
                        log_probs, kM, kN, kK, 4, 1) == Status::kOk,
                "quantized lm head candidates");
  const float beam_cumulative[] = {0.0f, -0.5f};
  int beam_tokens[2], beam_parents[2];
  float beam_scores[2];
  ok &= require(quantized_lm_head_beam_advance(
                        QuantFormat::kQ8_0, packed.data(), x.data(),
                        bias.data(), beam_cumulative, beam_tokens,
                        beam_parents, beam_scores, 1, 2, kN, kK) ==
                        Status::kOk &&
                    beam_tokens[0] >= 0 && beam_tokens[0] < kN &&
                    beam_tokens[1] >= 0 && beam_tokens[1] < kN &&
                    beam_scores[0] >= beam_scores[1],
                "quantized lm head beam advance");

  const float constrained_weights[] = {1, 0, 0, 0, 0, 1, 0, 0,
                                       0, 0, 1, 0, 0, 0, 0, 1};
  const float constrained_hidden[] = {1, 2, 3, 4};
  std::uint8_t forbidden[16] = {};
  forbidden[3] = 1;
  const int previous[] = {0};
  int selected = -1;
  float selected_log_probability = 0.0f;
  ok &= require(lm_head_constrained(
                        constrained_hidden, constrained_weights, nullptr,
                        forbidden, previous, &selected,
                        &selected_log_probability, 1, 4, 4) == Status::kOk &&
                    selected == 2 && std::isfinite(selected_log_probability),
                "constrained lm head");

  if (!ok) return 1;
  std::cout << "quantized projection tests passed\n";
  return 0;
}
