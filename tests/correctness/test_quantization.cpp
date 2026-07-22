#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"
#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using quixicore_cpu::Status;

bool close(float a, float b, float tolerance = 1e-5f) {
  return std::fabs(a - b) <= tolerance;
}

bool require(bool value, const char* message) {
  if (!value) std::cerr << "FAIL: " << message << '\n';
  return value;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;

  ok &= require(float8_encode(448.0f, Float8Format::kE4M3FN) == 0x7e,
                "e4m3 max code");
  ok &= require(close(float8_decode(0x7e, Float8Format::kE4M3FN), 448.0f),
                "e4m3 max decode");
  ok &= require(close(float8_decode(0x3c, Float8Format::kE4M3FN), 1.5f),
                "e4m3 normal decode");
  ok &= require(close(float8_decode(0x3e, Float8Format::kE5M2), 1.5f),
                "e5m2 normal decode");

  const std::vector<float> x = {-2.0f, -0.5f, 0.0f, 2.0f,
                                1.0f,  2.0f,  3.0f, 4.0f};
  std::vector<std::int8_t> i8(x.size());
  std::vector<float> scales(4), dequant(x.size());
  ok &= require(quantize_int8(x.data(), i8.data(), scales.data(), 2, 4, 2) ==
                    Status::kOk,
                "group int8 quant");
  ok &= require(dequantize_int8(i8.data(), scales.data(), dequant.data(), 2, 4,
                                2) == Status::kOk,
                "group int8 dequant");
  ok &= require(i8[0] == -127 && i8[3] == 127 && i8[7] == 127,
                "group int8 endpoints");

  std::vector<int> zero(2);
  ok &= require(quantize_int8_asymmetric(x.data(), i8.data(), scales.data(),
                                         zero.data(), 2, 4) == Status::kOk,
                "asymmetric int8 quant");
  ok &= require(dequantize_int8_asymmetric(i8.data(), scales.data(), zero.data(),
                                           dequant.data(), 2, 4) == Status::kOk,
                "asymmetric int8 dequant");
  ok &= require(close(dequant[0], -2.0f, 0.02f) &&
                    close(dequant[3], 2.0f, 0.02f),
                "asymmetric endpoints");

  std::vector<std::uint8_t> fp8(x.size());
  ok &= require(quantize_float8(x.data(), fp8.data(), scales.data(), 2, 4, 2,
                                Float8Format::kE4M3FN, true) == Status::kOk,
                "group fp8 quant");
  ok &= require(dequantize_float8(fp8.data(), scales.data(), dequant.data(), 2,
                                  4, 2) == Status::kOk,
                "group fp8 dequant");
  ok &= require(close(std::log2(scales[0]),
                      std::nearbyint(std::log2(scales[0]))),
                "ue8m0 scale");

  std::vector<std::uint8_t> int4_packed(4);
  std::vector<float> int4_scales(4), int4_dequant(8);
  ok &= require(quantize_int4_group(x.data(), int4_packed.data(),
                                    int4_scales.data(), 2, 4, 2) ==
                        Status::kOk &&
                    dequantize_int4_group(
                        int4_packed.data(), int4_scales.data(),
                        int4_dequant.data(), 2, 4, 2) == Status::kOk,
                "group int4 round trip");
  ok &= require(close(int4_dequant[0], -2.0f, 0.3f) &&
                    close(int4_dequant[7], 4.0f, 0.6f),
                "group int4 endpoints");

  std::vector<float> fq(x.size());
  ok &= require(fake_quant_int8(x.data(), fq.data(), i8.data(), scales.data(),
                                2, 4) == Status::kOk,
                "fake quant int8");
  for (std::size_t i = 0; i < x.size(); ++i) {
    ok &= require(close(fq[i], scales[i / 4] * i8[i]),
                  "fake quant reconstruction");
  }

  std::vector<float> ternary(64);
  for (int i = 0; i < 64; ++i) ternary[i] = (i % 3 - 1) * 0.25f;
  std::vector<std::uint8_t> ternary_packed(20);
  std::vector<float> ternary_dequant(64), ternary_unpacked(64);
  ok &= require(ternary_pack(ternary.data(), ternary_packed.data(),
                             ternary_dequant.data(), 1, 64, 64) == Status::kOk,
                "ternary pack");
  ok &= require(ternary_unpack(ternary_packed.data(), ternary_unpacked.data(),
                               1, 64) == Status::kOk,
                "ternary unpack");
  ok &= require(ternary_dequant == ternary_unpacked,
                "ternary emitted dequant agrees");
  std::uint32_t counts[3] = {};
  std::uint32_t flips = 99;
  ok &= require(ternary_stats(ternary_packed.data(), counts, 1, 64) ==
                    Status::kOk && counts[0] + counts[1] + counts[2] == 64,
                "ternary stats");
  ok &= require(ternary_code_flip_count(ternary_packed.data(),
                                        ternary_packed.data(), &flips, 1, 64) ==
                        Status::kOk && flips == 0,
                "ternary flip count");

  std::vector<float> tq(256);
  for (int i = 0; i < 256; ++i) tq[i] = (i % 5 - 2) * 0.5f;
  std::vector<std::uint8_t> tq_packed(66);
  std::vector<float> tq_dequant(256), tq_unpacked(256);
  ok &= require(tq2_0_pack(tq.data(), tq_packed.data(), tq_dequant.data(), 1,
                           256) == Status::kOk,
                "tq2 pack");
  ok &= require(tq2_0_unpack(tq_packed.data(), tq_unpacked.data(), 1, 256) ==
                    Status::kOk,
                "tq2 unpack");
  ok &= require(tq_dequant == tq_unpacked, "tq2 emitted dequant agrees");
  std::vector<std::uint8_t> public_packed(66);
  ok &= require(qgemv_pack(QuantFormat::kTQ2_0, tq.data(), 1, 256,
                           public_packed.data()) == Status::kOk &&
                    public_packed == tq_packed,
                "tq2 public pack");
  std::vector<float> activation(256, 1.0f);
  float output = 0.0f;
  ok &= require(qgemv(QuantFormat::kTQ2_0, tq_packed.data(), activation.data(),
                      &output, 1, 256) == Status::kOk,
                "tq2 qgemv");
  float expected = 0.0f;
  for (float value : tq_unpacked) expected += value;
  ok &= require(close(output, expected, 1e-4f), "tq2 qgemv value");

  const std::vector<float> gated_x = {-1.0f, 0.0f, 1.0f, 2.0f};
  const std::vector<float> gated_g = {2.0f, 3.0f, -1.0f, 0.5f};
  std::vector<std::int8_t> gated_codes(4);
  std::vector<float> gated_scales(1), gated_dequant(4);
  ok &= require(silu_mul_quant_int8(
                        gated_x.data(), gated_g.data(), gated_codes.data(),
                        gated_scales.data(), 1, 4,
                        false) == Status::kOk,
                "fused silu int8 quant");
  ok &= require(dequantize_int8(gated_codes.data(), gated_scales.data(),
                                gated_dequant.data(), 1, 4) == Status::kOk,
                "fused silu int8 dequant");
  ok &= require(close(gated_dequant[1], 0.0f) && gated_dequant[2] < 0.0f &&
                    gated_dequant[3] > 0.0f,
                "fused silu quant signs");
  std::vector<std::uint8_t> gated_fp8(4);
  ok &= require(silu_mul_quant_float8(
                        gated_x.data(), gated_g.data(), gated_fp8.data(),
                        gated_scales.data(), 1, 4, 4, true, false) ==
                        Status::kOk,
                "fused silu fp8 quant");
  std::vector<float> gated_fake(4);
  std::vector<std::int8_t> gated_fake_codes(4);
  float gated_fake_scale = 0.0f;
  ok &= require(silu_mul_fake_quant_int8(
                        gated_x.data(), gated_g.data(), gated_fake.data(),
                        gated_fake_codes.data(), &gated_fake_scale, 1, 4,
                        false) == Status::kOk,
                "fused silu fake quant");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(gated_fake[i],
                        gated_fake_scale * gated_fake_codes[i]),
                  "fused silu fake reconstruction");
  }
  std::vector<float> fp8_fake(4);
  std::vector<std::uint8_t> fp8_fake_codes(4);
  float fp8_fake_scale = 0.0f;
  ok &= require(fake_quant_float8(
                        gated_x.data(), fp8_fake.data(),
                        fp8_fake_codes.data(), &fp8_fake_scale, 4,
                        Float8Format::kE4M3FN) == Status::kOk,
                "fp8 fake quant");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(fp8_fake[i],
                        fp8_fake_scale * float8_decode(
                                             fp8_fake_codes[i],
                                             Float8Format::kE4M3FN)),
                  "fp8 fake reconstruction");
  }

  const float norm_x[] = {1, 2, 3, 4};
  const float norm_residual[] = {0.5f, -0.5f, 1.0f, -1.0f};
  const float norm_weight[] = {1.0f, 0.5f, 1.5f, 2.0f};
  const float norm_bias[] = {0.25f, -0.25f, 0.5f, -0.5f};
  float norm_reference[4], residual_reference[4];
  ok &= require(layer_norm_add(
                        norm_x, norm_residual, norm_weight, norm_bias,
                        norm_reference, residual_reference, 1, 4) ==
                        Status::kOk,
                "layer norm add reference");
  std::int8_t norm_i8[4];
  float norm_i8_scales[2], norm_residual_out[4], norm_dequant[4];
  ok &= require(layer_norm_add_quant_int8(
                        norm_x, norm_residual, norm_weight, norm_bias,
                        norm_i8, norm_residual_out, norm_i8_scales, 1, 4,
                        1e-5f, 2) == Status::kOk &&
                    dequantize_int8(norm_i8, norm_i8_scales, norm_dequant,
                                    1, 4, 2) == Status::kOk,
                "layer norm int8 quant epilogue");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(norm_residual_out[i], residual_reference[i]) &&
                      close(norm_dequant[i], norm_reference[i], 0.03f),
                  "layer norm int8 quant value");
  }
  std::uint8_t norm_fp8[4];
  float norm_fp8_scales[2];
  ok &= require(layer_norm_add_quant_float8(
                        norm_x, norm_residual, norm_weight, norm_bias,
                        norm_fp8, norm_residual_out, norm_fp8_scales, 1, 4,
                        1e-5f, 2, false, Float8Format::kE4M3FN) ==
                        Status::kOk &&
                    dequantize_float8(
                        norm_fp8, norm_fp8_scales, norm_dequant, 1, 4, 2,
                        Float8Format::kE4M3FN) == Status::kOk,
                "layer norm fp8 quant epilogue");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(norm_dequant[i], norm_reference[i], 0.08f),
                  "layer norm fp8 quant value");
  }
  ok &= require(rms_norm_add(
                        norm_x, norm_residual, norm_weight, norm_reference,
                        residual_reference, 1, 4) == Status::kOk &&
                    rms_norm_add_quant_float8(
                        norm_x, norm_residual, norm_weight, norm_fp8,
                        norm_residual_out, norm_fp8_scales, 1, 4, 1e-5f, 2,
                        false, Float8Format::kE4M3FN) == Status::kOk &&
                    dequantize_float8(
                        norm_fp8, norm_fp8_scales, norm_dequant, 1, 4, 2,
                        Float8Format::kE4M3FN) == Status::kOk,
                "RMS norm fp8 quant epilogue");
  for (int i = 0; i < 4; ++i) {
    ok &= require(close(norm_dequant[i], norm_reference[i], 0.08f),
                  "RMS norm fp8 quant value");
  }

  constexpr long long kHeadSize = 64;
  constexpr int kKeyBits = 4;
  constexpr int kValueBits = 4;
  long long key_bytes = 0, value_bytes = 0;
  ok &= require(turboquant_packed_bytes(kHeadSize, kKeyBits, &key_bytes) ==
                        Status::kOk &&
                    turboquant_packed_bytes(kHeadSize, kValueBits,
                                            &value_bytes) == Status::kOk,
                "turboquant packed sizes");
  std::vector<float> tq_key(kHeadSize), tq_value(kHeadSize), tq_sign(kHeadSize);
  for (long long i = 0; i < kHeadSize; ++i) {
    tq_key[i] = static_cast<float>(i - 31) / 16.0f;
    tq_value[i] = std::sin(static_cast<float>(i) * 0.2f);
    tq_sign[i] = (i & 1) ? -1.0f : 1.0f;
  }
  std::vector<float> centroids(1 << kValueBits);
  for (int i = 0; i < (1 << kValueBits); ++i) {
    centroids[i] = -2.5f + 5.0f * i / ((1 << kValueBits) - 1);
  }
  const int slot_mapping[] = {1};
  const int gather[] = {1};
  std::vector<std::uint8_t> tq_key_cache(2 * key_bytes);
  std::vector<std::uint8_t> tq_value_cache(2 * value_bytes);
  std::vector<float> tq_key_scale(4), tq_value_scale(4), tq_key_zero(4);
  std::vector<float> tq_key_out(kHeadSize), tq_value_out(kHeadSize);
  ok &= require(turboquant_encode(
                        tq_key.data(), tq_value.data(), slot_mapping,
                        centroids.data(), tq_sign.data(), tq_key_cache.data(),
                        tq_value_cache.data(), tq_key_scale.data(),
                        tq_value_scale.data(), tq_key_zero.data(), 1, 2, 1,
                        kHeadSize, kKeyBits, false, kValueBits) == Status::kOk,
                "turboquant encode");
  ok &= require(turboquant_decode(
                        tq_key_cache.data(), tq_value_cache.data(),
                        tq_key_scale.data(), tq_value_scale.data(),
                        tq_key_zero.data(), gather, centroids.data(),
                        tq_sign.data(), tq_key_out.data(), tq_value_out.data(),
                        2, 1, 1, kHeadSize, kKeyBits, false, kValueBits) ==
                        Status::kOk,
                "turboquant decode");
  float key_error = 0.0f, value_error = 0.0f;
  for (long long i = 0; i < kHeadSize; ++i) {
    key_error = std::max(key_error, std::fabs(tq_key[i] - tq_key_out[i]));
    value_error =
        std::max(value_error, std::fabs(tq_value[i] - tq_value_out[i]));
  }
  ok &= require(key_error < 0.2f && value_error < 0.6f,
                "turboquant round trip error");

  if (!ok) return 1;
  std::cout << "runtime and ternary quantization tests passed\n";
  return 0;
}
