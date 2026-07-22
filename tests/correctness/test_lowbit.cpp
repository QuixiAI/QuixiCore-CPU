#include "quixicore_cpu/lowbit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using quixicore_cpu::LowBitFormat;
using quixicore_cpu::Status;

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

bool close(float lhs, float rhs, float relative = 2e-4f) {
  return std::fabs(lhs - rhs) <=
         relative * std::max(1.0f, std::fabs(rhs));
}

std::uint32_t state = 0xC0FFEEu;

float random_value() {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (static_cast<float>(state & 0xffffu) / 32768.0f - 1.0f);
}

bool test_format(LowBitFormat format, long long dim, long long group_size) {
  constexpr long long kWeightRows = 5;
  constexpr long long kInputRows = 3;
  bool ok = true;
  std::size_t weight_bytes = 0;
  std::size_t scale_count = 0;
  ok &= require(quixicore_cpu::lowbit_packed_size(
                    format, kWeightRows, dim, group_size, &weight_bytes,
                    &scale_count) == Status::kOk,
                "low-bit packed size");
  std::vector<float> weights(static_cast<std::size_t>(kWeightRows * dim));
  std::vector<float> x(static_cast<std::size_t>(kInputRows * dim));
  for (float& value : weights) value = random_value() * 0.15f;
  for (float& value : x) value = random_value();
  weights[3] = 1.7f;
  weights[static_cast<std::size_t>(2 * dim + 5)] = -2.2f;

  std::vector<std::uint8_t> packed(weight_bytes);
  std::vector<float> scales(scale_count);
  std::vector<float> unpacked(weights.size());
  ok &= require(quixicore_cpu::lowbit_pack(
                    format, weights.data(), packed.data(), scales.data(),
                    kWeightRows, dim, group_size) == Status::kOk,
                "low-bit pack");
  ok &= require(quixicore_cpu::lowbit_unpack(
                    format, packed.data(), scales.data(), unpacked.data(),
                    kWeightRows, dim, group_size) == Status::kOk,
                "low-bit unpack");
  for (float value : unpacked) {
    ok &= require(std::isfinite(value), "low-bit unpack finite");
  }

  std::vector<float> output(
      static_cast<std::size_t>(kInputRows * kWeightRows));
  ok &= require(quixicore_cpu::lowbit_gemm(
                    format, packed.data(), scales.data(), x.data(),
                    output.data(), kInputRows, kWeightRows, dim,
                    group_size) == Status::kOk,
                "low-bit GEMM");
  for (long long row = 0; row < kInputRows; ++row) {
    for (long long output_index = 0; output_index < kWeightRows;
         ++output_index) {
      double expected = 0.0;
      for (long long input = 0; input < dim; ++input) {
        expected += double(x[row * dim + input]) *
                    unpacked[output_index * dim + input];
      }
      ok &= require(close(output[row * kWeightRows + output_index],
                          static_cast<float>(expected)),
                    "low-bit GEMM value");
    }
  }

  std::vector<float> accumulator(static_cast<std::size_t>(dim), 0.25f);
  ok &= require(quixicore_cpu::lowbit_axpy_row(
                    format, packed.data(), scales.data(), 2, -0.75f,
                    accumulator.data(), kWeightRows, dim,
                    group_size) == Status::kOk,
                "low-bit packed-row AXPY");
  for (long long input = 0; input < dim; ++input) {
    ok &= require(close(accumulator[input],
                        0.25f - 0.75f * unpacked[2 * dim + input], 1e-6f),
                  "low-bit packed-row AXPY value");
  }

  const int selected[] = {4, 1, 3};
  float selected_output[3] = {};
  ok &= require(quixicore_cpu::lowbit_gemv_rows(
                    format, packed.data(), scales.data(), x.data(), selected,
                    selected_output, 3, kWeightRows, dim,
                    group_size) == Status::kOk,
                "low-bit subset GEMV");
  for (int item = 0; item < 3; ++item) {
    double expected = 0.0;
    for (long long input = 0; input < dim; ++input) {
      expected += double(x[input]) * unpacked[selected[item] * dim + input];
    }
    ok &= require(close(selected_output[item], static_cast<float>(expected)),
                  "low-bit subset GEMV value");
  }

  if (format == LowBitFormat::kInt4Row ||
      format == LowBitFormat::kInt4Group) {
    std::vector<float> up_weights(weights.size());
    for (float& value : up_weights) value = random_value() * 0.2f;
    std::vector<std::uint8_t> up_packed(weight_bytes);
    std::vector<float> up_scales(scale_count);
    std::vector<float> up_unpacked(weights.size());
    std::vector<float> gate_output(output.size()), up_output(output.size());
    ok &= require(quixicore_cpu::lowbit_pack(
                      format, up_weights.data(), up_packed.data(),
                      up_scales.data(), kWeightRows, dim,
                      group_size) == Status::kOk &&
                      quixicore_cpu::lowbit_unpack(
                          format, up_packed.data(), up_scales.data(),
                          up_unpacked.data(), kWeightRows, dim,
                          group_size) == Status::kOk,
                  "low-bit pair setup");
    ok &= require(quixicore_cpu::lowbit_gemm_pair(
                      format, packed.data(), scales.data(), up_packed.data(),
                      up_scales.data(), x.data(), gate_output.data(),
                      up_output.data(), kInputRows, kWeightRows, dim,
                      group_size) == Status::kOk,
                  "low-bit fused pair");
    for (long long row = 0; row < kInputRows; ++row) {
      for (long long output_index = 0; output_index < kWeightRows;
           ++output_index) {
        double gate_expected = 0.0;
        double up_expected = 0.0;
        for (long long input = 0; input < dim; ++input) {
          gate_expected += double(x[row * dim + input]) *
                           unpacked[output_index * dim + input];
          up_expected += double(x[row * dim + input]) *
                         up_unpacked[output_index * dim + input];
        }
        ok &= require(close(gate_output[row * kWeightRows + output_index],
                            static_cast<float>(gate_expected)) &&
                          close(up_output[row * kWeightRows + output_index],
                                static_cast<float>(up_expected)),
                      "low-bit fused pair values");
      }
    }
  }
  return ok;
}

bool test_e8iq3() {
  constexpr long long kRows = 2;
  constexpr long long kDim = 300;
  bool ok = true;
  std::size_t bytes = 0;
  ok &= require(quixicore_cpu::e8iq3_packed_size(kRows, kDim, &bytes) ==
                    Status::kOk &&
                    bytes == 2u * 2u * 98u,
                "E8/IQ3 packed size");
  std::vector<std::uint8_t> packed(bytes, 0);
  // Grid zero has four doubled magnitudes of 4. code=0 and d=1 produce 0.5.
  for (long long row = 0; row < kRows; ++row) {
    for (int block = 0; block < 2; ++block) {
      const std::size_t offset =
          static_cast<std::size_t>((row * 2 + block) * 98 + 96);
      const std::uint16_t half = row == 0 ? 0x3c00u : 0x4000u;
      std::memcpy(packed.data() + offset, &half, sizeof(half));
    }
  }
  std::vector<float> unpacked(static_cast<std::size_t>(kRows * kDim));
  ok &= require(quixicore_cpu::e8iq3_unpack(
                    packed.data(), unpacked.data(), kRows, kDim) ==
                    Status::kOk,
                "E8/IQ3 unpack");
  for (long long input = 0; input < kDim; ++input) {
    ok &= require(unpacked[input] == 0.5f &&
                      unpacked[kDim + input] == 1.0f,
                  "E8/IQ3 known block decode");
  }
  std::vector<float> x(static_cast<std::size_t>(2 * kDim));
  for (long long input = 0; input < kDim; ++input) {
    x[input] = static_cast<float>((input % 11) - 5) * 0.1f;
    x[kDim + input] = static_cast<float>((input % 7) - 3) * 0.2f;
  }
  float output[4] = {};
  ok &= require(quixicore_cpu::e8iq3_gemm(
                    packed.data(), x.data(), output, 2, kRows, kDim) ==
                    Status::kOk,
                "E8/IQ3 GEMM");
  for (int row = 0; row < 2; ++row) {
    double sum = 0.0;
    for (long long input = 0; input < kDim; ++input) {
      sum += x[row * kDim + input];
    }
    ok &= require(close(output[row * 2], static_cast<float>(0.5 * sum),
                        1e-5f) &&
                      close(output[row * 2 + 1], static_cast<float>(sum),
                            1e-5f),
                  "E8/IQ3 GEMM values");
  }

  const float rotation_input[] = {1, 2, 3, 4, 5, 6};
  const float rotation_expected[] = {
      -0.70710677f, 2.12132049f, 3.0f, 5.0f, 4.0f, -6.0f};
  float rotated[6] = {};
  float restored[6] = {};
  ok &= require(quixicore_cpu::e8iq3_rotate(
                    rotation_input, rotated, 1, 6) == Status::kOk,
                "E8/IQ3 rotation");
  for (int i = 0; i < 6; ++i) {
    ok &= require(close(rotated[i], rotation_expected[i], 1e-6f),
                  "E8/IQ3 rotation oracle");
  }
  ok &= require(quixicore_cpu::e8iq3_rotate(
                    rotated, restored, 1, 6, true) == Status::kOk,
                "E8/IQ3 inverse rotation");
  for (int i = 0; i < 6; ++i) {
    ok &= require(close(restored[i], rotation_input[i], 1e-6f),
                  "E8/IQ3 rotation round trip");
  }
  return ok;
}

bool test_e8iq3_pack() {
  constexpr long long kRows = 2;
  constexpr long long kDim = 512;
  bool ok = true;
  std::vector<float> weights(static_cast<std::size_t>(kRows * kDim));
  for (long long i = 0; i < kRows * kDim; ++i) {
    weights[static_cast<std::size_t>(i)] =
        0.07f * std::sin(0.071f * static_cast<float>(i)) +
        0.03f * std::cos(0.113f * static_cast<float>(i));
  }
  std::size_t bytes = 0;
  ok &= require(quixicore_cpu::e8iq3_packed_size(kRows, kDim, &bytes) ==
                    Status::kOk &&
                    bytes == 4u * 98u,
                "E8/IQ3 encoder size");
  std::vector<std::uint8_t> packed(bytes), repeat(bytes);
  std::vector<float> unpacked(weights.size());
  ok &= require(quixicore_cpu::e8iq3_pack(
                    weights.data(), packed.data(), kRows, kDim) == Status::kOk &&
                    quixicore_cpu::e8iq3_pack(
                        weights.data(), repeat.data(), kRows, kDim) ==
                        Status::kOk &&
                    packed == repeat,
                "E8/IQ3 deterministic encode");
  ok &= require(quixicore_cpu::e8iq3_unpack(
                    packed.data(), unpacked.data(), kRows, kDim) == Status::kOk,
                "E8/IQ3 encoded decode");
  double error = 0.0;
  double signal = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const double delta = unpacked[i] - weights[i];
    error += delta * delta;
    signal += double(weights[i]) * weights[i];
  }
  const double relative_rmse = std::sqrt(error / signal);
  ok &= require(relative_rmse > 0.02 && relative_rmse < 0.30,
                "E8/IQ3 reconstruction quality");
  for (std::size_t group = 0; group < unpacked.size() / 8; ++group) {
    int negative = 0;
    for (int element = 0; element < 8; ++element) {
      negative += unpacked[group * 8 + element] < 0.0f ? 1 : 0;
    }
    ok &= require((negative & 1) == 0, "E8/IQ3 sign parity");
  }
  std::uint8_t ignored[196] = {};
  ok &= require(quixicore_cpu::e8iq3_pack(
                    weights.data(), ignored, 1, 300) == Status::kInvalidShape,
                "E8/IQ3 encoder full-block guard");

  // Frozen byte oracle from Colibri tools/iq3_pack.py for a simple f32 row.
  const char expected_hex[] =
      "1fd73fab9556d211d764ab9a56d211da64879a55fa11da9587d255d742c09556"
      "d21fd73fab9556d211d764ab9a56d211da64879a55fa42da9587d255d742c095"
      "8761382e789ee720c3709c279ec3302c713ccf238761382e789e6728e1389e274f19";
  std::vector<float> oracle_weights(256);
  for (int i = 0; i < 256; ++i) {
    oracle_weights[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 37 + 11) % 257 - 128) * 0.0007f;
  }
  std::uint8_t oracle_packed[98] = {};
  ok &= require(quixicore_cpu::e8iq3_pack(
                    oracle_weights.data(), oracle_packed, 1, 256) == Status::kOk,
                "E8/IQ3 frozen oracle encode");
  for (int i = 0; i < 98; ++i) {
    const auto nibble = [](char value) -> unsigned {
      return value >= '0' && value <= '9'
                 ? static_cast<unsigned>(value - '0')
                 : static_cast<unsigned>(value - 'a' + 10);
    };
    const std::uint8_t expected = static_cast<std::uint8_t>(
        (nibble(expected_hex[2 * i]) << 4) | nibble(expected_hex[2 * i + 1]));
    ok &= require(oracle_packed[i] == expected,
                  "E8/IQ3 Colibri byte oracle");
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_format(LowBitFormat::kInt2Row, 65, 0);
  ok &= test_format(LowBitFormat::kInt3Group64, 100, 0);
  ok &= test_format(LowBitFormat::kInt4Row, 65, 0);
  ok &= test_format(LowBitFormat::kInt4Group, 67, 16);
  ok &= test_e8iq3();
  ok &= test_e8iq3_pack();
  ok &= require(std::strcmp(
                    quixicore_cpu::lowbit_gemm_variant(
                        LowBitFormat::kInt4Row),
                    "unsupported") != 0,
                "low-bit variant is available");
  if (ok) {
    std::cout << "low-bit kernels ("
              << quixicore_cpu::lowbit_gemm_variant(
                     LowBitFormat::kInt4Row)
              << "): ok\n";
  }
  return ok ? 0 : 1;
}
