#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "quixicore_cpu/base_q.h"
#include "quixicore_cpu/quantization.h"
#include "quixicore_cpu/threading.h"

namespace {

using quixicore_cpu::BaseQScaleType;
using quixicore_cpu::BaseQTensorView;
using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

#define REQUIRE(expr)                                                     \
  do {                                                                    \
    if (!(expr)) {                                                        \
      std::cerr << "requirement failed: " #expr " at " << __FILE__ << ':' \
                << __LINE__ << '\n';                                      \
      return false;                                                       \
    }                                                                     \
  } while (false)

bool close(float a, float b, float tolerance = 2e-4f) {
  return std::fabs(a - b) <=
         tolerance * std::max({1.0f, std::fabs(a), std::fabs(b)});
}

struct Fixture {
  int bits = 0;
  long long rows = 3;
  long long columns = 128;
  int group_size = 32;
  BaseQScaleType scale_type = BaseQScaleType::kBF16;
  bool symmetric = false;
  std::vector<std::uint8_t> codes;
  std::vector<std::uint16_t> scales16;
  std::vector<std::uint16_t> biases16;
  std::vector<std::uint8_t> scales8;
  std::vector<std::uint8_t> biases8;
  std::vector<float> decoded;

  const void* scales() const {
    return scale_type == BaseQScaleType::kBF16 ||
                   scale_type == BaseQScaleType::kF16
               ? static_cast<const void*>(scales16.data())
               : static_cast<const void*>(scales8.data());
  }

  const void* biases() const {
    if (symmetric) return nullptr;
    return scale_type == BaseQScaleType::kBF16 ||
                   scale_type == BaseQScaleType::kF16
               ? static_cast<const void*>(biases16.data())
               : static_cast<const void*>(biases8.data());
  }

  BaseQTensorView view() const {
    const std::size_t groups =
        static_cast<std::size_t>(rows * (columns / group_size));
    return {codes.data(), codes.size(), scales(),
            groups,       biases(),     symmetric ? 0 : groups,
            rows,         columns,      bits,
            group_size,   scale_type,   symmetric};
  }
};

void insert_code(std::vector<std::uint8_t>& bytes, std::size_t row_offset,
                 long long column, int bits, std::uint32_t code) {
  const int bit = static_cast<int>(column * bits);
  const std::size_t byte = row_offset + static_cast<std::size_t>(bit >> 3);
  const int shift = bit & 7;
  const std::uint16_t word = static_cast<std::uint16_t>(code << shift);
  bytes[byte] |= static_cast<std::uint8_t>(word);
  if (shift + bits > 8) bytes[byte + 1] |= static_cast<std::uint8_t>(word >> 8);
}

float decoded_scale(const Fixture& fixture, std::size_t group, bool bias) {
  if (fixture.scale_type == BaseQScaleType::kBF16) {
    const auto& values = bias ? fixture.biases16 : fixture.scales16;
    return quixicore_cpu::bf16_to_float(values[group]);
  }
  if (fixture.scale_type == BaseQScaleType::kF16) {
    const auto& values = bias ? fixture.biases16 : fixture.scales16;
    return quixicore_cpu::f16_to_float(values[group]);
  }
  const auto& values = bias ? fixture.biases8 : fixture.scales8;
  if (fixture.scale_type == BaseQScaleType::kE8M0)
    return std::ldexp(1.0f, static_cast<int>(values[group]) - 127);
  return quixicore_cpu::float8_decode(values[group],
                                      quixicore_cpu::Float8Format::kE4M3FN);
}

Fixture make_fixture(int bits, bool symmetric = false,
                     BaseQScaleType scale_type = BaseQScaleType::kBF16,
                     int group_size = 32, long long rows = 3) {
  Fixture fixture;
  fixture.bits = bits;
  fixture.symmetric = symmetric;
  fixture.scale_type = scale_type;
  fixture.group_size = group_size;
  fixture.rows = rows;
  const std::size_t row_bytes =
      static_cast<std::size_t>(fixture.columns * bits / 8);
  fixture.codes.assign(static_cast<std::size_t>(fixture.rows) * row_bytes, 0);
  const std::size_t group_count = static_cast<std::size_t>(
      fixture.rows * (fixture.columns / fixture.group_size));
  if (scale_type == BaseQScaleType::kBF16 ||
      scale_type == BaseQScaleType::kF16) {
    fixture.scales16.resize(group_count);
    if (!symmetric) fixture.biases16.resize(group_count);
    for (std::size_t group = 0; group < group_count; ++group) {
      const float scale = 0.0625f * static_cast<float>((group % 5) + 1);
      const float bias = -0.25f + 0.03125f * static_cast<float>(group % 7);
      fixture.scales16[group] = scale_type == BaseQScaleType::kBF16
                                    ? quixicore_cpu::float_to_bf16(scale)
                                    : quixicore_cpu::float_to_f16(scale);
      if (!symmetric) {
        fixture.biases16[group] = scale_type == BaseQScaleType::kBF16
                                      ? quixicore_cpu::float_to_bf16(bias)
                                      : quixicore_cpu::float_to_f16(bias);
      }
    }
  } else {
    fixture.scales8.resize(group_count);
    if (!symmetric) fixture.biases8.resize(group_count);
    for (std::size_t group = 0; group < group_count; ++group) {
      if (scale_type == BaseQScaleType::kE8M0) {
        fixture.scales8[group] = static_cast<std::uint8_t>(124 + group % 4);
        if (!symmetric)
          fixture.biases8[group] = static_cast<std::uint8_t>(123 + group % 5);
      } else {
        constexpr std::uint8_t values[] = {0x28, 0x30, 0x38, 0x40};
        fixture.scales8[group] = values[group % 4];
        if (!symmetric) fixture.biases8[group] = values[(group + 1) % 4];
      }
    }
  }
  fixture.decoded.resize(
      static_cast<std::size_t>(fixture.rows * fixture.columns));
  const std::uint32_t mask = (1u << bits) - 1u;
  for (long long row = 0; row < fixture.rows; ++row) {
    for (long long column = 0; column < fixture.columns; ++column) {
      const std::uint32_t code =
          static_cast<std::uint32_t>((row * 19 + column * 5 + 3) & mask);
      insert_code(fixture.codes, static_cast<std::size_t>(row) * row_bytes,
                  column, bits, code);
      const std::size_t group = static_cast<std::size_t>(
          row * (fixture.columns / fixture.group_size) +
          column / fixture.group_size);
      const float scale = decoded_scale(fixture, group, false);
      fixture
          .decoded[static_cast<std::size_t>(row * fixture.columns + column)] =
          symmetric
              ? static_cast<float>(static_cast<int>(code) - (1 << (bits - 1))) *
                    scale
              : static_cast<float>(code) * scale +
                    decoded_scale(fixture, group, true);
    }
  }
  return fixture;
}

bool test_dequant_matrix() {
  for (int group_size : {32, 64, 128}) {
    for (int bits : {2, 3, 4, 5, 6, 8}) {
      for (BaseQScaleType scale_type :
           {BaseQScaleType::kBF16, BaseQScaleType::kF16,
            BaseQScaleType::kE8M0}) {
        for (bool symmetric : {false, true}) {
          Fixture fixture =
              make_fixture(bits, symmetric, scale_type, group_size);
          std::vector<float> output(fixture.decoded.size());
          REQUIRE(quixicore_cpu::base_q_dequant(
                      fixture.view(),
                      {output.data(), FloatStorageType::kF32,
                       static_cast<long long>(output.size())}) == Status::kOk);
          for (std::size_t index = 0; index < output.size(); ++index)
            REQUIRE(close(output[index], fixture.decoded[index], 1e-6f));
        }
      }
    }
  }
  for (int group_size : {32, 64, 128}) {
    for (bool symmetric : {false, true}) {
      Fixture fixture =
          make_fixture(8, symmetric, BaseQScaleType::kE4M3, group_size);
      std::vector<float> output(fixture.decoded.size());
      REQUIRE(quixicore_cpu::base_q_dequant(
                  fixture.view(), {output.data(), FloatStorageType::kF32,
                                   static_cast<long long>(output.size())}) ==
              Status::kOk);
      for (std::size_t index = 0; index < output.size(); ++index)
        REQUIRE(close(output[index], fixture.decoded[index], 1e-6f));
    }
  }

  Fixture typed_fixture = make_fixture(4);
  std::vector<std::uint16_t> output16(typed_fixture.decoded.size());
  for (FloatStorageType output_type :
       {FloatStorageType::kF16, FloatStorageType::kBF16}) {
    REQUIRE(quixicore_cpu::base_q_dequant(
                typed_fixture.view(),
                {output16.data(), output_type,
                 static_cast<long long>(output16.size())}) == Status::kOk);
    for (std::size_t index = 0; index < output16.size(); ++index) {
      const float got = output_type == FloatStorageType::kF16
                            ? quixicore_cpu::f16_to_float(output16[index])
                            : quixicore_cpu::bf16_to_float(output16[index]);
      const float expected =
          output_type == FloatStorageType::kF16
              ? quixicore_cpu::f16_to_float(
                    quixicore_cpu::float_to_f16(typed_fixture.decoded[index]))
              : quixicore_cpu::bf16_to_float(
                    quixicore_cpu::float_to_bf16(typed_fixture.decoded[index]));
      REQUIRE(got == expected);
    }
  }
  return true;
}

std::vector<float> dense_projection(const Fixture& fixture,
                                    const std::vector<float>& x, long long m) {
  std::vector<float> output(static_cast<std::size_t>(m * fixture.rows));
  for (long long input_row = 0; input_row < m; ++input_row) {
    for (long long weight_row = 0; weight_row < fixture.rows; ++weight_row) {
      float sums[4] = {};
      for (long long column = 0; column < fixture.columns; ++column) {
        sums[column & 3] +=
            fixture.decoded[static_cast<std::size_t>(
                weight_row * fixture.columns + column)] *
            x[static_cast<std::size_t>(input_row * fixture.columns + column)];
      }
      output[static_cast<std::size_t>(input_row * fixture.rows + weight_row)] =
          (sums[0] + sums[1]) + (sums[2] + sums[3]);
    }
  }
  return output;
}

bool test_projection_and_storage() {
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    Fixture fixture = make_fixture(bits);
    constexpr long long m = 7;
    std::vector<float> input(static_cast<std::size_t>(m * fixture.columns));
    std::vector<std::uint16_t> input16(input.size());
    for (FloatStorageType input_type :
         {FloatStorageType::kF16, FloatStorageType::kBF16}) {
      for (std::size_t index = 0; index < input.size(); ++index) {
        const float value =
            static_cast<float>(static_cast<int>(index % 23) - 11) / 13.0f;
        input16[index] = input_type == FloatStorageType::kF16
                             ? quixicore_cpu::float_to_f16(value)
                             : quixicore_cpu::float_to_bf16(value);
        input[index] = input_type == FloatStorageType::kF16
                           ? quixicore_cpu::f16_to_float(input16[index])
                           : quixicore_cpu::bf16_to_float(input16[index]);
      }
      const std::vector<float> expected = dense_projection(fixture, input, m);
      std::vector<float> output(expected.size());
      REQUIRE(
          quixicore_cpu::base_q_gemm(fixture.view(),
                                     {input16.data(), input_type,
                                      static_cast<long long>(input16.size())},
                                     {output.data(), FloatStorageType::kF32,
                                      static_cast<long long>(output.size())},
                                     m) == Status::kOk);
      for (std::size_t index = 0; index < output.size(); ++index)
        REQUIRE(close(output[index], expected[index]));

      for (FloatStorageType output_type :
           {FloatStorageType::kF16, FloatStorageType::kBF16}) {
        std::vector<std::uint16_t> typed(expected.size());
        REQUIRE(
            quixicore_cpu::base_q_gemm(fixture.view(),
                                       {input16.data(), input_type,
                                        static_cast<long long>(input16.size())},
                                       {typed.data(), output_type,
                                        static_cast<long long>(typed.size())},
                                       m) == Status::kOk);
        for (std::size_t index = 0; index < typed.size(); ++index) {
          const float got = output_type == FloatStorageType::kF16
                                ? quixicore_cpu::f16_to_float(typed[index])
                                : quixicore_cpu::bf16_to_float(typed[index]);
          REQUIRE(close(got, expected[index], 5e-3f));
        }
      }
    }
  }
  return true;
}

bool test_embedding_and_fusions() {
  Fixture fixture = make_fixture(5);
  const int ids[] = {2, -1, 99, 0};
  std::vector<float> embedded(4 * static_cast<std::size_t>(fixture.columns));
  REQUIRE(quixicore_cpu::base_q_embedding(
              fixture.view(), ids, 4,
              {embedded.data(), FloatStorageType::kF32,
               static_cast<long long>(embedded.size())}) == Status::kOk);
  for (long long token = 0; token < 4; ++token) {
    for (long long column = 0; column < fixture.columns; ++column) {
      const float expected = ids[token] < 0 || ids[token] >= fixture.rows
                                 ? 0.0f
                                 : fixture.decoded[static_cast<std::size_t>(
                                       ids[token] * fixture.columns + column)];
      REQUIRE(close(
          embedded[static_cast<std::size_t>(token * fixture.columns + column)],
          expected, 1e-6f));
    }
  }

  std::vector<float> input(static_cast<std::size_t>(fixture.columns));
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 9.0f;
  const std::vector<float> projected = dense_projection(fixture, input, 1);
  Fixture q_fixture = make_fixture(5, false, BaseQScaleType::kBF16, 32, 2);
  Fixture v_fixture = make_fixture(5, false, BaseQScaleType::kBF16, 32, 4);
  const std::vector<float> q_expected = dense_projection(q_fixture, input, 1);
  const std::vector<float> v_expected = dense_projection(v_fixture, input, 1);
  std::vector<float> q(q_fixture.rows), k(fixture.rows), v(v_fixture.rows);
  REQUIRE(quixicore_cpu::base_q_gemv_qkv(
              q_fixture.view(), fixture.view(), v_fixture.view(),
              {input.data(), FloatStorageType::kF32, fixture.columns},
              {q.data(), FloatStorageType::kF32, q_fixture.rows},
              {k.data(), FloatStorageType::kF32, fixture.rows},
              {v.data(), FloatStorageType::kF32, v_fixture.rows}) ==
          Status::kOk);
  for (long long row = 0; row < q_fixture.rows; ++row)
    REQUIRE(close(q[row], q_expected[row]));
  for (long long row = 0; row < fixture.rows; ++row)
    REQUIRE(close(k[row], projected[row]));
  for (long long row = 0; row < v_fixture.rows; ++row)
    REQUIRE(close(v[row], v_expected[row]));

  std::vector<float> swiglu(fixture.rows);
  REQUIRE(quixicore_cpu::base_q_gemv_swiglu(
              fixture.view(), fixture.view(),
              {input.data(), FloatStorageType::kF32, fixture.columns},
              {swiglu.data(), FloatStorageType::kF32, fixture.rows}) ==
          Status::kOk);
  for (long long row = 0; row < fixture.rows; ++row) {
    const float silu = projected[row] / (1.0f + std::exp(-projected[row]));
    REQUIRE(close(swiglu[row], silu * projected[row]));
  }
  return true;
}

bool test_lm_head_argmax() {
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (bool symmetric : {false, true}) {
      Fixture fixture =
          make_fixture(bits, symmetric, BaseQScaleType::kF16, 32, 19);
      constexpr long long batch = 3;
      for (FloatStorageType input_type :
           {FloatStorageType::kF32, FloatStorageType::kF16,
            FloatStorageType::kBF16}) {
        std::vector<float> input(
            static_cast<std::size_t>(batch * fixture.columns));
        std::vector<std::uint16_t> input16(input.size());
        for (std::size_t index = 0; index < input.size(); ++index) {
          const float source =
              static_cast<float>(static_cast<int>((index * 11 + 7) % 37) - 18) /
              29.0f;
          if (input_type == FloatStorageType::kF32) {
            input[index] = source;
          } else {
            input16[index] = input_type == FloatStorageType::kF16
                                 ? quixicore_cpu::float_to_f16(source)
                                 : quixicore_cpu::float_to_bf16(source);
            input[index] = input_type == FloatStorageType::kF16
                               ? quixicore_cpu::f16_to_float(input16[index])
                               : quixicore_cpu::bf16_to_float(input16[index]);
          }
        }
        const std::vector<float> logits =
            dense_projection(fixture, input, batch);
        std::vector<int> expected(static_cast<std::size_t>(batch));
        for (long long input_row = 0; input_row < batch; ++input_row) {
          int best = 0;
          const auto rounded_logit = [&](long long token) {
            const float value = logits[static_cast<std::size_t>(
                input_row * fixture.rows + token)];
            if (input_type == FloatStorageType::kF16) {
              return quixicore_cpu::f16_to_float(
                  quixicore_cpu::float_to_f16(value));
            }
            if (input_type == FloatStorageType::kBF16) {
              return quixicore_cpu::bf16_to_float(
                  quixicore_cpu::float_to_bf16(value));
            }
            return value;
          };
          float best_value = rounded_logit(0);
          for (long long token = 1; token < fixture.rows; ++token) {
            const float value = rounded_logit(token);
            if (value > best_value) {
              best_value = value;
              best = static_cast<int>(token);
            }
          }
          expected[static_cast<std::size_t>(input_row)] = best;
        }
        std::vector<int> actual(static_cast<std::size_t>(batch), -1);
        REQUIRE(quixicore_cpu::base_q_lm_head_argmax(
                    fixture.view(),
                    {input_type == FloatStorageType::kF32
                         ? static_cast<const void*>(input.data())
                         : static_cast<const void*>(input16.data()),
                     input_type, static_cast<long long>(input.size())},
                    actual.data(), batch) == Status::kOk);
        REQUIRE(actual == expected);
        if (bits == 4 && !symmetric && input_type == FloatStorageType::kF32) {
          quixicore_cpu::set_num_threads(8);
          std::fill(actual.begin(), actual.end(), -1);
          REQUIRE(quixicore_cpu::base_q_lm_head_argmax(
                      fixture.view(),
                      {input.data(), FloatStorageType::kF32,
                       static_cast<long long>(input.size())},
                      actual.data(), batch) == Status::kOk);
          quixicore_cpu::set_num_threads(1);
          REQUIRE(actual == expected);
        }
      }
    }
  }

  Fixture ties = make_fixture(4, false, BaseQScaleType::kF16, 32, 17);
  std::fill(ties.codes.begin(), ties.codes.end(), 0);
  std::fill(ties.scales16.begin(), ties.scales16.end(), 0);
  std::fill(ties.biases16.begin(), ties.biases16.end(), 0);
  std::vector<float> ones(static_cast<std::size_t>(2 * ties.columns), 1.0f);
  int tied_tokens[] = {-1, -1};
  REQUIRE(quixicore_cpu::base_q_lm_head_argmax(
              ties.view(),
              {ones.data(), FloatStorageType::kF32, 2 * ties.columns},
              tied_tokens, 2) == Status::kOk);
  REQUIRE(tied_tokens[0] == 0 && tied_tokens[1] == 0);
  return true;
}

bool test_grouped_expert_projection() {
  constexpr long long experts = 2;
  constexpr long long output_rows = 64;
  constexpr long long intermediate = 32;
  constexpr long long total_rows = 64;
  const int expert_of_tile[] = {1, 0};
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (bool symmetric : {false, true}) {
      Fixture fixture = make_fixture(bits, symmetric, BaseQScaleType::kF16, 32,
                                     experts * output_rows);
      std::vector<float> input(
          static_cast<std::size_t>(total_rows * fixture.columns));
      for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            static_cast<float>(static_cast<int>((index * 7 + 3) % 29) - 14) /
            31.0f;
      }
      std::vector<float> rect(
          static_cast<std::size_t>(total_rows * output_rows));
      std::vector<float> swiglu(
          static_cast<std::size_t>(total_rows * intermediate));
      REQUIRE(quixicore_cpu::base_q_moe_gemm(
                  fixture.view(), experts,
                  {input.data(), FloatStorageType::kF32,
                   static_cast<long long>(input.size())},
                  expert_of_tile, total_rows,
                  {rect.data(), FloatStorageType::kF32,
                   static_cast<long long>(rect.size())}) == Status::kOk);
      REQUIRE(quixicore_cpu::base_q_moe_swiglu(
                  fixture.view(), experts,
                  {input.data(), FloatStorageType::kF32,
                   static_cast<long long>(input.size())},
                  expert_of_tile, total_rows,
                  {swiglu.data(), FloatStorageType::kF32,
                   static_cast<long long>(swiglu.size())}) == Status::kOk);
      for (long long row = 0; row < total_rows; ++row) {
        const long long expert = expert_of_tile[row / 32];
        for (long long output_row = 0; output_row < output_rows; ++output_row) {
          float sums[4] = {};
          const long long weight_row = expert * output_rows + output_row;
          for (long long column = 0; column < fixture.columns; ++column) {
            sums[column & 3] += input[static_cast<std::size_t>(
                                    row * fixture.columns + column)] *
                                fixture.decoded[static_cast<std::size_t>(
                                    weight_row * fixture.columns + column)];
          }
          const float expected = (sums[0] + sums[1]) + (sums[2] + sums[3]);
          REQUIRE(close(
              rect[static_cast<std::size_t>(row * output_rows + output_row)],
              expected));
        }
        for (long long output_row = 0; output_row < intermediate;
             ++output_row) {
          const float gate =
              rect[static_cast<std::size_t>(row * output_rows + output_row)];
          const float up = rect[static_cast<std::size_t>(
              row * output_rows + intermediate + output_row)];
          const float expected = gate / (1.0f + std::exp(-gate)) * up;
          REQUIRE(close(
              swiglu[static_cast<std::size_t>(row * intermediate + output_row)],
              expected));
        }
      }
    }
  }

  Fixture typed =
      make_fixture(4, false, BaseQScaleType::kF16, 32, experts * output_rows);
  std::vector<std::uint16_t> input16(
      static_cast<std::size_t>(total_rows * typed.columns));
  std::vector<std::uint16_t> output16(
      static_cast<std::size_t>(total_rows * output_rows));
  std::vector<std::uint16_t> swiglu16(
      static_cast<std::size_t>(total_rows * intermediate));
  for (FloatStorageType type :
       {FloatStorageType::kF16, FloatStorageType::kBF16}) {
    for (std::size_t index = 0; index < input16.size(); ++index) {
      const float value =
          static_cast<float>(static_cast<int>(index % 23) - 11) / 19.0f;
      input16[index] = type == FloatStorageType::kF16
                           ? quixicore_cpu::float_to_f16(value)
                           : quixicore_cpu::float_to_bf16(value);
    }
    REQUIRE(quixicore_cpu::base_q_moe_gemm(
                typed.view(), experts,
                {input16.data(), type, static_cast<long long>(input16.size())},
                expert_of_tile, total_rows,
                {output16.data(), type,
                 static_cast<long long>(output16.size())}) == Status::kOk);
    REQUIRE(quixicore_cpu::base_q_moe_swiglu(
                typed.view(), experts,
                {input16.data(), type, static_cast<long long>(input16.size())},
                expert_of_tile, total_rows,
                {swiglu16.data(), type,
                 static_cast<long long>(swiglu16.size())}) == Status::kOk);
    for (long long row : {0LL, 31LL, 32LL, 63LL}) {
      const long long expert = expert_of_tile[row / 32];
      for (long long output_row : {0LL, 17LL, 63LL}) {
        float sums[4] = {};
        const long long weight_row = expert * output_rows + output_row;
        for (long long column = 0; column < typed.columns; ++column) {
          const std::size_t input_index =
              static_cast<std::size_t>(row * typed.columns + column);
          const float activation =
              type == FloatStorageType::kF16
                  ? quixicore_cpu::f16_to_float(input16[input_index])
                  : quixicore_cpu::bf16_to_float(input16[input_index]);
          sums[column & 3] +=
              activation * typed.decoded[static_cast<std::size_t>(
                               weight_row * typed.columns + column)];
        }
        const float value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        const float expected = type == FloatStorageType::kF16
                                   ? quixicore_cpu::f16_to_float(
                                         quixicore_cpu::float_to_f16(value))
                                   : quixicore_cpu::bf16_to_float(
                                         quixicore_cpu::float_to_bf16(value));
        const std::size_t output_index =
            static_cast<std::size_t>(row * output_rows + output_row);
        const float actual =
            type == FloatStorageType::kF16
                ? quixicore_cpu::f16_to_float(output16[output_index])
                : quixicore_cpu::bf16_to_float(output16[output_index]);
        REQUIRE(actual == expected);
      }
      for (long long output_row : {0LL, 17LL, 31LL}) {
        float gate_sums[4] = {};
        float up_sums[4] = {};
        const long long gate_row = expert * output_rows + output_row;
        const long long up_row = gate_row + intermediate;
        for (long long column = 0; column < typed.columns; ++column) {
          const std::size_t input_index =
              static_cast<std::size_t>(row * typed.columns + column);
          const float activation =
              type == FloatStorageType::kF16
                  ? quixicore_cpu::f16_to_float(input16[input_index])
                  : quixicore_cpu::bf16_to_float(input16[input_index]);
          gate_sums[column & 3] +=
              activation * typed.decoded[static_cast<std::size_t>(
                               gate_row * typed.columns + column)];
          up_sums[column & 3] +=
              activation * typed.decoded[static_cast<std::size_t>(
                               up_row * typed.columns + column)];
        }
        const float gate =
            (gate_sums[0] + gate_sums[1]) + (gate_sums[2] + gate_sums[3]);
        const float up = (up_sums[0] + up_sums[1]) + (up_sums[2] + up_sums[3]);
        const float value = gate / (1.0f + std::exp(-gate)) * up;
        const float expected = type == FloatStorageType::kF16
                                   ? quixicore_cpu::f16_to_float(
                                         quixicore_cpu::float_to_f16(value))
                                   : quixicore_cpu::bf16_to_float(
                                         quixicore_cpu::float_to_bf16(value));
        const std::size_t output_index =
            static_cast<std::size_t>(row * intermediate + output_row);
        const float actual =
            type == FloatStorageType::kF16
                ? quixicore_cpu::f16_to_float(swiglu16[output_index])
                : quixicore_cpu::bf16_to_float(swiglu16[output_index]);
        REQUIRE(actual == expected);
      }
    }
  }
  quixicore_cpu::set_num_threads(8);
  REQUIRE(quixicore_cpu::base_q_moe_gemm(
              typed.view(), experts,
              {input16.data(), FloatStorageType::kBF16,
               static_cast<long long>(input16.size())},
              expert_of_tile, total_rows,
              {output16.data(), FloatStorageType::kBF16,
               static_cast<long long>(output16.size())}) == Status::kOk);
  quixicore_cpu::set_num_threads(1);
  return true;
}

bool test_validation() {
  Fixture fixture = make_fixture(4);
  std::vector<float> output(fixture.decoded.size());
  BaseQTensorView invalid = fixture.view();
  invalid.code_bytes -= 1;
  REQUIRE(quixicore_cpu::base_q_dequant(
              invalid, {output.data(), FloatStorageType::kF32,
                        static_cast<long long>(output.size())}) ==
          Status::kInvalidArgument);
  invalid = fixture.view();
  invalid.biases = nullptr;
  invalid.bias_count = 0;
  REQUIRE(quixicore_cpu::base_q_dequant(
              invalid, {output.data(), FloatStorageType::kF32,
                        static_cast<long long>(output.size())}) ==
          Status::kInvalidArgument);
  invalid = fixture.view();
  invalid.scale_type = BaseQScaleType::kE4M3;
  REQUIRE(quixicore_cpu::base_q_dequant(
              invalid, {output.data(), FloatStorageType::kF32,
                        static_cast<long long>(output.size())}) ==
          Status::kUnsupportedFormat);
  invalid = fixture.view();
  invalid.scale_type = static_cast<BaseQScaleType>(99);
  REQUIRE(quixicore_cpu::base_q_dequant(
              invalid, {output.data(), FloatStorageType::kF32,
                        static_cast<long long>(output.size())}) ==
          Status::kUnsupportedFormat);

  Fixture other = make_fixture(5);
  std::vector<float> input(static_cast<std::size_t>(fixture.columns), 1.0f);
  std::vector<float> q(fixture.rows, 123.0f);
  std::vector<float> k(fixture.rows, 123.0f);
  std::vector<float> v(fixture.rows, 123.0f);
  REQUIRE(quixicore_cpu::base_q_gemv_qkv(
              fixture.view(), other.view(), fixture.view(),
              {input.data(), FloatStorageType::kF32, fixture.columns},
              {q.data(), FloatStorageType::kF32, fixture.rows},
              {k.data(), FloatStorageType::kF32, fixture.rows},
              {v.data(), FloatStorageType::kF32, fixture.rows}) ==
          Status::kInvalidShape);
  REQUIRE(std::all_of(q.begin(), q.end(),
                      [](float value) { return value == 123.0f; }));
  int token = 123;
  REQUIRE(quixicore_cpu::base_q_lm_head_argmax(
              fixture.view(),
              {input.data(), FloatStorageType::kF32, fixture.columns - 1},
              &token, 1) == Status::kInvalidShape);
  REQUIRE(token == 123);
  REQUIRE(quixicore_cpu::base_q_lm_head_argmax(
              fixture.view(),
              {input.data(), FloatStorageType::kF32, fixture.columns}, nullptr,
              1) == Status::kInvalidArgument);
  Fixture expert_weights =
      make_fixture(4, false, BaseQScaleType::kF16, 32, 2 * 64);
  std::vector<float> expert_input(64 * expert_weights.columns, 1.0f);
  std::vector<float> expert_output(64 * 64, 123.0f);
  const int invalid_experts[] = {0, 2};
  REQUIRE(quixicore_cpu::base_q_moe_gemm(
              expert_weights.view(), 2,
              {expert_input.data(), FloatStorageType::kF32,
               static_cast<long long>(expert_input.size())},
              invalid_experts, 64,
              {expert_output.data(), FloatStorageType::kF32,
               static_cast<long long>(expert_output.size())}) ==
          Status::kInvalidArgument);
  REQUIRE(std::all_of(expert_output.begin(), expert_output.end(),
                      [](float value) { return value == 123.0f; }));
  return true;
}

}  // namespace

int main() {
  if (!test_dequant_matrix()) return 1;
  if (!test_projection_and_storage()) return 1;
  if (!test_embedding_and_fusions()) return 1;
  if (!test_lm_head_argmax()) return 1;
  if (!test_grouped_expert_projection()) return 1;
  if (!test_validation()) return 1;
  std::cout << "base_q: ok\n";
  return 0;
}
