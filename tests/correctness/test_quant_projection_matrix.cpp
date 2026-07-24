#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quant_import.h"

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ':' << __LINE__ \
                << '\n';                                                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::CpuPackedWeights;
using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::LinearActivation;
using quixicore_cpu::Status;

struct LayoutCase {
  CanonicalQuantLayout layout;
  long long group_size;
  const char* name;
};

constexpr LayoutCase kLayouts[] = {
    {CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
    {CanonicalQuantLayout::kUInt4Affine, 32, "u4"},
    {CanonicalQuantLayout::kInt8Symmetric, 16, "int8"},
    {CanonicalQuantLayout::kInt8Affine, 32, "int8_affine"},
    {CanonicalQuantLayout::kFP8E4M3FN, 16, "fp8_e4m3"},
    {CanonicalQuantLayout::kFP8E5M2, 16, "fp8_e5m2"},
    {CanonicalQuantLayout::kFP4E2M1, 32, "fp4"},
    {CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, "mxfp8"},
    {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
    {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, "nvfp4"},
    {CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
};

bool close(float actual, float expected, float tolerance = 4e-5f) {
  return std::fabs(actual - expected) <=
         tolerance * (1.0f + std::fabs(expected));
}

float apply_activation(float value, LinearActivation activation) {
  switch (activation) {
    case LinearActivation::kNone:
      return value;
    case LinearActivation::kGeluErf:
      return 0.5f * value * (1.0f + std::erf(value * 0.7071067811865475f));
    case LinearActivation::kGeluTanh:
      return 0.5f * value *
             (1.0f + std::tanh(0.7978845608028654f *
                               (value + 0.044715f * value * value * value)));
    case LinearActivation::kSilu:
      return value / (1.0f + std::exp(-value));
    case LinearActivation::kRelu2:
      return value > 0.0f ? value * value : 0.0f;
  }
  return value;
}

std::vector<float> make_values(long long rows, long long columns, int salt) {
  std::vector<float> values(static_cast<std::size_t>(rows * columns));
  for (long long row = 0; row < rows; ++row) {
    for (long long column = 0; column < columns; ++column) {
      const int code =
          static_cast<int>((row * 29 + column * 17 + salt * 11) % 61);
      values[static_cast<std::size_t>(row * columns + column)] =
          static_cast<float>(code - 30) / 13.0f +
          0.03125f * static_cast<float>((row + salt) % 5);
    }
  }
  return values;
}

Status pack_layout(const LayoutCase& test, const std::vector<float>& values,
                   long long rows, long long columns,
                   CanonicalQuantTensor* tensor) {
  return quixicore_cpu::quantize_canonical(
      {values.data(), FloatStorageType::kF32,
       static_cast<long long>(values.size())},
      rows, columns, test.layout, test.group_size, tensor,
      test.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3);
}

std::vector<float> decode(const CanonicalQuantTensor& tensor) {
  std::vector<float> values(static_cast<std::size_t>(
      tensor.metadata.logical_rows * tensor.metadata.logical_columns));
  if (quixicore_cpu::dequantize_canonical(
          tensor, values.data(), static_cast<long long>(values.size())) !=
      Status::kOk) {
    return {};
  }
  return values;
}

std::vector<float> oracle(const std::vector<float>& weights,
                          const std::vector<float>& activations, long long m,
                          long long n, long long k,
                          const std::vector<int>* order = nullptr) {
  std::vector<float> output(static_cast<std::size_t>(m * n));
  for (long long row = 0; row < m; ++row) {
    for (long long output_column = 0; output_column < n; ++output_column) {
      float sum = 0.0f;
      for (long long packed_column = 0; packed_column < k; ++packed_column) {
        const long long logical_column =
            order == nullptr ? packed_column : (*order)[packed_column];
        sum += weights[static_cast<std::size_t>(output_column * k +
                                                packed_column)] *
               activations[static_cast<std::size_t>(row * k + logical_column)];
      }
      output[static_cast<std::size_t>(row * n + output_column)] = sum;
    }
  }
  return output;
}

std::vector<std::uint16_t> encode_storage(const std::vector<float>& values,
                                          FloatStorageType type) {
  std::vector<std::uint16_t> encoded(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    encoded[index] = type == FloatStorageType::kF16
                         ? quixicore_cpu::float_to_f16(values[index])
                         : quixicore_cpu::float_to_bf16(values[index]);
  }
  return encoded;
}

std::vector<float> decode_storage(const std::vector<std::uint16_t>& values,
                                  FloatStorageType type) {
  std::vector<float> decoded(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    decoded[index] = type == FloatStorageType::kF16
                         ? quixicore_cpu::f16_to_float(values[index])
                         : quixicore_cpu::bf16_to_float(values[index]);
  }
  return decoded;
}

bool test_weight_only_matrix() {
  constexpr long long n = 5;
  constexpr long long k = 32;
  const std::array<long long, 3> m_values = {1, 16, 128};
  const std::vector<float> weight_source = make_values(n, k, 1);
  for (const LayoutCase& test : kLayouts) {
    CanonicalQuantTensor tensor;
    REQUIRE(pack_layout(test, weight_source, n, k, &tensor) == Status::kOk);
    const std::vector<float> weights = decode(tensor);
    REQUIRE(weights.size() == static_cast<std::size_t>(n * k));
    CpuPackedWeights prepared;
    REQUIRE(prepared.prepare(tensor) == Status::kOk);
    REQUIRE(prepared.info().quant_metadata.layout == test.layout);
    REQUIRE(prepared.info().quant_metadata.group_size ==
            tensor.metadata.group_size);
    const std::vector<float> bias = make_values(1, n, 7);

    for (long long m : m_values) {
      const std::vector<float> source = make_values(m, k, 3);
      for (FloatStorageType type :
           {FloatStorageType::kF32, FloatStorageType::kF16,
            FloatStorageType::kBF16}) {
        std::vector<std::uint16_t> encoded;
        std::vector<float> exact = source;
        const void* input = source.data();
        if (type != FloatStorageType::kF32) {
          encoded = encode_storage(source, type);
          exact = decode_storage(encoded, type);
          input = encoded.data();
        }
        const std::vector<float> expected = oracle(weights, exact, m, n, k);
        std::vector<float> actual(expected.size());
        REQUIRE(quixicore_cpu::qgemm_prepacked_storage(
                    prepared,
                    {input, type, static_cast<long long>(exact.size())},
                    {actual.data(), FloatStorageType::kF32,
                     static_cast<long long>(actual.size())},
                    m) == Status::kOk);
        for (std::size_t index = 0; index < actual.size(); ++index) {
          if (!close(actual[index], expected[index])) {
            std::cerr << test.name << " m=" << m
                      << " storage=" << static_cast<int>(type)
                      << " index=" << index << " actual=" << actual[index]
                      << " expected=" << expected[index] << '\n';
            return false;
          }
        }
        const std::vector<float> projection_actual = actual;

        for (LinearActivation activation :
             {LinearActivation::kNone, LinearActivation::kGeluErf,
              LinearActivation::kGeluTanh, LinearActivation::kSilu,
              LinearActivation::kRelu2}) {
          std::fill(actual.begin(), actual.end(), 0.0f);
          REQUIRE(quixicore_cpu::qgemm_prepacked_epilogue_storage(
                      prepared,
                      {input, type, static_cast<long long>(exact.size())},
                      bias.data(),
                      {actual.data(), FloatStorageType::kF32,
                       static_cast<long long>(actual.size())},
                      m, activation) == Status::kOk);
          for (std::size_t index = 0; index < actual.size(); ++index) {
            const float epilogue_expected = apply_activation(
                expected[index] + bias[index % static_cast<std::size_t>(n)],
                activation);
            REQUIRE(close(actual[index], epilogue_expected, 2e-4f));
          }
        }

        if (type != FloatStorageType::kF32) {
          std::vector<std::uint16_t> rounded(expected.size());
          REQUIRE(quixicore_cpu::qgemm_prepacked_storage(
                      prepared,
                      {input, type, static_cast<long long>(exact.size())},
                      {rounded.data(), type,
                       static_cast<long long>(rounded.size())},
                      m) == Status::kOk);
          for (std::size_t index = 0; index < rounded.size(); ++index) {
            const std::uint16_t expected_bits =
                type == FloatStorageType::kF16
                    ? quixicore_cpu::float_to_f16(projection_actual[index])
                    : quixicore_cpu::float_to_bf16(projection_actual[index]);
            REQUIRE(rounded[index] == expected_bits);
          }
          REQUIRE(quixicore_cpu::qgemm_prepacked_epilogue_storage(
                      prepared,
                      {input, type, static_cast<long long>(exact.size())},
                      bias.data(),
                      {actual.data(), FloatStorageType::kF32,
                       static_cast<long long>(actual.size())},
                      m, LinearActivation::kSilu) == Status::kOk);
          REQUIRE(quixicore_cpu::qgemm_prepacked_epilogue_storage(
                      prepared,
                      {input, type, static_cast<long long>(exact.size())},
                      bias.data(),
                      {rounded.data(), type,
                       static_cast<long long>(rounded.size())},
                      m, LinearActivation::kSilu) == Status::kOk);
          for (std::size_t index = 0; index < rounded.size(); ++index) {
            const std::uint16_t expected_bits =
                type == FloatStorageType::kF16
                    ? quixicore_cpu::float_to_f16(actual[index])
                    : quixicore_cpu::float_to_bf16(actual[index]);
            REQUIRE(rounded[index] == expected_bits);
          }
        }
      }

      std::vector<float> expected = oracle(weights, source, m, n, k);
      std::vector<float> actual(expected.size());
      REQUIRE(quixicore_cpu::qgemm_prepacked(prepared, source.data(),
                                             actual.data(), m) == Status::kOk);
      for (std::size_t index = 0; index < actual.size(); ++index)
        REQUIRE(close(actual[index], expected[index]));
      if (m == 1) {
        std::fill(actual.begin(), actual.end(), 0.0f);
        REQUIRE(quixicore_cpu::qgemv_prepacked(prepared, source.data(),
                                               actual.data()) == Status::kOk);
        for (std::size_t index = 0; index < actual.size(); ++index)
          REQUIRE(close(actual[index], expected[index]));
      }
    }
  }
  return true;
}

bool test_gate_up_matrix() {
  constexpr long long n = 5;
  constexpr long long k = 32;
  for (const LayoutCase& test : kLayouts) {
    CanonicalQuantTensor gate_tensor;
    CanonicalQuantTensor up_tensor;
    REQUIRE(pack_layout(test, make_values(n, k, 19), n, k, &gate_tensor) ==
            Status::kOk);
    REQUIRE(pack_layout(test, make_values(n, k, 23), n, k, &up_tensor) ==
            Status::kOk);
    CpuPackedWeights gate_prepared;
    CpuPackedWeights up_prepared;
    REQUIRE(gate_prepared.prepare(gate_tensor) == Status::kOk);
    REQUIRE(up_prepared.prepare(up_tensor) == Status::kOk);
    const std::vector<float> gate_weights = decode(gate_tensor);
    const std::vector<float> up_weights = decode(up_tensor);

    for (long long m : {1LL, 16LL, 128LL}) {
      const std::vector<float> source = make_values(m, k, 29);
      for (FloatStorageType input_type :
           {FloatStorageType::kF32, FloatStorageType::kF16,
            FloatStorageType::kBF16}) {
        const void* input = source.data();
        std::vector<std::uint16_t> encoded;
        std::vector<float> exact = source;
        if (input_type != FloatStorageType::kF32) {
          encoded = encode_storage(source, input_type);
          exact = decode_storage(encoded, input_type);
          input = encoded.data();
        }
        const std::vector<float> expected_gate =
            oracle(gate_weights, exact, m, n, k);
        const std::vector<float> expected_up =
            oracle(up_weights, exact, m, n, k);
        std::vector<float> gate(expected_gate.size());
        std::vector<float> up(expected_up.size());
        REQUIRE(quixicore_cpu::qgemm_prepacked_gate_up_storage(
                    gate_prepared, up_prepared, {input, input_type, m * k},
                    {gate.data(), FloatStorageType::kF32, m * n},
                    {up.data(), FloatStorageType::kF32, m * n},
                    m) == Status::kOk);
        for (std::size_t index = 0; index < gate.size(); ++index) {
          if (!close(gate[index], expected_gate[index]) ||
              !close(up[index], expected_up[index])) {
            std::cerr << "gate/up " << test.name << " m=" << m
                      << " storage=" << static_cast<int>(input_type)
                      << " index=" << index << '\n';
            return false;
          }
        }
        std::vector<float> swiglu(gate.size());
        REQUIRE(quixicore_cpu::qgemm_prepacked_swiglu_storage(
                    gate_prepared, up_prepared, {input, input_type, m * k},
                    {swiglu.data(), FloatStorageType::kF32, m * n},
                    m) == Status::kOk);
        for (std::size_t index = 0; index < swiglu.size(); ++index) {
          const float silu = gate[index] / (1.0f + std::exp(-gate[index]));
          REQUIRE(close(swiglu[index], silu * up[index], 2e-4f));
        }
        if (input_type != FloatStorageType::kF32) {
          std::vector<std::uint16_t> gate_typed(gate.size());
          std::vector<std::uint16_t> up_typed(up.size());
          REQUIRE(quixicore_cpu::qgemm_prepacked_gate_up_storage(
                      gate_prepared, up_prepared, {input, input_type, m * k},
                      {gate_typed.data(), input_type, m * n},
                      {up_typed.data(), input_type, m * n}, m) == Status::kOk);
          for (std::size_t index = 0; index < gate.size(); ++index) {
            const std::uint16_t expected_gate_bits =
                input_type == FloatStorageType::kF16
                    ? quixicore_cpu::float_to_f16(gate[index])
                    : quixicore_cpu::float_to_bf16(gate[index]);
            const std::uint16_t expected_up_bits =
                input_type == FloatStorageType::kF16
                    ? quixicore_cpu::float_to_f16(up[index])
                    : quixicore_cpu::float_to_bf16(up[index]);
            REQUIRE(gate_typed[index] == expected_gate_bits);
            REQUIRE(up_typed[index] == expected_up_bits);
          }
          std::vector<std::uint16_t> swiglu_typed(swiglu.size());
          REQUIRE(quixicore_cpu::qgemm_prepacked_swiglu_storage(
                      gate_prepared, up_prepared, {input, input_type, m * k},
                      {swiglu_typed.data(), input_type, m * n},
                      m) == Status::kOk);
          for (std::size_t index = 0; index < swiglu.size(); ++index) {
            const std::uint16_t expected_bits =
                input_type == FloatStorageType::kF16
                    ? quixicore_cpu::float_to_f16(swiglu[index])
                    : quixicore_cpu::float_to_bf16(swiglu[index]);
            REQUIRE(swiglu_typed[index] == expected_bits);
          }
        }
      }
    }
  }
  return true;
}

bool test_qkv_matrix() {
  constexpr long long nq = 7;
  constexpr long long nk = 3;
  constexpr long long nv = 5;
  constexpr long long k = 32;
  for (const LayoutCase& test : kLayouts) {
    CanonicalQuantTensor q_tensor;
    CanonicalQuantTensor k_tensor;
    CanonicalQuantTensor v_tensor;
    REQUIRE(pack_layout(test, make_values(nq, k, 47), nq, k, &q_tensor) ==
            Status::kOk);
    REQUIRE(pack_layout(test, make_values(nk, k, 53), nk, k, &k_tensor) ==
            Status::kOk);
    REQUIRE(pack_layout(test, make_values(nv, k, 59), nv, k, &v_tensor) ==
            Status::kOk);
    CpuPackedWeights q_prepared;
    CpuPackedWeights k_prepared;
    CpuPackedWeights v_prepared;
    REQUIRE(q_prepared.prepare(q_tensor) == Status::kOk);
    REQUIRE(k_prepared.prepare(k_tensor) == Status::kOk);
    REQUIRE(v_prepared.prepare(v_tensor) == Status::kOk);
    const std::vector<float> q_weights = decode(q_tensor);
    const std::vector<float> k_weights = decode(k_tensor);
    const std::vector<float> v_weights = decode(v_tensor);

    for (long long m : {1LL, 16LL, 128LL}) {
      const std::vector<float> source = make_values(m, k, 61);
      for (FloatStorageType input_type :
           {FloatStorageType::kF32, FloatStorageType::kF16,
            FloatStorageType::kBF16}) {
        const void* input = source.data();
        std::vector<std::uint16_t> encoded;
        std::vector<float> exact = source;
        if (input_type != FloatStorageType::kF32) {
          encoded = encode_storage(source, input_type);
          exact = decode_storage(encoded, input_type);
          input = encoded.data();
        }
        const std::vector<float> expected_q =
            oracle(q_weights, exact, m, nq, k);
        const std::vector<float> expected_k =
            oracle(k_weights, exact, m, nk, k);
        const std::vector<float> expected_v =
            oracle(v_weights, exact, m, nv, k);
        std::vector<float> q(expected_q.size());
        std::vector<float> key(expected_k.size());
        std::vector<float> value(expected_v.size());
        REQUIRE(quixicore_cpu::qgemm_prepacked_qkv_storage(
                    q_prepared, k_prepared, v_prepared,
                    {input, input_type, m * k},
                    {q.data(), FloatStorageType::kF32, m * nq},
                    {key.data(), FloatStorageType::kF32, m * nk},
                    {value.data(), FloatStorageType::kF32, m * nv},
                    m) == Status::kOk);
        for (std::size_t index = 0; index < q.size(); ++index)
          REQUIRE(close(q[index], expected_q[index]));
        for (std::size_t index = 0; index < key.size(); ++index)
          REQUIRE(close(key[index], expected_k[index]));
        for (std::size_t index = 0; index < value.size(); ++index)
          REQUIRE(close(value[index], expected_v[index]));

        std::vector<std::uint16_t> q_f16(q.size());
        std::vector<std::uint16_t> k_bf16(key.size());
        std::vector<std::uint16_t> v_f16(value.size());
        REQUIRE(quixicore_cpu::qgemm_prepacked_qkv_storage(
                    q_prepared, k_prepared, v_prepared,
                    {input, input_type, m * k},
                    {q_f16.data(), FloatStorageType::kF16, m * nq},
                    {k_bf16.data(), FloatStorageType::kBF16, m * nk},
                    {v_f16.data(), FloatStorageType::kF16, m * nv},
                    m) == Status::kOk);
        for (std::size_t index = 0; index < q.size(); ++index)
          REQUIRE(q_f16[index] == quixicore_cpu::float_to_f16(q[index]));
        for (std::size_t index = 0; index < key.size(); ++index)
          REQUIRE(k_bf16[index] == quixicore_cpu::float_to_bf16(key[index]));
        for (std::size_t index = 0; index < value.size(); ++index)
          REQUIRE(v_f16[index] == quixicore_cpu::float_to_f16(value[index]));
        if (m == 1) {
          std::fill(q.begin(), q.end(), 0.0f);
          std::fill(key.begin(), key.end(), 0.0f);
          std::fill(value.begin(), value.end(), 0.0f);
          REQUIRE(
              quixicore_cpu::qgemv_prepacked_qkv_storage(
                  q_prepared, k_prepared, v_prepared, {input, input_type, k},
                  {q.data(), FloatStorageType::kF32, nq},
                  {key.data(), FloatStorageType::kF32, nk},
                  {value.data(), FloatStorageType::kF32, nv}) == Status::kOk);
          for (std::size_t index = 0; index < q.size(); ++index)
            REQUIRE(close(q[index], expected_q[index]));
          for (std::size_t index = 0; index < key.size(); ++index)
            REQUIRE(close(key[index], expected_k[index]));
          for (std::size_t index = 0; index < value.size(); ++index)
            REQUIRE(close(value[index], expected_v[index]));
        }
      }
    }
  }
  return true;
}

bool test_qkv_validation() {
  constexpr long long nq = 7;
  constexpr long long nk = 3;
  constexpr long long nv = 5;
  constexpr long long k = 32;
  CanonicalQuantTensor q_tensor;
  CanonicalQuantTensor k_tensor;
  CanonicalQuantTensor v_tensor;
  CanonicalQuantTensor incompatible_tensor;
  REQUIRE(pack_layout(kLayouts[0], make_values(nq, k, 67), nq, k, &q_tensor) ==
          Status::kOk);
  REQUIRE(pack_layout(kLayouts[0], make_values(nk, k, 71), nk, k, &k_tensor) ==
          Status::kOk);
  REQUIRE(pack_layout(kLayouts[0], make_values(nv, k, 73), nv, k, &v_tensor) ==
          Status::kOk);
  REQUIRE(pack_layout(kLayouts[4], make_values(nk, k, 79), nk, k,
                      &incompatible_tensor) == Status::kOk);
  CpuPackedWeights q_prepared;
  CpuPackedWeights k_prepared;
  CpuPackedWeights v_prepared;
  CpuPackedWeights incompatible_prepared;
  REQUIRE(q_prepared.prepare(q_tensor) == Status::kOk);
  REQUIRE(k_prepared.prepare(k_tensor) == Status::kOk);
  REQUIRE(v_prepared.prepare(v_tensor) == Status::kOk);
  REQUIRE(incompatible_prepared.prepare(incompatible_tensor) == Status::kOk);

  const std::vector<float> input = make_values(1, k, 83);
  std::vector<float> q(nq, 17.0f);
  std::vector<float> key(nk, 17.0f);
  std::vector<float> value(nv, 17.0f);
  const auto call = [&](const CpuPackedWeights& key_weights,
                        FloatStorageType input_type) {
    return quixicore_cpu::qgemv_prepacked_qkv_storage(
        q_prepared, key_weights, v_prepared, {input.data(), input_type, k},
        {q.data(), FloatStorageType::kF32, nq},
        {key.data(), FloatStorageType::kF32, nk},
        {value.data(), FloatStorageType::kF32, nv});
  };
  REQUIRE(call(incompatible_prepared, FloatStorageType::kF32) ==
          Status::kInvalidShape);
  REQUIRE(call(k_prepared, static_cast<FloatStorageType>(99)) ==
          Status::kUnsupportedFormat);
  for (float item : q) REQUIRE(item == 17.0f);
  for (float item : key) REQUIRE(item == 17.0f);
  for (float item : value) REQUIRE(item == 17.0f);
  return true;
}

bool test_qkv_rope_kv_matrix() {
  constexpr long long query_heads = 2;
  constexpr long long kv_heads = 1;
  constexpr long long head_dim = 4;
  constexpr long long query_dim = query_heads * head_dim;
  constexpr long long kv_dim = kv_heads * head_dim;
  constexpr long long k = 32;
  constexpr long long slots = 3;
  constexpr long long max_position = 3;
  constexpr int position = 1;
  constexpr int slot = 2;
  const float cosine[] = {1.0f, 1.0f, 0.8f, 0.6f, 0.4f, 0.2f};
  const float sine[] = {0.0f, 0.0f, 0.6f, 0.8f, 0.9f, 0.98f};
  for (const LayoutCase& test : kLayouts) {
    CanonicalQuantTensor q_tensor;
    CanonicalQuantTensor k_tensor;
    CanonicalQuantTensor v_tensor;
    REQUIRE(pack_layout(test, make_values(query_dim, k, 89), query_dim, k,
                        &q_tensor) == Status::kOk);
    REQUIRE(pack_layout(test, make_values(kv_dim, k, 97), kv_dim, k,
                        &k_tensor) == Status::kOk);
    REQUIRE(pack_layout(test, make_values(kv_dim, k, 101), kv_dim, k,
                        &v_tensor) == Status::kOk);
    CpuPackedWeights q_prepared;
    CpuPackedWeights k_prepared;
    CpuPackedWeights v_prepared;
    REQUIRE(q_prepared.prepare(q_tensor) == Status::kOk);
    REQUIRE(k_prepared.prepare(k_tensor) == Status::kOk);
    REQUIRE(v_prepared.prepare(v_tensor) == Status::kOk);
    const std::vector<float> q_weights = decode(q_tensor);
    const std::vector<float> k_weights = decode(k_tensor);
    const std::vector<float> v_weights = decode(v_tensor);

    const std::vector<float> source = make_values(1, k, 103);
    for (FloatStorageType input_type :
         {FloatStorageType::kF32, FloatStorageType::kF16,
          FloatStorageType::kBF16}) {
      const void* input = source.data();
      std::vector<std::uint16_t> encoded;
      std::vector<float> exact = source;
      if (input_type != FloatStorageType::kF32) {
        encoded = encode_storage(source, input_type);
        exact = decode_storage(encoded, input_type);
        input = encoded.data();
      }
      const std::vector<float> raw_q =
          oracle(q_weights, exact, 1, query_dim, k);
      const std::vector<float> raw_k = oracle(k_weights, exact, 1, kv_dim, k);
      const std::vector<float> raw_v = oracle(v_weights, exact, 1, kv_dim, k);
      std::vector<float> expected_q(query_dim);
      std::vector<float> expected_key(slots * kv_dim, 17.0f);
      std::vector<float> expected_value(slots * kv_dim, 17.0f);
      const long long half = head_dim / 2;
      const float* cos_row = cosine + position * half;
      const float* sin_row = sine + position * half;
      for (long long head = 0; head < query_heads; ++head) {
        for (long long dim = 0; dim < half; ++dim) {
          const long long row0 = head * head_dim + dim;
          const long long row1 = row0 + half;
          expected_q[row0] =
              raw_q[row0] * cos_row[dim] - raw_q[row1] * sin_row[dim];
          expected_q[row1] =
              raw_q[row1] * cos_row[dim] + raw_q[row0] * sin_row[dim];
        }
      }
      for (long long dim = 0; dim < half; ++dim) {
        const long long row0 = dim;
        const long long row1 = row0 + half;
        expected_key[slot * kv_dim + row0] =
            raw_k[row0] * cos_row[dim] - raw_k[row1] * sin_row[dim];
        expected_key[slot * kv_dim + row1] =
            raw_k[row1] * cos_row[dim] + raw_k[row0] * sin_row[dim];
      }
      for (long long row = 0; row < kv_dim; ++row)
        expected_value[slot * kv_dim + row] = raw_v[row];

      std::vector<float> q(query_dim);
      std::vector<float> key(slots * kv_dim, 17.0f);
      std::vector<float> value(slots * kv_dim, 17.0f);
      REQUIRE(quixicore_cpu::qgemv_prepacked_qkv_rope_kv_storage(
                  q_prepared, k_prepared, v_prepared, {input, input_type, k},
                  cosine, sine, {q.data(), FloatStorageType::kF32, query_dim},
                  {key.data(), FloatStorageType::kF32, slots * kv_dim},
                  {value.data(), FloatStorageType::kF32, slots * kv_dim},
                  query_heads, kv_heads, head_dim, slots, max_position,
                  position, slot) == Status::kOk);
      for (std::size_t index = 0; index < q.size(); ++index)
        REQUIRE(close(q[index], expected_q[index]));
      for (std::size_t index = 0; index < key.size(); ++index)
        REQUIRE(close(key[index], expected_key[index]));
      for (std::size_t index = 0; index < value.size(); ++index)
        REQUIRE(close(value[index], expected_value[index]));

      std::vector<std::uint16_t> q_f16(query_dim);
      std::vector<std::uint16_t> key_bf16(slots * kv_dim,
                                          quixicore_cpu::float_to_bf16(17.0f));
      std::vector<std::uint16_t> value_f16(slots * kv_dim,
                                           quixicore_cpu::float_to_f16(17.0f));
      REQUIRE(quixicore_cpu::qgemv_prepacked_qkv_rope_kv_storage(
                  q_prepared, k_prepared, v_prepared, {input, input_type, k},
                  cosine, sine,
                  {q_f16.data(), FloatStorageType::kF16, query_dim},
                  {key_bf16.data(), FloatStorageType::kBF16, slots * kv_dim},
                  {value_f16.data(), FloatStorageType::kF16, slots * kv_dim},
                  query_heads, kv_heads, head_dim, slots, max_position,
                  position, slot) == Status::kOk);
      for (std::size_t index = 0; index < q_f16.size(); ++index)
        REQUIRE(q_f16[index] == quixicore_cpu::float_to_f16(q[index]));
      for (std::size_t index = 0; index < key_bf16.size(); ++index)
        REQUIRE(key_bf16[index] == quixicore_cpu::float_to_bf16(key[index]));
      for (std::size_t index = 0; index < value_f16.size(); ++index)
        REQUIRE(value_f16[index] == quixicore_cpu::float_to_f16(value[index]));
    }

    std::vector<float> q(query_dim);
    std::vector<float> key(slots * kv_dim, 19.0f);
    std::vector<float> value(slots * kv_dim, 19.0f);
    REQUIRE(quixicore_cpu::qgemv_prepacked_qkv_rope_kv_storage(
                q_prepared, k_prepared, v_prepared,
                {source.data(), FloatStorageType::kF32, k}, cosine, sine,
                {q.data(), FloatStorageType::kF32, query_dim},
                {key.data(), FloatStorageType::kF32, slots * kv_dim},
                {value.data(), FloatStorageType::kF32, slots * kv_dim},
                query_heads, kv_heads, head_dim, slots, max_position, position,
                -1) == Status::kOk);
    for (float item : key) REQUIRE(item == 19.0f);
    for (float item : value) REQUIRE(item == 19.0f);
  }
  return true;
}

struct QuantOutputCase {
  CanonicalQuantLayout layout;
  long long group_size;
  bool scale_2d;
  const char* name;
};

bool test_swiglu_quantized_matrix() {
  constexpr long long n = 32;
  constexpr long long k = 32;
  const QuantOutputCase outputs[] = {
      {CanonicalQuantLayout::kInt4Symmetric, 16, false, "a4"},
      {CanonicalQuantLayout::kUInt4Affine, 16, false, "u4"},
      {CanonicalQuantLayout::kInt8Symmetric, 16, false, "a8"},
      {CanonicalQuantLayout::kInt8Symmetric, 0, false, "a8_row"},
      {CanonicalQuantLayout::kInt8Affine, 0, false, "a8_affine"},
      {CanonicalQuantLayout::kFP8E4M3FN, 16, false, "fp8_e4m3"},
      {CanonicalQuantLayout::kFP8E4M3FN, 0, false, "fp8_e4m3_tensor"},
      {CanonicalQuantLayout::kFP8E5M2, 16, false, "fp8_e5m2"},
      {CanonicalQuantLayout::kFP4E2M1, 16, false, "fp4"},
      {CanonicalQuantLayout::kFP4E2M1, 0, false, "fp4_tensor"},
      {CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false, "mxfp8"},
      {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false, "mxfp4"},
      {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, false, "nvfp4_1d"},
      {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true, "nvfp4_2d"},
  };
  for (const LayoutCase& weight : kLayouts) {
    CanonicalQuantTensor gate_tensor;
    CanonicalQuantTensor up_tensor;
    REQUIRE(pack_layout(weight, make_values(n, k, 37), n, k, &gate_tensor) ==
            Status::kOk);
    REQUIRE(pack_layout(weight, make_values(n, k, 41), n, k, &up_tensor) ==
            Status::kOk);
    CpuPackedWeights gate_prepared;
    CpuPackedWeights up_prepared;
    REQUIRE(gate_prepared.prepare(gate_tensor) == Status::kOk);
    REQUIRE(up_prepared.prepare(up_tensor) == Status::kOk);
    for (long long m : {1LL, 16LL, 128LL}) {
      const std::vector<float> source = make_values(m, k, 43);
      for (FloatStorageType input_type :
           {FloatStorageType::kF32, FloatStorageType::kF16,
            FloatStorageType::kBF16}) {
        const void* input = source.data();
        std::vector<std::uint16_t> encoded;
        if (input_type != FloatStorageType::kF32) {
          encoded = encode_storage(source, input_type);
          input = encoded.data();
        }
        std::vector<float> swiglu(static_cast<std::size_t>(m * n));
        REQUIRE(quixicore_cpu::qgemm_prepacked_swiglu_storage(
                    gate_prepared, up_prepared, {input, input_type, m * k},
                    {swiglu.data(), FloatStorageType::kF32, m * n},
                    m) == Status::kOk);
        for (const QuantOutputCase& quant : outputs) {
          CanonicalQuantTensor expected;
          REQUIRE(quixicore_cpu::quantize_canonical(
                      {swiglu.data(), FloatStorageType::kF32, m * n}, m, n,
                      quant.layout, quant.group_size, &expected,
                      quant.scale_2d) == Status::kOk);
          CanonicalQuantTensor actual;
          quixicore_cpu::Workspace workspace;
          REQUIRE(quixicore_cpu::qgemm_prepacked_swiglu_quantized(
                      gate_prepared, up_prepared, {input, input_type, m * k},
                      quant.layout, quant.group_size, &actual, m,
                      quant.scale_2d, &workspace) == Status::kOk);
          REQUIRE(workspace.used() == 0);
          REQUIRE(quixicore_cpu::validate_canonical_quant_tensor(actual) ==
                  Status::kOk);
          const std::vector<float> expected_values = decode(expected);
          const std::vector<float> actual_values = decode(actual);
          REQUIRE(expected_values.size() == actual_values.size());
          for (std::size_t index = 0; index < actual_values.size(); ++index) {
            if (!close(actual_values[index], expected_values[index], 3e-2f)) {
              std::cerr << "SwiGLU quant " << weight.name << " -> "
                        << quant.name << " m=" << m
                        << " storage=" << static_cast<int>(input_type)
                        << " index=" << index
                        << " actual=" << actual_values[index]
                        << " expected=" << expected_values[index] << '\n';
              return false;
            }
          }
          if (m == 1) {
            CanonicalQuantTensor gemv;
            REQUIRE(quixicore_cpu::qgemv_prepacked_swiglu_quantized(
                        gate_prepared, up_prepared, {input, input_type, k},
                        quant.layout, quant.group_size, &gemv,
                        quant.scale_2d) == Status::kOk);
            REQUIRE(gemv.data == actual.data);
            REQUIRE(gemv.scale_codes == actual.scale_codes);
            REQUIRE(gemv.scales == actual.scales);
            REQUIRE(gemv.zero_points == actual.zero_points);
          }
        }
      }
    }
  }
  return true;
}

struct DualCase {
  LayoutCase weight;
  LayoutCase activation;
  const char* name;
};

bool test_quantized_activation_matrix() {
  constexpr long long n = 3;
  constexpr long long k = 32;
  const DualCase cases[] = {
      {kLayouts[0], kLayouts[0], "w4a4"},
      {kLayouts[1], kLayouts[2], "w4a8"},
      {kLayouts[2], kLayouts[3], "w8a8"},
      {kLayouts[4], kLayouts[4], "fp8_e4e4"},
      {kLayouts[5], kLayouts[4], "fp8_e5e4"},
      {kLayouts[7], kLayouts[7], "mxfp8"},
      {kLayouts[8], kLayouts[8], "mxfp4"},
      {kLayouts[9], kLayouts[9], "nvfp4"},
      {kLayouts[10], kLayouts[2], "bitnet_a8"},
      {kLayouts[10], kLayouts[6], "bitnet_a4"},
  };
  const std::vector<float> weight_source = make_values(n, k, 5);
  for (const DualCase& test : cases) {
    CanonicalQuantTensor weight_tensor;
    REQUIRE(pack_layout(test.weight, weight_source, n, k, &weight_tensor) ==
            Status::kOk);
    CpuPackedWeights prepared;
    REQUIRE(prepared.prepare(weight_tensor) == Status::kOk);
    const std::vector<float> weights = decode(weight_tensor);
    for (long long m : {1LL, 16LL, 128LL}) {
      const std::vector<float> source = make_values(m, k, 7);
      CanonicalQuantTensor activation;
      REQUIRE(pack_layout(test.activation, source, m, k, &activation) ==
              Status::kOk);
      const std::vector<float> decoded_activation = decode(activation);
      const std::vector<float> expected =
          oracle(weights, decoded_activation, m, n, k);
      std::vector<float> actual(expected.size());
      REQUIRE(quixicore_cpu::qgemm_prepacked_quantized(
                  prepared, activation, actual.data()) == Status::kOk);
      for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!close(actual[index], expected[index])) {
          std::cerr << test.name << " m=" << m << " index=" << index
                    << " actual=" << actual[index]
                    << " expected=" << expected[index] << '\n';
          return false;
        }
      }
      if (m == 1) {
        std::fill(actual.begin(), actual.end(), 0.0f);
        REQUIRE(quixicore_cpu::qgemv_prepacked_quantized(
                    prepared, activation, actual.data()) == Status::kOk);
        for (std::size_t index = 0; index < actual.size(); ++index)
          REQUIRE(close(actual[index], expected[index]));
      }
    }
  }
  return true;
}

bool test_fp8_projection_codebook() {
  constexpr long long n = 1;
  constexpr long long k = 256;
  const std::vector<float> source = make_values(n, k, 17);
  for (CanonicalQuantLayout layout :
       {CanonicalQuantLayout::kFP8E4M3FN, CanonicalQuantLayout::kFP8E5M2}) {
    CanonicalQuantTensor tensor;
    REQUIRE(quixicore_cpu::quantize_canonical(
                {source.data(), FloatStorageType::kF32, k}, n, k, layout, k,
                &tensor) == Status::kOk);
    for (int code = 0; code < 256; ++code) {
      const int magnitude = code & 0x7f;
      const bool nonfinite = layout == CanonicalQuantLayout::kFP8E4M3FN
                                 ? magnitude == 0x7f
                                 : ((magnitude >> 2) & 31) == 31;
      tensor.data[static_cast<std::size_t>(code)] =
          nonfinite ? 0 : static_cast<std::uint8_t>(code);
    }
    REQUIRE(quixicore_cpu::validate_canonical_quant_tensor(tensor) ==
            Status::kOk);
    CpuPackedWeights prepared;
    REQUIRE(prepared.prepare(tensor) == Status::kOk);
    std::vector<float> activation(k);
    for (long long item = 0; item < k; ++item) {
      activation[static_cast<std::size_t>(item)] =
          0.25f + static_cast<float>((item * 13) % 37) / 29.0f;
    }
    const std::vector<float> weights = decode(tensor);
    const std::vector<float> expected = oracle(weights, activation, 1, n, k);
    float actual = 0.0f;
    REQUIRE(quixicore_cpu::qgemv_prepacked(prepared, activation.data(),
                                           &actual) == Status::kOk);
    REQUIRE(close(actual, expected[0]));

    for (int code = 0; code < 256; ++code) {
      tensor.data[static_cast<std::size_t>(code)] =
          static_cast<std::uint8_t>(code);
    }
    CpuPackedWeights special_prepared;
    REQUIRE(special_prepared.prepare(tensor) == Status::kOk);
    actual = 0.0f;
    REQUIRE(quixicore_cpu::qgemv_prepacked(special_prepared, activation.data(),
                                           &actual) == Status::kOk);
    REQUIRE(std::isnan(actual));
  }
  return true;
}

bool test_act_order_and_validation() {
  constexpr long long m = 16;
  constexpr long long n = 5;
  constexpr long long k = 32;
  const std::vector<float> weight_source = make_values(n, k, 9);
  CanonicalQuantTensor tensor;
  const LayoutCase grouped_int4 = {CanonicalQuantLayout::kInt4Symmetric, 16,
                                   "int4_actorder"};
  REQUIRE(pack_layout(grouped_int4, weight_source, n, k, &tensor) ==
          Status::kOk);
  tensor.group_index.resize(k);
  tensor.act_order.resize(k);
  for (long long column = 0; column < k; ++column) {
    tensor.group_index[static_cast<std::size_t>(column)] = column < 16 ? 1 : 0;
    tensor.act_order[static_cast<std::size_t>(column)] =
        static_cast<int>(k - 1 - column);
  }
  tensor.metadata.group_index_count = k;
  tensor.metadata.act_order_count = k;
  REQUIRE(quixicore_cpu::validate_canonical_quant_tensor(tensor) ==
          Status::kOk);
  CpuPackedWeights prepared;
  REQUIRE(prepared.prepare(tensor) == Status::kOk);
  REQUIRE(prepared.info().act_order_offset !=
          std::numeric_limits<std::size_t>::max());
  const std::vector<float> weights = decode(tensor);
  const std::vector<float> source = make_values(m, k, 11);
  const std::vector<float> expected =
      oracle(weights, source, m, n, k, &tensor.act_order);
  std::vector<float> actual(expected.size());
  quixicore_cpu::Workspace workspace;
  REQUIRE(quixicore_cpu::qgemm_prepacked(prepared, source.data(), actual.data(),
                                         m, &workspace) == Status::kOk);
  REQUIRE(workspace.used() == 0);
  for (std::size_t index = 0; index < actual.size(); ++index)
    REQUIRE(close(actual[index], expected[index]));

  REQUIRE(quixicore_cpu::qgemm_prepacked_storage(
              prepared, {source.data(), FloatStorageType::kF32, m * k - 1},
              {actual.data(), FloatStorageType::kF32, m * n},
              m) == Status::kInvalidShape);
  REQUIRE(quixicore_cpu::qgemm_prepacked_storage(
              prepared, {nullptr, FloatStorageType::kF32, m * k},
              {actual.data(), FloatStorageType::kF32, m * n},
              m) == Status::kInvalidArgument);
  CanonicalQuantTensor wrong_activation;
  const std::vector<float> wrong_source = make_values(1, 16, 13);
  REQUIRE(pack_layout(kLayouts[2], wrong_source, 1, 16, &wrong_activation) ==
          Status::kOk);
  REQUIRE(quixicore_cpu::qgemm_prepacked_quantized(prepared, wrong_activation,
                                                   actual.data()) ==
          Status::kInvalidShape);
  REQUIRE(quixicore_cpu::qgemv_prepacked_quantized(
              prepared, tensor, actual.data()) == Status::kInvalidShape);
  return true;
}

}  // namespace

int main() {
  if (!test_weight_only_matrix()) return 1;
  if (!test_gate_up_matrix()) return 1;
  if (!test_qkv_matrix()) return 1;
  if (!test_qkv_validation()) return 1;
  if (!test_qkv_rope_kv_matrix()) return 1;
  if (!test_swiglu_quantized_matrix()) return 1;
  if (!test_quantized_activation_matrix()) return 1;
  if (!test_fp8_projection_codebook()) return 1;
  if (!test_act_order_and_validation()) return 1;
  return 0;
}
