#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "quixicore_cpu/quant_import.h"
#include "quixicore_cpu/quantization.h"

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
using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

struct OutputCase {
  CanonicalQuantLayout layout;
  long long group_size;
  bool scale_2d;
  const char* name;
};

constexpr OutputCase kOutputs[] = {
    {CanonicalQuantLayout::kInt4Symmetric, 32, false, "int4"},
    {CanonicalQuantLayout::kUInt4Affine, 32, false, "u4"},
    {CanonicalQuantLayout::kInt8Symmetric, 32, false, "int8"},
    {CanonicalQuantLayout::kInt8Affine, 0, false, "int8_affine"},
    {CanonicalQuantLayout::kFP8E4M3FN, 32, false, "fp8_e4m3"},
    {CanonicalQuantLayout::kFP8E5M2, 0, false, "fp8_e5m2_tensor"},
    {CanonicalQuantLayout::kFP4E2M1, 32, false, "fp4"},
    {CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false, "mxfp8"},
    {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false, "mxfp4"},
    {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, false, "nvfp4_1d"},
    {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true, "nvfp4_2d"},
};

struct StorageCase {
  FloatStorageType x;
  FloatStorageType residual;
  FloatStorageType parameter;
  FloatStorageType output;
  const char* name;
};

constexpr StorageCase kStorage[] = {
    {FloatStorageType::kF32, FloatStorageType::kF32, FloatStorageType::kF32,
     FloatStorageType::kF32, "f32"},
    {FloatStorageType::kF16, FloatStorageType::kF16, FloatStorageType::kF16,
     FloatStorageType::kF16, "f16"},
    {FloatStorageType::kBF16, FloatStorageType::kBF16, FloatStorageType::kBF16,
     FloatStorageType::kBF16, "bf16"},
    {FloatStorageType::kF16, FloatStorageType::kBF16, FloatStorageType::kF32,
     FloatStorageType::kBF16, "mixed"},
};

std::vector<float> make_values(long long count, int salt, float scale,
                               float offset = 0.0f) {
  std::vector<float> values(static_cast<std::size_t>(count));
  for (long long index = 0; index < count; ++index) {
    const int code = static_cast<int>((index * 37 + salt * 19) % 101) - 50;
    values[static_cast<std::size_t>(index)] =
        offset + scale * static_cast<float>(code) / 31.0f;
  }
  return values;
}

struct Stored {
  std::vector<float> f32;
  std::vector<std::uint16_t> u16;
  FloatStorageType type = FloatStorageType::kF32;

  const void* data() const {
    return type == FloatStorageType::kF32
               ? static_cast<const void*>(f32.data())
               : static_cast<const void*>(u16.data());
  }
};

Stored store_input(const std::vector<float>& values, FloatStorageType type) {
  Stored stored;
  stored.type = type;
  if (type == FloatStorageType::kF32) {
    stored.f32 = values;
  } else {
    stored.u16.resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      stored.u16[index] = type == FloatStorageType::kF16
                              ? quixicore_cpu::float_to_f16(values[index])
                              : quixicore_cpu::float_to_bf16(values[index]);
    }
  }
  return stored;
}

std::vector<float> load_stored(const Stored& stored) {
  if (stored.type == FloatStorageType::kF32) return stored.f32;
  std::vector<float> values(stored.u16.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = stored.type == FloatStorageType::kF16
                        ? quixicore_cpu::f16_to_float(stored.u16[index])
                        : quixicore_cpu::bf16_to_float(stored.u16[index]);
  }
  return values;
}

struct StoredOutput {
  std::vector<float> f32;
  std::vector<std::uint16_t> u16;
  FloatStorageType type = FloatStorageType::kF32;

  explicit StoredOutput(long long count, FloatStorageType storage_type)
      : type(storage_type) {
    if (type == FloatStorageType::kF32) {
      f32.resize(static_cast<std::size_t>(count));
    } else {
      u16.resize(static_cast<std::size_t>(count));
    }
  }

  void* data() {
    return type == FloatStorageType::kF32 ? static_cast<void*>(f32.data())
                                          : static_cast<void*>(u16.data());
  }

  std::vector<float> load() const {
    if (type == FloatStorageType::kF32) return f32;
    std::vector<float> values(u16.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = type == FloatStorageType::kF16
                          ? quixicore_cpu::f16_to_float(u16[index])
                          : quixicore_cpu::bf16_to_float(u16[index]);
    }
    return values;
  }
};

float round_storage(float value, FloatStorageType type) {
  if (type == FloatStorageType::kF32) return value;
  return type == FloatStorageType::kF16
             ? quixicore_cpu::f16_to_float(quixicore_cpu::float_to_f16(value))
             : quixicore_cpu::bf16_to_float(
                   quixicore_cpu::float_to_bf16(value));
}

std::vector<float> normalized_oracle(
    const std::vector<float>& x, const std::vector<float>& residual,
    const std::vector<float>& weight, const std::vector<float>& bias,
    FloatStorageType residual_output_type, long long rows, long long hidden,
    bool layer_norm, std::vector<float>* residual_output) {
  residual_output->resize(static_cast<std::size_t>(rows * hidden));
  std::vector<float> normalized(static_cast<std::size_t>(rows * hidden));
  for (long long row = 0; row < rows; ++row) {
    const long long base = row * hidden;
    double mean = 0.0;
    double m2 = 0.0;
    double sumsq = 0.0;
    for (long long item = 0; item < hidden; ++item) {
      const long long flat = base + item;
      const float value =
          round_storage(x[static_cast<std::size_t>(flat)] +
                            residual[static_cast<std::size_t>(flat)],
                        residual_output_type);
      (*residual_output)[static_cast<std::size_t>(flat)] = value;
      if (layer_norm) {
        const double delta = static_cast<double>(value) - mean;
        mean += delta / static_cast<double>(item + 1);
        m2 += delta * (static_cast<double>(value) - mean);
      } else {
        sumsq += static_cast<double>(value) * value;
      }
    }
    const double inverse = 1.0 / std::sqrt((layer_norm ? m2 : sumsq) /
                                               static_cast<double>(hidden) +
                                           1e-5);
    for (long long item = 0; item < hidden; ++item) {
      const long long flat = base + item;
      const float centered =
          layer_norm
              ? static_cast<float>(((*residual_output)[flat] - mean) * inverse)
              : static_cast<float>((*residual_output)[flat] * inverse);
      normalized[static_cast<std::size_t>(flat)] =
          centered * weight[static_cast<std::size_t>(item)] +
          (layer_norm ? bias[static_cast<std::size_t>(item)] : 0.0f);
    }
  }
  return normalized;
}

bool close(float actual, float expected, float tolerance) {
  return std::fabs(actual - expected) <=
         tolerance * (1.0f + std::fabs(expected));
}

bool run_matrix(bool layer_norm) {
  constexpr long long rows = 17;
  constexpr long long hidden = 64;
  constexpr long long count = rows * hidden;
  const std::vector<float> source_x = make_values(count, 3, 0.7f);
  const std::vector<float> source_residual = make_values(count, 5, 0.19f);
  const std::vector<float> source_weight = make_values(hidden, 7, 0.2f, 0.8f);
  const std::vector<float> source_bias = make_values(hidden, 11, 0.05f);

  for (const StorageCase& storage : kStorage) {
    const Stored x = store_input(source_x, storage.x);
    const Stored residual = store_input(source_residual, storage.residual);
    const Stored weight = store_input(source_weight, storage.parameter);
    const Stored bias = store_input(source_bias, storage.parameter);
    const std::vector<float> x_values = load_stored(x);
    const std::vector<float> residual_values = load_stored(residual);
    const std::vector<float> weight_values = load_stored(weight);
    const std::vector<float> bias_values = load_stored(bias);
    std::vector<float> expected_residual;
    const std::vector<float> normalized = normalized_oracle(
        x_values, residual_values, weight_values, bias_values, storage.output,
        rows, hidden, layer_norm, &expected_residual);

    for (const OutputCase& quant : kOutputs) {
      StoredOutput residual_output(count, storage.output);
      CanonicalQuantTensor actual;
      const FloatStorageInput x_view{x.data(), storage.x, count};
      const FloatStorageInput residual_view{residual.data(), storage.residual,
                                            count};
      const FloatStorageInput weight_view{weight.data(), storage.parameter,
                                          hidden};
      const FloatStorageInput bias_view{bias.data(), storage.parameter, hidden};
      const FloatStorageOutput residual_output_view{residual_output.data(),
                                                    storage.output, count};
      const Status status =
          layer_norm ? quixicore_cpu::layer_norm_add_quantized_storage(
                           x_view, residual_view, weight_view, bias_view,
                           residual_output_view, quant.layout, quant.group_size,
                           &actual, rows, hidden, 1e-5f, quant.scale_2d)
                     : quixicore_cpu::rms_norm_add_quantized_storage(
                           x_view, residual_view, weight_view,
                           residual_output_view, quant.layout, quant.group_size,
                           &actual, rows, hidden, 1e-5f, quant.scale_2d);
      if (status != Status::kOk) {
        std::cerr << (layer_norm ? "layer" : "rms") << ' ' << storage.name
                  << " -> " << quant.name
                  << " status=" << static_cast<int>(status) << '\n';
        return false;
      }
      REQUIRE(quixicore_cpu::validate_canonical_quant_tensor(actual) ==
              Status::kOk);
      const std::vector<float> actual_residual = residual_output.load();
      for (std::size_t index = 0; index < actual_residual.size(); ++index) {
        REQUIRE(actual_residual[index] == expected_residual[index]);
      }

      CanonicalQuantTensor expected;
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {normalized.data(), FloatStorageType::kF32, count}, rows,
                  hidden, quant.layout, quant.group_size, &expected,
                  quant.scale_2d) == Status::kOk);
      std::vector<float> actual_values(static_cast<std::size_t>(count));
      std::vector<float> expected_values(static_cast<std::size_t>(count));
      REQUIRE(quixicore_cpu::dequantize_canonical(actual, actual_values.data(),
                                                  count) == Status::kOk);
      REQUIRE(quixicore_cpu::dequantize_canonical(
                  expected, expected_values.data(), count) == Status::kOk);
      for (std::size_t index = 0; index < actual_values.size(); ++index) {
        if (!close(actual_values[index], expected_values[index], 0.04f)) {
          std::cerr << (layer_norm ? "layer" : "rms") << ' ' << storage.name
                    << " -> " << quant.name << " index=" << index
                    << " actual=" << actual_values[index]
                    << " expected=" << expected_values[index] << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

bool test_validation() {
  constexpr long long rows = 1;
  constexpr long long hidden = 64;
  std::vector<float> values(hidden, 1.0f);
  std::vector<float> residual_out(hidden, 9.0f);
  CanonicalQuantTensor output;
  const FloatStorageInput input{values.data(), FloatStorageType::kF32, hidden};
  REQUIRE(quixicore_cpu::rms_norm_add_quantized_storage(
              input, input, input,
              {residual_out.data(), FloatStorageType::kF32, hidden},
              CanonicalQuantLayout::kBitNetTernary, 32, &output, rows,
              hidden) == Status::kUnsupportedFormat);
  REQUIRE(quixicore_cpu::rms_norm_add_quantized_storage(
              {values.data(), FloatStorageType::kF32, hidden - 1}, input, input,
              {residual_out.data(), FloatStorageType::kF32, hidden},
              CanonicalQuantLayout::kInt4Symmetric, 32, &output, rows,
              hidden) == Status::kInvalidShape);
  REQUIRE(quixicore_cpu::rms_norm_add_quantized_storage(
              input, input, input,
              {residual_out.data(), FloatStorageType::kF32, hidden},
              CanonicalQuantLayout::kInt4Symmetric, 32, &output, rows, hidden,
              1e-5f, true) == Status::kInvalidArgument);
  return true;
}

}  // namespace

int main() {
  if (!run_matrix(false) || !run_matrix(true) || !test_validation()) return 1;
  std::cout << "canonical norm-add-quant matrix tests passed\n";
  return 0;
}
