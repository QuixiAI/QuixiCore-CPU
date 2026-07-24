#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/threading.h"

namespace {

using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

float round_storage(float value, FloatStorageType type) {
  if (type == FloatStorageType::kF16) {
    return quixicore_cpu::f16_to_float(quixicore_cpu::float_to_f16(value));
  }
  if (type == FloatStorageType::kBF16) {
    return quixicore_cpu::bf16_to_float(quixicore_cpu::float_to_bf16(value));
  }
  return value;
}

struct Typed {
  std::vector<float> decoded;
  std::vector<std::uint16_t> bits;
  FloatStorageType type;
  const void* data() const {
    return type == FloatStorageType::kF32
               ? static_cast<const void*>(decoded.data())
               : static_cast<const void*>(bits.data());
  }
};

Typed typed(const std::vector<float>& values, FloatStorageType type) {
  Typed result{values, {}, type};
  if (type == FloatStorageType::kF32) return result;
  result.bits.resize(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result.bits[i] = type == FloatStorageType::kF16
                         ? quixicore_cpu::float_to_f16(values[i])
                         : quixicore_cpu::float_to_bf16(values[i]);
    result.decoded[i] = type == FloatStorageType::kF16
                            ? quixicore_cpu::f16_to_float(result.bits[i])
                            : quixicore_cpu::bf16_to_float(result.bits[i]);
  }
  return result;
}

void reference(const std::vector<float>& x,
               const std::vector<std::uint16_t>& a,
               const std::vector<std::uint16_t>& b,
               const std::vector<float>* base, std::vector<float>& out,
               long long rows, long long input_dim, long long output_dim,
               long long rank, float scale) {
  for (long long row = 0; row < rows; ++row) {
    std::vector<std::uint16_t> low(rank);
    for (long long r = 0; r < rank; ++r) {
      float sum = 0.0f;
      for (long long k = 0; k < input_dim; ++k) {
        sum += x[row * input_dim + k] *
               quixicore_cpu::f16_to_float(a[r * input_dim + k]);
      }
      low[r] = quixicore_cpu::float_to_f16(sum);
    }
    for (long long n = 0; n < output_dim; ++n) {
      float sum = 0.0f;
      for (long long r = 0; r < rank; ++r) {
        sum += quixicore_cpu::f16_to_float(low[r]) *
               quixicore_cpu::f16_to_float(b[n * rank + r]);
      }
      const float delta = quixicore_cpu::f16_to_float(
          quixicore_cpu::float_to_f16(sum));
      const long long index = row * output_dim + n;
      out[index] = (base == nullptr ? 0.0f : (*base)[index]) + scale * delta;
    }
  }
}

bool run_case(FloatStorageType type, long long rows, long long input_dim,
              long long output_dim, long long rank, bool with_base,
              int threads) {
  std::vector<float> x(rows * input_dim), base(rows * output_dim);
  std::vector<std::uint16_t> a(rank * input_dim), b(output_dim * rank);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.15f * std::sin(0.017f * static_cast<float>(i + 1));
  }
  for (std::size_t i = 0; i < base.size(); ++i) {
    base[i] = 0.2f * std::cos(0.013f * static_cast<float>(i + 3));
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = quixicore_cpu::float_to_f16(
        0.1f * std::sin(0.011f * static_cast<float>(i + 5)));
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = quixicore_cpu::float_to_f16(
        0.1f * std::cos(0.019f * static_cast<float>(i + 7)));
  }
  const Typed tx = typed(x, type);
  const Typed tbase = typed(base, type);
  std::vector<float> out(rows * output_dim), expected(out.size());
  std::vector<std::uint16_t> out_bits(out.size());
  void* out_data = type == FloatStorageType::kF32
                       ? static_cast<void*>(out.data())
                       : static_cast<void*>(out_bits.data());
  const FloatStorageInput base_input =
      with_base ? FloatStorageInput{tbase.data(), type,
                                    static_cast<long long>(base.size())}
                : FloatStorageInput{nullptr, type, 0};
  quixicore_cpu::set_num_threads(threads);
  bool ok = require(
      quixicore_cpu::lora_apply_direct_f16_storage(
          FloatStorageInput{tx.data(), type, static_cast<long long>(x.size())},
          a.data(), b.data(), base_input,
          FloatStorageOutput{out_data, type,
                             static_cast<long long>(out.size())},
          rows, input_dim, output_dim, rank, 0.75f) == Status::kOk,
      "lora typed status");
  if (type != FloatStorageType::kF32) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = type == FloatStorageType::kF16
                   ? quixicore_cpu::f16_to_float(out_bits[i])
                   : quixicore_cpu::bf16_to_float(out_bits[i]);
    }
  }
  reference(tx.decoded, a, b, with_base ? &tbase.decoded : nullptr, expected,
            rows, input_dim, output_dim, rank, 0.75f);
  const float tolerance = type == FloatStorageType::kF32 ? 2e-5f : 3e-2f;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const float rounded = round_storage(expected[i], type);
    ok &= require(std::fabs(out[i] - rounded) <=
                      tolerance + tolerance * std::fabs(rounded),
                  "lora typed oracle");
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  for (FloatStorageType type : {FloatStorageType::kF32,
                                FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    ok &= run_case(type, 1, 257, 193, 4, false, 1);
    ok &= run_case(type, 1, 257, 193, 4, true, 1);
    ok &= run_case(type, 4, 1024, 513, 16, false, 4);
    ok &= run_case(type, 4, 1024, 513, 16, true, 4);
  }
  float x[2] = {}, out[2] = {};
  std::uint16_t adapter[2] = {};
  ok &= require(quixicore_cpu::lora_apply_direct_f16(
                    x, adapter, adapter, nullptr, out, 1, 2, 1, 0,
                    1.0f) == Status::kInvalidShape,
                "lora rejects zero rank");
  ok &= require(quixicore_cpu::lora_apply_direct_f16(
                    x, nullptr, adapter, nullptr, out, 1, 2, 1, 1,
                    1.0f) == Status::kInvalidArgument,
                "lora rejects null adapter");
  ok &= require(quixicore_cpu::lora_apply_direct_f16(
                    x, adapter, adapter, nullptr, out, 1, 2, 1, 1,
                    std::numeric_limits<float>::infinity()) ==
                    Status::kInvalidArgument,
                "lora rejects nonfinite scale");
  quixicore_cpu::set_num_threads(1);
  if (!ok) return 1;
  std::cout << "LoRA tests passed\n";
  return 0;
}
