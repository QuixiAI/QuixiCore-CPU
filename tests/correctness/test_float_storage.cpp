#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/rms_norm.h"

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ":"      \
                << __LINE__ << '\n';                                      \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

using namespace quixicore_cpu;

bool close(float actual, float expected, float tolerance) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::fabs(actual - expected) <=
             tolerance * (1.0f + std::fabs(expected));
}

std::vector<std::uint16_t> encode(const std::vector<float>& values,
                                  FloatStorageType type) {
  std::vector<std::uint16_t> out(values.size());
  const Status status =
      float_storage_from_f32(type, values.data(), out.data(), values.size());
  if (status != Status::kOk) std::abort();
  return out;
}

std::vector<float> decode(const std::vector<std::uint16_t>& values,
                          FloatStorageType type) {
  std::vector<float> out(values.size());
  const Status status =
      float_storage_to_f32(type, values.data(), out.data(), values.size());
  if (status != Status::kOk) std::abort();
  return out;
}

Status add_kernel(const float* const* inputs, float* const* outputs,
                  void* context) {
  const long long count = *static_cast<const long long*>(context);
  for (long long i = 0; i < count; ++i) {
    outputs[0][i] = inputs[0][i] + inputs[1][i];
  }
  return Status::kOk;
}

Status fail_kernel(const float* const*, float* const* outputs, void*) {
  outputs[0][0] = 99.0f;
  return Status::kInvalidArgument;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;

  // Public scalar codecs cover special values, subnormals, signed zero, and
  // round-to-nearest-even at both 16-bit precisions.
  REQUIRE(float_to_f16(1.0f) == 0x3c00u);
  REQUIRE(float_to_f16(std::ldexp(1.0f, -24)) == 0x0001u);
  REQUIRE(float_to_f16(-0.0f) == 0x8000u);
  REQUIRE(std::signbit(f16_to_float(0x8000u)));
  REQUIRE(std::isinf(f16_to_float(float_to_f16(INFINITY))));
  REQUIRE(std::isnan(f16_to_float(float_to_f16(NAN))));
  REQUIRE(float_to_bf16(1.00390625f) == 0x3f80u);  // ties to even
  REQUIRE(float_to_bf16(1.01171875f) == 0x3f82u);  // odd lower rounds up
  REQUIRE(float_to_bf16(-0.0f) == 0x8000u);
  REQUIRE(std::signbit(bf16_to_float(0x8000u)));
  REQUIRE(std::isnan(bf16_to_float(float_to_bf16(NAN))));

  std::vector<float> source = {-8.0f, -1.25f, -0.0f, 0.125f,
                               1.0f, 3.1415926f, 32.0f};
  for (FloatStorageType type : {FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    const auto bits = encode(source, type);
    const auto roundtrip = decode(bits, type);
    const float tolerance =
        type == FloatStorageType::kF16 ? 1e-3f : 8e-3f;
    for (std::size_t i = 0; i < source.size(); ++i) {
      REQUIRE(close(roundtrip[i], source[i], tolerance));
    }
  }
  std::vector<float> copied(source.size());
  REQUIRE(float_storage_to_f32(FloatStorageType::kF32, source.data(),
                               copied.data(), source.size()) == Status::kOk);
  REQUIRE(copied == source);
  REQUIRE(float_storage_to_f32(FloatStorageType::kF16, nullptr, copied.data(),
                               copied.size()) == Status::kInvalidArgument);
  REQUIRE(float_storage_from_f32(FloatStorageType::kBF16, source.data(),
                                 nullptr, source.size()) ==
          Status::kInvalidArgument);
  REQUIRE(float_storage_to_f32(FloatStorageType::kF16, nullptr, nullptr, 0) ==
          Status::kOk);

  {
    const std::vector<float> special = {
        -INFINITY,
        -65504.0f,
        -0.0f,
        0.0f,
        std::ldexp(1.0f, -25),
        std::ldexp(1.0f, -24),
        std::ldexp(1.0f, -14),
        1.00048828125f,
        65504.0f,
        INFINITY,
        NAN,
    };
    std::vector<std::uint16_t> encoded(special.size());
    REQUIRE(float_storage_from_f32(FloatStorageType::kF16, special.data(),
                                   encoded.data(), special.size()) ==
            Status::kOk);
    for (std::size_t i = 0; i + 1 < special.size(); ++i) {
      REQUIRE(encoded[i] == float_to_f16(special[i]));
    }
    REQUIRE(std::isnan(f16_to_float(encoded.back())));
    std::vector<float> decoded(special.size());
    REQUIRE(float_storage_to_f32(FloatStorageType::kF16, encoded.data(),
                                 decoded.data(), decoded.size()) == Status::kOk);
    for (std::size_t i = 0; i < encoded.size(); ++i) {
      const float expected = f16_to_float(encoded[i]);
      if (std::isnan(expected)) {
        REQUIRE(std::isnan(decoded[i]));
      } else {
        std::uint32_t actual_bits = 0, expected_bits = 0;
        std::memcpy(&actual_bits, &decoded[i], sizeof actual_bits);
        std::memcpy(&expected_bits, &expected, sizeof expected_bits);
        REQUIRE(actual_bits == expected_bits);
      }
    }
  }

  // The generic seam supports mixed storage and exact in-place aliases.
  const auto half = encode(source, FloatStorageType::kF16);
  auto bfloat = encode(source, FloatStorageType::kBF16);
  std::vector<std::uint16_t> sum(source.size(), 0);
  const FloatStorageInput inputs[] = {
      {half.data(), FloatStorageType::kF16,
       static_cast<long long>(half.size())},
      {bfloat.data(), FloatStorageType::kBF16,
       static_cast<long long>(bfloat.size())}};
  const FloatStorageOutput output{sum.data(), FloatStorageType::kBF16,
                                  static_cast<long long>(sum.size())};
  long long count = static_cast<long long>(source.size());
  REQUIRE(dispatch_float_storage(inputs, 2, &output, 1, add_kernel, &count) ==
          Status::kOk);
  const auto decoded_sum = decode(sum, FloatStorageType::kBF16);
  const auto decoded_half = decode(half, FloatStorageType::kF16);
  const auto decoded_bfloat = decode(bfloat, FloatStorageType::kBF16);
  for (std::size_t i = 0; i < source.size(); ++i) {
    REQUIRE(close(decoded_sum[i], decoded_half[i] + decoded_bfloat[i], 8e-3f));
  }

  const auto before_failure = sum;
  REQUIRE(dispatch_float_storage(nullptr, 0, &output, 1, fail_kernel,
                                 nullptr) == Status::kInvalidArgument);
  REQUIRE(sum == before_failure);  // 16-bit output commits only on success

  FloatStorageInput inplace_input{bfloat.data(), FloatStorageType::kBF16,
                                  static_cast<long long>(bfloat.size())};
  FloatStorageOutput inplace_output{bfloat.data(), FloatStorageType::kBF16,
                                    static_cast<long long>(bfloat.size())};
  REQUIRE(unary_storage(inplace_input, inplace_output, UnaryOp::kNegate) ==
          Status::kOk);
  const auto negated = decode(bfloat, FloatStorageType::kBF16);
  for (std::size_t i = 0; i < source.size(); ++i) {
    REQUIRE(close(negated[i], -decoded_bfloat[i], 8e-3f));
  }
  FloatStorageOutput partial{bfloat.data() + 1, FloatStorageType::kBF16,
                             static_cast<long long>(bfloat.size() - 1)};
  REQUIRE(dispatch_float_storage(&inplace_input, 1, &partial, 1, add_kernel,
                                 &count) == Status::kInvalidArgument);

  // Named wrappers exercise activation, reduction, norm, dense GEMM,
  // attention, and quantized projection families.
  {
    std::vector<float> logits = {-2.0f, -1.0f, 0.0f, 1.0f,
                                 2.0f,  3.0f,  4.0f, 5.0f};
    auto logits16 = encode(logits, FloatStorageType::kF16);
    std::vector<std::uint16_t> probabilities(8);
    REQUIRE(softmax_storage({logits16.data(), FloatStorageType::kF16, 8},
                            {probabilities.data(), FloatStorageType::kBF16, 8},
                            2, 4) == Status::kOk);
    const auto p = decode(probabilities, FloatStorageType::kBF16);
    for (int row = 0; row < 2; ++row) {
      float total = 0.0f;
      for (int col = 0; col < 4; ++col) total += p[row * 4 + col];
      REQUIRE(close(total, 1.0f, 8e-3f));
    }

    std::vector<float> weight = {1.0f, 0.75f, 1.25f, 0.5f};
    auto weight_bf16 = encode(weight, FloatStorageType::kBF16);
    std::vector<std::uint16_t> norm_out(8);
    REQUIRE(rms_norm_storage(
                {logits16.data(), FloatStorageType::kF16, 8},
                {weight_bf16.data(), FloatStorageType::kBF16, 4},
                {norm_out.data(), FloatStorageType::kF16, 8}, 2, 4) ==
            Status::kOk);
    std::vector<float> norm_ref(8);
    const auto logits_decoded = decode(logits16, FloatStorageType::kF16);
    const auto weight_decoded = decode(weight_bf16, FloatStorageType::kBF16);
    REQUIRE(rms_norm(logits_decoded.data(), weight_decoded.data(),
                     norm_ref.data(), 2, 4) == Status::kOk);
    const auto norm_actual = decode(norm_out, FloatStorageType::kF16);
    for (int i = 0; i < 8; ++i) {
      REQUIRE(close(norm_actual[i], norm_ref[i], 1e-3f));
    }
  }

  // An operation with three read/write tensors demonstrates that the generic
  // helper covers APIs beyond the named inference conveniences.
  {
    std::vector<float> parameters = {1.0f, -2.0f, 0.5f, 4.0f};
    std::vector<float> gradients = {0.1f, -0.2f, 0.3f, -0.4f};
    std::vector<float> moments(4, 0.0f);
    auto p16 = encode(parameters, FloatStorageType::kBF16);
    auto g16 = encode(gradients, FloatStorageType::kF16);
    auto m1 = encode(moments, FloatStorageType::kBF16);
    auto m2 = encode(moments, FloatStorageType::kBF16);
    const FloatStorageInput adam_inputs[] = {
        {p16.data(), FloatStorageType::kBF16, 4},
        {g16.data(), FloatStorageType::kF16, 4},
        {m1.data(), FloatStorageType::kBF16, 4},
        {m2.data(), FloatStorageType::kBF16, 4}};
    const FloatStorageOutput adam_outputs[] = {
        {p16.data(), FloatStorageType::kBF16, 4},
        {m1.data(), FloatStorageType::kBF16, 4},
        {m2.data(), FloatStorageType::kBF16, 4}};
    REQUIRE(with_float_storage(
                adam_inputs, 4, adam_outputs, 3,
                [](const float* const* in, float* const* out) {
                  // Exact aliases make out[0/1/2] the same decoded storage as
                  // in[0/2/3], matching AdamW's in-place contract.
                  if (in[0] != out[0] || in[2] != out[1] ||
                      in[3] != out[2]) {
                    return Status::kInvalidArgument;
                  }
                  return adamw(out[0], in[1], out[1], out[2], 4, 1e-3f,
                               0.9f, 0.999f, 1e-8f, 0.01f, 1);
                }) == Status::kOk);
    const auto updated = decode(p16, FloatStorageType::kBF16);
    REQUIRE(updated != parameters);
    for (float value : updated) REQUIRE(std::isfinite(value));
  }

  {
    const long long m = 2, n = 3, k = 4;
    std::vector<float> a = {1, 2, 3, 4, -1, 0.5f, 2, -2};
    std::vector<float> b = {1, 0, -1, 2, 0.5f, 1, -1, 2, 0, 0.25f, 1, 3};
    auto a16 = encode(a, FloatStorageType::kF16);
    auto b16 = encode(b, FloatStorageType::kBF16);
    std::vector<std::uint16_t> c16(m * n);
    REQUIRE(dense_gemm_storage(
                {a16.data(), FloatStorageType::kF16, m * k},
                {b16.data(), FloatStorageType::kBF16, k * n},
                {c16.data(), FloatStorageType::kF16, m * n}, m, n, k) ==
            Status::kOk);
    std::vector<float> ref(m * n);
    const auto ad = decode(a16, FloatStorageType::kF16);
    const auto bd = decode(b16, FloatStorageType::kBF16);
    REQUIRE(dense_gemm(ad.data(), bd.data(), ref.data(), m, n, k) ==
            Status::kOk);
    const auto actual = decode(c16, FloatStorageType::kF16);
    for (int i = 0; i < m * n; ++i) REQUIRE(close(actual[i], ref[i], 1e-3f));
  }

  {
    const long long heads = 2, sequence = 3, dim = 4;
    std::vector<float> q(heads * sequence * dim);
    std::vector<float> k(q.size()), v(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
      q[i] = 0.01f * static_cast<float>(i + 1);
      k[i] = 0.02f * static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
      v[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5) % 11) - 5);
    }
    auto q16 = encode(q, FloatStorageType::kF16);
    auto k16 = encode(k, FloatStorageType::kBF16);
    auto v16 = encode(v, FloatStorageType::kF16);
    std::vector<std::uint16_t> out16(q.size());
    REQUIRE(attention_storage(
                {q16.data(), FloatStorageType::kF16,
                 static_cast<long long>(q.size())},
                {k16.data(), FloatStorageType::kBF16,
                 static_cast<long long>(k.size())},
                {v16.data(), FloatStorageType::kF16,
                 static_cast<long long>(v.size())},
                {out16.data(), FloatStorageType::kBF16,
                 static_cast<long long>(out16.size())},
                heads, heads, sequence, sequence, dim, true) == Status::kOk);
    const auto actual = decode(out16, FloatStorageType::kBF16);
    for (float value : actual) REQUIRE(std::isfinite(value));
  }

  {
    const long long m = 2, n = 3, k = 32;
    std::vector<float> weights(n * k), x(m * k);
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i] = 0.25f * std::sin(static_cast<float>(i));
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = 0.5f * std::cos(static_cast<float>(i));
    }
    std::size_t packed_size = 0;
    REQUIRE(qgemv_packed_size(QuantFormat::kQ8_0, n, k, &packed_size) ==
            Status::kOk);
    std::vector<std::uint8_t> packed(packed_size);
    REQUIRE(qgemv_pack(QuantFormat::kQ8_0, weights.data(), n, k,
                       packed.data()) == Status::kOk);
    auto x16 = encode(x, FloatStorageType::kF16);
    std::vector<std::uint16_t> y16(m * n);
    REQUIRE(qgemv_storage(
                QuantFormat::kQ8_0, packed.data(),
                {x16.data(), FloatStorageType::kF16, k},
                {y16.data(), FloatStorageType::kBF16, n}, n, k) ==
            Status::kOk);
    REQUIRE(qgemm_storage(
                QuantFormat::kQ8_0, packed.data(),
                {x16.data(), FloatStorageType::kF16, m * k},
                {y16.data(), FloatStorageType::kBF16, m * n}, m, n, k) ==
            Status::kOk);
    for (float value : decode(y16, FloatStorageType::kBF16)) {
      REQUIRE(std::isfinite(value));
    }
  }

  FloatStorageWorkspace workspace;
  REQUIRE(workspace.reserve(1024) == Status::kOk);
  REQUIRE(workspace.capacity() >= 1024);
  REQUIRE(std::strcmp(float_storage_variant(FloatStorageType::kF32),
                      "zero_copy") == 0);
  REQUIRE(std::strlen(float_storage_variant(FloatStorageType::kF16)) > 0);
  REQUIRE(dense_gemm_storage({nullptr, FloatStorageType::kF16,
                              std::numeric_limits<long long>::max()},
                             {nullptr, FloatStorageType::kF16, 1},
                             {nullptr, FloatStorageType::kF16, 1},
                             std::numeric_limits<long long>::max(), 2, 2) ==
          Status::kInvalidShape);
  return 0;
}
