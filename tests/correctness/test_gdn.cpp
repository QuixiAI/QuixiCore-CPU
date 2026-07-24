#include <algorithm>
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

bool close(float actual, float expected, float atol = 2e-5f,
           float rtol = 2e-5f) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

float sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
  const float exponential = std::exp(value);
  return exponential / (1.0f + exponential);
}

struct TypedData {
  std::vector<float> f32;
  std::vector<std::uint16_t> bits;
  FloatStorageType type = FloatStorageType::kF32;

  const void* data() const {
    return type == FloatStorageType::kF32
               ? static_cast<const void*>(f32.data())
               : static_cast<const void*>(bits.data());
  }
};

TypedData typed(const std::vector<float>& input, FloatStorageType type) {
  TypedData result;
  result.type = type;
  result.f32 = input;
  if (type != FloatStorageType::kF32) {
    result.bits.resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      result.bits[index] = type == FloatStorageType::kF16
                               ? quixicore_cpu::float_to_f16(input[index])
                               : quixicore_cpu::float_to_bf16(input[index]);
      result.f32[index] =
          type == FloatStorageType::kF16
              ? quixicore_cpu::f16_to_float(result.bits[index])
              : quixicore_cpu::bf16_to_float(result.bits[index]);
    }
  }
  return result;
}

std::vector<float> decode_output(const std::vector<float>& f32,
                                 const std::vector<std::uint16_t>& bits,
                                 FloatStorageType type) {
  if (type == FloatStorageType::kF32) return f32;
  std::vector<float> output(bits.size());
  for (std::size_t index = 0; index < bits.size(); ++index) {
    output[index] = type == FloatStorageType::kF16
                        ? quixicore_cpu::f16_to_float(bits[index])
                        : quixicore_cpu::bf16_to_float(bits[index]);
  }
  return output;
}

float storage_round(float value, FloatStorageType type) {
  if (type == FloatStorageType::kF16) {
    return quixicore_cpu::f16_to_float(quixicore_cpu::float_to_f16(value));
  }
  if (type == FloatStorageType::kBF16) {
    return quixicore_cpu::bf16_to_float(quixicore_cpu::float_to_bf16(value));
  }
  return value;
}

bool test_sigmoid_mul() {
  bool ok = true;
  constexpr long long count = 257;
  std::vector<float> gate(count), value(count), grad(count), out(count),
      grad_gate(count), grad_value(count);
  for (long long index = 0; index < count; ++index) {
    gate[index] = 0.03f * static_cast<float>(index - 128);
    value[index] = std::sin(0.07f * static_cast<float>(index + 1));
    grad[index] = std::cos(0.11f * static_cast<float>(index + 2));
  }
  ok &= require(quixicore_cpu::sigmoid_mul(gate.data(), value.data(),
                                           out.data(), count) == Status::kOk,
                "sigmoid_mul status");
  ok &= require(quixicore_cpu::sigmoid_mul_backward(
                    grad.data(), gate.data(), value.data(), grad_gate.data(),
                    grad_value.data(), count) == Status::kOk,
                "sigmoid_mul backward status");
  for (long long index = 0; index < count; ++index) {
    const float probability = sigmoid(gate[index]);
    ok &= require(close(out[index], probability * value[index]),
                  "sigmoid_mul oracle");
    ok &=
        require(close(grad_gate[index], grad[index] * value[index] *
                                            probability * (1.0f - probability)),
                "sigmoid_mul gate gradient");
    ok &= require(close(grad_value[index], grad[index] * probability),
                  "sigmoid_mul value gradient");
  }
  for (FloatStorageType type : {FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    const TypedData typed_gate = typed(gate, type);
    const TypedData typed_value = typed(value, type);
    const TypedData typed_grad = typed(grad, type);
    std::vector<std::uint16_t> out_bits(count), gate_bits(count),
        value_bits(count);
    ok &= require(
        quixicore_cpu::sigmoid_mul_storage(
            FloatStorageInput{typed_gate.data(), type, count},
            FloatStorageInput{typed_value.data(), type, count},
            FloatStorageOutput{out_bits.data(), type, count}) == Status::kOk,
        "typed sigmoid_mul status");
    ok &= require(
        quixicore_cpu::sigmoid_mul_backward_storage(
            FloatStorageInput{typed_grad.data(), type, count},
            FloatStorageInput{typed_gate.data(), type, count},
            FloatStorageInput{typed_value.data(), type, count},
            FloatStorageOutput{gate_bits.data(), type, count},
            FloatStorageOutput{value_bits.data(), type, count}) == Status::kOk,
        "typed sigmoid_mul backward status");
    const std::vector<float> typed_out = decode_output({}, out_bits, type);
    const std::vector<float> typed_gate_grad =
        decode_output({}, gate_bits, type);
    const std::vector<float> typed_value_grad =
        decode_output({}, value_bits, type);
    for (long long index = 0; index < count; ++index) {
      const float probability = sigmoid(typed_gate.f32[index]);
      ok &= require(
          typed_out[index] == storage_round(
                                  probability * typed_value.f32[index], type),
          "typed sigmoid_mul oracle");
      ok &= require(
          typed_gate_grad[index] ==
              storage_round(typed_grad.f32[index] * typed_value.f32[index] *
                                probability * (1.0f - probability),
                            type),
          "typed sigmoid gate gradient");
      ok &= require(
          typed_value_grad[index] ==
              storage_round(typed_grad.f32[index] * probability, type),
          "typed sigmoid value gradient");
    }
  }
  return ok;
}

void recur_oracle(const std::vector<float>& q, const std::vector<float>& k,
                  const std::vector<float>& v, const std::vector<float>& decay,
                  const std::vector<float>& beta,
                  const std::vector<float>& pool, const int* cumulative,
                  const int* slots, std::vector<float>& out,
                  std::vector<float>& pool_out, long long requests,
                  long long pool_slots, long long key_heads,
                  long long value_heads, long long key_dim, long long value_dim,
                  bool load_initial) {
  pool_out = pool;
  std::fill(out.begin(), out.end(), 0.0f);
  const long long group = value_heads / key_heads;
  for (long long request = 0; request < requests; ++request) {
    for (long long value_head = 0; value_head < value_heads; ++value_head) {
      const long long key_head = value_head / group;
      for (long long dv = 0; dv < value_dim; ++dv) {
        float* state = pool_out.data() +
                       ((static_cast<long long>(slots[request]) * value_heads +
                         value_head) *
                            value_dim +
                        dv) *
                           key_dim;
        if (!load_initial) std::fill_n(state, key_dim, 0.0f);
        for (long long token = cumulative[request];
             token < cumulative[request + 1]; ++token) {
          const float* kr = k.data() + (token * key_heads + key_head) * key_dim;
          const float* qr = q.data() + (token * key_heads + key_head) * key_dim;
          double memory = 0.0;
          for (long long dk = 0; dk < key_dim; ++dk) {
            state[dk] *= decay[token * value_heads + value_head];
            memory += static_cast<double>(state[dk]) * kr[dk];
          }
          const double correction =
              (v[(token * value_heads + value_head) * value_dim + dv] -
               memory) *
              beta[token * value_heads + value_head];
          double result = 0.0;
          for (long long dk = 0; dk < key_dim; ++dk) {
            state[dk] += static_cast<float>(kr[dk] * correction);
            result += static_cast<double>(state[dk]) * qr[dk];
          }
          out[(token * value_heads + value_head) * value_dim + dv] =
              static_cast<float>(result);
        }
      }
    }
  }
  (void)pool_slots;
}

bool test_recur() {
  bool ok = true;
  constexpr long long requests = 2;
  constexpr long long pool_slots = 4;
  constexpr long long key_heads = 1;
  constexpr long long value_heads = 2;
  constexpr long long key_dim = 64;
  constexpr long long value_dim = 5;
  const int cumulative[] = {0, 2, 5};
  const int slots[] = {3, 1};
  constexpr long long tokens = 5;
  std::vector<float> q(tokens * key_heads * key_dim), k(q.size()),
      v(tokens * value_heads * value_dim), decay(tokens * value_heads),
      beta(decay.size()), pool(pool_slots * value_heads * value_dim * key_dim),
      pool_original, out(v.size()), reference(v.size()), pool_out(pool.size()),
      reference_pool(pool.size());
  for (std::size_t index = 0; index < q.size(); ++index) {
    q[index] = 0.08f * std::sin(0.013f * static_cast<float>(index + 1));
    k[index] = 0.07f * std::cos(0.017f * static_cast<float>(index + 3));
  }
  for (std::size_t index = 0; index < v.size(); ++index) {
    v[index] = 0.12f * std::sin(0.031f * static_cast<float>(index + 5));
  }
  for (std::size_t index = 0; index < decay.size(); ++index) {
    decay[index] = 0.91f + 0.008f * static_cast<float>(index % 7);
    beta[index] = 0.2f + 0.05f * static_cast<float>(index % 9);
  }
  for (std::size_t index = 0; index < pool.size(); ++index) {
    pool[index] = 0.01f * std::cos(0.007f * static_cast<float>(index + 2));
  }
  pool_original = pool;
  ok &= require(
      quixicore_cpu::gdn_recur(
          q.data(), k.data(), v.data(), decay.data(), beta.data(), pool.data(),
          cumulative, slots, out.data(), pool_out.data(), requests, pool_slots,
          key_heads, value_heads, key_dim, value_dim, true) == Status::kOk,
      "gdn_recur status");
  recur_oracle(q, k, v, decay, beta, pool, cumulative, slots, reference,
               reference_pool, requests, pool_slots, key_heads, value_heads,
               key_dim, value_dim, true);
  for (std::size_t index = 0; index < out.size(); ++index) {
    ok &= require(close(out[index], reference[index], 1e-6f, 1e-6f),
                  "gdn_recur output oracle");
  }
  for (std::size_t index = 0; index < pool_out.size(); ++index) {
    ok &= require(close(pool_out[index], reference_pool[index], 1e-6f, 1e-6f),
                  "gdn_recur state oracle");
  }
  ok &= require(pool == pool_original, "gdn_recur functional input pool");
  const long long slot_stride = value_heads * value_dim * key_dim;
  ok &= require(std::equal(pool_out.begin(), pool_out.begin() + slot_stride,
                           pool.begin()),
                "gdn_recur untouched slot");
  ok &= require(quixicore_cpu::gdn_recur(
                    q.data(), k.data(), v.data(), decay.data(), beta.data(),
                    pool.data(), cumulative, slots, out.data(), pool.data(),
                    requests, pool_slots, key_heads, value_heads, key_dim,
                    value_dim, true) == Status::kInvalidArgument,
                "gdn_recur rejects pool alias");
  for (FloatStorageType type : {FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    const TypedData typed_q = typed(q, type);
    const TypedData typed_k = typed(k, type);
    const TypedData typed_v = typed(v, type);
    const TypedData typed_decay = typed(decay, type);
    const TypedData typed_beta = typed(beta, type);
    std::vector<std::uint16_t> typed_out_bits(out.size());
    std::vector<float> typed_state(pool.size()), f32_out(out.size()),
        f32_state(pool.size());
    ok &= require(
        quixicore_cpu::gdn_recur(
            typed_q.f32.data(), typed_k.f32.data(), typed_v.f32.data(),
            typed_decay.f32.data(), typed_beta.f32.data(), pool.data(),
            cumulative, slots, f32_out.data(), f32_state.data(), requests,
            pool_slots, key_heads, value_heads, key_dim, value_dim, true) ==
            Status::kOk,
        "typed gdn_recur f32 staging oracle");
    ok &= require(
        quixicore_cpu::gdn_recur_storage(
            FloatStorageInput{typed_q.data(), type,
                              static_cast<long long>(typed_q.f32.size())},
            FloatStorageInput{typed_k.data(), type,
                              static_cast<long long>(typed_k.f32.size())},
            FloatStorageInput{typed_v.data(), type,
                              static_cast<long long>(typed_v.f32.size())},
            FloatStorageInput{
                typed_decay.data(), type,
                static_cast<long long>(typed_decay.f32.size())},
            FloatStorageInput{typed_beta.data(), type,
                              static_cast<long long>(typed_beta.f32.size())},
            pool.data(), cumulative, slots,
            FloatStorageOutput{typed_out_bits.data(), type,
                               static_cast<long long>(out.size())},
            typed_state.data(), requests, pool_slots, key_heads, value_heads,
            key_dim, value_dim, true) == Status::kOk,
        "typed gdn_recur storage status");
    const std::vector<float> typed_out =
        decode_output({}, typed_out_bits, type);
    for (std::size_t index = 0; index < typed_out.size(); ++index) {
      ok &= require(typed_out[index] == storage_round(f32_out[index], type),
                    "typed gdn_recur output staging");
    }
    for (std::size_t index = 0; index < typed_state.size(); ++index) {
      ok &= require(typed_state[index] == f32_state[index],
                    "typed gdn_recur FP32 state staging");
    }
  }
  return ok;
}

bool test_helpers_for_type(FloatStorageType type) {
  bool ok = true;
  constexpr long long tokens = 5;
  constexpr long long key_heads = 2;
  constexpr long long value_heads = 4;
  constexpr long long key_dim = 64;
  constexpr long long value_dim = 64;
  constexpr long long channels =
      2 * key_heads * key_dim + value_heads * value_dim;
  std::vector<float> mixed(tokens * channels);
  for (std::size_t index = 0; index < mixed.size(); ++index) {
    mixed[index] = 0.4f * std::sin(0.009f * static_cast<float>(index + 1));
  }
  const TypedData mixed_typed = typed(mixed, type);
  const long long qk_count = tokens * key_heads * key_dim;
  const long long value_count = tokens * value_heads * value_dim;
  std::vector<float> q(qk_count), k(qk_count), value(value_count);
  std::vector<std::uint16_t> q_bits(qk_count), k_bits(qk_count),
      value_bits(value_count);
  void* q_data = type == FloatStorageType::kF32 ? static_cast<void*>(q.data())
                                                : q_bits.data();
  void* k_data = type == FloatStorageType::kF32 ? static_cast<void*>(k.data())
                                                : k_bits.data();
  void* value_data = type == FloatStorageType::kF32
                         ? static_cast<void*>(value.data())
                         : value_bits.data();
  ok &= require(quixicore_cpu::gdn_qkv_prepare_storage(
                    FloatStorageInput{mixed_typed.data(), type,
                                      static_cast<long long>(mixed.size())},
                    FloatStorageOutput{q_data, type, qk_count},
                    FloatStorageOutput{k_data, type, qk_count},
                    FloatStorageOutput{value_data, type, value_count}, tokens,
                    key_heads, value_heads, key_dim, value_dim) == Status::kOk,
                "typed gdn_qkv_prepare status");
  q = decode_output(q, q_bits, type);
  k = decode_output(k, k_bits, type);
  value = decode_output(value, value_bits, type);
  const float q_scale = 1.0f / key_dim;
  const float k_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  for (long long token = 0; token < tokens; ++token) {
    for (long long head = 0; head < key_heads; ++head) {
      for (int is_key = 0; is_key < 2; ++is_key) {
        const long long source =
            token * channels + is_key * key_heads * key_dim + head * key_dim;
        float sum = 0.0f;
        for (long long dim = 0; dim < key_dim; ++dim) {
          sum += mixed_typed.f32[source + dim] * mixed_typed.f32[source + dim];
        }
        const float scale =
            (is_key ? k_scale : q_scale) / std::sqrt(sum / key_dim + 1e-6f);
        const auto& actual = is_key ? k : q;
        for (long long dim = 0; dim < key_dim; ++dim) {
          float expected = mixed_typed.f32[source + dim] * scale;
          if (type == FloatStorageType::kF16) {
            expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected));
          } else if (type == FloatStorageType::kBF16) {
            expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected));
          }
          const float tolerance =
              type == FloatStorageType::kF32 ? 2e-5f : 5e-3f;
          ok &= require(close(actual[(token * key_heads + head) * key_dim + dim],
                              expected, tolerance, tolerance),
                        "typed gdn_qkv_prepare oracle");
        }
      }
    }
  }
  for (long long index = 0; index < value_count; ++index) {
    const long long token = index / (value_heads * value_dim);
    const long long local = index % (value_heads * value_dim);
    ok &= require(
        value[index] ==
            mixed_typed.f32[token * channels + 2 * key_heads * key_dim + local],
        "typed gdn value split");
  }

  std::vector<float> z(value_count), weight(value_dim);
  for (std::size_t index = 0; index < z.size(); ++index) {
    z[index] = 0.3f * std::cos(0.015f * static_cast<float>(index + 4));
  }
  for (long long dim = 0; dim < value_dim; ++dim) {
    weight[dim] = 0.8f + 0.004f * static_cast<float>(dim);
  }
  const TypedData y_typed = typed(value, type);
  const TypedData z_typed = typed(z, type);
  const TypedData weight_typed = typed(weight, type);
  std::vector<float> norm_out(value_count);
  std::vector<std::uint16_t> norm_bits(value_count);
  void* norm_data = type == FloatStorageType::kF32
                        ? static_cast<void*>(norm_out.data())
                        : norm_bits.data();
  ok &= require(quixicore_cpu::gdn_gated_rmsnorm_storage(
                    FloatStorageInput{y_typed.data(), type, value_count},
                    FloatStorageInput{z_typed.data(), type, value_count},
                    FloatStorageInput{weight_typed.data(), type, value_dim},
                    FloatStorageOutput{norm_data, type, value_count}, tokens,
                    value_heads, value_dim) == Status::kOk,
                "typed gdn gated norm status");
  norm_out = decode_output(norm_out, norm_bits, type);
  for (long long row = 0; row < tokens * value_heads; ++row) {
    float sum = 0.0f;
    for (long long dim = 0; dim < value_dim; ++dim) {
      const float element = y_typed.f32[row * value_dim + dim];
      sum += element * element;
    }
    const float inverse = 1.0f / std::sqrt(sum / value_dim + 1e-6f);
    for (long long dim = 0; dim < value_dim; ++dim) {
      const long long index = row * value_dim + dim;
      float expected = y_typed.f32[index] * inverse * weight_typed.f32[dim] *
                       z_typed.f32[index] * sigmoid(z_typed.f32[index]);
      if (type == FloatStorageType::kF16) {
        expected =
            quixicore_cpu::f16_to_float(quixicore_cpu::float_to_f16(expected));
      } else if (type == FloatStorageType::kBF16) {
        expected = quixicore_cpu::bf16_to_float(
            quixicore_cpu::float_to_bf16(expected));
      }
      const float tolerance =
          type == FloatStorageType::kF32 ? 2e-5f : 1.6e-2f;
      ok &= require(close(norm_out[index], expected, tolerance, tolerance),
                    "typed gdn gated norm oracle");
    }
  }
  return ok;
}

bool test_conv_and_controls() {
  bool ok = true;
  constexpr long long requests = 3;
  constexpr long long slots_count = 4;
  constexpr long long channels = 17;
  constexpr long long kernel_size = 4;
  const int cumulative[] = {0, 1, 4, 6};
  const int slots[] = {2, -1, 0};
  constexpr long long tokens = 6;
  std::vector<float> x(tokens * channels), weight(channels * kernel_size),
      pool(slots_count * channels * (kernel_size - 1)), pool_original,
      out(x.size()), pool_out(pool.size());
  for (std::size_t index = 0; index < x.size(); ++index) {
    x[index] = 0.3f * std::sin(0.021f * static_cast<float>(index + 1));
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = 0.2f * std::cos(0.037f * static_cast<float>(index + 2));
  }
  for (std::size_t index = 0; index < pool.size(); ++index) {
    pool[index] = 0.1f * std::sin(0.011f * static_cast<float>(index + 4));
  }
  pool_original = pool;
  ok &= require(quixicore_cpu::gdn_short_conv(
                    x.data(), weight.data(), pool.data(), cumulative, slots,
                    out.data(), pool_out.data(), requests, slots_count,
                    channels, kernel_size, true, true) == Status::kOk,
                "gdn short conv status");
  ok &= require(pool == pool_original, "gdn short conv functional pool");
  for (long long token = 1; token < 4; ++token) {
    ok &= require(std::all_of(out.begin() + token * channels,
                              out.begin() + (token + 1) * channels,
                              [](float value) { return value == 0.0f; }),
                  "gdn negative slot zero output");
  }
  for (long long request : {0LL, 2LL}) {
    for (long long channel = 0; channel < channels; ++channel) {
      float history[3];
      const long long slot = slots[request];
      const long long state = (slot * channels + channel) * 3;
      std::copy_n(pool.data() + state, 3, history);
      for (long long token = cumulative[request];
           token < cumulative[request + 1]; ++token) {
        float expected =
            x[token * channels + channel] * weight[channel * kernel_size + 3];
        for (long long item = 0; item < 3; ++item) {
          expected += history[item] * weight[channel * kernel_size + item];
        }
        expected *= sigmoid(expected);
        ok &= require(close(out[token * channels + channel], expected),
                      "gdn short conv oracle");
        history[0] = history[1];
        history[1] = history[2];
        history[2] = x[token * channels + channel];
      }
      for (long long item = 0; item < 3; ++item) {
        ok &= require(pool_out[state + item] == history[item],
                      "gdn short conv state oracle");
      }
    }
  }

  for (FloatStorageType type : {FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    const TypedData typed_x = typed(x, type);
    const TypedData typed_weight = typed(weight, type);
    std::vector<std::uint16_t> typed_out_bits(out.size());
    std::vector<float> typed_state(pool.size()), f32_out(out.size()),
        f32_state(pool.size());
    ok &= require(
        quixicore_cpu::gdn_short_conv(
            typed_x.f32.data(), typed_weight.f32.data(), pool.data(),
            cumulative, slots, f32_out.data(), f32_state.data(), requests,
            slots_count, channels, kernel_size, true, true) == Status::kOk,
        "typed short conv f32 staging oracle");
    ok &= require(
        quixicore_cpu::gdn_short_conv_storage(
            FloatStorageInput{typed_x.data(), type,
                              static_cast<long long>(typed_x.f32.size())},
            FloatStorageInput{
                typed_weight.data(), type,
                static_cast<long long>(typed_weight.f32.size())},
            pool.data(), cumulative, slots,
            FloatStorageOutput{typed_out_bits.data(), type,
                               static_cast<long long>(out.size())},
            typed_state.data(), requests, slots_count, channels, kernel_size,
            true, true) == Status::kOk,
        "typed short conv storage status");
    const std::vector<float> typed_out =
        decode_output({}, typed_out_bits, type);
    for (std::size_t index = 0; index < typed_out.size(); ++index) {
      ok &= require(typed_out[index] == storage_round(f32_out[index], type),
                    "typed short conv output staging");
    }
    for (std::size_t index = 0; index < typed_state.size(); ++index) {
      ok &= require(typed_state[index] == f32_state[index],
                    "typed short conv FP32 state staging");
    }
  }

  constexpr long long heads = 8;
  std::vector<float> a(tokens * heads), b(a.size()), a_log(heads),
      dt_bias(heads), decay(a.size()), beta(a.size());
  for (std::size_t index = 0; index < a.size(); ++index) {
    a[index] = 0.5f * std::sin(0.17f * static_cast<float>(index + 1));
    b[index] = 0.7f * std::cos(0.13f * static_cast<float>(index + 2));
  }
  for (long long head = 0; head < heads; ++head) {
    a_log[head] = -1.0f + 0.2f * static_cast<float>(head);
    dt_bias[head] = -0.3f + 0.07f * static_cast<float>(head);
  }
  ok &= require(quixicore_cpu::gdn_gate_beta(
                    a.data(), b.data(), a_log.data(), dt_bias.data(),
                    decay.data(), beta.data(), tokens, heads) == Status::kOk,
                "gdn gate beta status");
  for (long long index = 0; index < tokens * heads; ++index) {
    const long long head = index % heads;
    const float softplus = std::log1p(std::exp(a[index] + dt_bias[head]));
    ok &=
        require(close(decay[index], std::exp(-std::exp(a_log[head]) * softplus),
                      2e-6f, 2e-6f),
                "gdn decay oracle");
    ok &= require(close(beta[index], sigmoid(b[index]), 2e-6f, 2e-6f),
                  "gdn beta oracle");
  }
  for (FloatStorageType type : {FloatStorageType::kF16,
                                FloatStorageType::kBF16}) {
    const TypedData typed_a = typed(a, type);
    const TypedData typed_b = typed(b, type);
    std::vector<float> typed_decay(a.size()), typed_beta(a.size());
    ok &= require(
        quixicore_cpu::gdn_gate_beta_storage(
            FloatStorageInput{typed_a.data(), type,
                              static_cast<long long>(typed_a.f32.size())},
            FloatStorageInput{typed_b.data(), type,
                              static_cast<long long>(typed_b.f32.size())},
            a_log.data(), dt_bias.data(), typed_decay.data(),
            typed_beta.data(), tokens, heads) == Status::kOk,
        "typed gdn gate beta status");
    for (long long index = 0; index < tokens * heads; ++index) {
      const long long head = index % heads;
      const float softplus =
          std::log1p(std::exp(typed_a.f32[index] + dt_bias[head]));
      ok &= require(close(typed_decay[index],
                          std::exp(-std::exp(a_log[head]) * softplus), 2e-6f,
                          2e-6f),
                    "typed gdn decay oracle");
      ok &= require(close(typed_beta[index], sigmoid(typed_b.f32[index]),
                          2e-6f, 2e-6f),
                    "typed gdn beta oracle");
    }
  }
  return ok;
}

}  // namespace

int main() {
  quixicore_cpu::set_num_threads(1);
  bool ok = true;
  ok &= test_sigmoid_mul();
  ok &= test_recur();
  ok &= test_conv_and_controls();
  ok &= test_helpers_for_type(FloatStorageType::kF32);
  ok &= test_helpers_for_type(FloatStorageType::kF16);
  ok &= test_helpers_for_type(FloatStorageType::kBF16);
  quixicore_cpu::set_num_threads(4);
  ok &= test_recur();
  ok &= test_conv_and_controls();
  quixicore_cpu::set_num_threads(1);
  if (!ok) return 1;
  std::cout << "GDN tests passed\n";
  return 0;
}
