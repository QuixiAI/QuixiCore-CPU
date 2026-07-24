#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quantization.h"

namespace {

using quixicore_cpu::Float8Format;
using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

std::vector<std::uint16_t> encode_storage(const std::vector<float>& values,
                                          FloatStorageType type) {
  std::vector<std::uint16_t> output(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = type == FloatStorageType::kF16
                    ? quixicore_cpu::float_to_f16(values[i])
                    : quixicore_cpu::float_to_bf16(values[i]);
  }
  return output;
}

std::vector<float> decode_storage(const void* data, FloatStorageType type,
                                  std::size_t count) {
  std::vector<float> output(count);
  if (type == FloatStorageType::kF32) {
    std::copy_n(static_cast<const float*>(data), count, output.data());
    return output;
  }
  const auto* values = static_cast<const std::uint16_t*>(data);
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = type == FloatStorageType::kF16
                    ? quixicore_cpu::f16_to_float(values[i])
                    : quixicore_cpu::bf16_to_float(values[i]);
  }
  return output;
}

const void* storage_data(const std::vector<float>& f32,
                         const std::vector<std::uint16_t>& packed,
                         FloatStorageType type) {
  return type == FloatStorageType::kF32
             ? static_cast<const void*>(f32.data())
             : static_cast<const void*>(packed.data());
}

bool test_format(Float8Format format, long long dim) {
  constexpr long long kTokens = 7;
  constexpr long long kHeads = 2;
  constexpr long long kSlots = 8;
  constexpr long long kPage = 4;
  constexpr long long kQueryHeads = 4;
  const long long elements = kTokens * kHeads * dim;
  const long long cache_elements = kSlots * kHeads * dim;
  const long long query_elements = kQueryHeads * dim;
  std::vector<float> key(static_cast<std::size_t>(elements));
  std::vector<float> value(key.size());
  std::vector<float> query(static_cast<std::size_t>(query_elements));
  for (long long i = 0; i < elements; ++i) {
    key[i] = 0.8f * std::sin(0.031f * static_cast<float>(i + 1));
    value[i] = 0.7f * std::cos(0.027f * static_cast<float>(i + 3));
  }
  for (long long i = 0; i < query_elements; ++i) {
    query[i] = 0.2f * std::sin(0.043f * static_cast<float>(i + 5));
  }
  const int slots[] = {0, 1, 2, 3, 4, 5, 6};
  const int block_table[] = {0, 1};
  const int context_lens[] = {7};
  const FloatStorageType types[] = {
      FloatStorageType::kF32, FloatStorageType::kF16, FloatStorageType::kBF16};
  bool ok = true;

  for (FloatStorageType input_type : types) {
    const auto key_packed = encode_storage(key, input_type);
    const auto value_packed = encode_storage(value, input_type);
    const auto rounded_key = decode_storage(
        storage_data(key, key_packed, input_type), input_type, key.size());
    const auto rounded_value =
        decode_storage(storage_data(value, value_packed, input_type),
                       input_type, value.size());
    const FloatStorageInput key_view{storage_data(key, key_packed, input_type),
                                     input_type, elements};
    const FloatStorageInput value_view{
        storage_data(value, value_packed, input_type), input_type, elements};
    std::vector<float> key_scales(kHeads), value_scales(kHeads);
    ok &= require(
        quixicore_cpu::kv_cache_scales_fp8_storage(
            key_view, value_view, key_scales.data(), value_scales.data(),
            kTokens, kHeads, dim, format) == Status::kOk,
        "typed dynamic FP8 scales");
    const float maximum = format == Float8Format::kE5M2 ? 57344.0f : 448.0f;
    for (long long head = 0; head < kHeads; ++head) {
      float expected_key = 0.0f;
      float expected_value = 0.0f;
      for (long long token = 0; token < kTokens; ++token) {
        const long long base = (token * kHeads + head) * dim;
        for (long long d = 0; d < dim; ++d) {
          expected_key =
              std::max(expected_key, std::fabs(rounded_key[base + d]));
          expected_value =
              std::max(expected_value, std::fabs(rounded_value[base + d]));
        }
      }
      ok &= require(
          std::fabs(key_scales[head] - expected_key / maximum) < 1e-8f &&
              std::fabs(value_scales[head] - expected_value / maximum) < 1e-8f,
          "per-head FP8 dynamic scale value");
    }

    std::vector<std::uint8_t> key_cache(
        static_cast<std::size_t>(cache_elements));
    std::vector<std::uint8_t> value_cache(key_cache.size());
    ok &= require(quixicore_cpu::kv_cache_scatter_fp8_storage(
                      key_view, value_view, slots, key_scales.data(),
                      value_scales.data(), key_cache.data(), value_cache.data(),
                      kSlots, kTokens, kHeads, dim, format) == Status::kOk,
                  "typed FP8 scatter");

    for (FloatStorageType output_type : types) {
      std::vector<float> gathered_key_f32;
      std::vector<float> gathered_value_f32;
      std::vector<std::uint16_t> gathered_key_16;
      std::vector<std::uint16_t> gathered_value_16;
      void* key_output = nullptr;
      void* value_output = nullptr;
      if (output_type == FloatStorageType::kF32) {
        gathered_key_f32.resize(key.size());
        gathered_value_f32.resize(value.size());
        key_output = gathered_key_f32.data();
        value_output = gathered_value_f32.data();
      } else {
        gathered_key_16.resize(key.size());
        gathered_value_16.resize(value.size());
        key_output = gathered_key_16.data();
        value_output = gathered_value_16.data();
      }
      ok &= require(quixicore_cpu::kv_cache_gather_fp8_storage(
                        key_cache.data(), value_cache.data(), slots,
                        key_scales.data(), value_scales.data(),
                        FloatStorageOutput{key_output, output_type, elements},
                        FloatStorageOutput{value_output, output_type, elements},
                        kSlots, kTokens, kHeads, dim, format) == Status::kOk,
                    "typed FP8 gather");
      const auto gathered_key =
          decode_storage(key_output, output_type, key.size());
      const auto gathered_value =
          decode_storage(value_output, output_type, value.size());
      for (long long i = 0; i < elements; ++i) {
        const long long head = (i / dim) % kHeads;
        float expected_key =
            quixicore_cpu::float8_decode(key_cache[i], format) *
            key_scales[head];
        float expected_value =
            quixicore_cpu::float8_decode(value_cache[i], format) *
            value_scales[head];
        if (output_type == FloatStorageType::kF16) {
          expected_key = quixicore_cpu::f16_to_float(
              quixicore_cpu::float_to_f16(expected_key));
          expected_value = quixicore_cpu::f16_to_float(
              quixicore_cpu::float_to_f16(expected_value));
        } else if (output_type == FloatStorageType::kBF16) {
          expected_key = quixicore_cpu::bf16_to_float(
              quixicore_cpu::float_to_bf16(expected_key));
          expected_value = quixicore_cpu::bf16_to_float(
              quixicore_cpu::float_to_bf16(expected_value));
        }
        ok &= require(gathered_key[i] == expected_key &&
                          gathered_value[i] == expected_value,
                      "typed FP8 gather rounding boundary");
      }
    }

    for (FloatStorageType query_type : types) {
      const auto query_packed = encode_storage(query, query_type);
      const auto rounded_query =
          decode_storage(storage_data(query, query_packed, query_type),
                         query_type, query.size());
      std::vector<float> reference(query.size());
      ok &= require(
          quixicore_cpu::paged_attention_fp8(
              rounded_query.data(), key_cache.data(), value_cache.data(),
              block_table, context_lens, key_scales.data(), value_scales.data(),
              reference.data(), 2, 1, kQueryHeads, kHeads, dim, kPage, 2,
              format) == Status::kOk,
          "FP32 FP8 attention oracle");
      for (FloatStorageType output_type : types) {
        std::vector<float> output_f32;
        std::vector<std::uint16_t> output_16;
        void* output = nullptr;
        if (output_type == FloatStorageType::kF32) {
          output_f32.resize(query.size());
          output = output_f32.data();
        } else {
          output_16.resize(query.size());
          output = output_16.data();
        }
        ok &= require(
            quixicore_cpu::paged_attention_fp8_storage(
                FloatStorageInput{storage_data(query, query_packed, query_type),
                                  query_type, query_elements},
                key_cache.data(), value_cache.data(), block_table, context_lens,
                key_scales.data(), value_scales.data(),
                FloatStorageOutput{output, output_type, query_elements}, 2, 1,
                kQueryHeads, kHeads, dim, kPage, 2, format) == Status::kOk,
            "typed direct FP8 attention");
        const auto actual =
            decode_storage(output, output_type, reference.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
          float expected = reference[i];
          if (output_type == FloatStorageType::kF16) {
            expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected));
          } else if (output_type == FloatStorageType::kBF16) {
            expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected));
          }
          ok &= require(actual[i] == expected,
                        "typed FP8 attention output rounding boundary");
        }
      }
    }
  }
  return ok;
}

bool test_zero_and_validation() {
  constexpr long long kDim = 64;
  std::vector<float> zero(kDim, 0.0f);
  std::vector<std::uint8_t> key_cache(kDim, 0xff);
  std::vector<std::uint8_t> value_cache(kDim, 0xff);
  const int slot[] = {0};
  float key_scale = -1.0f;
  float value_scale = -1.0f;
  bool ok = true;
  ok &=
      require(quixicore_cpu::kv_cache_scales_fp8_storage(
                  FloatStorageInput{zero.data(), FloatStorageType::kF32, kDim},
                  FloatStorageInput{zero.data(), FloatStorageType::kF32, kDim},
                  &key_scale, &value_scale, 1, 1, kDim,
                  Float8Format::kE4M3FN) == Status::kOk &&
                  key_scale == 0.0f && value_scale == 0.0f,
              "zero FP8 dynamic scale");
  ok &= require(
      quixicore_cpu::kv_cache_scatter_fp8_storage(
          FloatStorageInput{zero.data(), FloatStorageType::kF32, kDim},
          FloatStorageInput{zero.data(), FloatStorageType::kF32, kDim}, slot,
          &key_scale, &value_scale, key_cache.data(), value_cache.data(), 1, 1,
          1, kDim, Float8Format::kE4M3FN) == Status::kOk &&
          std::all_of(key_cache.begin(), key_cache.end(),
                      [](std::uint8_t code) { return code == 0; }),
      "zero-scale FP8 scatter");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  ok &= require(quixicore_cpu::kv_cache_scatter_fp8(
                    zero.data(), zero.data(), slot, &nan, &value_scale,
                    key_cache.data(), value_cache.data(), 1, 1, 1, kDim,
                    Float8Format::kE4M3FN) == Status::kInvalidArgument,
                "FP8 scatter scale validation");
  return ok;
}

float e8m0_decode(std::uint8_t code) {
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

bool test_mxfp8(long long dim) {
  constexpr long long kTokens = 7;
  constexpr long long kHeads = 2;
  constexpr long long kSlots = 8;
  constexpr long long kPage = 4;
  constexpr long long kQueryHeads = 4;
  constexpr long long kBlockBytes = 33;
  const long long groups = dim / 32;
  const long long elements = kTokens * kHeads * dim;
  const long long cache_bytes = kSlots * kHeads * groups * kBlockBytes;
  const long long query_elements = kQueryHeads * dim;
  std::vector<float> key(static_cast<std::size_t>(elements));
  std::vector<float> value(key.size());
  std::vector<float> query(static_cast<std::size_t>(query_elements));
  for (long long i = 0; i < elements; ++i) {
    key[i] = 0.8f * std::sin(0.019f * static_cast<float>(i + 1));
    value[i] = 0.7f * std::cos(0.023f * static_cast<float>(i + 3));
  }
  for (long long i = 0; i < query_elements; ++i) {
    query[i] = 0.2f * std::sin(0.037f * static_cast<float>(i + 5));
  }
  const int slots[] = {0, 1, 2, 3, 4, 5, 6};
  const int block_table[] = {0, 1};
  const int context_lens[] = {7};
  const FloatStorageType types[] = {
      FloatStorageType::kF32, FloatStorageType::kF16, FloatStorageType::kBF16};
  bool ok = true;

  for (FloatStorageType input_type : types) {
    const auto key_packed = encode_storage(key, input_type);
    const auto value_packed = encode_storage(value, input_type);
    std::vector<std::uint8_t> key_cache(static_cast<std::size_t>(cache_bytes));
    std::vector<std::uint8_t> value_cache(key_cache.size());
    ok &= require(
        quixicore_cpu::kv_cache_scatter_mxfp8_storage(
            FloatStorageInput{storage_data(key, key_packed, input_type),
                              input_type, elements},
            FloatStorageInput{storage_data(value, value_packed, input_type),
                              input_type, elements},
            slots, key_cache.data(), value_cache.data(), kSlots, kTokens,
            kHeads, dim) == Status::kOk,
        "typed MXFP8 scatter");
    for (long long block = 0; block < kTokens * kHeads * groups; ++block) {
      ok &= require(key_cache[block * kBlockBytes] != 255 &&
                        value_cache[block * kBlockBytes] != 255,
                    "MXFP8 valid E8M0 scale code");
    }

    std::vector<float> decoded_key(static_cast<std::size_t>(elements));
    std::vector<float> decoded_value(decoded_key.size());
    ok &= require(
        quixicore_cpu::kv_cache_gather_mxfp8(
            key_cache.data(), value_cache.data(), slots, decoded_key.data(),
            decoded_value.data(), kSlots, kTokens, kHeads, dim) == Status::kOk,
        "MXFP8 FP32 gather oracle");
    for (long long token = 0; token < kTokens; ++token) {
      for (long long head = 0; head < kHeads; ++head) {
        for (long long group = 0; group < groups; ++group) {
          const long long cache =
              ((token * kHeads + head) * groups + group) * kBlockBytes;
          const long long output = (token * kHeads + head) * dim + group * 32;
          const float key_scale = e8m0_decode(key_cache[cache]);
          const float value_scale = e8m0_decode(value_cache[cache]);
          for (long long d = 0; d < 32; ++d) {
            ok &= require(decoded_key[output + d] ==
                                  key_scale * quixicore_cpu::float8_decode(
                                                  key_cache[cache + 1 + d],
                                                  Float8Format::kE4M3FN) &&
                              decoded_value[output + d] ==
                                  value_scale * quixicore_cpu::float8_decode(
                                                    value_cache[cache + 1 + d],
                                                    Float8Format::kE4M3FN),
                          "MXFP8 canonical block decode");
          }
        }
      }
    }

    for (FloatStorageType output_type : types) {
      std::vector<float> gathered_key_f32;
      std::vector<float> gathered_value_f32;
      std::vector<std::uint16_t> gathered_key_16;
      std::vector<std::uint16_t> gathered_value_16;
      void* key_output = nullptr;
      void* value_output = nullptr;
      if (output_type == FloatStorageType::kF32) {
        gathered_key_f32.resize(key.size());
        gathered_value_f32.resize(value.size());
        key_output = gathered_key_f32.data();
        value_output = gathered_value_f32.data();
      } else {
        gathered_key_16.resize(key.size());
        gathered_value_16.resize(value.size());
        key_output = gathered_key_16.data();
        value_output = gathered_value_16.data();
      }
      ok &= require(quixicore_cpu::kv_cache_gather_mxfp8_storage(
                        key_cache.data(), value_cache.data(), slots,
                        FloatStorageOutput{key_output, output_type, elements},
                        FloatStorageOutput{value_output, output_type, elements},
                        kSlots, kTokens, kHeads, dim) == Status::kOk,
                    "typed MXFP8 gather");
      const auto gathered_key =
          decode_storage(key_output, output_type, key.size());
      const auto gathered_value =
          decode_storage(value_output, output_type, value.size());
      for (long long i = 0; i < elements; ++i) {
        float expected_key = decoded_key[i];
        float expected_value = decoded_value[i];
        if (output_type == FloatStorageType::kF16) {
          expected_key = quixicore_cpu::f16_to_float(
              quixicore_cpu::float_to_f16(expected_key));
          expected_value = quixicore_cpu::f16_to_float(
              quixicore_cpu::float_to_f16(expected_value));
        } else if (output_type == FloatStorageType::kBF16) {
          expected_key = quixicore_cpu::bf16_to_float(
              quixicore_cpu::float_to_bf16(expected_key));
          expected_value = quixicore_cpu::bf16_to_float(
              quixicore_cpu::float_to_bf16(expected_value));
        }
        ok &= require(gathered_key[i] == expected_key &&
                          gathered_value[i] == expected_value,
                      "typed MXFP8 gather rounding boundary");
      }
    }

    for (FloatStorageType query_type : types) {
      const auto query_packed = encode_storage(query, query_type);
      const auto rounded_query =
          decode_storage(storage_data(query, query_packed, query_type),
                         query_type, query.size());
      std::vector<float> reference(query.size());
      ok &= require(
          quixicore_cpu::paged_attention(
              rounded_query.data(), decoded_key.data(), decoded_value.data(),
              block_table, context_lens, reference.data(), 2, 1, kQueryHeads,
              kHeads, dim, kPage, 2) == Status::kOk,
          "MXFP8 materialized attention oracle");
      for (FloatStorageType output_type : types) {
        std::vector<float> output_f32;
        std::vector<std::uint16_t> output_16;
        void* output = nullptr;
        if (output_type == FloatStorageType::kF32) {
          output_f32.resize(query.size());
          output = output_f32.data();
        } else {
          output_16.resize(query.size());
          output = output_16.data();
        }
        ok &= require(
            quixicore_cpu::paged_attention_mxfp8_storage(
                FloatStorageInput{storage_data(query, query_packed, query_type),
                                  query_type, query_elements},
                key_cache.data(), value_cache.data(), block_table, context_lens,
                FloatStorageOutput{output, output_type, query_elements}, 2, 1,
                kQueryHeads, kHeads, dim, kPage, 2) == Status::kOk,
            "typed direct MXFP8 attention");
        const auto actual =
            decode_storage(output, output_type, reference.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
          float expected = reference[i];
          if (output_type == FloatStorageType::kF16) {
            expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected));
          } else if (output_type == FloatStorageType::kBF16) {
            expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected));
          }
          ok &= require(std::fabs(actual[i] - expected) <= 3e-5f,
                        "typed MXFP8 attention output");
        }
      }
    }
  }
  return ok;
}

void fwht(std::vector<float>* values) {
  for (std::size_t width = 1; width < values->size(); width *= 2) {
    for (std::size_t base = 0; base < values->size(); base += 2 * width) {
      for (std::size_t item = 0; item < width; ++item) {
        const float a = (*values)[base + item];
        const float b = (*values)[base + width + item];
        (*values)[base + item] = a + b;
        (*values)[base + width + item] = a - b;
      }
    }
  }
}

bool test_turboquant(long long dim, int key_bits, bool key_signed,
                     int value_bits, bool typed_matrix) {
  constexpr long long kTokens = 7;
  constexpr long long kHeads = 2;
  constexpr long long kSlots = 8;
  constexpr long long kPage = 4;
  constexpr long long kQueryHeads = 4;
  const long long groups = dim / 32;
  const long long elements = kTokens * kHeads * dim;
  const long long query_elements = kQueryHeads * dim;
  const long long key_bytes = (dim * key_bits + 7) / 8;
  const long long value_bytes = (dim * value_bits + 7) / 8;
  std::vector<float> key(static_cast<std::size_t>(elements));
  std::vector<float> value(key.size());
  std::vector<float> query(static_cast<std::size_t>(query_elements));
  std::vector<float> signs(static_cast<std::size_t>(dim));
  std::vector<float> centroids(static_cast<std::size_t>(1 << value_bits));
  for (long long i = 0; i < elements; ++i) {
    key[i] = 0.8f * std::sin(0.017f * static_cast<float>(i + 1));
    value[i] = 0.7f * std::cos(0.021f * static_cast<float>(i + 3));
  }
  for (long long i = 0; i < query_elements; ++i) {
    query[i] = 0.2f * std::sin(0.033f * static_cast<float>(i + 5));
  }
  for (long long i = 0; i < dim; ++i) {
    signs[i] = (i * 17 + 3) % 5 < 2 ? -1.0f : 1.0f;
  }
  for (std::size_t i = 0; i < centroids.size(); ++i) {
    centroids[i] = -2.5f + 5.0f * static_cast<float>(i) /
                               static_cast<float>(centroids.size() - 1);
  }
  const int slots[] = {0, 1, 2, 3, 4, 5, 6};
  const int block_table[] = {0, 1};
  const int context_lens[] = {7};
  std::vector<std::uint8_t> key_cache(
      static_cast<std::size_t>(kSlots * kHeads * key_bytes));
  std::vector<std::uint8_t> value_cache(
      static_cast<std::size_t>(kSlots * kHeads * value_bytes));
  std::vector<float> key_scale(
      static_cast<std::size_t>(kSlots * kHeads * groups));
  std::vector<float> value_scale(key_scale.size());
  std::vector<float> key_zero(key_scale.size());
  bool ok = true;
  ok &= require(
      quixicore_cpu::turboquant_encode(
          key.data(), value.data(), slots, centroids.data(), signs.data(),
          key_cache.data(), value_cache.data(), key_scale.data(),
          value_scale.data(), key_zero.data(), kTokens, kSlots, kHeads, dim,
          key_bits, key_signed, value_bits) == Status::kOk,
      "TurboQuant encode setup");
  std::vector<float> decoded_key(static_cast<std::size_t>(elements));
  std::vector<float> decoded_value(decoded_key.size());
  ok &= require(quixicore_cpu::turboquant_decode(
                    key_cache.data(), value_cache.data(), key_scale.data(),
                    value_scale.data(), key_zero.data(), slots,
                    centroids.data(), signs.data(), decoded_key.data(),
                    decoded_value.data(), kSlots, kTokens, kHeads, dim,
                    key_bits, key_signed, value_bits) == Status::kOk,
                "TurboQuant decode oracle");

  std::vector<float> reference(static_cast<std::size_t>(query_elements));
  std::vector<float> actual(reference.size());
  ok &=
      require(quixicore_cpu::paged_attention(
                  query.data(), decoded_key.data(), decoded_value.data(),
                  block_table, context_lens, reference.data(), 2, 1,
                  kQueryHeads, kHeads, dim, kPage, 2) == Status::kOk &&
                  quixicore_cpu::paged_attention_turboquant(
                      query.data(), key_cache.data(), value_cache.data(),
                      key_scale.data(), value_scale.data(), key_zero.data(),
                      centroids.data(), signs.data(), block_table, context_lens,
                      actual.data(), 2, 1, kQueryHeads, kHeads, dim, kPage, 2,
                      key_bits, key_signed, value_bits) == Status::kOk,
              "direct TurboQuant attention");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ok &= require(std::fabs(actual[i] - reference[i]) <=
                      3e-5f + 3e-4f * std::fabs(reference[i]),
                  "TurboQuant rotated-domain attention oracle");
  }

  std::vector<float> transform_input(static_cast<std::size_t>(dim));
  std::vector<float> transform_expected(transform_input.size());
  std::vector<float> transform_actual(transform_input.size());
  std::copy_n(query.data(), dim, transform_input.data());
  for (long long i = 0; i < dim; ++i) {
    transform_expected[i] = transform_input[i] * signs[i];
  }
  fwht(&transform_expected);
  const float normalization = 1.0f / std::sqrt(static_cast<float>(dim));
  for (float& item : transform_expected) item *= normalization;
  ok &= require(quixicore_cpu::turboquant_query_transform(
                    transform_input.data(), signs.data(),
                    transform_actual.data(), 1, 1, dim) == Status::kOk,
                "TurboQuant query transform");
  for (long long i = 0; i < dim; ++i) {
    ok &= require(transform_actual[i] == transform_expected[i],
                  "TurboQuant normalized signed FWHT");
  }

  if (typed_matrix) {
    const FloatStorageType types[] = {FloatStorageType::kF32,
                                      FloatStorageType::kF16,
                                      FloatStorageType::kBF16};
    for (FloatStorageType input_type : types) {
      const auto key_packed = encode_storage(key, input_type);
      const auto value_packed = encode_storage(value, input_type);
      const auto rounded_key = decode_storage(
          storage_data(key, key_packed, input_type), input_type, key.size());
      const auto rounded_value =
          decode_storage(storage_data(value, value_packed, input_type),
                         input_type, value.size());
      std::vector<std::uint8_t> typed_key_cache(key_cache.size());
      std::vector<std::uint8_t> typed_value_cache(value_cache.size());
      std::vector<std::uint8_t> rounded_key_cache(key_cache.size());
      std::vector<std::uint8_t> rounded_value_cache(value_cache.size());
      std::vector<float> typed_key_scale(key_scale.size());
      std::vector<float> typed_value_scale(value_scale.size());
      std::vector<float> typed_key_zero(key_zero.size());
      std::vector<float> rounded_key_scale(key_scale.size());
      std::vector<float> rounded_value_scale(value_scale.size());
      std::vector<float> rounded_key_zero(key_zero.size());
      ok &= require(
          quixicore_cpu::turboquant_encode_storage(
              FloatStorageInput{storage_data(key, key_packed, input_type),
                                input_type, elements},
              FloatStorageInput{storage_data(value, value_packed, input_type),
                                input_type, elements},
              slots, centroids.data(), signs.data(), typed_key_cache.data(),
              typed_value_cache.data(), typed_key_scale.data(),
              typed_value_scale.data(), typed_key_zero.data(), kTokens, kSlots,
              kHeads, dim, key_bits, key_signed, value_bits) == Status::kOk &&
              quixicore_cpu::turboquant_encode(
                  rounded_key.data(), rounded_value.data(), slots,
                  centroids.data(), signs.data(), rounded_key_cache.data(),
                  rounded_value_cache.data(), rounded_key_scale.data(),
                  rounded_value_scale.data(), rounded_key_zero.data(), kTokens,
                  kSlots, kHeads, dim, key_bits, key_signed,
                  value_bits) == Status::kOk,
          "typed TurboQuant encode");
      ok &= require(typed_key_cache == rounded_key_cache &&
                        typed_value_cache == rounded_value_cache &&
                        typed_key_scale == rounded_key_scale &&
                        typed_value_scale == rounded_value_scale &&
                        typed_key_zero == rounded_key_zero,
                    "typed TurboQuant encode oracle");

      std::vector<float> rounded_decoded_key(decoded_key.size());
      std::vector<float> rounded_decoded_value(decoded_value.size());
      ok &= require(quixicore_cpu::turboquant_decode(
                        typed_key_cache.data(), typed_value_cache.data(),
                        typed_key_scale.data(), typed_value_scale.data(),
                        typed_key_zero.data(), slots, centroids.data(),
                        signs.data(), rounded_decoded_key.data(),
                        rounded_decoded_value.data(), kSlots, kTokens, kHeads,
                        dim, key_bits, key_signed, value_bits) == Status::kOk,
                    "typed TurboQuant decode oracle");
      for (FloatStorageType output_type : types) {
        std::vector<float> output_key_f32;
        std::vector<float> output_value_f32;
        std::vector<std::uint16_t> output_key_16;
        std::vector<std::uint16_t> output_value_16;
        void* output_key = nullptr;
        void* output_value = nullptr;
        if (output_type == FloatStorageType::kF32) {
          output_key_f32.resize(decoded_key.size());
          output_value_f32.resize(decoded_value.size());
          output_key = output_key_f32.data();
          output_value = output_value_f32.data();
        } else {
          output_key_16.resize(decoded_key.size());
          output_value_16.resize(decoded_value.size());
          output_key = output_key_16.data();
          output_value = output_value_16.data();
        }
        ok &= require(
            quixicore_cpu::turboquant_decode_storage(
                typed_key_cache.data(), typed_value_cache.data(),
                typed_key_scale.data(), typed_value_scale.data(),
                typed_key_zero.data(), slots, centroids.data(), signs.data(),
                FloatStorageOutput{output_key, output_type, elements},
                FloatStorageOutput{output_value, output_type, elements}, kSlots,
                kTokens, kHeads, dim, key_bits, key_signed,
                value_bits) == Status::kOk,
            "typed TurboQuant decode");
        const auto typed_decoded_key =
            decode_storage(output_key, output_type, decoded_key.size());
        const auto typed_decoded_value =
            decode_storage(output_value, output_type, decoded_value.size());
        for (std::size_t i = 0; i < typed_decoded_key.size(); ++i) {
          float expected_key = rounded_decoded_key[i];
          float expected_value = rounded_decoded_value[i];
          if (output_type == FloatStorageType::kF16) {
            expected_key = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected_key));
            expected_value = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected_value));
          } else if (output_type == FloatStorageType::kBF16) {
            expected_key = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected_key));
            expected_value = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected_value));
          }
          ok &= require(typed_decoded_key[i] == expected_key &&
                            typed_decoded_value[i] == expected_value,
                        "typed TurboQuant decode output rounding");
        }
      }
    }
    for (FloatStorageType query_type : types) {
      const auto query_packed = encode_storage(query, query_type);
      const auto rounded_query =
          decode_storage(storage_data(query, query_packed, query_type),
                         query_type, query.size());
      ok &= require(
          quixicore_cpu::paged_attention(
              rounded_query.data(), decoded_key.data(), decoded_value.data(),
              block_table, context_lens, reference.data(), 2, 1, kQueryHeads,
              kHeads, dim, kPage, 2) == Status::kOk,
          "typed TurboQuant attention oracle");
      for (FloatStorageType output_type : types) {
        std::vector<float> output_f32;
        std::vector<std::uint16_t> output_16;
        void* output = nullptr;
        if (output_type == FloatStorageType::kF32) {
          output_f32.resize(query.size());
          output = output_f32.data();
        } else {
          output_16.resize(query.size());
          output = output_16.data();
        }
        ok &= require(
            quixicore_cpu::paged_attention_turboquant_storage(
                FloatStorageInput{storage_data(query, query_packed, query_type),
                                  query_type, query_elements},
                key_cache.data(), value_cache.data(), key_scale.data(),
                value_scale.data(), key_zero.data(), centroids.data(),
                signs.data(), block_table, context_lens,
                FloatStorageOutput{output, output_type, query_elements}, 2, 1,
                kQueryHeads, kHeads, dim, kPage, 2, key_bits, key_signed,
                value_bits) == Status::kOk,
            "typed TurboQuant attention");
        const auto typed_actual =
            decode_storage(output, output_type, reference.size());
        for (std::size_t i = 0; i < typed_actual.size(); ++i) {
          float expected = reference[i];
          if (output_type == FloatStorageType::kF16) {
            expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected));
          } else if (output_type == FloatStorageType::kBF16) {
            expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected));
          }
          ok &= require(std::fabs(typed_actual[i] - expected) <= 3e-5f,
                        "typed TurboQuant output rounding");
        }
      }
    }
  }
  return ok;
}

bool test_bitnet_kv3(long long dim,
                     const quixicore_cpu::BitNetKv3Config& config,
                     bool typed_matrix) {
  constexpr long long kTokens = 7;
  constexpr long long kHeads = 2;
  constexpr long long kSlots = 8;
  constexpr long long kPage = 4;
  constexpr long long kQueryHeads = 4;
  const long long groups = dim / config.group_size;
  const long long elements = kTokens * kHeads * dim;
  const long long query_elements = kQueryHeads * dim;
  const long long packed_bytes = (dim * 3 + 7) / 8;
  const long long metadata_elements = kSlots * kHeads * groups;
  const bool unsigned_symmetric =
      config.signedness == quixicore_cpu::BitNetKv3Signedness::kUnsigned &&
      config.zero_point_mode == quixicore_cpu::BitNetKv3ZeroPointMode::kNone;
  const bool integer_zero =
      config.zero_point_mode == quixicore_cpu::BitNetKv3ZeroPointMode::kInteger;
  std::vector<float> key(static_cast<std::size_t>(elements));
  std::vector<float> value(key.size());
  std::vector<float> query(static_cast<std::size_t>(query_elements));
  for (long long i = 0; i < elements; ++i) {
    key[i] = 0.8f * std::sin(0.017f * static_cast<float>(i + 1));
    value[i] = 0.7f * std::cos(0.021f * static_cast<float>(i + 3));
    if (unsigned_symmetric) {
      key[i] = std::fabs(key[i]);
      value[i] = std::fabs(value[i]);
    }
  }
  for (long long i = 0; i < query_elements; ++i) {
    query[i] = 0.2f * std::sin(0.033f * static_cast<float>(i + 5));
  }
  const int slots[] = {0, 1, 2, 3, 4, 5, 6};
  const int block_table[] = {0, 1};
  const int context_lens[] = {7};
  std::vector<std::uint8_t> key_cache(
      static_cast<std::size_t>(kSlots * kHeads * packed_bytes));
  std::vector<std::uint8_t> value_cache(key_cache.size());
  std::vector<float> key_scale_f32(static_cast<std::size_t>(metadata_elements));
  std::vector<float> value_scale_f32(key_scale_f32.size());
  std::vector<std::uint16_t> key_scale_f16(key_scale_f32.size());
  std::vector<std::uint16_t> value_scale_f16(key_scale_f32.size());
  void* key_scale =
      config.scale_type == quixicore_cpu::BitNetKv3ScaleType::kFP32
          ? static_cast<void*>(key_scale_f32.data())
          : static_cast<void*>(key_scale_f16.data());
  void* value_scale =
      config.scale_type == quixicore_cpu::BitNetKv3ScaleType::kFP32
          ? static_cast<void*>(value_scale_f32.data())
          : static_cast<void*>(value_scale_f16.data());
  std::vector<int> key_zero(static_cast<std::size_t>(metadata_elements));
  std::vector<int> value_zero(key_zero.size());
  int* key_zero_ptr = integer_zero ? key_zero.data() : nullptr;
  int* value_zero_ptr = integer_zero ? value_zero.data() : nullptr;
  bool ok = true;
  ok &= require(
      quixicore_cpu::kv_cache_scatter_bitnet_kv3(
          key.data(), value.data(), slots, key_cache.data(), value_cache.data(),
          key_scale, value_scale, key_zero_ptr, value_zero_ptr, kSlots, kTokens,
          kHeads, dim, config) == Status::kOk,
      "BitNet KV3 scatter");

  const auto scale_at = [&](const void* data, long long index) {
    return config.scale_type == quixicore_cpu::BitNetKv3ScaleType::kFP32
               ? static_cast<const float*>(data)[index]
               : quixicore_cpu::f16_to_float(
                     static_cast<const std::uint16_t*>(data)[index]);
  };
  const auto code_at = [](const std::uint8_t* data, long long element) {
    const long long bit = element * 3;
    const long long byte = bit / 8;
    const int shift = static_cast<int>(bit % 8);
    unsigned raw = data[byte];
    if (shift > 5) raw |= unsigned(data[byte + 1]) << 8;
    return (raw >> shift) & 7u;
  };
  const auto signed_code = [&](unsigned raw) {
    return config.signedness == quixicore_cpu::BitNetKv3Signedness::kSigned &&
                   raw >= 4
               ? static_cast<int>(raw) - 8
               : static_cast<int>(raw);
  };
  std::vector<float> expected_key(static_cast<std::size_t>(elements));
  std::vector<float> expected_value(expected_key.size());
  for (long long row = 0; row < kTokens; ++row) {
    for (long long head = 0; head < kHeads; ++head) {
      const long long cache_row = slots[row] * kHeads + head;
      const long long output = (row * kHeads + head) * dim;
      const long long metadata = cache_row * groups;
      const auto* key_codes = key_cache.data() + cache_row * packed_bytes;
      const auto* value_codes = value_cache.data() + cache_row * packed_bytes;
      for (long long column = 0; column < dim; ++column) {
        const long long group = column / config.group_size;
        const int kz = integer_zero ? key_zero[metadata + group] : 0;
        const int vz = integer_zero ? value_zero[metadata + group] : 0;
        expected_key[output + column] =
            (signed_code(code_at(key_codes, column)) - kz) *
            scale_at(key_scale, metadata + group);
        expected_value[output + column] =
            (signed_code(code_at(value_codes, column)) - vz) *
            scale_at(value_scale, metadata + group);
      }
    }
  }
  std::vector<float> decoded_key(expected_key.size());
  std::vector<float> decoded_value(expected_value.size());
  ok &= require(quixicore_cpu::kv_cache_gather_bitnet_kv3(
                    key_cache.data(), value_cache.data(), slots, key_scale,
                    value_scale, key_zero_ptr, value_zero_ptr,
                    decoded_key.data(), decoded_value.data(), kSlots, kTokens,
                    kHeads, dim, config) == Status::kOk,
                "BitNet KV3 gather");
  ok &= require(decoded_key == expected_key && decoded_value == expected_value,
                "BitNet KV3 independent decode oracle");

  std::vector<float> reference(static_cast<std::size_t>(query_elements));
  std::vector<float> actual(reference.size());
  ok &= require(
      quixicore_cpu::paged_attention(
          query.data(), decoded_key.data(), decoded_value.data(), block_table,
          context_lens, reference.data(), 2, 1, kQueryHeads, kHeads, dim, kPage,
          2) == Status::kOk &&
          quixicore_cpu::paged_attention_bitnet_kv3(
              query.data(), key_cache.data(), value_cache.data(), key_scale,
              value_scale, key_zero_ptr, value_zero_ptr, block_table,
              context_lens, actual.data(), 2, 1, kQueryHeads, kHeads, dim,
              kPage, 2, config) == Status::kOk,
      "direct BitNet KV3 attention");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ok &= require(std::fabs(actual[i] - reference[i]) <=
                      3e-5f + 3e-4f * std::fabs(reference[i]),
                  "BitNet KV3 direct attention oracle");
  }

  if (typed_matrix) {
    const FloatStorageType types[] = {FloatStorageType::kF32,
                                      FloatStorageType::kF16,
                                      FloatStorageType::kBF16};
    for (FloatStorageType input_type : types) {
      const auto key_packed = encode_storage(key, input_type);
      const auto value_packed = encode_storage(value, input_type);
      const auto rounded_key = decode_storage(
          storage_data(key, key_packed, input_type), input_type, key.size());
      const auto rounded_value =
          decode_storage(storage_data(value, value_packed, input_type),
                         input_type, value.size());
      std::vector<std::uint8_t> typed_key_cache(key_cache.size());
      std::vector<std::uint8_t> typed_value_cache(value_cache.size());
      std::vector<std::uint8_t> rounded_key_cache(key_cache.size());
      std::vector<std::uint8_t> rounded_value_cache(value_cache.size());
      std::vector<std::uint16_t> typed_key_scale(key_scale_f16.size());
      std::vector<std::uint16_t> typed_value_scale(value_scale_f16.size());
      std::vector<std::uint16_t> rounded_key_scale(key_scale_f16.size());
      std::vector<std::uint16_t> rounded_value_scale(value_scale_f16.size());
      ok &= require(
          quixicore_cpu::kv_cache_scatter_bitnet_kv3_storage(
              FloatStorageInput{storage_data(key, key_packed, input_type),
                                input_type, elements},
              FloatStorageInput{storage_data(value, value_packed, input_type),
                                input_type, elements},
              slots, typed_key_cache.data(), typed_value_cache.data(),
              typed_key_scale.data(), typed_value_scale.data(), nullptr,
              nullptr, kSlots, kTokens, kHeads, dim, config) == Status::kOk &&
              quixicore_cpu::kv_cache_scatter_bitnet_kv3(
                  rounded_key.data(), rounded_value.data(), slots,
                  rounded_key_cache.data(), rounded_value_cache.data(),
                  rounded_key_scale.data(), rounded_value_scale.data(), nullptr,
                  nullptr, kSlots, kTokens, kHeads, dim, config) == Status::kOk,
          "typed BitNet KV3 scatter");
      ok &= require(typed_key_cache == rounded_key_cache &&
                        typed_value_cache == rounded_value_cache &&
                        typed_key_scale == rounded_key_scale &&
                        typed_value_scale == rounded_value_scale,
                    "typed BitNet KV3 scatter oracle");
      std::vector<float> typed_reference_key(expected_key.size());
      std::vector<float> typed_reference_value(expected_value.size());
      ok &= require(
          quixicore_cpu::kv_cache_gather_bitnet_kv3(
              typed_key_cache.data(), typed_value_cache.data(), slots,
              typed_key_scale.data(), typed_value_scale.data(), nullptr,
              nullptr, typed_reference_key.data(), typed_reference_value.data(),
              kSlots, kTokens, kHeads, dim, config) == Status::kOk,
          "typed BitNet KV3 gather oracle");
      for (FloatStorageType output_type : types) {
        std::vector<float> output_key_f32;
        std::vector<float> output_value_f32;
        std::vector<std::uint16_t> output_key_16;
        std::vector<std::uint16_t> output_value_16;
        void* output_key = nullptr;
        void* output_value = nullptr;
        if (output_type == FloatStorageType::kF32) {
          output_key_f32.resize(expected_key.size());
          output_value_f32.resize(expected_value.size());
          output_key = output_key_f32.data();
          output_value = output_value_f32.data();
        } else {
          output_key_16.resize(expected_key.size());
          output_value_16.resize(expected_value.size());
          output_key = output_key_16.data();
          output_value = output_value_16.data();
        }
        ok &= require(
            quixicore_cpu::kv_cache_gather_bitnet_kv3_storage(
                typed_key_cache.data(), typed_value_cache.data(), slots,
                typed_key_scale.data(), typed_value_scale.data(), nullptr,
                nullptr, FloatStorageOutput{output_key, output_type, elements},
                FloatStorageOutput{output_value, output_type, elements}, kSlots,
                kTokens, kHeads, dim, config) == Status::kOk,
            "typed BitNet KV3 gather");
        const auto typed_key =
            decode_storage(output_key, output_type, expected_key.size());
        const auto typed_value =
            decode_storage(output_value, output_type, expected_value.size());
        for (std::size_t i = 0; i < typed_key.size(); ++i) {
          float key_expected = typed_reference_key[i];
          float value_expected = typed_reference_value[i];
          if (output_type == FloatStorageType::kF16) {
            key_expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(key_expected));
            value_expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(value_expected));
          } else if (output_type == FloatStorageType::kBF16) {
            key_expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(key_expected));
            value_expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(value_expected));
          }
          ok &= require(
              typed_key[i] == key_expected && typed_value[i] == value_expected,
              "typed BitNet KV3 gather rounding");
        }
      }
    }

    for (FloatStorageType query_type : types) {
      const auto query_packed = encode_storage(query, query_type);
      const auto rounded_query =
          decode_storage(storage_data(query, query_packed, query_type),
                         query_type, query.size());
      ok &= require(
          quixicore_cpu::paged_attention(
              rounded_query.data(), decoded_key.data(), decoded_value.data(),
              block_table, context_lens, reference.data(), 2, 1, kQueryHeads,
              kHeads, dim, kPage, 2) == Status::kOk,
          "typed BitNet KV3 attention oracle");
      for (FloatStorageType output_type : types) {
        std::vector<float> output_f32;
        std::vector<std::uint16_t> output_16;
        void* output = nullptr;
        if (output_type == FloatStorageType::kF32) {
          output_f32.resize(reference.size());
          output = output_f32.data();
        } else {
          output_16.resize(reference.size());
          output = output_16.data();
        }
        ok &= require(
            quixicore_cpu::paged_attention_bitnet_kv3_storage(
                FloatStorageInput{storage_data(query, query_packed, query_type),
                                  query_type, query_elements},
                key_cache.data(), value_cache.data(), key_scale, value_scale,
                key_zero_ptr, value_zero_ptr, block_table, context_lens,
                FloatStorageOutput{output, output_type, query_elements}, 2, 1,
                kQueryHeads, kHeads, dim, kPage, 2, config) == Status::kOk,
            "typed BitNet KV3 attention");
        const auto typed_actual =
            decode_storage(output, output_type, reference.size());
        for (std::size_t i = 0; i < typed_actual.size(); ++i) {
          float expected = reference[i];
          if (output_type == FloatStorageType::kF16) {
            expected = quixicore_cpu::f16_to_float(
                quixicore_cpu::float_to_f16(expected));
          } else if (output_type == FloatStorageType::kBF16) {
            expected = quixicore_cpu::bf16_to_float(
                quixicore_cpu::float_to_bf16(expected));
          }
          ok &= require(std::fabs(typed_actual[i] - expected) <= 3e-5f,
                        "typed BitNet KV3 attention rounding");
        }
      }
    }
  }
  return ok;
}

bool test_q8_0_kv() {
  constexpr long long kBlocks = 4;
  constexpr long long kPage = 4;
  constexpr long long kSlots = kBlocks * kPage;
  constexpr long long kTokens = 11;
  constexpr long long kHeads = 3;
  constexpr long long kDim = 64;
  constexpr long long kGroups = kDim / 32;
  constexpr long long kElements = kTokens * kHeads * kDim;
  constexpr long long kCacheElements = kSlots * kHeads * kDim;
  constexpr long long kScaleElements = kSlots * kHeads * kGroups;
  std::vector<float> key(kElements), value(kElements);
  for (long long i = 0; i < kElements; ++i) {
    key[i] = 0.4f * std::sin(0.031f * static_cast<float>(i + 1));
    value[i] = 0.4f * std::cos(0.027f * static_cast<float>(i + 3));
  }
  const int slots[] = {0, 1, 2, -1, 4, 5, 7, 8, 9, 10, 14};
  const FloatStorageType types[] = {
      FloatStorageType::kF32, FloatStorageType::kF16, FloatStorageType::kBF16};
  bool ok = true;
  for (const FloatStorageType type : types) {
    const auto key_packed = encode_storage(key, type);
    const auto value_packed = encode_storage(value, type);
    const auto rounded_key =
        decode_storage(storage_data(key, key_packed, type), type, key.size());
    const auto rounded_value = decode_storage(
        storage_data(value, value_packed, type), type, value.size());
    std::vector<std::int8_t> key_codes(kCacheElements),
        value_codes(kCacheElements);
    std::vector<std::uint16_t> key_scales(kScaleElements),
        value_scales(kScaleElements);
    ok &= require(quixicore_cpu::kv_cache_scatter_q8_0_storage(
                      FloatStorageInput{storage_data(key, key_packed, type),
                                        type, kElements},
                      FloatStorageInput{storage_data(value, value_packed, type),
                                        type, kElements},
                      slots, key_codes.data(), key_scales.data(),
                      value_codes.data(), value_scales.data(), kBlocks, kTokens,
                      kHeads, kDim, kPage) == Status::kOk,
                  "Q8_0 typed scatter");

    std::vector<std::int8_t> expected_key_codes(kCacheElements, 0),
        expected_value_codes(kCacheElements, 0);
    std::vector<std::uint16_t> expected_key_scales(kScaleElements, 0),
        expected_value_scales(kScaleElements, 0);
    for (long long token = 0; token < kTokens; ++token) {
      if (slots[token] < 0) continue;
      for (long long head = 0; head < kHeads; ++head) {
        for (long long group = 0; group < kGroups; ++group) {
          float key_amax = 0.0f, value_amax = 0.0f;
          for (long long lane = 0; lane < 32; ++lane) {
            const long long source =
                (token * kHeads + head) * kDim + group * 32 + lane;
            key_amax = std::max(key_amax, std::fabs(rounded_key[source]));
            value_amax = std::max(value_amax, std::fabs(rounded_value[source]));
          }
          const float key_scale = key_amax / 127.0f;
          const float value_scale = value_amax / 127.0f;
          const float key_inverse = key_scale > 0.0f ? 1.0f / key_scale : 0.0f;
          const float value_inverse =
              value_scale > 0.0f ? 1.0f / value_scale : 0.0f;
          const long long scale =
              (static_cast<long long>(slots[token]) * kHeads + head) * kGroups +
              group;
          expected_key_scales[scale] = quixicore_cpu::float_to_f16(key_scale);
          expected_value_scales[scale] =
              quixicore_cpu::float_to_f16(value_scale);
          for (long long lane = 0; lane < 32; ++lane) {
            const long long source =
                (token * kHeads + head) * kDim + group * 32 + lane;
            const long long destination =
                (static_cast<long long>(slots[token]) * kHeads + head) * kDim +
                group * 32 + lane;
            const auto encode = [](float number, float inverse) {
              const float rounded = std::copysign(
                  std::floor(std::fabs(number * inverse) + 0.5f), number);
              return static_cast<std::int8_t>(
                  std::clamp(rounded, -127.0f, 127.0f));
            };
            expected_key_codes[destination] =
                encode(rounded_key[source], key_inverse);
            expected_value_codes[destination] =
                encode(rounded_value[source], value_inverse);
          }
        }
      }
    }
    ok &= require(key_codes == expected_key_codes, "Q8_0 exact key codes");
    ok &=
        require(value_codes == expected_value_codes, "Q8_0 exact value codes");
    ok &=
        require(key_scales == expected_key_scales, "Q8_0 exact key scale bits");
    ok &= require(value_scales == expected_value_scales,
                  "Q8_0 exact value scale bits");

    const int block_table[] = {0, 1, -1};
    const int cumulative[] = {0, 12};
    std::vector<float> key_out(12 * kHeads * kDim), value_out(key_out.size());
    ok &= require(quixicore_cpu::kv_cache_gather_q8_0(
                      key_codes.data(), key_scales.data(), value_codes.data(),
                      value_scales.data(), block_table, cumulative,
                      key_out.data(), value_out.data(), kBlocks, 12, 1, kHeads,
                      kDim, kPage, 3) == Status::kOk,
                  "Q8_0 gather");
    for (long long token = 0; token < 12; ++token) {
      const int block = block_table[token / kPage];
      for (long long head = 0; head < kHeads; ++head) {
        for (long long dim = 0; dim < kDim; ++dim) {
          const long long output = (token * kHeads + head) * kDim + dim;
          float expected_key = 0.0f, expected_value = 0.0f;
          if (block >= 0) {
            const long long slot = block * kPage + token % kPage;
            const long long code = (slot * kHeads + head) * kDim + dim;
            const long long scale = (slot * kHeads + head) * kGroups + dim / 32;
            expected_key = key_codes[code] *
                           quixicore_cpu::f16_to_float(key_scales[scale]);
            expected_value = value_codes[code] *
                             quixicore_cpu::f16_to_float(value_scales[scale]);
          }
          ok &= require(key_out[output] == expected_key,
                        "Q8_0 gather key decode");
          ok &= require(value_out[output] == expected_value,
                        "Q8_0 gather value decode");
        }
      }
    }

    const long long pairs[] = {0, 3, 1, 2};
    std::vector<std::int8_t> copied_key(kCacheElements),
        copied_value(kCacheElements);
    std::vector<std::uint16_t> copied_key_scales(kScaleElements),
        copied_value_scales(kScaleElements);
    ok &= require(
        quixicore_cpu::kv_cache_copy_blocks_q8_0(
            key_codes.data(), key_scales.data(), value_codes.data(),
            value_scales.data(), copied_key.data(), copied_key_scales.data(),
            copied_value.data(), copied_value_scales.data(), pairs, 2, kBlocks,
            kPage, kHeads, kDim) == Status::kOk,
        "Q8_0 functional block copy");
    const long long codes_per_block = kPage * kHeads * kDim;
    const long long scales_per_block = kPage * kHeads * kGroups;
    ok &= require(
        std::equal(copied_key.begin() + 3 * codes_per_block,
                   copied_key.begin() + 4 * codes_per_block, key_codes.begin()),
        "Q8_0 copied code block");
    ok &= require(std::equal(copied_key_scales.begin() + 2 * scales_per_block,
                             copied_key_scales.begin() + 3 * scales_per_block,
                             key_scales.begin() + scales_per_block),
                  "Q8_0 copied scale block");

    if (type != FloatStorageType::kF32) continue;
    constexpr long long kQueryHeads = 6;
    constexpr long long kBatch = 2;
    std::vector<float> query(kBatch * kQueryHeads * kDim),
        attention_out(query.size()), attention_reference(query.size());
    for (std::size_t i = 0; i < query.size(); ++i) {
      query[i] = 0.2f * std::sin(0.019f * static_cast<float>(i + 5));
    }
    const int attention_blocks[] = {0, 1, 2, 3};
    const int context_lens[] = {7, 5};
    ok &= require(quixicore_cpu::paged_attention_q8_0(
                      query.data(), key_codes.data(), key_scales.data(),
                      value_codes.data(), value_scales.data(), attention_blocks,
                      context_lens, attention_out.data(), kBlocks, kBatch,
                      kQueryHeads, kHeads, kDim, kPage, 2) == Status::kOk,
                  "Q8_0 paged attention");
    const double score_scale = 1.0 / std::sqrt(static_cast<double>(kDim));
    for (long long request = 0; request < kBatch; ++request) {
      for (long long head = 0; head < kQueryHeads; ++head) {
        const long long kv_head = head / (kQueryHeads / kHeads);
        std::vector<double> probabilities(context_lens[request]);
        double maximum = -std::numeric_limits<double>::infinity();
        for (long long position = 0; position < context_lens[request];
             ++position) {
          const int block = attention_blocks[request * 2 + position / kPage];
          const long long slot = block * kPage + position % kPage;
          double dot = 0.0;
          for (long long dim = 0; dim < kDim; ++dim) {
            const long long code = (slot * kHeads + kv_head) * kDim + dim;
            const long long scale =
                (slot * kHeads + kv_head) * kGroups + dim / 32;
            dot += query[(request * kQueryHeads + head) * kDim + dim] *
                   key_codes[code] *
                   quixicore_cpu::f16_to_float(key_scales[scale]);
          }
          probabilities[position] = dot * score_scale;
          maximum = std::max(maximum, probabilities[position]);
        }
        double denominator = 0.0;
        for (double& probability : probabilities) {
          probability = std::exp(probability - maximum);
          denominator += probability;
        }
        for (long long position = 0; position < context_lens[request];
             ++position) {
          const int block = attention_blocks[request * 2 + position / kPage];
          const long long slot = block * kPage + position % kPage;
          for (long long dim = 0; dim < kDim; ++dim) {
            const long long code = (slot * kHeads + kv_head) * kDim + dim;
            const long long scale =
                (slot * kHeads + kv_head) * kGroups + dim / 32;
            attention_reference[(request * kQueryHeads + head) * kDim + dim] +=
                static_cast<float>(
                    probabilities[position] / denominator * value_codes[code] *
                    quixicore_cpu::f16_to_float(value_scales[scale]));
          }
        }
      }
    }
    for (std::size_t i = 0; i < attention_out.size(); ++i) {
      ok &=
          require(std::fabs(attention_out[i] - attention_reference[i]) <= 2e-5f,
                  "Q8_0 paged attention oracle");
    }
    const int sparse_blocks[] = {-1, -1, -1, -1};
    std::fill(attention_out.begin(), attention_out.end(), 1.0f);
    ok &= require(quixicore_cpu::paged_attention_q8_0(
                      query.data(), key_codes.data(), key_scales.data(),
                      value_codes.data(), value_scales.data(), sparse_blocks,
                      context_lens, attention_out.data(), kBlocks, kBatch,
                      kQueryHeads, kHeads, kDim, kPage, 2) == Status::kOk,
                  "Q8_0 sparse paged attention");
    ok &= require(std::all_of(attention_out.begin(), attention_out.end(),
                              [](float value) { return value == 0.0f; }),
                  "Q8_0 sparse pages zero-fill");
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_format(Float8Format::kE4M3FN, 64);
  ok &= test_format(Float8Format::kE5M2, 64);
  ok &= test_format(Float8Format::kE4M3FN, 128);
  ok &= test_format(Float8Format::kE5M2, 128);
  ok &= test_zero_and_validation();
  ok &= test_mxfp8(64);
  ok &= test_mxfp8(128);
  ok &= test_q8_0_kv();
  for (int bits = 2; bits <= 8; ++bits) {
    ok &= test_turboquant(64, bits, false, bits, bits == 4);
  }
  ok &= test_turboquant(64, 8, true, 4, false);
  ok &= test_turboquant(128, 4, false, 4, false);
  ok &= test_turboquant(256, 4, false, 4, false);
  ok &= test_bitnet_kv3(64,
                        {32, quixicore_cpu::BitNetKv3ScaleType::kFP16,
                         quixicore_cpu::BitNetKv3Signedness::kSigned,
                         quixicore_cpu::BitNetKv3ZeroPointMode::kNone},
                        true);
  ok &= test_bitnet_kv3(64,
                        {16, quixicore_cpu::BitNetKv3ScaleType::kFP32,
                         quixicore_cpu::BitNetKv3Signedness::kUnsigned,
                         quixicore_cpu::BitNetKv3ZeroPointMode::kNone},
                        false);
  ok &= test_bitnet_kv3(128,
                        {32, quixicore_cpu::BitNetKv3ScaleType::kFP32,
                         quixicore_cpu::BitNetKv3Signedness::kUnsigned,
                         quixicore_cpu::BitNetKv3ZeroPointMode::kInteger},
                        false);
  ok &= test_bitnet_kv3(256,
                        {64, quixicore_cpu::BitNetKv3ScaleType::kFP16,
                         quixicore_cpu::BitNetKv3Signedness::kSigned,
                         quixicore_cpu::BitNetKv3ZeroPointMode::kInteger},
                        false);
  if (!ok) return 1;
  std::cout << "quantized cache attention tests passed\n";
  return 0;
}
