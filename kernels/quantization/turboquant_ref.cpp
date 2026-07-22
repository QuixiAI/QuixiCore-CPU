#include "quixicore_cpu/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"

namespace quixicore_cpu {
namespace {

float half_round(float value) { return fp16_to_fp32(fp32_to_fp16(value)); }

unsigned unpack_bits(const std::uint8_t* bytes, long long element, int bits) {
  const long long bit_position = element * bits;
  const long long byte = bit_position / 8;
  const int offset = static_cast<int>(bit_position % 8);
  unsigned raw = bytes[byte];
  if (offset + bits > 8) raw |= unsigned(bytes[byte + 1]) << 8;
  return (raw >> offset) & ((1u << bits) - 1u);
}

void pack_bits(const std::vector<unsigned>& values, int bits,
               std::uint8_t* bytes) {
  const long long count = static_cast<long long>(values.size());
  const long long byte_count = (count * bits + 7) / 8;
  std::fill_n(bytes, byte_count, std::uint8_t{0});
  const unsigned mask = (1u << bits) - 1u;
  for (long long element = 0; element < count; ++element) {
    const long long bit_position = element * bits;
    const long long byte = bit_position / 8;
    const int offset = static_cast<int>(bit_position % 8);
    const unsigned value = values[element] & mask;
    bytes[byte] |= static_cast<std::uint8_t>(value << offset);
    if (offset + bits > 8) {
      bytes[byte + 1] |= static_cast<std::uint8_t>(value >> (8 - offset));
    }
  }
}

void fwht(std::vector<float>* values) {
  const long long count = static_cast<long long>(values->size());
  for (long long width = 1; width < count; width *= 2) {
    for (long long base = 0; base < count; base += 2 * width) {
      for (long long i = 0; i < width; ++i) {
        const float a = (*values)[base + i];
        const float b = (*values)[base + width + i];
        (*values)[base + i] = a + b;
        (*values)[base + width + i] = a - b;
      }
    }
  }
}

bool valid_tq(long long head_size, int key_bits, int value_bits) {
  return (head_size == 64 || head_size == 128 || head_size == 256) &&
         key_bits >= 2 && key_bits <= 8 && value_bits >= 2 && value_bits <= 8;
}

}  // namespace

Status turboquant_packed_bytes(long long head_size, int bits,
                               long long* packed_bytes) {
  if (head_size <= 0 || bits < 2 || bits > 8) return Status::kInvalidShape;
  if (packed_bytes == nullptr) return Status::kInvalidArgument;
  *packed_bytes = (head_size * bits + 7) / 8;
  return Status::kOk;
}

Status turboquant_encode(
    const float* key, const float* value, const int* slot_mapping,
    const float* value_centroids, const float* signs,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    float* key_scale_cache, float* value_scale_cache,
    float* key_zero_cache, long long tokens, long long slots,
    long long heads, long long head_size, int key_bits,
    bool key_signed, int value_bits) {
  if (!detail::valid_product({tokens, slots, heads, head_size}) ||
      !valid_tq(head_size, key_bits, value_bits)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key, value, slot_mapping, value_centroids, signs,
                           key_cache, value_cache, key_scale_cache,
                           value_scale_cache, key_zero_cache)) {
    return Status::kInvalidArgument;
  }
  const int centroid_count = 1 << value_bits;
  for (int i = 0; i < centroid_count; ++i) {
    if (!std::isfinite(value_centroids[i]) ||
        (i > 0 && value_centroids[i] < value_centroids[i - 1])) {
      return Status::kInvalidArgument;
    }
  }
  for (long long i = 0; i < head_size; ++i) {
    if (!std::isfinite(signs[i])) return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (slot_mapping[token] >= slots) return Status::kInvalidArgument;
  }
  const long long groups = head_size / 32;
  const long long key_bytes = (head_size * key_bits + 7) / 8;
  const long long value_bytes = (head_size * value_bits + 7) / 8;
  const float normalization = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (long long token = 0; token < tokens; ++token) {
    const int slot = slot_mapping[token];
    if (slot < 0) continue;
    for (long long head = 0; head < heads; ++head) {
      const long long source_base = (token * heads + head) * head_size;
      const long long scale_base = (static_cast<long long>(slot) * heads + head) * groups;
      std::vector<unsigned> key_codes(static_cast<std::size_t>(head_size));
      for (long long group = 0; group < groups; ++group) {
        const float* source = key + source_base + group * 32;
        const auto bounds = std::minmax_element(source, source + 32);
        float scale;
        float zero;
        if (key_signed) {
          const int maximum = (1 << (key_bits - 1)) - 1;
          scale = half_round(half_round(*bounds.second - *bounds.first) /
                             half_round(2.0f * maximum));
          zero = scale != 0.0f
                     ? std::nearbyint(half_round(*bounds.second + *bounds.first) /
                                      half_round(2.0f * scale))
                     : 0.0f;
          for (int i = 0; i < 32; ++i) {
            const int code = scale != 0.0f
                                 ? std::clamp(static_cast<int>(std::nearbyint(
                                       half_round(source[i]) / scale - zero)),
                                              -maximum, maximum)
                                 : 0;
            key_codes[group * 32 + i] =
                static_cast<unsigned>(code) & ((1u << key_bits) - 1u);
          }
        } else {
          const int maximum = (1 << key_bits) - 1;
          scale = half_round(half_round(*bounds.second - *bounds.first) /
                             half_round(static_cast<float>(maximum)));
          zero = scale != 0.0f
                     ? std::nearbyint(half_round(*bounds.first) / scale)
                     : 0.0f;
          for (int i = 0; i < 32; ++i) {
            const int code = scale != 0.0f
                                 ? std::clamp(static_cast<int>(std::nearbyint(
                                       half_round(source[i]) / scale - zero)),
                                              0, maximum)
                                 : 0;
            key_codes[group * 32 + i] = static_cast<unsigned>(code);
          }
        }
        key_scale_cache[scale_base + group] = scale;
        key_zero_cache[scale_base + group] = half_round(zero);
      }
      pack_bits(key_codes, key_bits,
                key_cache + (static_cast<long long>(slot) * heads + head) * key_bytes);

      std::vector<float> rotated(static_cast<std::size_t>(head_size));
      for (long long i = 0; i < head_size; ++i) {
        rotated[i] = value[source_base + i] * signs[i];
      }
      fwht(&rotated);
      for (float& item : rotated) item = half_round(item * normalization);
      std::vector<unsigned> value_codes(static_cast<std::size_t>(head_size));
      for (long long group = 0; group < groups; ++group) {
        double square_sum = 0.0;
        for (int i = 0; i < 32; ++i) {
          square_sum += rotated[group * 32 + i] * rotated[group * 32 + i];
        }
        const float scale = half_round(std::sqrt(square_sum / 32.0));
        value_scale_cache[scale_base + group] = scale;
        for (int i = 0; i < 32; ++i) {
          const float normalized = scale != 0.0f
                                       ? half_round(rotated[group * 32 + i] / scale)
                                       : 0.0f;
          int index = 0;
          while (index + 1 < centroid_count &&
                 normalized > 0.5f *
                     (value_centroids[index] + value_centroids[index + 1])) {
            ++index;
          }
          value_codes[group * 32 + i] = static_cast<unsigned>(index);
        }
      }
      pack_bits(value_codes, value_bits,
                value_cache + (static_cast<long long>(slot) * heads + head) * value_bytes);
    }
  }
  return Status::kOk;
}

Status turboquant_decode(
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const float* key_scale_cache, const float* value_scale_cache,
    const float* key_zero_cache, const int* slots_to_gather,
    const float* value_centroids, const float* signs, float* key_out,
    float* value_out, long long cache_slots, long long rows,
    long long heads, long long head_size, int key_bits,
    bool key_signed, int value_bits) {
  if (!detail::valid_product({cache_slots, rows, heads, head_size}) ||
      !valid_tq(head_size, key_bits, value_bits)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(key_cache, value_cache, key_scale_cache,
                           value_scale_cache, key_zero_cache, slots_to_gather,
                           value_centroids, signs, key_out, value_out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (slots_to_gather[row] < 0 || slots_to_gather[row] >= cache_slots) {
      return Status::kInvalidArgument;
    }
  }
  const long long groups = head_size / 32;
  const long long key_bytes = (head_size * key_bits + 7) / 8;
  const long long value_bytes = (head_size * value_bits + 7) / 8;
  const float normalization = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (long long row = 0; row < rows; ++row) {
    const long long slot = slots_to_gather[row];
    for (long long head = 0; head < heads; ++head) {
      const std::uint8_t* keys =
          key_cache + (slot * heads + head) * key_bytes;
      const std::uint8_t* values =
          value_cache + (slot * heads + head) * value_bytes;
      const long long scale_base = (slot * heads + head) * groups;
      const long long output_base = (row * heads + head) * head_size;
      std::vector<float> rotated(static_cast<std::size_t>(head_size));
      for (long long i = 0; i < head_size; ++i) {
        unsigned key_code = unpack_bits(keys, i, key_bits);
        float q;
        if (key_signed && key_bits == 8) {
          q = static_cast<std::int8_t>(key_code);
        } else {
          q = static_cast<float>(key_code);
        }
        const long long group = i / 32;
        key_out[output_base + i] =
            (q + key_zero_cache[scale_base + group]) *
            key_scale_cache[scale_base + group];
        const unsigned value_code = unpack_bits(values, i, value_bits);
        rotated[i] = value_centroids[value_code] *
                     value_scale_cache[scale_base + group];
      }
      fwht(&rotated);
      for (long long i = 0; i < head_size; ++i) {
        value_out[output_base + i] = rotated[i] * normalization * signs[i];
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
