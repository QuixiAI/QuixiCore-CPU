#pragma once

#include <cstdint>

#include "quixicore_cpu/float_storage.h"

namespace quixicore_cpu::detail {

template <FloatStorageType Type>
inline float load_storage(const void* data, long long index);

template <>
inline float load_storage<FloatStorageType::kF32>(const void* data,
                                                   long long index) {
  return static_cast<const float*>(data)[index];
}

template <>
inline float load_storage<FloatStorageType::kF16>(const void* data,
                                                   long long index) {
  return f16_to_float(static_cast<const std::uint16_t*>(data)[index]);
}

template <>
inline float load_storage<FloatStorageType::kBF16>(const void* data,
                                                    long long index) {
  return bf16_to_float(static_cast<const std::uint16_t*>(data)[index]);
}

template <FloatStorageType Type>
inline void store_storage(void* data, long long index, float value);

template <>
inline void store_storage<FloatStorageType::kF32>(void* data, long long index,
                                                   float value) {
  static_cast<float*>(data)[index] = value;
}

template <>
inline void store_storage<FloatStorageType::kF16>(void* data, long long index,
                                                   float value) {
  static_cast<std::uint16_t*>(data)[index] = float_to_f16(value);
}

template <>
inline void store_storage<FloatStorageType::kBF16>(void* data, long long index,
                                                    float value) {
  static_cast<std::uint16_t*>(data)[index] = float_to_bf16(value);
}

}  // namespace quixicore_cpu::detail
