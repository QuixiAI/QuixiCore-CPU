#pragma once

#include <cstdint>
#include <cstring>

namespace quixicore_cpu {

// Portable IEEE-754 binary16 <-> binary32 conversion by bit manipulation.
// Reference paths must not assume hardware f16 support on any target.

inline float fp16_to_fp32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // signed zero
    } else {
      // Subnormal half: renormalize into a normal float.
      int shift = 0;
      uint32_t m = mant;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      bits = sign | ((113u - shift) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof out);
  return out;
}

inline uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof x);
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  x &= 0x7FFFFFFFu;

  if (x >= 0x47800000u) {  // overflow -> inf; nan stays nan
    return static_cast<uint16_t>(
        sign | (x > 0x7F800000u ? 0x7E00u : 0x7C00u));
  }
  if (x < 0x38800000u) {  // half subnormal or zero
    if (x < 0x33000000u) {
      return sign;  // rounds to signed zero
    }
    // A float with exponent e must move its implicit leading 1 right by
    // (113 - e) additional places beyond the normal 13-bit f32->f16
    // mantissa truncation. Keeping the two shifts separate here is
    // important: using 126-e and then adding 13 again both zeros valid half
    // subnormals and can shift a uint32_t by 32 or more (undefined behavior).
    const uint32_t shift = 113u - (x >> 23);
    const uint32_t mant = (x & 0x7FFFFFu) | 0x800000u;
    uint16_t h = static_cast<uint16_t>(sign | (mant >> (shift + 13)));
    const uint32_t rem = mant & ((1u << (shift + 13)) - 1u);
    const uint32_t half = 1u << (shift + 12);
    if (rem > half || (rem == half && (h & 1u))) {
      ++h;  // round to nearest even
    }
    return h;
  }
  uint16_t h = static_cast<uint16_t>(sign | (((x >> 23) - 112u) << 10) |
                                     ((x >> 13) & 0x3FFu));
  const uint32_t rem = x & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
    ++h;  // carry propagates correctly into the exponent by IEEE layout
  }
  return h;
}

}  // namespace quixicore_cpu
