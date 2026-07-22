#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "quixicore_cpu/qgemv.h"

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ":"      \
                << __LINE__ << '\n';                                      \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

void put_half_one(std::vector<std::uint8_t>& bytes, std::size_t offset) {
  bytes[offset] = 0x00;
  bytes[offset + 1] = 0x3c;
}

void put_half_two(std::vector<std::uint8_t>& bytes, std::size_t offset) {
  bytes[offset] = 0x00;
  bytes[offset + 1] = 0x40;
}

struct Fixture {
  quixicore_cpu::QuantFormat format;
  long long block;
  std::size_t bytes;
  float first;
};

}  // namespace

int main() {
  using namespace quixicore_cpu;
  const Fixture fixtures[] = {
      {QuantFormat::kQ4_1, 32, 20, 2.0f},
      {QuantFormat::kQ5_0, 32, 22, -16.0f},
      {QuantFormat::kQ5_1, 32, 24, 2.0f},
      {QuantFormat::kU4B8, 128, 66, -8.0f},
      {QuantFormat::kU4, 128, 68, -2.0f},
      {QuantFormat::kHQQ, 64, 36, -2.0f},
      {QuantFormat::kFP8E4M3, 32, 34, 1.0f},
      {QuantFormat::kFP8E5M2, 32, 34, 1.0f},
      {QuantFormat::kFP8Block, 128, 130, 1.0f},
      {QuantFormat::kFP8Raw, 128, 128, 1.0f},
      {QuantFormat::kFP4E2M1, 32, 18, 1.0f},
      {QuantFormat::kMXFP8, 32, 33, 1.0f},
      {QuantFormat::kNVFP4, 16, 9, 1.0f},
      {QuantFormat::kMXFP4, 32, 17, 1.0f},
      {QuantFormat::kMXFP6E3M2, 32, 25, 1.0f},
      {QuantFormat::kMXFP6E2M3, 32, 25, 1.0f},
      {QuantFormat::kBitnet, 32, 10, 1.0f},
      {QuantFormat::kQ2_K, 256, 84, 1.0f},
      {QuantFormat::kQ3_K, 256, 110, 128.0f},
      {QuantFormat::kQ4_K, 256, 144, 1.0f},
      {QuantFormat::kQ5_K, 256, 176, 1.0f},
      {QuantFormat::kQ6_K, 256, 210, -32.0f},
      {QuantFormat::kIQ4_NL, 32, 18, -127.0f},
      {QuantFormat::kIQ4_XS, 256, 136, 4064.0f},
      {QuantFormat::kIQ2_XXS, 256, 66, 1.0f},
      {QuantFormat::kIQ2_XS, 256, 74, 1.0f},
      {QuantFormat::kIQ3_XXS, 256, 98, 1.0f},
      {QuantFormat::kIQ1_S, 256, 50, -0.875f},
  };

  for (const Fixture& fixture : fixtures) {
    std::size_t packed_size = 0;
    REQUIRE(qgemv_packed_size(fixture.format, 1, fixture.block, &packed_size) ==
            Status::kOk);
    REQUIRE(packed_size == fixture.bytes);
    std::vector<std::uint8_t> packed(packed_size, 0);
    switch (fixture.format) {
      case QuantFormat::kQ4_1:
        put_half_one(packed, 0);
        put_half_two(packed, 2);
        break;
      case QuantFormat::kQ5_0:
        put_half_one(packed, 0);
        break;
      case QuantFormat::kQ5_1:
        put_half_one(packed, 0);
        put_half_two(packed, 2);
        break;
      case QuantFormat::kU4B8:
        put_half_one(packed, 0);
        break;
      case QuantFormat::kU4:
      case QuantFormat::kHQQ:
        put_half_one(packed, 0);
        put_half_two(packed, 2);
        break;
      case QuantFormat::kFP8E4M3:
      case QuantFormat::kFP8Block:
        put_half_one(packed, 0);
        packed[2] = 0x38;
        break;
      case QuantFormat::kFP8E5M2:
        put_half_one(packed, 0);
        packed[2] = 0x3c;
        break;
      case QuantFormat::kFP8Raw:
        packed[0] = 0x38;
        break;
      case QuantFormat::kFP4E2M1:
        put_half_one(packed, 0);
        packed[2] = 2;
        break;
      case QuantFormat::kMXFP8:
        packed[0] = 127;
        packed[1] = 0x38;
        break;
      case QuantFormat::kNVFP4:
        packed[0] = 0x38;
        packed[1] = 2;
        break;
      case QuantFormat::kMXFP4:
        packed[0] = 127;
        packed[1] = 2;
        break;
      case QuantFormat::kMXFP6E3M2:
        packed[0] = 127;
        packed[1] = 12;
        break;
      case QuantFormat::kMXFP6E2M3:
        packed[0] = 127;
        packed[1] = 8;
        break;
      case QuantFormat::kBitnet:
        put_half_one(packed, 0);
        packed[2] = 2;
        break;
      case QuantFormat::kQ2_K:
        packed[0] = 1;
        packed[16] = 1;
        put_half_one(packed, 80);
        break;
      case QuantFormat::kQ3_K:
        put_half_one(packed, 108);
        break;
      case QuantFormat::kQ4_K:
        put_half_one(packed, 0);
        packed[4] = 1;
        packed[16] = 1;
        break;
      case QuantFormat::kQ5_K:
        put_half_one(packed, 0);
        packed[4] = 1;
        packed[48] = 1;
        break;
      case QuantFormat::kQ6_K:
        packed[192] = 1;
        put_half_one(packed, 208);
        break;
      case QuantFormat::kIQ4_NL:
        put_half_one(packed, 0);
        break;
      case QuantFormat::kIQ4_XS:
        put_half_one(packed, 0);
        break;
      case QuantFormat::kIQ2_XXS:
      case QuantFormat::kIQ2_XS:
      case QuantFormat::kIQ3_XXS:
      case QuantFormat::kIQ1_S:
        put_half_one(packed, 0);
        break;
      default:
        REQUIRE(false);
    }
    std::vector<float> unpacked(static_cast<std::size_t>(fixture.block));
    REQUIRE(qgemv_unpack(fixture.format, packed.data(), 1, fixture.block,
                         unpacked.data()) == Status::kOk);
    REQUIRE(std::fabs(unpacked[0] - fixture.first) < 1e-5f);
    std::vector<float> ones(static_cast<std::size_t>(fixture.block), 1.0f);
    float output = 0.0f;
    REQUIRE(qgemv(fixture.format, packed.data(), ones.data(), &output, 1,
                  fixture.block) == Status::kOk);
    const double expected =
        std::accumulate(unpacked.begin(), unpacked.end(), 0.0);
    REQUIRE(std::fabs(output - expected) <=
            1e-4 + 1e-5 * std::fabs(expected));
    REQUIRE(std::string(qgemv_variant(fixture.format)) != "unsupported");
    REQUIRE(qgemv_pack(fixture.format, ones.data(), 1, fixture.block,
                       packed.data()) == Status::kUnsupportedFormat);
  }

  std::cout << "GGUF format tests passed\n";
  return 0;
}
