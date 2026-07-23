#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

bool has_public_packer(quixicore_cpu::QuantFormat format) {
  using quixicore_cpu::QuantFormat;
  switch (format) {
    case QuantFormat::kQ1_0:
    case QuantFormat::kQ2_0:
    case QuantFormat::kQ4_1:
    case QuantFormat::kQ5_0:
    case QuantFormat::kQ5_1:
    case QuantFormat::kMXFP4:
    case QuantFormat::kNVFP4:
    case QuantFormat::kTQ1_0:
      return true;
    default:
      return false;
  }
}

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
      {QuantFormat::kNVFP4, 64, 36, 1.0f},
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
      {QuantFormat::kIQ3_S, 256, 110, 1.0f},
      {QuantFormat::kIQ2_S, 256, 82, 1.0f},
      {QuantFormat::kIQ1_S, 256, 50, -0.875f},
      {QuantFormat::kIQ1_M, 256, 56, -0.875f},
      {QuantFormat::kQ1_0, 128, 18, 1.0f},
      {QuantFormat::kQ2_0, 64, 18, 1.0f},
      {QuantFormat::kTQ1_0, 256, 54, -1.0f},
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
        packed[4] = 2;
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
      case QuantFormat::kIQ3_S:
      case QuantFormat::kIQ2_S:
      case QuantFormat::kIQ1_S:
        put_half_one(packed, 0);
        break;
      case QuantFormat::kIQ1_M:
        // IQ1_M distributes the fp16 super-scale high nibbles across its four
        // packed scale words. These bytes reconstruct fp16(1.0) == 0x3c00.
        packed[53] = 0xc0;
        packed[55] = 0x30;
        break;
      case QuantFormat::kQ1_0:
        put_half_one(packed, 0);
        packed[2] = 1;
        break;
      case QuantFormat::kQ2_0:
        put_half_one(packed, 0);
        packed[2] = 2;
        break;
      case QuantFormat::kTQ1_0:
        put_half_one(packed, 52);
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

    // Exercise arbitrary payload bits while keeping floating scales finite.
    // This compares the selected ISA route against the centralized element
    // decoder and catches packed-lane or sub-block ordering errors.
    std::vector<std::uint8_t> stress(packed_size);
    for (std::size_t index = 0; index < stress.size(); ++index) {
      stress[index] = static_cast<std::uint8_t>((index * 37 + 19) % 113);
    }
    switch (fixture.format) {
      case QuantFormat::kQ4_1:
      case QuantFormat::kQ5_1:
      case QuantFormat::kU4:
      case QuantFormat::kHQQ:
        put_half_one(stress, 2);
        [[fallthrough]];
      case QuantFormat::kQ1_0:
      case QuantFormat::kQ2_0:
      case QuantFormat::kQ5_0:
      case QuantFormat::kU4B8:
      case QuantFormat::kFP8E4M3:
      case QuantFormat::kFP8E5M2:
      case QuantFormat::kFP8Block:
      case QuantFormat::kFP4E2M1:
      case QuantFormat::kBitnet:
      case QuantFormat::kIQ4_NL:
      case QuantFormat::kIQ4_XS:
      case QuantFormat::kIQ2_XXS:
      case QuantFormat::kIQ2_XS:
      case QuantFormat::kIQ3_XXS:
      case QuantFormat::kIQ3_S:
      case QuantFormat::kIQ2_S:
      case QuantFormat::kIQ1_S:
        put_half_one(stress, 0);
        break;
      case QuantFormat::kQ2_K:
        put_half_one(stress, 80);
        put_half_one(stress, 82);
        break;
      case QuantFormat::kQ3_K:
        put_half_one(stress, 108);
        break;
      case QuantFormat::kQ4_K:
      case QuantFormat::kQ5_K:
        put_half_one(stress, 0);
        put_half_one(stress, 2);
        break;
      case QuantFormat::kQ6_K:
        put_half_one(stress, 208);
        break;
      case QuantFormat::kMXFP8:
      case QuantFormat::kMXFP4:
      case QuantFormat::kMXFP6E3M2:
      case QuantFormat::kMXFP6E2M3:
        stress[0] = 127;
        break;
      case QuantFormat::kNVFP4:
        std::fill(stress.begin(), stress.begin() + 4, 0x38);
        break;
      case QuantFormat::kIQ1_M:
        std::fill(stress.begin() + 48, stress.begin() + 56, 0);
        stress[53] = 0xc0;
        stress[55] = 0x30;
        break;
      case QuantFormat::kTQ1_0:
        put_half_one(stress, 52);
        break;
      case QuantFormat::kTQ2_0:
        put_half_one(stress, 64);
        break;
      case QuantFormat::kFP8Raw:
        break;
      default:
        REQUIRE(false);
    }
    REQUIRE(qgemv_unpack(fixture.format, stress.data(), 1, fixture.block,
                         unpacked.data()) == Status::kOk);
    std::vector<float> probe(static_cast<std::size_t>(fixture.block));
    for (long long index = 0; index < fixture.block; ++index) {
      probe[static_cast<std::size_t>(index)] =
          static_cast<float>((index * 17) % 31) / 15.0f - 1.0f;
    }
    output = 0.0f;
    REQUIRE(qgemv(fixture.format, stress.data(), probe.data(), &output, 1,
                  fixture.block) == Status::kOk);
    double stress_expected = 0.0;
    for (long long index = 0; index < fixture.block; ++index) {
      stress_expected += static_cast<double>(
          unpacked[static_cast<std::size_t>(index)]) *
          probe[static_cast<std::size_t>(index)];
    }
    REQUIRE(std::fabs(output - stress_expected) <=
            3e-4 * (1.0 + std::fabs(stress_expected)));

    const Status pack_status =
        qgemv_pack(fixture.format, ones.data(), 1, fixture.block,
                   packed.data());
    REQUIRE(pack_status == (has_public_packer(fixture.format)
                                ? Status::kOk
                                : Status::kUnsupportedFormat));
    if (pack_status == Status::kOk) {
      REQUIRE(qgemv_unpack(fixture.format, packed.data(), 1, fixture.block,
                           unpacked.data()) == Status::kOk);
      for (float value : unpacked) {
        REQUIRE(std::isfinite(value));
        REQUIRE(std::fabs(value - 1.0f) <= 0.04f);
      }
    }
  }

  // Source-revision golden bytes for the simple canonical packers. These pin
  // nibble ordering, scale placement, sub-block ordering, and TQ base-3
  // rotation independently of our dequantizer.
  const auto require_all_ones_golden = [](QuantFormat format, long long block,
                                          const std::vector<std::uint8_t>& expected) {
    std::vector<float> values(static_cast<std::size_t>(block), 1.0f);
    std::vector<std::uint8_t> actual(expected.size());
    return qgemv_pack(format, values.data(), 1, block, actual.data()) ==
               Status::kOk &&
           actual == expected;
  };
  std::vector<std::uint8_t> q1(18, 0xff);
  q1[0] = 0x00;
  q1[1] = 0x3c;
  REQUIRE(require_all_ones_golden(QuantFormat::kQ1_0, 128, q1));
  std::vector<std::uint8_t> q2(18, 0xaa);
  q2[0] = 0x00;
  q2[1] = 0x3c;
  REQUIRE(require_all_ones_golden(QuantFormat::kQ2_0, 64, q2));
  std::vector<std::uint8_t> q4_1(20, 0);
  q4_1[2] = 0x00;
  q4_1[3] = 0x3c;
  REQUIRE(require_all_ones_golden(QuantFormat::kQ4_1, 32, q4_1));
  std::vector<std::uint8_t> q5_0(22, 0);
  q5_0[0] = 0x00;
  q5_0[1] = 0xac;
  REQUIRE(require_all_ones_golden(QuantFormat::kQ5_0, 32, q5_0));
  std::vector<std::uint8_t> q5_1(24, 0);
  q5_1[2] = 0x00;
  q5_1[3] = 0x3c;
  REQUIRE(require_all_ones_golden(QuantFormat::kQ5_1, 32, q5_1));
  std::vector<std::uint8_t> mxfp4(17, 0x66);
  mxfp4[0] = 125;
  REQUIRE(require_all_ones_golden(QuantFormat::kMXFP4, 32, mxfp4));
  std::vector<std::uint8_t> nvfp4(36, 0x77);
  std::fill(nvfp4.begin(), nvfp4.begin() + 4, 0x23);
  REQUIRE(require_all_ones_golden(QuantFormat::kNVFP4, 64, nvfp4));
  std::vector<std::uint8_t> tq1(54, 0xff);
  std::fill(tq1.begin() + 48, tq1.begin() + 52, 0xfd);
  tq1[52] = 0x00;
  tq1[53] = 0x3c;
  REQUIRE(require_all_ones_golden(QuantFormat::kTQ1_0, 256, tq1));

  for (const Fixture& fixture : fixtures) {
    if (!has_public_packer(fixture.format)) continue;
    std::vector<float> values(static_cast<std::size_t>(fixture.block), 0.0f);
    std::vector<std::uint8_t> bytes(fixture.bytes);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(qgemv_pack(fixture.format, values.data(), 1, fixture.block,
                       bytes.data()) == Status::kInvalidArgument);
    values[0] = std::numeric_limits<float>::infinity();
    REQUIRE(qgemv_pack(fixture.format, values.data(), 1, fixture.block,
                       bytes.data()) == Status::kInvalidArgument);
    std::fill(values.begin(), values.end(),
              std::numeric_limits<float>::denorm_min());
    REQUIRE(qgemv_pack(fixture.format, values.data(), 1, fixture.block,
                       bytes.data()) == Status::kOk);
  }

  std::cout << "GGUF format tests passed\n";
  return 0;
}
