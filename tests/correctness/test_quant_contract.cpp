#include "kernels/quantization/quant_microkernel.h"
#include "quixicore_cpu/quant_contract.h"
#include "quixicore_cpu/quantization.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
using quixicore_cpu::Status;

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::vector<std::string> fields;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, delimiter))
    fields.push_back(field);
  return fields;
}

std::vector<std::uint8_t> hex_bytes(const std::string &value) {
  std::vector<std::uint8_t> bytes;
  if (value == "-")
    return bytes;
  if (value.size() % 2 != 0)
    return {};
  for (std::size_t offset = 0; offset < value.size(); offset += 2) {
    bytes.push_back(static_cast<std::uint8_t>(
        std::stoul(value.substr(offset, 2), nullptr, 16)));
  }
  return bytes;
}

std::vector<float> floats(const std::string &value) {
  std::vector<float> result;
  for (const auto &field : split(value, ','))
    result.push_back(std::stof(field));
  return result;
}

std::unordered_map<std::string, std::string>
parameters(const std::string &value) {
  std::unordered_map<std::string, std::string> result;
  for (const auto &field : split(value, ';')) {
    const auto equal = field.find('=');
    if (equal != std::string::npos) {
      result.emplace(field.substr(0, equal), field.substr(equal + 1));
    }
  }
  return result;
}

float e8m0(std::uint8_t code) {
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

unsigned unpack_bits(const std::vector<std::uint8_t> &bytes,
                     std::size_t element, int bits) {
  const std::size_t bit = element * static_cast<std::size_t>(bits);
  unsigned value = bytes[bit / 8];
  if (bit % 8 + static_cast<std::size_t>(bits) > 8) {
    value |= static_cast<unsigned>(bytes[bit / 8 + 1]) << 8;
  }
  return (value >> (bit % 8)) & ((1u << bits) - 1u);
}

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <= 1e-6f * (1.0f + std::fabs(expected));
}

bool decode_row(const std::vector<std::string> &fields) {
  using namespace quixicore_cpu;
  REQUIRE(fields.size() == 6);
  const std::string &layout = fields[0];
  const auto bytes = hex_bytes(fields[2]);
  const auto scale_bytes = hex_bytes(fields[3]);
  const auto params = parameters(fields[4]);
  const auto expected = floats(fields[5]);
  std::vector<float> actual;
  actual.reserve(expected.size());

  if (layout == "int4_symmetric" || layout == "uint4_affine" ||
      layout == "fp4_e2m1") {
    const float scale = std::stof(params.at("scale"));
    const float zero =
        params.count("zero") ? std::stof(params.at("zero")) : 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const unsigned code = (bytes[i / 2] >> (4 * (i & 1))) & 15u;
      if (layout == "int4_symmetric") {
        const int signed_code =
            (code & 8u) ? static_cast<int>(code) - 16 : static_cast<int>(code);
        actual.push_back(scale * signed_code);
      } else if (layout == "uint4_affine") {
        actual.push_back(scale * (static_cast<float>(code) - zero));
      } else {
        actual.push_back(scale * fp4_e2m1_decode(code));
      }
    }
  } else if (layout == "int8_symmetric" || layout == "int8_affine") {
    const float scale = std::stof(params.at("scale"));
    const int zero = params.count("zero") ? std::stoi(params.at("zero")) : 0;
    for (std::uint8_t byte : bytes) {
      actual.push_back(
          scale * (static_cast<int>(static_cast<std::int8_t>(byte)) - zero));
    }
  } else if (layout == "fp8_e4m3fn" || layout == "fp8_e5m2") {
    const auto format =
        layout == "fp8_e4m3fn" ? Float8Format::kE4M3FN : Float8Format::kE5M2;
    const float scale = std::stof(params.at("scale"));
    for (std::uint8_t byte : bytes) {
      actual.push_back(scale * float8_decode(byte, format));
    }
  } else if (layout == "mxfp8_e4m3_e8m0") {
    REQUIRE(scale_bytes.size() == 1);
    const float scale = e8m0(scale_bytes[0]);
    for (std::uint8_t byte : bytes) {
      actual.push_back(scale * float8_decode(byte, Float8Format::kE4M3FN));
    }
  } else if (layout == "mxfp4_e2m1_e8m0" || layout == "nvfp4_e2m1_e4m3") {
    REQUIRE(scale_bytes.size() == 1);
    float scale = 0.0f;
    if (layout == "mxfp4_e2m1_e8m0") {
      scale = e8m0(scale_bytes[0]);
    } else {
      scale = std::stof(params.at("global")) *
              float8_decode(scale_bytes[0], Float8Format::kE4M3FN);
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const unsigned code = (bytes[i / 2] >> (4 * (i & 1))) & 15u;
      actual.push_back(scale * fp4_e2m1_decode(code));
    }
  } else if (layout == "bitnet_ternary") {
    REQUIRE(bytes.size() == 10);
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const unsigned code = (bytes[2 + i / 4] >> (2 * (i & 3))) & 3u;
      actual.push_back(static_cast<float>(static_cast<int>(code) - 1));
    }
  } else if (layout == "turboquant_key" || layout == "turboquant_value") {
    const int bits = std::stoi(params.at("bits"));
    const float scale = std::stof(params.at("scale"));
    std::vector<float> centroids;
    if (layout == "turboquant_value")
      centroids = floats(params.at("centroids"));
    const float zero =
        params.count("zero") ? std::stof(params.at("zero")) : 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const unsigned code = unpack_bits(bytes, i, bits);
      actual.push_back(layout == "turboquant_key"
                           ? scale * (static_cast<float>(code) + zero)
                           : scale * centroids[code]);
    }
  } else {
    REQUIRE(false && "unknown golden layout");
  }

  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    REQUIRE(close(actual[i], expected[i]));
  return true;
}

bool test_fixture() {
  std::ifstream input(std::string(QUIXICORE_CPU_TESTDATA_DIR) +
                      "/quant_contract_golden.tsv");
  REQUIRE(input.good());
  std::string line;
  REQUIRE(static_cast<bool>(std::getline(input, line)));
  REQUIRE(
      line ==
      "layout\tcase\telement_bytes_hex\tscale_bytes_hex\tparameters\tdecoded");
  std::size_t rows = 0;
  while (std::getline(input, line)) {
    if (line.empty())
      continue;
    REQUIRE(decode_row(split(line, '\t')));
    ++rows;
  }
  REQUIRE(rows == 13);
  return true;
}

bool test_descriptors() {
  using namespace quixicore_cpu;
  const CanonicalQuantLayout layouts[] = {
      CanonicalQuantLayout::kInt4Symmetric,
      CanonicalQuantLayout::kUInt4Affine,
      CanonicalQuantLayout::kInt8Symmetric,
      CanonicalQuantLayout::kInt8Affine,
      CanonicalQuantLayout::kFP8E4M3FN,
      CanonicalQuantLayout::kFP8E5M2,
      CanonicalQuantLayout::kFP4E2M1,
      CanonicalQuantLayout::kMXFP8E4M3E8M0,
      CanonicalQuantLayout::kMXFP4E2M1E8M0,
      CanonicalQuantLayout::kNVFP4E2M1E4M3,
      CanonicalQuantLayout::kBitNetTernary,
      CanonicalQuantLayout::kTurboQuantKey,
      CanonicalQuantLayout::kTurboQuantValue,
  };
  for (CanonicalQuantLayout layout : layouts) {
    CanonicalQuantDescriptor descriptor;
    REQUIRE(canonical_quant_descriptor(layout, &descriptor) == Status::kOk);
    REQUIRE(descriptor.name != nullptr);
    REQUIRE(std::string(descriptor.name) ==
            canonical_quant_layout_name(layout));
    REQUIRE(descriptor.contract_version == 1);
  }
  REQUIRE(canonical_quant_descriptor(layouts[0], nullptr) ==
          Status::kInvalidArgument);

  QuantTensorMetadata mx;
  mx.layout = CanonicalQuantLayout::kMXFP8E4M3E8M0;
  mx.logical_rows = 2;
  mx.logical_columns = 64;
  mx.group_size = 32;
  mx.scale_mode = QuantScaleMode::kMicroscaleK32;
  mx.scale_encoding = QuantScaleEncoding::kE8M0;
  mx.scale_count = 4;
  REQUIRE(validate_quant_metadata(mx) == Status::kOk);
  mx.logical_columns = 63;
  REQUIRE(validate_quant_metadata(mx) == Status::kInvalidShape);

  QuantTensorMetadata tq;
  tq.layout = CanonicalQuantLayout::kTurboQuantValue;
  tq.logical_rows = 4;
  tq.logical_columns = 64;
  tq.group_size = 32;
  tq.scale_mode = QuantScaleMode::kTurboQuantK32;
  tq.scale_encoding = QuantScaleEncoding::kFP16;
  tq.scale_count = 8;
  tq.value_bits = 2;
  tq.centroid_count = 4;
  REQUIRE(validate_quant_metadata(tq) == Status::kOk);
  tq.centroid_count = 3;
  REQUIRE(validate_quant_metadata(tq) == Status::kInvalidShape);

  quant::QuantMicrokernelSet interface;
  interface.name = "contract_probe";
  interface.layout = CanonicalQuantLayout::kInt4Symmetric;
  interface.row_tile = 4;
  interface.column_tile = 8;
  interface.k_tile = 32;
  REQUIRE(interface.block_dot_f32 == nullptr && interface.tile == nullptr &&
          interface.paired_projection == nullptr &&
          interface.selected_rows == nullptr);
  return true;
}

} // namespace

int main() {
  if (!test_fixture() || !test_descriptors())
    return 1;
  std::cout << "quant contract golden tests passed\n";
  return 0;
}
