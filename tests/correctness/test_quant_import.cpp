#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quant_import.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#define REQUIRE(condition)                                                    \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ':'          \
                << __LINE__ << '\n';                                          \
      return false;                                                           \
    }                                                                         \
  } while (0)

namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

bool close(float actual, float expected, float tolerance) {
  return std::fabs(actual - expected) <=
         tolerance * (1.0f + std::fabs(expected));
}

std::uint8_t nibble(const std::vector<std::uint8_t>& data,
                    std::size_t element) {
  return static_cast<std::uint8_t>(
      (data[element / 2] >> (4 * (element & 1))) & 0x0f);
}

bool test_canonical_packers() {
  using namespace quixicore_cpu;
  constexpr long long rows = 2;
  constexpr long long columns = 32;
  std::vector<float> values(static_cast<std::size_t>(rows * columns));
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 4.0f;
  }
  std::vector<std::uint16_t> f16(values.size());
  std::vector<std::uint16_t> bf16(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    f16[index] = float_to_f16(values[index]);
    bf16[index] = float_to_bf16(values[index]);
  }

  struct Case {
    CanonicalQuantLayout layout;
    long long group_size;
    float tolerance;
  };
  const Case cases[] = {
      {CanonicalQuantLayout::kInt4Symmetric, 16, 0.35f},
      {CanonicalQuantLayout::kUInt4Affine, 16, 0.35f},
      {CanonicalQuantLayout::kInt8Symmetric, 16, 0.03f},
      {CanonicalQuantLayout::kInt8Affine, 32, 0.03f},
      {CanonicalQuantLayout::kFP8E4M3FN, 0, 0.08f},
      {CanonicalQuantLayout::kFP8E5M2, 16, 0.18f},
      {CanonicalQuantLayout::kFP4E2M1, 16, 0.5f},
      {CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, 0.08f},
      {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, 0.5f},
      {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, 0.5f},
      {CanonicalQuantLayout::kBitNetTernary, 32, 1.1f},
  };
  for (const Case& test : cases) {
    CanonicalQuantTensor f32_tensor;
    CanonicalQuantTensor f16_tensor;
    CanonicalQuantTensor bf16_tensor;
    REQUIRE(quantize_canonical(
                {values.data(), FloatStorageType::kF32,
                 static_cast<long long>(values.size())},
                rows, columns, test.layout, test.group_size, &f32_tensor) ==
            Status::kOk);
    REQUIRE(quantize_canonical(
                {f16.data(), FloatStorageType::kF16,
                 static_cast<long long>(f16.size())},
                rows, columns, test.layout, test.group_size, &f16_tensor) ==
            Status::kOk);
    REQUIRE(quantize_canonical(
                {bf16.data(), FloatStorageType::kBF16,
                 static_cast<long long>(bf16.size())},
                rows, columns, test.layout, test.group_size, &bf16_tensor) ==
            Status::kOk);
    REQUIRE(validate_canonical_quant_tensor(f32_tensor) == Status::kOk);
    REQUIRE(f32_tensor.data == f16_tensor.data);
    REQUIRE(f32_tensor.data == bf16_tensor.data);
    CpuPackedWeights prepared;
    REQUIRE(prepared.prepare(f32_tensor, CpuPanelLayout::kNeonRows4) ==
            Status::kOk);
    REQUIRE(prepared.info().has_canonical_layout);
    REQUIRE(prepared.info().canonical_layout == test.layout);
    REQUIRE(std::memcmp(prepared.contract_data(), f32_tensor.data.data(),
                        f32_tensor.data.size()) == 0);
    std::vector<float> decoded(values.size());
    REQUIRE(dequantize_canonical(f32_tensor, decoded.data(),
                                 static_cast<long long>(decoded.size())) ==
            Status::kOk);
    for (std::size_t index = 0; index < values.size(); ++index) {
      REQUIRE(close(decoded[index], values[index], test.tolerance));
    }
  }

  CanonicalQuantTensor nv2d;
  REQUIRE(quantize_canonical(
              {values.data(), FloatStorageType::kF32,
               static_cast<long long>(values.size())},
              rows, columns, CanonicalQuantLayout::kNVFP4E2M1E4M3, 16,
              &nv2d, true) == Status::kOk);
  REQUIRE(nv2d.metadata.scale_2d && nv2d.metadata.scale_domain_rows == 16);

  for (QuantScaleMode mode : {
           QuantScaleMode::kTensor, QuantScaleMode::kRow,
           QuantScaleMode::kChannel, QuantScaleMode::kGroup,
           QuantScaleMode::kBlock}) {
    CanonicalQuantTensor scaled;
    const long long group =
        mode == QuantScaleMode::kGroup || mode == QuantScaleMode::kBlock
            ? 16
            : 0;
    const int block_rows = mode == QuantScaleMode::kBlock ? 2 : 1;
    REQUIRE(quantize_canonical_fp_scaled(
                {values.data(), FloatStorageType::kF32,
                 static_cast<long long>(values.size())},
                rows, columns, CanonicalQuantLayout::kFP8E4M3FN, mode, group,
                block_rows, &scaled) == Status::kOk);
    REQUIRE(scaled.metadata.scale_mode == mode);
    const std::size_t expected_scales =
        mode == QuantScaleMode::kTensor
            ? 1
            : (mode == QuantScaleMode::kRow ||
                       mode == QuantScaleMode::kChannel
                   ? rows
                   : (mode == QuantScaleMode::kBlock ? 2 : 4));
    REQUIRE(scaled.scales.size() == expected_scales);
    std::vector<float> decoded(values.size());
    REQUIRE(dequantize_canonical(scaled, decoded.data(), decoded.size()) ==
            Status::kOk);
    for (std::size_t index = 0; index < values.size(); ++index)
      REQUIRE(close(decoded[index], values[index], 0.08f));
  }
  return true;
}

bool test_awq_fragment() {
  using namespace quixicore_cpu;
  constexpr std::size_t k = 8;
  constexpr std::size_t n = 8;
  std::array<std::uint32_t, k> qweight{};
  qweight.fill(0x76543210U);
  const std::array<std::uint32_t, 1> qzeros = {0x33333333U};
  std::array<float, n> scales{};
  scales.fill(0.25f);
  AwqU4Source source;
  source.qweight = qweight.data();
  source.qweight_words = qweight.size();
  source.qzeros = qzeros.data();
  source.qzero_words = qzeros.size();
  source.scales = scales.data();
  source.scale_count = scales.size();
  source.input_features = k;
  source.output_features = n;
  source.group_size = k;
  CanonicalQuantTensor tensor;
  REQUIRE(import_awq_u4(source, &tensor) == Status::kOk);
  const std::array<int, 8> logical_codes = {0, 4, 1, 5, 2, 6, 3, 7};
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < k; ++column) {
      REQUIRE(nibble(tensor.data, row * k + column) == logical_codes[row]);
    }
  }
  REQUIRE(tensor.provenance == QuantImportProvenance::kAWQ);
  REQUIRE(tensor.zero_points.size() == n);
  std::vector<float> decoded(n * k);
  REQUIRE(dequantize_canonical(tensor, decoded.data(), decoded.size()) ==
          Status::kOk);
  for (std::size_t row = 0; row < n; ++row) {
    const float expected = 0.25f * (logical_codes[row] - 3);
    for (std::size_t column = 0; column < k; ++column)
      REQUIRE(close(decoded[row * k + column], expected, 1e-6f));
  }
  return true;
}

bool test_gptq_fragments() {
  using namespace quixicore_cpu;
  constexpr std::size_t k = 8;
  constexpr std::size_t n = 8;
  std::array<std::uint32_t, n> qweight{};
  qweight.fill(0x76543210U);
  const std::array<std::uint32_t, 1> qzeros_v1 = {0x22222222U};
  std::array<float, n> scales{};
  scales.fill(0.5f);
  GptqU4Source source;
  source.qweight = qweight.data();
  source.qweight_words = qweight.size();
  source.qzeros = qzeros_v1.data();
  source.qzero_words = qzeros_v1.size();
  source.scales = scales.data();
  source.scale_count = scales.size();
  source.input_features = k;
  source.output_features = n;
  source.group_size = k;
  CanonicalQuantTensor tensor;
  REQUIRE(import_gptq_u4(source, &tensor) == Status::kOk);
  const std::array<std::uint8_t, 4> expected_row = {0x10, 0x32, 0x54,
                                                   0x76};
  REQUIRE(std::memcmp(tensor.data.data(), expected_row.data(),
                      expected_row.size()) == 0);
  REQUIRE(tensor.zero_points[0] == 3.0f);
  std::vector<float> decoded(k * n);
  REQUIRE(dequantize_canonical(tensor, decoded.data(), decoded.size()) ==
          Status::kOk);
  REQUIRE(close(decoded[0], -1.5f, 1e-6f));
  REQUIRE(close(decoded[7], 2.0f, 1e-6f));

  constexpr std::size_t groups = 2;
  const std::array<std::uint32_t, groups> qzeros_v2 = {0x88888888U,
                                                       0x88888888U};
  std::array<float, groups * n> group_scales{};
  for (std::size_t output = 0; output < n; ++output) {
    group_scales[output] = 1.0f;
    group_scales[n + output] = 2.0f;
  }
  const std::array<int, k> group_index = {1, 1, 1, 1, 0, 0, 0, 0};
  const std::array<int, k> act_order = {4, 5, 6, 7, 0, 1, 2, 3};
  source.qzeros = qzeros_v2.data();
  source.qzero_words = qzeros_v2.size();
  source.scales = group_scales.data();
  source.scale_count = group_scales.size();
  source.group_size = 4;
  source.group_index = group_index.data();
  source.group_index_count = group_index.size();
  source.act_order = act_order.data();
  source.act_order_count = act_order.size();
  source.symmetric = true;
  source.gptq_v2 = true;
  REQUIRE(import_gptq_u4(source, &tensor) == Status::kOk);
  REQUIRE(tensor.metadata.group_index_count == k);
  REQUIRE(tensor.metadata.act_order_count == k);
  REQUIRE(tensor.group_index ==
          std::vector<int>(group_index.begin(), group_index.end()));
  REQUIRE(tensor.act_order ==
          std::vector<int>(act_order.begin(), act_order.end()));
  REQUIRE(dequantize_canonical(tensor, decoded.data(), decoded.size()) ==
          Status::kOk);
  REQUIRE(close(decoded[0], -16.0f, 1e-6f));
  REQUIRE(close(decoded[4], -4.0f, 1e-6f));

  REQUIRE(import_autoround_gptq_u4(source, &tensor) == Status::kOk);
  REQUIRE(tensor.provenance == QuantImportProvenance::kAutoRound);
  return true;
}

bool test_autoround_targets() {
  using namespace quixicore_cpu;
  constexpr long long rows = 1;
  constexpr long long columns = 32;
  std::array<float, columns> values{};
  for (long long item = 0; item < columns; ++item) {
    values[static_cast<std::size_t>(item)] =
        static_cast<float>((item % 9) - 4) * 0.5f;
  }
  const std::array<CanonicalQuantLayout, 8> layouts = {
      CanonicalQuantLayout::kInt4Symmetric,
      CanonicalQuantLayout::kUInt4Affine,
      CanonicalQuantLayout::kFP8E4M3FN,
      CanonicalQuantLayout::kFP8E5M2,
      CanonicalQuantLayout::kMXFP8E4M3E8M0,
      CanonicalQuantLayout::kMXFP4E2M1E8M0,
      CanonicalQuantLayout::kNVFP4E2M1E4M3,
      CanonicalQuantLayout::kBitNetTernary,
  };
  for (CanonicalQuantLayout layout : layouts) {
    CanonicalQuantTensor original;
    const long long group =
        layout == CanonicalQuantLayout::kFP8E4M3FN ||
                layout == CanonicalQuantLayout::kFP8E5M2
            ? 0
            : (layout == CanonicalQuantLayout::kNVFP4E2M1E4M3 ? 16 : 32);
    REQUIRE(quantize_canonical(
                {values.data(), FloatStorageType::kF32, columns}, rows,
                columns, layout, group, &original) == Status::kOk);
    AutoRoundCanonicalSource source;
    source.metadata = original.metadata;
    source.data = original.data.data();
    source.data_bytes = original.data.size();
    source.scale_codes = original.scale_codes.data();
    source.scale_code_count = original.scale_codes.size();
    source.scales = original.scales.data();
    source.scale_count = original.scales.size();
    source.zero_points = original.zero_points.data();
    source.zero_point_count = original.zero_points.size();
    CanonicalQuantTensor imported;
    REQUIRE(import_autoround_canonical(source, &imported) == Status::kOk);
    REQUIRE(imported.data == original.data);
    REQUIRE(imported.scale_codes == original.scale_codes);
    REQUIRE(imported.scales == original.scales);
    REQUIRE(imported.zero_points == original.zero_points);
    REQUIRE(imported.provenance == QuantImportProvenance::kAutoRound);
    std::vector<float> original_values(columns);
    std::vector<float> imported_values(columns);
    REQUIRE(dequantize_canonical(original, original_values.data(), columns) ==
            Status::kOk);
    REQUIRE(dequantize_canonical(imported, imported_values.data(), columns) ==
            Status::kOk);
    REQUIRE(original_values == imported_values);
  }
  const std::array<std::uint8_t, 4> raw_fp8 = {0x00, 0x38, 0x3c, 0x7e};
  AutoRoundCanonicalSource raw_source;
  raw_source.metadata.layout = CanonicalQuantLayout::kFP8E4M3FN;
  raw_source.metadata.logical_rows = 1;
  raw_source.metadata.logical_columns = raw_fp8.size();
  raw_source.data = raw_fp8.data();
  raw_source.data_bytes = raw_fp8.size();
  CanonicalQuantTensor raw_tensor;
  REQUIRE(import_autoround_canonical(raw_source, &raw_tensor) == Status::kOk);
  std::array<float, 4> raw_values{};
  REQUIRE(dequantize_canonical(raw_tensor, raw_values.data(),
                               raw_values.size()) == Status::kOk);
  REQUIRE(raw_values[0] == 0.0f && raw_values[1] == 1.0f &&
          raw_values[2] == 1.5f && raw_values[3] == 448.0f);
  return true;
}

bool test_smoothquant_and_preparation() {
  using namespace quixicore_cpu;
  constexpr long long rows = 2;
  constexpr long long columns = 8;
  const std::array<std::int8_t, rows * columns> weights = {
      -127, -64, -1, 0, 1, 2, 63, 127,
      5,    4,   3,  2, 1, 0, -1, -2};
  const std::array<float, rows> weight_scales = {0.25f, 0.125f};
  const std::array<float, 1> activation_scales = {0.0625f};
  const std::array<int, 1> activation_zeros = {-3};
  SmoothQuantW8A8Source source;
  source.weights = weights.data();
  source.weight_count = weights.size();
  source.weight_scales = weight_scales.data();
  source.weight_scale_count = weight_scales.size();
  source.activation_scales = activation_scales.data();
  source.activation_scale_count = activation_scales.size();
  source.activation_zero_points = activation_zeros.data();
  source.activation_zero_point_count = activation_zeros.size();
  source.rows = rows;
  source.columns = columns;
  CanonicalQuantTensor tensor;
  REQUIRE(import_smoothquant_w8a8(source, &tensor) == Status::kOk);
  REQUIRE(tensor.row_sums[0] == 1);
  REQUIRE(tensor.row_sums[1] == 12);
  REQUIRE(tensor.activation_zero_points[0] == -3);

  CpuPackedWeights prepared;
  REQUIRE(prepared.prepare(tensor, CpuPanelLayout::kNeonRows4) == Status::kOk);
  const CpuPackedWeightsInfo info = prepared.info();
  REQUIRE(info.has_canonical_layout);
  REQUIRE(info.canonical_layout == CanonicalQuantLayout::kInt8Symmetric);
  REQUIRE(info.prepared_version == 2);
  REQUIRE(info.logical_bytes > info.contract_bytes);
  REQUIRE(info.prepared_bytes >= info.panel_bytes);
  REQUIRE(info.scale_table_offset != static_cast<std::size_t>(-1));
  REQUIRE(info.row_sum_table_offset != static_cast<std::size_t>(-1));
  REQUIRE(info.activation_scale_offset != static_cast<std::size_t>(-1));
  REQUIRE(info.activation_zero_point_offset !=
          static_cast<std::size_t>(-1));
  REQUIRE(std::memcmp(prepared.contract_data(), tensor.data.data(),
                      tensor.data.size()) == 0);
  const auto* panel = static_cast<const std::uint8_t*>(prepared.panel_data());
  REQUIRE(std::memcmp(panel + info.scale_table_offset, tensor.scales.data(),
                      tensor.scales.size() * sizeof(float)) == 0);
  REQUIRE(std::memcmp(panel + info.row_sum_table_offset, tensor.row_sums.data(),
                      tensor.row_sums.size() * sizeof(std::int32_t)) == 0);
  std::array<float, columns> x{};
  std::array<float, rows> y{};
  REQUIRE(qgemm_prepacked(prepared, x.data(), y.data(), 1) ==
          Status::kUnsupportedFormat);
  return true;
}

bool test_bitnet_import() {
  using namespace quixicore_cpu;
  std::array<std::uint8_t, 10> packed = {
      0x00, 0x3c, 0x64, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55};
  BitNetI2Source source;
  source.packed = packed.data();
  source.packed_bytes = packed.size();
  source.rows = 1;
  source.columns = 32;
  CanonicalQuantTensor tensor;
  REQUIRE(import_bitnet_i2_s(source, &tensor) == Status::kOk);
  REQUIRE(tensor.data == std::vector<std::uint8_t>(packed.begin(), packed.end()));
  REQUIRE(tensor.provenance == QuantImportProvenance::kBitNet);
  std::array<float, 32> decoded{};
  REQUIRE(dequantize_canonical(tensor, decoded.data(), decoded.size()) ==
          Status::kOk);
  REQUIRE(decoded[0] == -1.0f && decoded[1] == 0.0f && decoded[2] == 1.0f);
  CpuPackedWeights prepared;
  REQUIRE(prepared.prepare(tensor) == Status::kOk);
  REQUIRE(std::memcmp(prepared.contract_data(), packed.data(), packed.size()) ==
          0);
  packed[2] = 0xff;
  REQUIRE(import_bitnet_i2_s(source, &tensor) == Status::kInvalidArgument);
  return true;
}

bool test_invalid_imports() {
  using namespace quixicore_cpu;
  CanonicalQuantTensor tensor;
  AwqU4Source awq;
  REQUIRE(import_awq_u4(awq, &tensor) == Status::kInvalidArgument);
  GptqU4Source gptq;
  REQUIRE(import_gptq_u4(gptq, &tensor) == Status::kInvalidArgument);
  SmoothQuantW8A8Source smooth;
  REQUIRE(import_smoothquant_w8a8(smooth, &tensor) ==
          Status::kInvalidArgument);
  return true;
}

}  // namespace

int main() {
  if (!test_canonical_packers() || !test_awq_fragment() ||
      !test_gptq_fragments() || !test_autoround_targets() ||
      !test_smoothquant_and_preparation() || !test_bitnet_import() ||
      !test_invalid_imports()) {
    return 1;
  }
  std::cout << "canonical quant lifecycle tests passed\n";
  return 0;
}
