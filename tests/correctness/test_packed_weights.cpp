#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/threading.h"
#include "quixicore_cpu/workspace.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define REQUIRE(condition)                                                  \
  do {                                                                      \
    if (!(condition)) {                                                     \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ':'         \
                << __LINE__ << '\n';                                        \
      return false;                                                         \
    }                                                                       \
  } while (0)

namespace {

std::uint32_t next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

bool run_format(quixicore_cpu::QuantFormat format) {
  using namespace quixicore_cpu;
  constexpr long long n = 19;
  constexpr long long k = 64;
  std::uint32_t state = 0xA341316Cu;
  std::vector<float> dense(static_cast<std::size_t>(n * k));
  for (float& value : dense) {
    value = static_cast<float>(next(&state) >> 8) / 8388608.0f - 1.0f;
  }
  std::size_t packed_bytes = 0;
  REQUIRE(qgemv_packed_size(format, n, k, &packed_bytes) == Status::kOk);
  std::vector<std::uint8_t> packed(packed_bytes);
  REQUIRE(qgemv_pack(format, dense.data(), n, k, packed.data()) == Status::kOk);

  for (CpuPanelLayout layout : {
           CpuPanelLayout::kPortableRows1, CpuPanelLayout::kNeonRows4,
           CpuPanelLayout::kAvx2Rows8, CpuPanelLayout::kAvx512Rows16}) {
    CpuPackedWeights weights;
    REQUIRE(weights.prepare(format, packed.data(), n, k, layout) == Status::kOk);
    REQUIRE(weights.ready());
    const CpuPackedWeightsInfo info = weights.info();
    REQUIRE(info.layout == layout);
    REQUIRE(info.rows == n && info.columns == k);
    REQUIRE(info.contract_bytes == packed_bytes);
    REQUIRE(std::memcmp(weights.contract_data(), packed.data(), packed_bytes) ==
            0);
    REQUIRE(reinterpret_cast<std::uintptr_t>(weights.panel_data()) % 64 == 0);

    const auto* panel = static_cast<const std::uint8_t*>(weights.panel_data());
    const long long panels = n / info.row_tile + (n % info.row_tile != 0);
    for (long long row_panel = 0; row_panel < panels; ++row_panel) {
      for (long long block = 0; block < info.blocks_per_row; ++block) {
        for (long long lane = 0; lane < info.row_tile; ++lane) {
          const long long row = row_panel * info.row_tile + lane;
          const std::size_t destination_block = static_cast<std::size_t>(
              (row_panel * info.blocks_per_row + block) * info.row_tile + lane);
          const std::uint8_t* got =
              panel + destination_block * info.block_bytes;
          if (row < n) {
            const std::size_t source_block =
                static_cast<std::size_t>(row * info.blocks_per_row + block);
            REQUIRE(std::memcmp(got,
                                packed.data() + source_block * info.block_bytes,
                                info.block_bytes) == 0);
          } else {
            for (std::size_t byte = 0; byte < info.block_bytes; ++byte) {
              REQUIRE(got[byte] == 0);
            }
          }
        }
      }
    }

    for (long long m : {1LL, 3LL, 17LL, 65LL}) {
      std::vector<float> x(static_cast<std::size_t>(m * k));
      for (float& value : x) {
        value = static_cast<float>(next(&state) >> 8) / 8388608.0f - 1.0f;
      }
      std::vector<float> expected(static_cast<std::size_t>(m * n));
      std::vector<float> actual(expected.size());
      for (int threads : {1, 4}) {
        set_num_threads(threads);
        REQUIRE(qgemm(format, packed.data(), x.data(), expected.data(), m, n,
                      k) == Status::kOk);
        Workspace workspace;
        REQUIRE(qgemm_prepacked(weights, x.data(), actual.data(), m,
                                &workspace) == Status::kOk);
        for (std::size_t index = 0; index < actual.size(); ++index) {
          REQUIRE(std::fabs(expected[index] - actual[index]) <=
                  3e-5f * (1.0f + std::fabs(expected[index])));
        }
        REQUIRE(workspace.used() == 0);
        if (m >= 64) {
          const std::size_t capacity = workspace.capacity();
          REQUIRE(capacity != 0);
          REQUIRE(qgemm_prepacked(weights, x.data(), actual.data(), m,
                                  &workspace) == Status::kOk);
          REQUIRE(workspace.capacity() == capacity);
          REQUIRE(workspace.used() == 0);
        }
      }
    }
  }
  return true;
}

bool run_all_panel_formats() {
  using namespace quixicore_cpu;
  constexpr QuantFormat formats[] = {
      QuantFormat::kQ8_0,      QuantFormat::kQ4_0,
      QuantFormat::kQ4_1,      QuantFormat::kQ5_0,
      QuantFormat::kQ5_1,      QuantFormat::kU4B8,
      QuantFormat::kU4,        QuantFormat::kHQQ,
      QuantFormat::kFP8E4M3,   QuantFormat::kFP8E5M2,
      QuantFormat::kFP8Block,  QuantFormat::kFP8Raw,
      QuantFormat::kFP4E2M1,   QuantFormat::kMXFP8,
      QuantFormat::kNVFP4,     QuantFormat::kMXFP4,
      QuantFormat::kMXFP6E3M2, QuantFormat::kMXFP6E2M3,
      QuantFormat::kBitnet,    QuantFormat::kQ2_K,
      QuantFormat::kQ3_K,      QuantFormat::kQ4_K,
      QuantFormat::kQ5_K,      QuantFormat::kQ6_K,
      QuantFormat::kIQ4_NL,    QuantFormat::kIQ4_XS,
      QuantFormat::kIQ2_XXS,   QuantFormat::kIQ2_XS,
      QuantFormat::kIQ3_XXS,   QuantFormat::kIQ1_S,
      QuantFormat::kTQ2_0,
  };
  constexpr long long n = 19;
  constexpr long long k = 256;
  for (QuantFormat format : formats) {
    std::size_t packed_bytes = 0;
    REQUIRE(qgemv_packed_size(format, n, k, &packed_bytes) == Status::kOk);
    std::vector<std::uint8_t> contract(packed_bytes);
    for (std::size_t index = 0; index < contract.size(); ++index) {
      contract[index] = static_cast<std::uint8_t>((index * 131 + 17) & 0xFF);
    }
    for (CpuPanelLayout layout : {
             CpuPanelLayout::kPortableRows1, CpuPanelLayout::kNeonRows4,
             CpuPanelLayout::kAvx2Rows8, CpuPanelLayout::kAvx512Rows16}) {
      CpuPackedWeights weights;
      REQUIRE(weights.prepare(format, contract.data(), n, k, layout) ==
              Status::kOk);
      const CpuPackedWeightsInfo info = weights.info();
      REQUIRE(info.contract_bytes == packed_bytes);
      REQUIRE(std::memcmp(weights.contract_data(), contract.data(),
                          packed_bytes) == 0);
      const auto* panel =
          static_cast<const std::uint8_t*>(weights.panel_data());
      const long long panels =
          n / info.row_tile + (n % info.row_tile != 0 ? 1 : 0);
      for (long long row_panel = 0; row_panel < panels; ++row_panel) {
        for (long long block = 0; block < info.blocks_per_row; ++block) {
          for (long long lane = 0; lane < info.row_tile; ++lane) {
            const long long row = row_panel * info.row_tile + lane;
            const std::size_t destination_block = static_cast<std::size_t>(
                (row_panel * info.blocks_per_row + block) * info.row_tile +
                lane);
            const std::uint8_t* actual =
                panel + destination_block * info.block_bytes;
            if (row < n) {
              const std::size_t source_block = static_cast<std::size_t>(
                  row * info.blocks_per_row + block);
              REQUIRE(std::memcmp(
                          actual,
                          contract.data() + source_block * info.block_bytes,
                          info.block_bytes) == 0);
            } else {
              for (std::size_t byte = 0; byte < info.block_bytes; ++byte) {
                REQUIRE(actual[byte] == 0);
              }
            }
          }
        }
      }
    }

    CpuPackedWeights compute_weights;
    const bool nonzero_compute =
        format == QuantFormat::kQ4_K || format == QuantFormat::kQ5_K ||
        format == QuantFormat::kQ6_K || format == QuantFormat::kIQ4_NL ||
        format == QuantFormat::kIQ4_XS || format == QuantFormat::kIQ2_XXS ||
        format == QuantFormat::kIQ2_XS || format == QuantFormat::kIQ3_XXS ||
        format == QuantFormat::kIQ1_S;
    if (!nonzero_compute) {
      std::fill(contract.begin(), contract.end(), 0);
    } else {
      CpuPackedWeights metadata;
      REQUIRE(metadata.prepare(format, contract.data(), n, k,
                               CpuPanelLayout::kPortableRows1) == Status::kOk);
      const std::size_t block_bytes = metadata.info().block_bytes;
      for (std::size_t block = 0; block < contract.size();
           block += block_bytes) {
        const std::size_t scale_offset =
            format == QuantFormat::kQ6_K ? 208 : 0;
        contract[block + scale_offset] = 0;
        contract[block + scale_offset + 1] = 0x3c;
        if (format == QuantFormat::kQ4_K || format == QuantFormat::kQ5_K) {
          contract[block + 2] = 0;
          contract[block + 3] = 0;
        }
      }
    }
    REQUIRE(compute_weights.prepare(format, contract.data(), n, k) ==
            Status::kOk);
    constexpr long long m = 3;
    std::vector<float> x(static_cast<std::size_t>(m * k));
    for (std::size_t index = 0; index < x.size(); ++index) {
      x[index] = static_cast<float>((index * 17) % 29) / 29.0f - 0.5f;
    }
    std::vector<float> expected(static_cast<std::size_t>(m * n));
    std::vector<float> actual(expected.size());
    REQUIRE(qgemm(format, contract.data(), x.data(), expected.data(), m, n,
                  k) == Status::kOk);
    REQUIRE(qgemm_prepacked(compute_weights, x.data(), actual.data(), m) ==
            Status::kOk);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (std::fabs(expected[index] - actual[index]) >
          2e-4f * (1.0f + std::fabs(expected[index]))) {
        std::cerr << "panel compute mismatch for format "
                  << static_cast<int>(format) << " at " << index
                  << ": expected " << expected[index] << ", got "
                  << actual[index] << '\n';
        return false;
      }
    }
  }
  return true;
}

bool run_invalid() {
  using namespace quixicore_cpu;
  CpuPackedWeights weights;
  std::uint8_t byte = 0;
  REQUIRE(weights.prepare(QuantFormat::kQ4_0, nullptr, 1, 32) ==
          Status::kInvalidArgument);
  REQUIRE(weights.prepare(QuantFormat::kQ4_0, &byte, 1, 31) ==
          Status::kInvalidShape);
  REQUIRE(weights.prepare(QuantFormat::kQ4_0, &byte, 1, 32,
                          static_cast<CpuPanelLayout>(99)) ==
          Status::kInvalidArgument);
  float value = 0.0f;
  REQUIRE(qgemm_prepacked(weights, &value, &value, 1) ==
          Status::kInvalidArgument);
  return true;
}

}  // namespace

int main() {
  if (!run_format(quixicore_cpu::QuantFormat::kQ4_0) ||
      !run_format(quixicore_cpu::QuantFormat::kQ8_0) ||
      !run_all_panel_formats() || !run_invalid()) {
    return 1;
  }
  quixicore_cpu::set_num_threads(1);
  std::cout << "CPU packed-weight panel tests passed\n";
  return 0;
}
