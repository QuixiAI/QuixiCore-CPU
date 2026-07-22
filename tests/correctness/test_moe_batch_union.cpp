#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <=
         2e-5f * std::max(1.0f, std::fabs(expected));
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  constexpr long long rows = 7;
  constexpr long long experts = 3;
  constexpr long long input_dim = 32;
  constexpr long long output_dim = 9;
  constexpr long long intermediate = 5;
  const int expert_ids[] = {2, 0, 2, 1, 2, 0, 2};
  std::vector<float> x(rows * input_dim);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 41) - 20) /
           23.0f;
  }

  auto check_projection = [&](long long outputs, bool swiglu) {
    std::vector<float> weights(experts * outputs * input_dim);
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i] =
          static_cast<float>(static_cast<int>((i * 13 + 3) % 37) - 18) /
          31.0f;
    }
    std::size_t expert_bytes = 0;
    if (qgemv_packed_size(QuantFormat::kQ8_0, outputs, input_dim,
                          &expert_bytes) != Status::kOk) {
      return false;
    }
    std::vector<std::uint8_t> packed(experts * expert_bytes);
    for (long long expert = 0; expert < experts; ++expert) {
      if (qgemv_pack(QuantFormat::kQ8_0,
                     weights.data() + expert * outputs * input_dim, outputs,
                     input_dim, packed.data() + expert * expert_bytes) !=
          Status::kOk) {
        return false;
      }
    }
    std::vector<float> got(rows * (swiglu ? intermediate : outputs));
    const Status status = swiglu
        ? moe_grouped_qswiglu(
              x.data(), packed.data(), expert_ids, nullptr, got.data(), rows,
              experts, input_dim, intermediate, QuantFormat::kQ8_0, false)
        : moe_grouped_qgemm(x.data(), packed.data(), expert_ids, nullptr,
                            got.data(), rows, experts, input_dim, outputs,
                            QuantFormat::kQ8_0, false);
    if (status != Status::kOk) return false;
    std::vector<float> projection(static_cast<std::size_t>(outputs));
    for (long long row = 0; row < rows; ++row) {
      if (qgemv(QuantFormat::kQ8_0,
                 packed.data() + expert_ids[row] * expert_bytes,
                 x.data() + row * input_dim, projection.data(), outputs,
                 input_dim) != Status::kOk) {
        return false;
      }
      if (swiglu) {
        for (long long item = 0; item < intermediate; ++item) {
          const float gate = projection[item];
          const float expected = gate / (1.0f + std::exp(-gate)) *
                                 projection[intermediate + item];
          if (!close(got[row * intermediate + item], expected)) return false;
        }
      } else {
        for (long long output = 0; output < outputs; ++output) {
          if (!close(got[row * outputs + output], projection[output])) {
            return false;
          }
        }
      }
    }
    return true;
  };

  if (!check_projection(output_dim, false) ||
      !check_projection(2 * intermediate, true)) {
    std::cerr << "FAIL: MoE batch-union projection\n";
    return 1;
  }
  std::cout << "MoE batch-union tests passed\n";
  return 0;
}
