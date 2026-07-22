#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <=
         2e-6f * std::max(1.0f, std::fabs(expected));
}

double uniform01(std::uint32_t seed, std::uint64_t row) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^ row ^
                    0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) *
         (1.0 / 9007199254740992.0);
}

int sorted_top_p_oracle(const float* logits, int vocab, float p,
                        float temperature, std::uint32_t seed) {
  std::vector<int> ids(static_cast<std::size_t>(vocab));
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
    return logits[lhs] == logits[rhs] ? lhs < rhs : logits[lhs] > logits[rhs];
  });
  const float maximum = logits[ids.front()];
  std::vector<double> weights(static_cast<std::size_t>(vocab));
  double total = 0.0;
  for (int token = 0; token < vocab; ++token) {
    weights[static_cast<std::size_t>(token)] =
        std::exp((static_cast<double>(logits[token]) - maximum) / temperature);
    total += weights[static_cast<std::size_t>(token)];
  }
  double cumulative = 0.0;
  std::size_t keep = 0;
  do {
    cumulative += weights[static_cast<std::size_t>(ids[keep++])];
  } while (keep < ids.size() && cumulative < p * total);
  ids.resize(keep);
  double kept = 0.0;
  for (int id : ids) kept += weights[static_cast<std::size_t>(id)];
  const double target = uniform01(seed, 0) * kept;
  cumulative = 0.0;
  for (int id : ids) {
    cumulative += weights[static_cast<std::size_t>(id)];
    if (target < cumulative) return id;
  }
  return ids.back();
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;

  const float input[] = {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6};
  float output[12] = {};
  constexpr float base = 10000.0f;
  ok &= require(rope_interleaved_to_split(input, output, 2, 1, 6, base, 3) ==
                    Status::kOk,
                "interleaved-to-split RoPE status");
  for (int token = 0; token < 2; ++token) {
    for (int pair = 0; pair < 3; ++pair) {
      const double angle = (3 + token) * std::pow(10000.0, -2.0 * pair / 6.0);
      const float a = input[token * 6 + pair * 2];
      const float b = input[token * 6 + pair * 2 + 1];
      ok &= require(close(output[token * 6 + pair],
                          a * std::cos(angle) - b * std::sin(angle)) &&
                        close(output[token * 6 + 3 + pair],
                              b * std::cos(angle) + a * std::sin(angle)),
                    "interleaved-to-split RoPE value");
    }
  }
  float in_place[12];
  std::copy_n(input, 12, in_place);
  ok &= require(rope_interleaved_to_split(in_place, in_place, 2, 1, 6, base, 3) ==
                        Status::kOk &&
                    std::equal(in_place, in_place + 12, output),
                "interleaved-to-split RoPE in-place");

  const float scores[] = {3, 1, 5, 5, 2, 5, -1, 4,
                          0, 7, 2, 6, 6, 1, 6, 3};
  int selected[10] = {};
  ok &= require(threshold_topk_indices(scores, selected, 2, 8, 5) ==
                    Status::kOk,
                "threshold top-k status");
  for (int row = 0; row < 2; ++row) {
    std::vector<float> sorted(scores + row * 8, scores + (row + 1) * 8);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float threshold = sorted[4];
    std::vector<int> expected;
    for (int column = 0; column < 8; ++column) {
      if (scores[row * 8 + column] > threshold) expected.push_back(column);
    }
    for (int column = 0; column < 8 && expected.size() < 5; ++column) {
      if (scores[row * 8 + column] == threshold) expected.push_back(column);
    }
    for (int item = 0; item < 5; ++item) {
      ok &= require(selected[row * 5 + item] == expected[item],
                    "threshold top-k value");
    }
  }

  std::vector<float> logits(257);
  for (int token = 0; token < 257; ++token) {
    logits[static_cast<std::size_t>(token)] =
        static_cast<float>((token * 73 + 19) % 101 - 50) / 11.0f;
  }
  logits[7] = logits[3];
  logits[91] = -std::numeric_limits<float>::infinity();
  for (float p : {0.01f, 0.37f, 0.9f, 1.0f}) {
    for (float temperature : {0.5f, 1.3f}) {
      for (std::uint32_t seed : {0u, 1u, 17u, 0xffffffffu}) {
        int actual = -1;
        ok &= require(top_p_sample(logits.data(), &actual, 1, logits.size(), p,
                                   temperature, seed) == Status::kOk &&
                          actual == sorted_top_p_oracle(
                                        logits.data(), logits.size(), p,
                                        temperature, seed),
                      "heap top-p matches sorted oracle");
      }
    }
  }

  if (!ok) return 1;
  std::cout << "Colibri-derived algorithm tests passed\n";
  return 0;
}
