#include <cmath>
#include <iostream>

#include "quixicore_cpu/ops.h"

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ":"      \
                << __LINE__ << '\n';                                      \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

bool close(float actual, float expected, float tolerance = 2e-5f) {
  return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  float output[32] = {};
  float state[8] = {};

  const float initial[] = {1, 2, 3, 4};
  const float key[] = {2, 3};
  const float value[] = {5, 7};
  const float query[] = {11, 13};
  const float gate[] = {.1f, .2f};
  REQUIRE(gated_linear_attention(key, value, query, gate, initial, output,
                                  state, 1, 1, 1, 2, .5f) == Status::kOk);
  // state rows: [.1, .2] + 2*[5,7], [.6,.8] + 3*[5,7]
  REQUIRE(close(state[0], 10.1f) && close(state[1], 14.2f) &&
          close(state[2], 15.6f) && close(state[3], 21.8f));
  REQUIRE(close(output[0], 156.95f) && close(output[1], 219.8f));

  const float receptance6[] = {11, 13};
  const float first[] = {.5f, .25f};
  const float decay6[] = {.1f, .2f};
  REQUIRE(rwkv_wkv6(key, value, receptance6, first, decay6, initial, output,
                    state, 1, 1, 1, 2) == Status::kOk);
  REQUIRE(close(output[0], 153.75f) && close(output[1], 219.25f));
  REQUIRE(close(state[0], 10.1f) && close(state[1], 14.2f) &&
          close(state[2], 15.6f) && close(state[3], 21.8f));

  const float receptance7[] = {2, 3};
  const float decay7[] = {.5f, .25f};
  const float key7[] = {4, 5};
  const float value7[] = {6, 7};
  const float a[] = {.1f, .2f};
  const float b[] = {.3f, .4f};
  REQUIRE(rwkv_wkv7(receptance7, decay7, key7, value7, a, b, initial,
                    output, state, 1, 1, 1, 2) == Status::kOk);
  REQUIRE(close(state[0], 24.65f) && close(state[1], 30.7f) &&
          close(state[2], 29.83f) && close(state[3], 36.44f));
  REQUIRE(close(output[0], 141.4f) && close(output[1], 168.98f));

  float mixes[24] = {};
  const float scale[] = {0, 0, 1};
  float base[24] = {};
  float comb[16] = {};
  REQUIRE(dsv4_hc_comb(mixes, scale, base, comb, 1, 0.0f, 2) == Status::kOk);
  for (float coefficient : comb) REQUIRE(close(coefficient, .25f));
  const float hyper[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float pre_weights[] = {1, 0, 0, 1};
  REQUIRE(dsv4_hc_pre(hyper, pre_weights, output, 1, 2) == Status::kOk);
  REQUIRE(close(output[0], 8) && close(output[1], 10));
  const float post_weights[] = {1, 2, 3, 4};
  const float pre_output[] = {10, 20};
  REQUIRE(dsv4_hc_post(pre_output, hyper, post_weights, comb, output, 1, 2) ==
          Status::kOk);
  const float post_expected[] = {14, 25, 24, 45, 34, 65, 44, 85};
  for (int i = 0; i < 8; ++i) REQUIRE(close(output[i], post_expected[i]));

  REQUIRE(gated_linear_attention(key, value, query, gate, initial, output,
                                  state, 0, 1, 1, 2) ==
          Status::kInvalidShape);
  REQUIRE(dsv4_hc_comb(mixes, scale, base, comb, 1, 0.0f, 0) ==
          Status::kInvalidShape);
  return 0;
}
