#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "quixicore_cpu/ops.h"

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ":"      \
                << __LINE__ << '\n';                                      \
      return 1;                                                           \
    }                                                                     \
  } while (0)

bool close(float actual, float expected, float tolerance = 1e-5f) {
  return std::fabs(actual - expected) <= tolerance;
}

int main() {
  using namespace quixicore_cpu;
  const float x[] = {-2.0f, -1.0f, 0.5f, 2.0f};
  const float y[] = {4.0f, 2.0f, 0.5f, -1.0f};
  float out[16] = {};
  REQUIRE(add_scalar(x, 1.0f, out, 4) == Status::kOk && close(out[0], -1.0f));
  REQUIRE(subtract(x, y, out, 4) == Status::kOk && close(out[3], 3.0f));
  REQUIRE(multiply(x, y, out, 4) == Status::kOk && close(out[1], -2.0f));
  REQUIRE(divide(x, y, out, 4) == Status::kOk && close(out[2], 1.0f));
  REQUIRE(scale(x, 2.0f, out, 4) == Status::kOk && close(out[3], 4.0f));
  REQUIRE(clamp(x, -1.0f, 1.0f, out, 4) == Status::kOk &&
          close(out[0], -1.0f) && close(out[3], 1.0f));
  REQUIRE(square(x, out, 4) == Status::kOk && close(out[0], 4.0f));
  const float positive[] = {1.0f, 4.0f, std::exp(1.0f), 9.0f};
  REQUIRE(square_root(positive, out, 4) == Status::kOk && close(out[1], 2.0f));
  REQUIRE(logarithm(positive, out, 4) == Status::kOk && close(out[2], 1.0f));
  const float angles[] = {0.0f, 1.57079632679f};
  REQUIRE(sine(angles, out, 2) == Status::kOk && close(out[1], 1.0f));
  REQUIRE(cosine(angles, out, 2) == Status::kOk && close(out[0], 1.0f));
  REQUIRE(leaky_relu(x, 0.1f, out, 4) == Status::kOk && close(out[0], -0.2f));
  REQUIRE(fill(out, 4, 3.0f) == Status::kOk && close(out[3], 3.0f));
  REQUIRE(arange(-1.0f, 0.5f, out, 4) == Status::kOk && close(out[3], 0.5f));

  const float matrix[] = {1, 2, 3, 4, 5, 6};
  REQUIRE(cumulative_sum(matrix, out, 2, 3) == Status::kOk &&
          close(out[2], 6.0f) && close(out[5], 15.0f));
  REQUIRE(reduce_sum_all(matrix, out, 6) == Status::kOk && close(out[0], 21));
  REQUIRE(reduce_mean(matrix, out, 2, 3) == Status::kOk &&
          close(out[0], 2.0f) && close(out[1], 5.0f));
  const std::int32_t xi[] = {1, 2, 3, 4};
  const std::int32_t yi[] = {1, 0, 3, 0};
  long long equal = 0;
  REQUIRE(count_equal(xi, yi, 4, &equal) == Status::kOk && equal == 2);
  const float unsorted[] = {3, 1, 2, 0, 5, 4};
  int order[6] = {};
  REQUIRE(argsort(unsorted, order, 2, 3) == Status::kOk &&
          order[0] == 1 && order[1] == 2 && order[2] == 0 &&
          order[3] == 0 && order[4] == 2 && order[5] == 1);

  const float a[] = {1, 2, 3, 4};
  const float b[] = {5, 6};
  REQUIRE(concat(a, b, out, 1, 2, 1, 2) == Status::kOk);
  const float concat_expected[] = {1, 2, 3, 4, 5, 6};
  REQUIRE(std::equal(out, out + 6, concat_expected));
  const float repeat_source[] = {1, 2, 3, 4};
  REQUIRE(repeat_2d(repeat_source, out, 2, 2, 4, 4) == Status::kOk &&
          close(out[0], 1) && close(out[3], 2) && close(out[12], 3));
  float repeat_grad[4] = {};
  std::fill_n(out, 16, 1.0f);
  REQUIRE(repeat_backward_2d(out, repeat_grad, 2, 2, 4, 4) == Status::kOk &&
          close(repeat_grad[0], 4.0f) && close(repeat_grad[3], 4.0f));

  const float diagonal[] = {2, 3};
  float square_matrix[4] = {};
  REQUIRE(diag_embed(diagonal, square_matrix, 1, 2) == Status::kOk &&
          close(square_matrix[0], 2) && close(square_matrix[3], 3) &&
          close(square_matrix[1], 0));
  const float mask_source[] = {1, 2, 3, 4, 5, 6};
  float mask_out[6] = {};
  REQUIRE(diag_mask(mask_source, mask_out, 2, 3, 0, false) == Status::kOk &&
          close(mask_out[1], 0) && close(mask_out[4], 5) && close(mask_out[5], 0));
  REQUIRE(triangular_fill(mask_source, mask_out, 2, 3, 0, true, -1.0f) ==
          Status::kOk);
  REQUIRE(close(mask_out[0], 1) && close(mask_out[3], -1) && close(mask_out[4], 5));
  REQUIRE(roll_2d(mask_source, mask_out, 2, 3, 1, -1) == Status::kOk &&
          close(mask_out[0], 5) && close(mask_out[5], 1));
  float padded[20] = {};
  REQUIRE(pad_2d(mask_source, padded, 2, 3, 1, 1, 1, 1, -1.0f) == Status::kOk &&
          close(padded[6], 1) && close(padded[13], 6));
  const float reflect_source[] = {1, 2, 3};
  float reflected[5] = {};
  REQUIRE(pad_reflect_1d(reflect_source, reflected, 1, 3, 1, 1) == Status::kOk);
  const float reflected_expected[] = {2, 1, 2, 3, 2};
  REQUIRE(std::equal(reflected, reflected + 5, reflected_expected));
  const float pixel[] = {1, 2, 3, 4};
  float upscaled[16] = {};
  REQUIRE(upscale_nearest_2d(pixel, upscaled, 1, 2, 2, 2, 2) == Status::kOk &&
          close(upscaled[0], 1) && close(upscaled[3], 2) && close(upscaled[15], 4));

  const float norm_input[] = {1, 2, 3, 4};
  float normalized[4] = {};
  REQUIRE(group_norm(norm_input, nullptr, nullptr, normalized, 1, 2, 2, 1) ==
          Status::kOk);
  float mean = 0.0f;
  for (float value : normalized) mean += value;
  REQUIRE(close(mean, 0.0f, 1e-4f));
  REQUIRE(l2_normalize(norm_input, normalized, 1, 4) == Status::kOk);
  float norm = 0.0f;
  for (float value : normalized) norm += value * value;
  REQUIRE(close(norm, 1.0f, 1e-5f));

  const float probabilities[] = {0.2f, 0.3f, 0.5f};
  const float grad[] = {1.0f, 2.0f, 4.0f};
  float grad_in[3] = {};
  REQUIRE(softmax_backward(grad, probabilities, grad_in, 1, 3) == Status::kOk);
  REQUIRE(close(grad_in[0] + grad_in[1] + grad_in[2], 0.0f, 1e-6f));
  std::vector<float> rope_input(8), rope_output(8), rope_recovered(8);
  for (int i = 0; i < 8; ++i) rope_input[i] = static_cast<float>(i + 1);
  REQUIRE(rope(rope_input.data(), rope_output.data(), 1, 1, 8, 10000.0f, 3) ==
          Status::kOk);
  REQUIRE(rope_backward(rope_output.data(), rope_recovered.data(), 1, 1, 8,
                        10000.0f, 3) == Status::kOk);
  for (int i = 0; i < 8; ++i) REQUIRE(close(rope_recovered[i], rope_input[i], 2e-5f));

  const float ox[] = {2, 3};
  const float oy[] = {4, 5, 6};
  float outer[6] = {};
  REQUIRE(outer_product(ox, oy, outer, 2, 3) == Status::kOk &&
          close(outer[0], 8) && close(outer[5], 18));
  float destination[] = {0, 0, 0, 0, 0, 0};
  const float rows[] = {1, 2, 3, 4};
  const int ids[] = {2, 0};
  REQUIRE(set_rows(rows, ids, destination, 2, 3, 2) == Status::kOk &&
          close(destination[4], 1) && close(destination[0], 3));
  REQUIRE(accumulate(destination, rows, 4, 0.5f) == Status::kOk &&
          close(destination[0], 3.5f));
  float parameters[] = {1, 2};
  const float gradients[] = {0.5f, -0.5f};
  REQUIRE(sgd(parameters, gradients, 2, 0.1f, 0.0f) == Status::kOk &&
          close(parameters[0], 0.95f) && close(parameters[1], 2.05f));

  REQUIRE(clamp(x, 2.0f, 1.0f, out, 4) == Status::kInvalidShape);
  REQUIRE(repeat_2d(x, out, 1, 4, 2, 7) == Status::kInvalidShape);
  REQUIRE(roll_2d(destination, destination, 2, 2, 1, 1) ==
          Status::kInvalidArgument);
  return 0;
}
