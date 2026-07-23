#include <algorithm>
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

bool close(float actual, float expected, float tolerance = 1e-5f) {
  return std::fabs(actual - expected) <= tolerance;
}

bool equal(const float* actual, const float* expected, int count,
           float tolerance = 1e-5f) {
  for (int i = 0; i < count; ++i) {
    if (!close(actual[i], expected[i], tolerance)) return false;
  }
  return true;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  float output[128] = {};

  const float ids_x[] = {1, 2, 3, 4};
  const float ids_rows[] = {10, 20, 30, 40, 50, 60};
  const int ids[] = {2, 0};
  const float ids_expected[] = {51, 62, 13, 24};
  REQUIRE(add_id(ids_x, ids_rows, ids, output, 2, 3, 2) == Status::kOk);
  REQUIRE(equal(output, ids_expected, 4));
  REQUIRE(tensor_copy(ids_x, output, 4) == Status::kOk);
  REQUIRE(equal(output, ids_x, 4));
  const float set_base[] = {0, 1, 2, 3, 4, 5, 6, 7};
  const float set_update[] = {10, 11, 12, 13};
  REQUIRE(tensor_set_4d(set_base, set_update, output, 8, 2, 2, 1, 1,
                        4, 0, 0, 1) == Status::kOk);
  const float set_expected[] = {0, 10, 11, 3, 4, 12, 13, 7};
  REQUIRE(equal(output, set_expected, 8));

  const float image[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  REQUIRE(im2col_2d(image, output, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0) ==
          Status::kOk);
  const float columns_expected[] = {
      1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9};
  REQUIRE(equal(output, columns_expected, 16));
  float recovered[9] = {};
  REQUIRE(col2im_2d(output, recovered, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0) ==
          Status::kOk);
  const float recovered_expected[] = {1, 4, 3, 8, 20, 12, 7, 16, 9};
  REQUIRE(equal(recovered, recovered_expected, 9));
  const float columns1d[] = {1, 2, 3, 4};
  REQUIRE(col2im_1d(columns1d, output, 2, 1, 2, 1, 0) == Status::kOk);
  const float col2im1d_expected[] = {1, 5, 4};
  REQUIRE(equal(output, col2im1d_expected, 3));

  const float kernel2[] = {1, 1, 1, 1};
  const float bias[] = {1};
  REQUIRE(conv2d(image, kernel2, bias, output, 1, 1, 1, 3, 3, 2, 2) ==
          Status::kOk);
  const float conv_expected[] = {13, 17, 25, 29};
  REQUIRE(equal(output, conv_expected, 4));
  const float depthwise_input[] = {1, 2, 3, 4, 10, 20, 30, 40};
  const float depthwise_weights[] = {1, 1, 1, 1, 2, 2, 2, 2};
  REQUIRE(depthwise_conv2d(depthwise_input, depthwise_weights, nullptr,
                           output, 1, 2, 1, 2, 2, 2, 2) == Status::kOk);
  const float depthwise_expected[] = {10, 200};
  REQUIRE(equal(output, depthwise_expected, 2));

  const float volume[] = {1, 2, 3, 4, 5, 6, 7, 8};
  REQUIRE(im2col_3d(volume, output, 1, 1, 2, 2, 2, 2, 2, 2,
                    1, 1, 1, 0, 0, 0) == Status::kOk);
  REQUIRE(equal(output, volume, 8));
  const float kernel3[] = {1, 1, 1, 1, 1, 1, 1, 1};
  REQUIRE(conv3d(volume, kernel3, nullptr, output, 1, 1, 1, 2, 2, 2,
                 2, 2, 2) == Status::kOk && close(output[0], 36));

  const float transposed1_input[] = {1, 2};
  const float transposed1_weight[] = {1, 2};
  REQUIRE(conv_transpose_1d(transposed1_input, transposed1_weight, nullptr,
                            output, 1, 1, 1, 2, 2) == Status::kOk);
  const float transposed1_expected[] = {1, 4, 4};
  REQUIRE(equal(output, transposed1_expected, 3));
  const float transposed2_input[] = {1, 2, 3, 4};
  REQUIRE(conv_transpose_2d(transposed2_input, kernel2, nullptr, output,
                            1, 1, 1, 2, 2, 2, 2) == Status::kOk);
  const float transposed2_expected[] = {1, 3, 2, 4, 10, 6, 3, 7, 4};
  REQUIRE(equal(output, transposed2_expected, 9));

  const float line[] = {1, 2, 3, 4};
  REQUIRE(pool1d(line, output, 1, 1, 4, 2, 2, 0) == Status::kOk);
  REQUIRE(close(output[0], 1.5f) && close(output[1], 3.5f));
  REQUIRE(pool1d(line, output, 1, 1, 4, 2, 2, 0, PoolMode::kMaximum) ==
          Status::kOk);
  REQUIRE(close(output[0], 2) && close(output[1], 4));
  REQUIRE(pool2d(transposed2_input, output, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0) ==
          Status::kOk && close(output[0], 2.5f));
  REQUIRE(pool2d(transposed2_input, output, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0,
                 PoolMode::kMaximum) == Status::kOk && close(output[0], 4));
  const float gradient[] = {1};
  float grad_in[4] = {};
  REQUIRE(pool2d_backward(transposed2_input, gradient, grad_in, 1, 1, 2, 2,
                          2, 2, 1, 1, 0, 0) == Status::kOk);
  const float average_grad[] = {.25f, .25f, .25f, .25f};
  REQUIRE(equal(grad_in, average_grad, 4));
  REQUIRE(pool2d_backward(transposed2_input, gradient, grad_in, 1, 1, 2, 2,
                          2, 2, 1, 1, 0, 0, PoolMode::kMaximum) == Status::kOk);
  const float maximum_grad[] = {0, 0, 0, 1};
  REQUIRE(equal(grad_in, maximum_grad, 4));

  const float timesteps[] = {0};
  REQUIRE(timestep_embedding(timesteps, output, 1, 4) == Status::kOk);
  const float timestep_expected[] = {1, 1, 0, 0};
  REQUIRE(equal(output, timestep_expected, 4));
  const float triangular[] = {2, 0, 3, 1};
  const float rhs[] = {4, 5};
  REQUIRE(solve_lower_triangular(triangular, rhs, output, 1, 2, 1) ==
          Status::kOk);
  REQUIRE(close(output[0], 2) && close(output[1], -1));

  const float relative_table[] = {10, 20, 30};
  REQUIRE(get_relative_position(relative_table, output, 2, 1) == Status::kOk);
  const float relative_expected[] = {20, 30, 10, 20};
  REQUIRE(equal(output, relative_expected, 4));
  const float attention[] = {1, 2, 3, 4};
  const float relative_height[] = {10, 20};
  const float relative_width[] = {100, 200};
  REQUIRE(add_relative_position_2d(attention, relative_height, relative_width,
                                   output, 1, 1, 1, 2, 2) == Status::kOk);
  const float added_relative_expected[] = {111, 212, 123, 224};
  REQUIRE(equal(output, added_relative_expected, 4));

  const float window_image[] = {1, 2, 3, 4, 5, 6};
  float windows[16] = {};
  REQUIRE(window_partition(window_image, windows, 2, 3, 1, 2) == Status::kOk);
  const float windows_expected[] = {1, 2, 4, 5, 3, 0, 6, 0};
  REQUIRE(equal(windows, windows_expected, 8));
  std::fill_n(output, 6, 0.0f);
  REQUIRE(window_unpartition(windows, output, 2, 3, 1, 2) == Status::kOk);
  REQUIRE(equal(output, window_image, 6));

  REQUIRE(get_relative_position(relative_table, output, -1, 1) ==
          Status::kInvalidShape);
  REQUIRE(conv_transpose_1d(transposed1_input, transposed1_weight, nullptr,
                            output, 1, 1, 1, 2, 2, 1, 3) ==
          Status::kInvalidShape);
  REQUIRE(add_id(ids_x, ids_rows, ids, output, 2, 2, 2) ==
          Status::kInvalidArgument);
  return 0;
}
