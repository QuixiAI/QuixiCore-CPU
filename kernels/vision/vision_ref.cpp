#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

void normalize_row(float* row, const float* weight, const float* bias,
                   long long width, float eps) {
  double mean = 0.0;
  for (long long i = 0; i < width; ++i) mean += row[i];
  mean /= width;
  double variance = 0.0;
  for (long long i = 0; i < width; ++i) {
    const double delta = row[i] - mean;
    variance += delta * delta;
  }
  const double inverse = 1.0 / std::sqrt(variance / width + eps);
  for (long long i = 0; i < width; ++i) {
    row[i] = static_cast<float>((row[i] - mean) * inverse * weight[i] +
                                (bias == nullptr ? 0.0f : bias[i]));
  }
}

float gelu_erf(float x) {
  return static_cast<float>(0.5 * x *
                            (1.0 + std::erf(x / std::sqrt(2.0))));
}

}  // namespace

Status patch_merge_layer_norm(const float* input, const float* weight,
                              const float* bias, float* out, long long batch,
                              long long height, long long width,
                              long long channels, float eps) {
  if (!detail::valid_product({batch, height, width, channels}) ||
      !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, weight, bias, out)) {
    return Status::kInvalidArgument;
  }
  const long long out_height = (height + 1) / 2;
  const long long out_width = (width + 1) / 2;
  const long long features = 4 * channels;
  threading::parallel_ranges(batch * out_height * out_width, 1,
                             [&](long long begin, long long end, int) {
    for (long long patch = begin; patch < end; ++patch) {
      const long long item = patch / (out_height * out_width);
      const long long spatial = patch % (out_height * out_width);
      const long long oy = spatial / out_width;
      const long long ox = spatial % out_width;
      float* destination = out + patch * features;
      for (long long by = 0; by < 2; ++by) {
        for (long long bx = 0; bx < 2; ++bx) {
          const long long source_y = 2 * oy + by;
          const long long source_x = 2 * ox + bx;
          float* block = destination + (by * 2 + bx) * channels;
          if (source_y < height && source_x < width) {
            const float* source =
                input + ((item * height + source_y) * width + source_x) *
                            channels;
            std::copy_n(source, channels, block);
          } else {
            std::fill_n(block, channels, 0.0f);
          }
        }
      }
      normalize_row(destination, weight, bias, features, eps);
    }
  });
  return Status::kOk;
}

Status space_to_depth_norm_linear(
    const float* input, const float* norm_weight, const float* norm_bias,
    const float* projection_weight, const float* projection_bias, float* out,
    long long batch, long long height, long long width, long long channels,
    long long out_channels, long long block_size, float eps) {
  if (!detail::valid_product({batch, height, width, channels}) ||
      out_channels <= 0 || block_size <= 0 || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input, norm_weight, projection_weight, out)) {
    return Status::kInvalidArgument;
  }
  const long long out_height = (height + block_size - 1) / block_size;
  const long long out_width = (width + block_size - 1) / block_size;
  const long long features = block_size * block_size * channels;
  threading::parallel_ranges(batch * out_height * out_width, 1,
                             [&](long long begin, long long end, int) {
    std::vector<float> patch(static_cast<std::size_t>(features));
    for (long long index = begin; index < end; ++index) {
      const long long item = index / (out_height * out_width);
      const long long spatial = index % (out_height * out_width);
      const long long oy = spatial / out_width;
      const long long ox = spatial % out_width;
      for (long long by = 0; by < block_size; ++by) {
        for (long long bx = 0; bx < block_size; ++bx) {
          const long long iy = oy * block_size + by;
          const long long ix = ox * block_size + bx;
          float* destination =
              patch.data() + (by * block_size + bx) * channels;
          if (iy < height && ix < width) {
            std::copy_n(input + ((item * height + iy) * width + ix) * channels,
                        channels, destination);
          } else {
            std::fill_n(destination, channels, 0.0f);
          }
        }
      }
      normalize_row(patch.data(), norm_weight, norm_bias, features, eps);
      for (long long output = 0; output < out_channels; ++output) {
        double accumulator =
            projection_bias == nullptr ? 0.0 : projection_bias[output];
        const float* projection = projection_weight + output * features;
        for (long long feature = 0; feature < features; ++feature) {
          accumulator += double(projection[feature]) * patch[feature];
        }
        out[index * out_channels + output] = static_cast<float>(accumulator);
      }
    }
  });
  return Status::kOk;
}

Status edge_mlp_256x7(const float* hidden, const float* first_weight,
                      const float* first_bias, const float* second_weight,
                      const float* second_bias, float* out, long long batch,
                      long long length) {
  constexpr long long kFeatures = 256;
  if (!detail::valid_product({batch, length, kFeatures})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(hidden, first_weight, first_bias, second_weight,
                           second_bias, out)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> left(static_cast<std::size_t>(batch * length * kFeatures));
  std::vector<float> right(left.size());
  threading::parallel_ranges(batch * length, 1,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* source = hidden + row * kFeatures;
      for (long long feature = 0; feature < kFeatures; ++feature) {
        const float* weights = first_weight + feature * 512;
        double lhs = 0.0;
        double rhs = first_bias[feature];
        for (long long input = 0; input < kFeatures; ++input) {
          lhs += double(source[input]) * weights[input];
          rhs += double(source[input]) * weights[kFeatures + input];
        }
        left[row * kFeatures + feature] = static_cast<float>(lhs);
        right[row * kFeatures + feature] = static_cast<float>(rhs);
      }
    }
  });
  threading::parallel_ranges(batch * length * length, 1,
                             [&](long long begin, long long end, int) {
    for (long long pair = begin; pair < end; ++pair) {
      const long long item = pair / (length * length);
      const long long pair_index = pair % (length * length);
      const long long lhs_index = pair_index / length;
      const long long rhs_index = pair_index % length;
      const float* lhs =
          left.data() + (item * length + lhs_index) * kFeatures;
      const float* rhs =
          right.data() + (item * length + rhs_index) * kFeatures;
      for (long long edge_class = 0; edge_class < 7; ++edge_class) {
        double accumulator = second_bias[edge_class];
        const float* weights = second_weight + edge_class * kFeatures;
        for (long long feature = 0; feature < kFeatures; ++feature) {
          accumulator += weights[feature] * gelu_erf(lhs[feature] + rhs[feature]);
        }
        out[(item * 7 + edge_class) * length * length + pair_index] =
            static_cast<float>(accumulator);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
